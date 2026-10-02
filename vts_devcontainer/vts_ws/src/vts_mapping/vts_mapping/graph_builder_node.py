"""Graph builder ROS2 node (dataset-agnostic).

Subscribes to standard topics with an explicit additive-budget adapter contract:

- ``/camera/image``           sensor_msgs/Image
- ``/odom``                   nav_msgs/Odometry  (carrying path_length_budget_v1)
- ``/ground_truth_pose``      geometry_msgs/PoseStamped (optional, attached
                              to nodes for evaluation only — never used by
                              the mapping algorithm)
- ``/dataset/sequence_done``  std_msgs/String    (JSON; end-of-run signal,
                              replacing the old 20 s image timeout)

For each completed sequence the resulting graph is saved to
``<output_dir>/graph_<i>.pkl`` and a JSON event is published on
``/mapping/graph_ready``. After the last sequence it publishes
``{"all_done": true}`` and exits cleanly.
"""

from __future__ import annotations

import io
import json
import os
import resource
import signal
import sys
import time
from contextlib import suppress
from pathlib import Path

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from vts_core.artifacts import atomic_bytes, atomic_json, file_digest
from vts_core.features import FeatureExtractor, build_extractor
from vts_core.mapper import TopologicalMapper
from vts_core.metrics import map_footprint
from vts_core.motion import Pose2D


def _yaw_from_quaternion(z: float, w: float) -> float:
    return float(2.0 * np.arctan2(z, w))


class GraphBuilderNode(Node):
    """Builds one topological graph per incoming sequence."""

    def __init__(self) -> None:
        super().__init__("graph_builder")

        self.declare_parameter("window_size", 30)
        self.declare_parameter("valley_k", 2.0)
        self.declare_parameter("valley_mode", "adaptive")
        self.declare_parameter("valley_delta", 0.1)
        self.declare_parameter("visual_outlier_k", 2.0)
        self.declare_parameter("uncertainty_contract", "")
        self.declare_parameter("descriptor_cache_dir", "output/cache/descriptors")
        self.declare_parameter("run_fingerprint", "unmanaged")
        if self.get_parameter("uncertainty_contract").value != "path_length_budget_v1":
            raise ValueError(
                "declare uncertainty_contract=path_length_budget_v1; ordinary pose marginals are unsupported"
            )
        self.declare_parameter("gate_mode", "both")
        self.declare_parameter("naive_threshold", 0.7)
        self.declare_parameter("visual_model", "cosine")
        self.declare_parameter("sequence_policy", "self_only")
        self.declare_parameter("require_visual_verification", False)
        self.declare_parameter("recovery_mode", 'off')
        self.declare_parameter("recovery_top_k", 5)
        self.declare_parameter("recovery_horizon", 2)
        self.declare_parameter("recovery_residual_m", 0.75)
        self.declare_parameter("recovery_min_span_m", 1.0)
        self.declare_parameter("recovery_min_turn_deg", 15.0)
        self.declare_parameter("recovery_visual_z", 2.0)
        self.declare_parameter("geometric_gate_scale", 1.0)
        self.declare_parameter("verification_threshold", 0.0)
        self.declare_parameter("defer_nodes", 2)
        self.declare_parameter("fixed_concentration", 384.0)
        self.declare_parameter("dino_model", "dinov3_vits16")
        self.declare_parameter("dino_device", "auto")
        self.declare_parameter("feature_backend", "dino_cls")
        self.declare_parameter("dino_layer", -1)
        self.declare_parameter("use_ground_truth", False)
        self.declare_parameter("use_frame_labels", False)
        self.declare_parameter("output_dir", "output/graphs")
        self.declare_parameter("run_name", "run")
        # Exit the process once the last sequence is done (lets scripted runs
        # like run_experiments.sh proceed without a manual Ctrl-C; the launch
        # file shuts the whole pipeline down when this node exits).
        self.declare_parameter("exit_when_done", True)

        self._window_size: int = (
            self.get_parameter("window_size").get_parameter_value().integer_value
        )
        self._valley_k: float = self.get_parameter("valley_k").get_parameter_value().double_value
        self._valley_mode = str(self.get_parameter("valley_mode").value)
        self._valley_delta = float(self.get_parameter("valley_delta").value)
        self._visual_outlier_k: float = (
            self.get_parameter("visual_outlier_k").get_parameter_value().double_value
        )
        self._gate_mode: str = self.get_parameter("gate_mode").get_parameter_value().string_value
        self._naive_threshold: float = (
            self.get_parameter("naive_threshold").get_parameter_value().double_value
        )
        self._visual_model = str(self.get_parameter("visual_model").value)
        dino_model: str = self.get_parameter("dino_model").get_parameter_value().string_value
        dino_device: str = self.get_parameter("dino_device").get_parameter_value().string_value
        feature_backend: str = (
            self.get_parameter("feature_backend").get_parameter_value().string_value
        )
        dino_layer: int = self.get_parameter("dino_layer").get_parameter_value().integer_value
        self._feature_backend = feature_backend
        self._dino_model = dino_model
        self._dino_layer = dino_layer
        self._use_ground_truth: bool = (
            self.get_parameter("use_ground_truth").get_parameter_value().bool_value
        )
        self._output_dir: str = self.get_parameter("output_dir").get_parameter_value().string_value
        self._run_name: str = self.get_parameter("run_name").get_parameter_value().string_value
        self._exit_when_done: bool = (
            self.get_parameter("exit_when_done").get_parameter_value().bool_value
        )
        self._shutdown_timer = None
        os.makedirs(self._output_dir, exist_ok=True)

        extractor_start: float = time.perf_counter()
        self._extractor: FeatureExtractor = build_extractor(
            dino_model,
            device=dino_device,
            backend=feature_backend,
            layer=dino_layer,
            cache_dir=str(self.get_parameter("descriptor_cache_dir").value) or None,
        )
        self._extractor_init_s: float = time.perf_counter() - extractor_start
        self.get_logger().info(
            f"Visual descriptor: {feature_backend}, model={dino_model}, "
            f"layer={dino_layer}, device={self._extractor.device} "
            f"(requested: {dino_device})"
        )
        self._bridge: CvBridge = CvBridge()

        self._sequence_index: int = 0
        self._processed: int = 0
        self._callback_times: list[float] = []
        self._first_frame_time: float | None = None
        self._cache_start = (0, 0)
        self._extraction_time_s: float = 0.0
        self._mapper: TopologicalMapper = self._new_mapper()
        self._latest_gt: Pose2D | None = None
        self._latest_label: str | None = None
        self._frame_labels: dict[str, str | None] = {}
        self._pending_sample: tuple | None = None
        self._use_frame_labels = bool(self.get_parameter("use_frame_labels").value)
        self._node_gt: dict[int, Pose2D] = {}

        # Create the acknowledgement publisher before exposing input
        # subscriptions. Once the dataset player observes those subscriptions,
        # this publisher is guaranteed to exist for the first processed frame.
        self._frame_processed_pub = self.create_publisher(String, "/mapping/frame_processed", 10)
        image_sub: Subscriber = Subscriber(self, Image, "/camera/image")
        odom_sub: Subscriber = Subscriber(self, Odometry, "/odom")
        synchronized_subscribers = [image_sub, odom_sub]
        if self._use_ground_truth:
            synchronized_subscribers.append(Subscriber(self, PoseStamped, "/ground_truth_pose"))
        self._synchronizer: ApproximateTimeSynchronizer = ApproximateTimeSynchronizer(
            synchronized_subscribers, queue_size=30, slop=0.05
        )
        self._synchronizer.registerCallback(self._on_frame)

        if not self._use_ground_truth:
            self.create_subscription(PoseStamped, "/ground_truth_pose", self._on_ground_truth, 10)
        self.create_subscription(String, "/dataset/room_label", self._on_room_label, 10)
        self.create_subscription(String, "/dataset/sequence_done", self._on_sequence_done, 10)
        self._ready_pub = self.create_publisher(String, "/mapping/graph_ready", 10)
        # Offline dataset players may use this acknowledgement for lossless
        # flow control. Live robots can ignore it; mapping input remains the
        # same standard image and odometry topics.

    # ------------------------------------------------------------------ #
    def _new_mapper(self) -> TopologicalMapper:
        return TopologicalMapper(
            window_size=self._window_size,
            valley_k=self._valley_k,
            valley_mode=self._valley_mode,
            valley_delta=self._valley_delta,
            visual_outlier_k=self._visual_outlier_k,
            gate_mode=self._gate_mode,
            naive_threshold=self._naive_threshold,
            visual_model=self._visual_model,
            sequence_policy=str(self.get_parameter("sequence_policy").value),
            require_visual_verification=bool(self.get_parameter("require_visual_verification").value),
            recovery_mode=str(self.get_parameter("recovery_mode").value),
            recovery_top_k=int(self.get_parameter("recovery_top_k").value),
            recovery_horizon=int(self.get_parameter("recovery_horizon").value),
            recovery_residual_m=float(self.get_parameter("recovery_residual_m").value),
            recovery_min_span_m=float(self.get_parameter("recovery_min_span_m").value),
            recovery_min_turn_deg=float(self.get_parameter("recovery_min_turn_deg").value),
            recovery_visual_z=float(self.get_parameter("recovery_visual_z").value),
            geometric_gate_scale=float(self.get_parameter("geometric_gate_scale").value),
            verification_threshold=float(self.get_parameter("verification_threshold").value),
            defer_nodes=int(self.get_parameter("defer_nodes").value),
            fixed_concentration=float(self.get_parameter("fixed_concentration").value),
            frame_id=f"{self._run_name}_seq{self._sequence_index}",
        )

    def _on_ground_truth(self, msg: PoseStamped) -> None:
        self._latest_gt = (
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            _yaw_from_quaternion(float(msg.pose.orientation.z), float(msg.pose.orientation.w)),
        )

    def _on_frame(
        self,
        image_msg: Image,
        odom_msg: Odometry,
        ground_truth_msg: PoseStamped | None = None,
    ) -> None:
        stamp = image_msg.header.stamp
        frame_key = f"{stamp.sec}:{stamp.nanosec}"
        if self._use_frame_labels and frame_key not in self._frame_labels:
            if self._pending_sample is not None:
                raise RuntimeError(
                    "received another frame before keyed metadata; lossless player protocol violated"
                )
            self._pending_sample = (image_msg, odom_msg, ground_truth_msg)
            return
        self._latest_label = self._frame_labels.pop(frame_key, None)
        callback_start = time.perf_counter()
        if self._first_frame_time is None:
            self._first_frame_time = callback_start
        if ground_truth_msg is not None:
            self._on_ground_truth(ground_truth_msg)
        image: np.ndarray = self._bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        pose: Pose2D = (
            float(odom_msg.pose.pose.position.x),
            float(odom_msg.pose.pose.position.y),
            _yaw_from_quaternion(
                float(odom_msg.pose.pose.orientation.z),
                float(odom_msg.pose.pose.orientation.w),
            ),
        )
        flat: list[float] = list(odom_msg.pose.covariance)
        covariance: np.ndarray = np.array(
            [
                [flat[0], flat[1], flat[5]],
                [flat[6], flat[7], flat[11]],
                [flat[30], flat[31], flat[35]],
            ],
            dtype=np.float64,
        )

        start: float = time.perf_counter()
        descriptor: np.ndarray = self._extractor.extract(image)
        elapsed: float = time.perf_counter() - start
        self._extraction_time_s += elapsed
        self._processed += 1
        if self._processed % 25 == 0:
            self.get_logger().info(
                f"{self._processed} frames processed, last extraction {elapsed:.3f}s"
            )
        touched_id: int | None = self._mapper.process_frame(
            image,
            descriptor,
            pose,
            covariance,
            gt_pose=self._latest_gt if ground_truth_msg is not None else None,
            room_label=self._latest_label,
        )
        if touched_id is not None:
            node = self._mapper.graph.nodes[touched_id]
            # Evaluation-only ground truth (never used to map). Use the node's
            # own gt_pose: the GT of its representative (medoid) frame, which
            # is the exact frame that fixed the node's pose. Falling back to
            # the latest GT only if the mapper carried none.
            node_gt: Pose2D | None = node.gt_pose
            if node_gt is not None:
                self._node_gt.setdefault(touched_id, node_gt)
        self._callback_times.append(time.perf_counter() - callback_start)
        self._frame_processed_pub.publish(String(data=frame_key))

    def _on_room_label(self, msg: String) -> None:
        if not self._use_frame_labels:
            return
        payload = json.loads(msg.data)
        key = str(payload["frame_key"])
        self._frame_labels[key] = payload["label"]
        if len(self._frame_labels) > 30:
            raise RuntimeError("unmatched label backlog exceeds synchronization queue")
        if self._pending_sample is not None:
            sample = self._pending_sample
            stamp = sample[0].header.stamp
            if key == f"{stamp.sec}:{stamp.nanosec}":
                self._pending_sample = None
                self._on_frame(*sample)

    # ------------------------------------------------------------------ #
    def _on_sequence_done(self, msg: String) -> None:
        info: dict[str, object] = json.loads(msg.data)
        expected_frames = info.get("frame_count")
        if expected_frames is not None:
            actual_frames = self._mapper._frame_count
            if actual_frames != int(expected_frames):
                raise RuntimeError(
                    "Incomplete frame stream: mapper processed "
                    f"{actual_frames}/{int(expected_frames)} frames"
                )
        # The detector confirms a boundary after the valley itself. Flush the
        # remaining post-valley frames before either graph snapshot is saved.
        flushed_id = self._mapper.finalize_nodes()
        if flushed_id is not None:
            flushed_gt = self._mapper.graph.nodes[flushed_id].gt_pose
            if flushed_gt is not None:
                self._node_gt.setdefault(flushed_id, flushed_gt)
        save_start = time.perf_counter()
        eig_path: str = os.path.join(self._output_dir, f"graph_{self._sequence_index}_lambda2.npy")
        payload = io.BytesIO()
        np.save(payload, np.array(self._mapper._monitor.eigenvalues))
        atomic_bytes(eig_path, payload.getvalue())
        self.get_logger().info(
            f"DIAG: frames_processed={self._mapper._frame_count}, "
            f"nodes={len(self._mapper.graph.nodes)}, "
            f"edges={len(self._mapper.graph.edges())}, "
            f"lambda2 series saved to {eig_path}"
        )
        graph_path: str = os.path.join(self._output_dir, f"graph_{self._sequence_index}.pkl")
        self._mapper.graph.save(graph_path)
        # Single-run pipeline: the per-sequence graph is also the final map the
        # language/evaluation tools consume (no separate alignment step).
        final_path: str = os.path.join(self._output_dir, "final_graph.pkl")
        self._mapper.graph.save(final_path)
        self._mapper.graph.without_images().save(
            os.path.join(self._output_dir, "compact_graph.pkl")
        )

        # Full runtime summary, including the frozen DINOv2 forward pass.
        perf: dict[str, object] = self._mapper.performance_stats()
        perf["encoder_time_ms_per_frame"] = (
            1000.0 * self._extraction_time_s / max(self._processed, 1)
        )
        perf["encoder_initialization_s"] = self._extractor_init_s
        perf["feature_backend"] = self._feature_backend
        perf["dino_model"] = self._dino_model
        perf["dino_layer"] = self._dino_layer
        perf["visual_model"] = self._visual_model
        perf["valley_mode"] = self._valley_mode
        perf["valley_k"] = self._valley_k
        perf["valley_delta"] = self._valley_delta
        perf["encoder_plus_mapper_ms_per_frame"] = (
            perf["encoder_time_ms_per_frame"] + perf["map_update_time_ms"]
        )
        footprint: dict[str, float] = map_footprint(self._mapper.graph)
        perf["map_disk_bytes_full"] = float(os.path.getsize(final_path))
        perf["map_disk_bytes_compact"] = footprint["compact_pickle_bytes"]
        perf.update(footprint)
        perf["encoder_identity"] = self._extractor.model_identity
        perf["descriptor_cache_enabled"] = hasattr(self._extractor, "hits")
        perf["descriptor_cache_hits"] = getattr(self._extractor, "hits", 0) - self._cache_start[0]
        perf["descriptor_cache_misses"] = (
            getattr(self._extractor, "misses", 0) - self._cache_start[1]
        )
        perf["callback_p95_ms"] = float(np.percentile(self._callback_times, 95) * 1000)
        perf["callback_p99_ms"] = float(np.percentile(self._callback_times, 99) * 1000)
        perf["observed_frames_per_second"] = self._processed / max(
            time.perf_counter() - self._first_frame_time, 1e-9
        )
        perf["graph_serialization_time_s"] = time.perf_counter() - save_start
        perf["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if sys.platform == "darwin" else 1024
        )
        perf_path: str = os.path.join(
            self._output_dir, f"graph_{self._sequence_index}_performance.json"
        )
        atomic_json(perf_path, perf)
        self.get_logger().info(
            "PERF: encoder_plus_mapper={encoder_plus_mapper_ms_per_frame:.2f} ms/frame, "
            "encoder={encoder_time_ms_per_frame:.2f} ms/frame, "
            "map_update={map_update_time_ms:.2f} ms/frame, "
            "loop_closure_search={loop_closure_time_ms:.2f} ms, "
            "map_size_full={full:.2f} MB, "
            "map_size_compact={desc:.3f} MB".format(
                **perf,
                full=perf["map_disk_bytes_full"] / 1e6,
                desc=perf["map_disk_bytes_compact"] / 1e6,
            )
        )
        gt_path: str = os.path.join(self._output_dir, f"graph_{self._sequence_index}_node_gt.json")
        atomic_json(gt_path, {str(k): list(v) for k, v in self._node_gt.items()})
        artifacts = [
            graph_path,
            final_path,
            eig_path,
            perf_path,
            gt_path,
            os.path.join(self._output_dir, "compact_graph.pkl"),
        ]
        artifacts.extend(
            str(path)
            for name in ("run_manifest.json", "effective_config.yaml")
            if (path := Path(self._output_dir) / name).is_file()
        )
        atomic_json(
            Path(self._output_dir) / "completion.json",
            {
                "schema": 1,
                "fingerprint": str(self.get_parameter("run_fingerprint").value),
                "frames": self._processed,
                "encoder_identity": self._extractor.model_identity,
                "artifacts": {Path(path).name: file_digest(path) for path in artifacts},
            },
        )

        event: dict[str, object] = {
            "graph_path": graph_path,
            "sequence_index": self._sequence_index,
            "all_done": bool(info.get("is_last", False)),
        }
        out: String = String()
        out.data = json.dumps(event)
        self._ready_pub.publish(out)
        self.get_logger().info(f"Graph ready: {event}")

        self._sequence_index += 1
        self._processed = 0
        self._extraction_time_s = 0.0
        self._mapper.close()
        self._mapper = self._new_mapper()
        self._callback_times = []
        self._first_frame_time = None
        self._cache_start = (
            getattr(self._extractor, "hits", 0),
            getattr(self._extractor, "misses", 0),
        )
        self._node_gt = {}
        self._frame_labels.clear()
        self._latest_gt = None

        if event["all_done"] and self._exit_when_done:
            # Grace period so DDS delivers the graph_ready message, then end
            # the spin. The launch file reacts to this process exiting by
            # shutting down the whole pipeline (player included).
            self.get_logger().info("All sequences done; exiting in 2 s.")
            self._shutdown_timer = self.create_timer(2.0, self._request_shutdown)

    def _request_shutdown(self) -> None:
        # ``rclpy.try_shutdown()`` from inside a timer callback can invalidate
        # the context without waking the executor that is currently spinning;
        # the process then remains alive and launch never receives
        # ``OnProcessExit``. SIGINT follows the already-tested manual Ctrl+C
        # path through ``main`` and therefore guarantees a clean process exit.
        if self._shutdown_timer is not None:
            self._shutdown_timer.cancel()
        self.get_logger().info("Shutdown grace period complete; exiting now.")
        os.kill(os.getpid(), signal.SIGINT)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: GraphBuilderNode = GraphBuilderNode()
    with suppress(KeyboardInterrupt, ExternalShutdownException):
        rclpy.spin(node)
    with suppress(Exception, KeyboardInterrupt):
        node._mapper.close()
        node.destroy_node()
    with suppress(Exception):
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
