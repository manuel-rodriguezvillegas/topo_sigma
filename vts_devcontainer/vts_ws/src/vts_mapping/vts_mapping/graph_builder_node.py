"""Graph builder ROS node: a thin wrapper around :class:`vts_core.mapping.OnlineMapper`.

Subscribes (time-synchronized) to ``/camera/image`` and ``/odom`` carrying the
``path_length_budget_v1`` cumulative budget, optionally ``/ground_truth_pose``
(stored for evaluation only, never used to map), and ``/dataset/sequence_done``.
Each frame is encoded with the frozen DINOv3 model and pushed to the same
engine the offline replays use. Each sequence produces in ``output_dir``:

- ``graph_<i>.npz``: places, in-session decisions and edges (TopoGraph)
- ``graph_<i>_views/<place>_<k>.jpg``: the representative views of each place
- ``graph_<i>.json``: node ground truth, lambda_2 series, timing and provenance
"""

from __future__ import annotations

import json
import os
import signal
import time
from contextlib import suppress
from pathlib import Path

import cv2
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

from vts_core.descriptors import build_encoder
from vts_core.graph import TopoGraph
from vts_core.mapping import MapperConfig, OnlineMapper
from vts_core.provenance import atomic_json, load_config, run_record


def yaw(z: float, w: float) -> float:
    return float(2.0 * np.arctan2(z, w))


def budget_from_covariance(flat: list[float]) -> np.ndarray:
    return np.asarray(flat, dtype=np.float64).reshape(6, 6)[np.ix_([0, 1, 5], [0, 1, 5])]


class GraphBuilderNode(Node):
    def __init__(self) -> None:
        super().__init__("graph_builder")
        self.declare_parameter("toposigma_config", "")
        self.declare_parameter("uncertainty_contract", "")
        self.declare_parameter("encoder_device", "auto")
        self.declare_parameter("output_dir", "output/ros")
        self.declare_parameter("use_ground_truth", True)
        self.declare_parameter("exit_when_done", True)
        if self.get_parameter("uncertainty_contract").value != "path_length_budget_v1":
            raise ValueError("declare uncertainty_contract=path_length_budget_v1; pose marginals are unsupported")
        self._config = load_config(str(self.get_parameter("toposigma_config").value) or None)
        self._mapper_config = MapperConfig.from_dict(self._config["mapping"])
        self._output = Path(str(self.get_parameter("output_dir").value))
        self._output.mkdir(parents=True, exist_ok=True)
        self._encoder = build_encoder("dinov3", str(self.get_parameter("encoder_device").value))
        self._bridge = CvBridge()
        self._sequence = 0
        self._reset()

        self._ack_pub = self.create_publisher(String, "/mapping/frame_processed", 10)
        self._ready_pub = self.create_publisher(String, "/mapping/graph_ready", 10)
        inputs = [Subscriber(self, Image, "/camera/image"), Subscriber(self, Odometry, "/odom")]
        if bool(self.get_parameter("use_ground_truth").value):
            inputs.append(Subscriber(self, PoseStamped, "/ground_truth_pose"))
        self._sync = ApproximateTimeSynchronizer(inputs, queue_size=30, slop=0.05)
        self._sync.registerCallback(self._on_frame)
        self.create_subscription(String, "/dataset/sequence_done", self._on_sequence_done, 10)
        self._shutdown_timer = None

    def _reset(self) -> None:
        self._mapper = OnlineMapper(self._mapper_config)
        self._images: dict[int, np.ndarray] = {}  # frames of the open segment, JPEG-encoded
        self._views: dict[int, list[bytes]] = {}
        self._gt: list[list[float] | None] = []
        self._encode_s = 0.0
        self._map_s = 0.0

    def _on_frame(self, image_msg: Image, odom_msg: Odometry, gt_msg: PoseStamped | None = None) -> None:
        image = self._bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        position, orientation = odom_msg.pose.pose.position, odom_msg.pose.pose.orientation
        pose = (float(position.x), float(position.y), yaw(orientation.z, orientation.w))
        started = time.perf_counter()
        descriptor = self._encoder.encode([image])[0]
        self._encode_s += time.perf_counter() - started
        frame = self._mapper.frames
        self._images[frame] = cv2.imencode(".jpg", image)[1]
        self._gt.append(None if gt_msg is None else [float(gt_msg.pose.position.x), float(gt_msg.pose.position.y),
                                                     yaw(gt_msg.pose.orientation.z, gt_msg.pose.orientation.w)])
        started = time.perf_counter()
        place = self._mapper.push(descriptor, pose, budget_from_covariance(odom_msg.pose.covariance))
        self._map_s += time.perf_counter() - started
        if place is not None:
            self._keep_views(place)
        stamp = image_msg.header.stamp
        self._ack_pub.publish(String(data=f"{stamp.sec}:{stamp.nanosec}"))

    def _keep_views(self, place) -> None:
        self._views[len(self._mapper.places) - 1] = [self._images[f].tobytes() for f in place.view_frames]
        for frame in [f for f in self._images if f < place.stop]:
            del self._images[frame]

    def _on_sequence_done(self, message: String) -> None:
        info = json.loads(message.data)
        if info.get("frame_count") is not None and int(info["frame_count"]) != self._mapper.frames:
            raise RuntimeError(f"incomplete stream: {self._mapper.frames}/{info['frame_count']} frames mapped")
        place = self._mapper.finish()
        if place is not None:
            self._keep_views(place)
        name = info.get("session") or f"sequence{self._sequence}"
        graph = TopoGraph()
        graph.add_session(name, self._mapper.places, self._mapper.decisions)
        stem = self._output / f"graph_{self._sequence}"
        graph.save(stem.with_suffix(".npz"))
        for index, views in self._views.items():
            for k, jpeg in enumerate(views):
                path = Path(f"{stem}_views") / f"{index}_{k}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(jpeg)
        frames = max(self._mapper.frames, 1)
        atomic_json(stem.with_suffix(".json"), {
            "run": run_record("ros graph_builder", {"mapping": self._config["mapping"]}, {}),
            "session": name, "frames": self._mapper.frames, "places": len(self._mapper.places),
            "accepted": sum(d.accepted for d in self._mapper.decisions),
            "place_gt": [self._gt[p.medoid] for p in self._mapper.places],
            "lambda2": self._mapper.segmenter.lambda2,
            "encoder_ms_per_frame": 1000 * self._encode_s / frames, "map_ms_per_frame": 1000 * self._map_s / frames,
            "encoder": self._encoder.identity,
        })
        event = {"graph_path": str(stem.with_suffix(".npz")), "sequence_index": self._sequence,
                 "all_done": bool(info.get("is_last", False))}
        self._ready_pub.publish(String(data=json.dumps(event)))
        self.get_logger().info(f"Graph ready: {event} ({len(self._mapper.places)} places)")
        self._sequence += 1
        self._reset()
        if event["all_done"] and bool(self.get_parameter("exit_when_done").value):
            self._shutdown_timer = self.create_timer(2.0, self._request_shutdown)

    def _request_shutdown(self) -> None:
        # SIGINT follows the tested Ctrl+C path, so launch sees the process exit.
        self._shutdown_timer.cancel()
        os.kill(os.getpid(), signal.SIGINT)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = GraphBuilderNode()
    with suppress(KeyboardInterrupt, ExternalShutdownException):
        rclpy.spin(node)
    with suppress(Exception, KeyboardInterrupt):
        node.destroy_node()
    with suppress(Exception):
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
