"""Replays dataset recordings as standard ROS topics (COLD and CID-SIMS).

Frames, ground truth and odometry come from :mod:`vts_core.datasets`, the
same adapters that build the offline session stores, so a ROS run and an
offline replay see identical inputs. Published per frame:

- ``/camera/image`` (bgr8), ``/odom`` with the cumulative uncertainty budget
  (``path_length_budget_v1``) in the covariance fields, and
  ``/ground_truth_pose`` (evaluation only), all with the same stamp.
- ``/dataset/sequence_done`` (JSON) after each recording.

Flow control is lossless: the next frame is published only after the mapper
acknowledges the current stamp on ``/mapping/frame_processed``.
"""

from __future__ import annotations

import json
import time
from contextlib import suppress

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from vts_core.datasets import catalogue, dataset_root, read_recording
from vts_core.odometry import MotionNoise, budget, interpolate
from vts_core.provenance import load_config


def _quaternion_zw(yaw: float) -> tuple[float, float]:
    return float(np.sin(yaw / 2.0)), float(np.cos(yaw / 2.0))


class SessionPlayerNode(Node):
    def __init__(self) -> None:
        super().__init__("session_player")
        self.declare_parameter("sessions", [""])
        self.declare_parameter("toposigma_config", "")
        self.declare_parameter("allow_test", False)
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("inter_sequence_pause_s", 2.0)
        self.declare_parameter("ack_timeout_s", 300.0)
        self.declare_parameter("startup_timeout_s", 900.0)

        self._config = load_config(str(self.get_parameter("toposigma_config").value) or None)
        self._sessions = [s for s in self.get_parameter("sessions").value if s]
        if not self._sessions:
            raise ValueError("set the 'sessions' parameter to one or more catalogue names")
        rate = float(self.get_parameter("publish_rate_hz").value)
        self._ack_timeout = float(self.get_parameter("ack_timeout_s").value)
        self._startup_timeout = float(self.get_parameter("startup_timeout_s").value)
        if not np.isfinite([rate, self._ack_timeout, self._startup_timeout]).all() or min(
                rate, self._ack_timeout, self._startup_timeout) <= 0:
            raise ValueError("publish rate and timeouts must be finite and positive")
        self._pause_ticks = int(float(self.get_parameter("inter_sequence_pause_s").value) * rate)

        self._bridge = CvBridge()
        self._image_pub = self.create_publisher(Image, "/camera/image", 10)
        self._odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self._gt_pub = self.create_publisher(PoseStamped, "/ground_truth_pose", 10)
        self._done_pub = self.create_publisher(String, "/dataset/sequence_done", 10)
        self.create_subscription(String, "/mapping/frame_processed", self._on_ack, 10)

        self._ack_token: str | None = None
        self._waiting = False
        self._connected = False
        self._progress = time.monotonic()
        self._pause = 0
        self._sequence = 0
        self._load(0)
        self._timer = self.create_timer(1.0 / rate, self._tick)

    def _load(self, index: int) -> None:
        name = self._sessions[index]
        recording = catalogue(self._config)[name]
        if recording.role == "test" and not bool(self.get_parameter("allow_test").value):
            raise PermissionError(f"{name} is test data; it needs a written protocol and allow_test")
        frames = read_recording(recording, self._config)
        settings = self._config["datasets"][recording.dataset]
        budgets = budget(frames.odom_poses, MotionNoise(*settings["motion_noise"]))
        self._poses, self._budgets = interpolate(frames.odom_t, frames.odom_poses, budgets, frames.timestamps,
                                                 settings["max_odometry_gap_s"], settings["exact_atol_s"])
        root = dataset_root(self._config, recording.dataset)
        self._paths = [str(root / p) for p in frames.image_paths]
        self._gt = frames.gt
        self._frame = 0
        self.get_logger().info(f"Session {name}: {len(self._paths)} frames")

    def _on_ack(self, message: String) -> None:
        if self._waiting and message.data == self._ack_token:
            self._waiting = False
            self._progress = time.monotonic()

    def _tick(self) -> None:
        if not self._connected:
            self._connected = all(p.get_subscription_count() > 0 for p in
                                  (self._image_pub, self._odom_pub, self._gt_pub, self._done_pub)) \
                and self.count_publishers("/mapping/frame_processed") > 0
            if not self._connected:
                if time.monotonic() - self._progress > self._startup_timeout:
                    raise TimeoutError("mapping consumers did not become ready")
                return
            self._progress = time.monotonic()
        if self._waiting:
            if time.monotonic() - self._progress > self._ack_timeout:
                raise TimeoutError(f"no acknowledgement for frame {self._ack_token}")
            return
        if self._pause > 0:
            self._pause -= 1
            return
        if self._frame >= len(self._paths):
            self._finish_sequence()
            return
        index = self._frame
        self._frame += 1
        image = cv2.imread(self._paths[index], cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"unreadable image: {self._paths[index]}")
        stamp = self.get_clock().now().to_msg()
        self._ack_token = f"{stamp.sec}:{stamp.nanosec}"
        self._progress = time.monotonic()
        self._waiting = True  # before publishing: a fast acknowledgement must not race

        gt = PoseStamped()
        gt.header.stamp, gt.header.frame_id = stamp, "world"
        gt.pose.position.x, gt.pose.position.y = float(self._gt[index, 0]), float(self._gt[index, 1])
        gt.pose.orientation.z, gt.pose.orientation.w = _quaternion_zw(self._gt[index, 2])
        self._gt_pub.publish(gt)

        image_msg = self._bridge.cv2_to_imgmsg(image, encoding="bgr8")
        image_msg.header.stamp, image_msg.header.frame_id = stamp, f"camera_seq{self._sequence}"
        self._image_pub.publish(image_msg)

        odom = Odometry()
        odom.header.stamp, odom.header.frame_id = stamp, "odom"
        pose, cov = self._poses[index], self._budgets[index]
        odom.pose.pose.position.x, odom.pose.pose.position.y = float(pose[0]), float(pose[1])
        odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = _quaternion_zw(pose[2])
        flat = np.zeros((6, 6))
        flat[np.ix_([0, 1, 5], [0, 1, 5])] = cov  # x, y, yaw blocks of the 6x6 covariance
        odom.pose.covariance = flat.ravel().tolist()
        self._odom_pub.publish(odom)

    def _finish_sequence(self) -> None:
        last = self._sequence >= len(self._sessions) - 1
        payload = json.dumps({"sequence_index": self._sequence, "session": self._sessions[self._sequence],
                              "is_last": last, "frame_count": len(self._paths)})
        self._done_pub.publish(String(data=payload))
        self.get_logger().info(f"Sequence done: {payload}")
        if last:
            self._timer.cancel()
            return
        self._sequence += 1
        self._pause = self._pause_ticks
        self._load(self._sequence)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = SessionPlayerNode()
    with suppress(KeyboardInterrupt):
        rclpy.spin(node)
    with suppress(KeyboardInterrupt):
        node.destroy_node()
    with suppress(Exception):
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
