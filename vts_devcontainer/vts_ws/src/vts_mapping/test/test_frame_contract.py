"""Graph builder callbacks with a stub ROS transport and a fake encoder (no DDS, no DINOv3)."""

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vts_core.graph import TopoGraph
from vts_core.mapping import MapperConfig, map_session


class FakeNode:
    def __init__(self, name):
        self.published = []
        self.params = {}

    def declare_parameter(self, name, value):
        self.params.setdefault(name, value)

    def get_parameter(self, name):
        return SimpleNamespace(value=self.params[name])

    def create_publisher(self, _type, topic, _depth):
        return SimpleNamespace(publish=lambda message: self.published.append((topic, message.data)))

    def create_subscription(self, *args):
        return None

    def create_timer(self, *args):
        return SimpleNamespace(cancel=lambda: None)

    def get_logger(self):
        return SimpleNamespace(info=lambda *a: None)


@pytest.fixture
def builder(monkeypatch, tmp_path):
    stubs = {
        "rclpy": {}, "rclpy.node": {"Node": FakeNode},
        "rclpy.executors": {"ExternalShutdownException": type("Shutdown", (Exception,), {})},
        "geometry_msgs.msg": {"PoseStamped": object}, "nav_msgs.msg": {"Odometry": object},
        "sensor_msgs.msg": {"Image": object}, "std_msgs.msg": {"String": lambda data: SimpleNamespace(data=data)},
        "cv_bridge": {"CvBridge": lambda: SimpleNamespace(imgmsg_to_cv2=lambda m, desired_encoding: m.image)},
        "message_filters": {"ApproximateTimeSynchronizer": lambda *a, **k: SimpleNamespace(registerCallback=id),
                            "Subscriber": lambda *a: None},
    }
    for name, contents in stubs.items():
        module = types.ModuleType(name)
        module.__dict__.update(contents)
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).parents[1] / "vts_mapping/graph_builder_node.py"
    spec = importlib.util.spec_from_file_location("builder_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "build_encoder", lambda name, device: None)
    original = FakeNode.__init__

    def init(self, name):
        original(self, name)
        self.params = {"uncertainty_contract": "path_length_budget_v1", "output_dir": str(tmp_path)}

    monkeypatch.setattr(FakeNode, "__init__", init)
    return module


def frames(rng, n=300, dim=16):
    centres = rng.normal(size=(6, dim))
    descriptors = np.repeat(centres, n // 6, axis=0) + 0.1 * rng.normal(size=(n, dim))
    descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True)
    poses = np.column_stack([np.linspace(0, 20, n), np.zeros(n), np.zeros(n)])
    budgets = np.linspace(0, 1, n)[:, None, None] * np.eye(3)
    return descriptors.astype(np.float32), poses, budgets


def message(pose, cov, k):
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[0, 0, :2] = divmod(k, 256)  # the fake encoder looks the frame up by this index
    flat = np.zeros((6, 6))
    flat[np.ix_([0, 1, 5], [0, 1, 5])] = cov
    odom = SimpleNamespace(pose=SimpleNamespace(
        pose=SimpleNamespace(position=SimpleNamespace(x=pose[0], y=pose[1]),
                             orientation=SimpleNamespace(z=np.sin(pose[2] / 2), w=np.cos(pose[2] / 2))),
        covariance=flat.ravel().tolist()))
    return SimpleNamespace(image=image, header=SimpleNamespace(stamp=SimpleNamespace(sec=k, nanosec=0))), odom


def test_builder_matches_offline_engine(builder, tmp_path):
    rng = np.random.default_rng(3)
    descriptors, poses, budgets = frames(rng)
    node = builder.GraphBuilderNode()
    node._encoder = SimpleNamespace(identity={"model": "fake"}, encode=lambda images: np.stack(
        [descriptors[256 * int(i[0, 0, 0]) + int(i[0, 0, 1])] for i in images]))
    for k, (p, b) in enumerate(zip(poses, budgets)):
        node._on_frame(*message(p, b, k))
    node._on_sequence_done(SimpleNamespace(data=json.dumps({"frame_count": len(descriptors), "is_last": False,
                                                            "session": "synthetic"})))
    acks = [data for topic, data in node.published if topic == "/mapping/frame_processed"]
    assert acks == [f"{k}:0" for k in range(len(descriptors))]
    graph = TopoGraph.load(tmp_path / "graph_0.npz")
    offline = map_session(descriptors, poses, budgets, MapperConfig())
    assert [(p.start, p.stop) for p in graph.places["synthetic"]] == [(p.start, p.stop) for p in offline.places]
    assert [vars(d) for d in graph.decisions["synthetic"]] == [vars(d) for d in offline.decisions]
    views = sorted((tmp_path / "graph_0_views").glob("*.jpg"))
    assert len(views) == sum(len(p.view_frames) for p in offline.places)
