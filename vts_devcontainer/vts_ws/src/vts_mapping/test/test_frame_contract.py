"""Callback contract tests with a lightweight ROS transport stub (no DDS)."""

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from vts_core.artifacts import completed_run
from vts_core.mapper import TopologicalMapper


@pytest.fixture
def builder_module(monkeypatch):
    modules = {
        "rclpy": {},
        "rclpy.node": {"Node": object},
        "rclpy.executors": {"ExternalShutdownException": type("ShutdownError", (Exception,), {})},
        "geometry_msgs.msg": {"PoseStamped": object},
        "nav_msgs.msg": {"Odometry": object},
        "sensor_msgs.msg": {"Image": object},
        "std_msgs.msg": {"String": SimpleNamespace},
        "cv_bridge": {"CvBridge": object},
        "message_filters": {"ApproximateTimeSynchronizer": object, "Subscriber": object},
    }
    for name, contents in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(contents)
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).parents[1] / "vts_mapping/graph_builder_node.py"
    spec = importlib.util.spec_from_file_location("builder_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def builder(module, tmp_path):
    instance = module.GraphBuilderNode.__new__(module.GraphBuilderNode)
    instance._use_frame_labels = True
    instance._frame_labels = {}
    instance._pending_sample = None
    instance._latest_label = None
    instance._first_frame_time = None
    instance._latest_gt = None
    instance._bridge = Mock()
    instance._bridge.imgmsg_to_cv2.return_value = np.ones((2, 2, 3), np.uint8)
    instance._extractor = Mock(model_identity={"test": True}, hits=0, misses=0)
    instance._extractor.extract.return_value = np.array([1.0, 0.0, 0.0], np.float32)
    instance._mapper = TopologicalMapper()
    instance._extraction_time_s = 0.0
    instance._processed = 0
    instance._node_gt = {}
    instance._callback_times = []
    instance._frame_processed_pub = Mock()
    instance._ready_pub = Mock()
    instance.get_logger = Mock(return_value=Mock())
    instance.get_parameter = lambda name: SimpleNamespace(value="test-fingerprint")
    instance._output_dir = str(tmp_path)
    instance._sequence_index = 0
    instance._extractor_init_s = 0.0
    instance._feature_backend = "dino_cls"
    instance._dino_model = "fixture"
    instance._dino_layer = -1
    instance._visual_model = "vmf"
    instance._valley_mode = "adaptive"
    instance._valley_k = 2.0
    instance._valley_delta = 0.1
    instance._cache_start = (0, 0)
    instance._new_mapper = TopologicalMapper
    instance._exit_when_done = False
    return instance


def messages():
    stamp = SimpleNamespace(sec=7, nanosec=42)
    image = SimpleNamespace(header=SimpleNamespace(stamp=stamp))
    pose = SimpleNamespace(
        position=SimpleNamespace(x=3.0, y=4.0), orientation=SimpleNamespace(z=0.0, w=1.0)
    )
    odom = SimpleNamespace(pose=SimpleNamespace(pose=pose, covariance=[0.0] * 36))
    ground_truth = SimpleNamespace(pose=pose)
    label = SimpleNamespace(data=json.dumps({"frame_key": "7:42", "label": "CR"}))
    return image, odom, ground_truth, label


@pytest.mark.parametrize("label_first", [True, False])
def test_label_join_is_independent_of_cross_topic_order(builder_module, tmp_path, label_first):
    instance = builder(builder_module, tmp_path)
    image, odom, gt, label = messages()
    if label_first:
        instance._on_room_label(label)
    instance._on_frame(image, odom, gt)
    if not label_first:
        assert instance._processed == 0
        instance._frame_processed_pub.publish.assert_not_called()
        instance._on_room_label(label)
    assert instance._processed == 1
    assert instance._mapper._pending_frames[0].room_label == "CR"
    assert instance._mapper._pending_frames[0].gt_pose == (3.0, 4.0, 0.0)
    assert instance._frame_processed_pub.publish.call_args.args[0].data == "7:42"
    instance._mapper.close()


def test_completion_is_published_after_all_artifacts(builder_module, tmp_path):
    instance = builder(builder_module, tmp_path)
    image, odom, gt, label = messages()
    instance._on_room_label(label)
    instance._on_frame(image, odom, gt)
    instance._on_sequence_done(
        SimpleNamespace(data=json.dumps({"frame_count": 1, "is_last": True}))
    )
    marker = completed_run(tmp_path, "test-fingerprint")
    assert marker is not None and marker["frames"] == 1
    assert set(marker["artifacts"]) == {
        "graph_0.pkl",
        "final_graph.pkl",
        "compact_graph.pkl",
        "graph_0_lambda2.npy",
        "graph_0_node_gt.json",
        "graph_0_performance.json",
    }
    instance._mapper.close()


def test_wrong_final_count_never_publishes_completion(builder_module, tmp_path):
    instance = builder(builder_module, tmp_path)
    with pytest.raises(RuntimeError, match="Incomplete frame stream"):
        instance._on_sequence_done(SimpleNamespace(data=json.dumps({"frame_count": 1})))
    assert completed_run(tmp_path) is None
    instance._mapper.close()


@pytest.mark.parametrize(
    "filename, class_name",
    [("cold_player_node.py", "ColdPlayerNode"), ("cid_sims_player_node.py", "CidSimsPlayerNode")],
)
def test_players_ignore_stale_ack_and_timeout_missing_ack(builder_module, filename, class_name):
    path = Path(__file__).parents[2] / "vts_players/vts_players" / filename
    spec = importlib.util.spec_from_file_location("player_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, class_name)
    player = cls.__new__(cls)
    player._waiting_for_frame_ack = True
    player._ack_token = "7:42"
    player._progress_time = 0.0
    player._consumer_connected = True
    player._ack_timeout = 0.01
    player._on_frame_processed(SimpleNamespace(data="previous frame"))
    assert player._waiting_for_frame_ack
    with pytest.raises(TimeoutError, match="no acknowledgement"):
        player._tick()
    player._on_frame_processed(SimpleNamespace(data="7:42"))
    assert not player._waiting_for_frame_ack
