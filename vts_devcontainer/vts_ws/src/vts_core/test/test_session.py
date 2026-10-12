import json

import numpy as np
import pytest
from conftest import make_place

from vts_core.association import Decision
from vts_core.datasets import RawFrames, recording_for
from vts_core.graph import TopoGraph
from vts_core.provenance import load_config
from vts_core.session import TestDataError, add_descriptors, open_session, write_session


@pytest.fixture
def config(tmp_path):
    return load_config({"store_root": str(tmp_path / "stores"), "output_root": str(tmp_path / "out")})


def raw(n=20):
    t = np.arange(n) * 0.2
    odom_t = np.arange(0, n * 0.2, 0.05)
    poses = np.column_stack([odom_t, np.zeros_like(odom_t), np.zeros_like(odom_t)])
    return RawFrames(t, np.zeros((n, 3)), [None] * n, [f"img{i}.png" for i in range(n)], odom_t, poses)


def test_catalogue_names_and_roles(config):
    cold = recording_for("cold", "cold-ljubljana_part_a_seq1_night3", config)
    assert cold.name == "cold_ljubljana_a_seq1_night3" and cold.role == "test"
    cid = recording_for("cid_sims", "apartment1_2", config)
    assert (cid.name, cid.environment, cid.role) == ("cid_apartment1_2", "cid_apartment1", "development")
    assert recording_for("cid_sims", "apartment3_3", config).role == "test"


def test_store_round_trip_and_tamper_detection(config):
    recording = recording_for("cid_sims", "apartment1_1", config)
    directory = write_session(config, recording, raw(), {"kind": "test"})
    descriptors = np.eye(20, 8, dtype=np.float32) + 1e-3
    descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True)
    add_descriptors(directory, "dinov3", descriptors, {"model": "fake"})
    session = open_session(config, recording.name)
    assert np.array_equal(session.descriptors(), descriptors)
    poses, budgets = session.odometry
    assert poses.shape == (20, 3) and np.all(np.diff(budgets[:, 0, 0]) >= 0)
    path = directory / "dinov3.npy"
    path.write_bytes(path.read_bytes()[:-4] + b"\0\0\0\0")
    with pytest.raises(ValueError):
        session.descriptors()


def test_test_sessions_are_refused(config):
    recording = recording_for("cold", "cold-ljubljana_part_a_seq2_cloudy2", config)
    write_session(config, recording, raw(), {"kind": "test"})
    with pytest.raises(TestDataError):
        open_session(config, recording.name)
    assert open_session(config, recording.name, allow_test=True).recording.role == "test"


def test_graph_round_trip(tmp_path, rng):
    places = [make_place([k, 0, 0], rng.normal(size=6), budget=0.1 * k, start=k) for k in range(5)]
    graph = TopoGraph()
    graph.add_session("s", places, [Decision(3, 0, True, 0.8, "accepted"), Decision(4, 1, False, 0.2, "rejected")])
    graph.add_session("t", places[:2])
    graph.add_edge(("t", 1), ("s", 3), "merge")
    graph.save(tmp_path / "g.npz")
    loaded = TopoGraph.load(tmp_path / "g.npz")
    assert loaded.edges == graph.edges and loaded.sessions == ["s", "t"]
    assert [vars(d) for d in loaded.decisions["s"]] == [vars(d) for d in graph.decisions["s"]]
    assert np.allclose(loaded.places["s"][2].mean, places[2].mean)
    assert json.loads(json.dumps(loaded.places["s"][4].view_frames)) == [4]
