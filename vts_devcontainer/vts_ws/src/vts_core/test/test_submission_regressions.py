"""Regression cases for the submission audit, independent of ROS and weights."""

import json
import pickle
from unittest.mock import Mock

import numpy as np
import pytest

from vts_core.artifacts import atomic_json, completed_run, file_digest
from vts_core.descriptor_cache import DescriptorCache
from vts_core.directional import fit_von_mises_fisher, vmf_log_overlap_ratio
from vts_core.frame_buffer import FrameBuffer, FrameRecord, StoredImage
from vts_core.mapper import TopologicalMapper
from vts_core.matching import gaussian_position_statistics
from vts_core.metrics import map_footprint, place_recognition_recall
from vts_core.motion import OdometryUncertaintyParams, OdometryUncertaintyTracker
from vts_core.selection import PendingSelection
from vts_core.topo_graph import TopoGraph, TopoNode


def node(index, x=0.0):
    feature = np.array([1, 0, 0, 0], dtype=np.float32)
    return TopoNode(
        index,
        (x, 0.0, 0.0),
        feature,
        visual_concentration=10.0,
        view_features=feature[None],
        pose_covariance=np.eye(2),
    )


def test_motion_budget_is_invariant_to_piecewise_path_resampling():
    budgets = []
    for count in (10, 100, 1000):
        tracker = OdometryUncertaintyTracker()
        for x in np.linspace(0, 10, count + 1):
            tracker.step((x, 0.0, 0.0))
        for yaw in np.linspace(0, np.pi / 2, count + 1):
            tracker.step((10.0, 0.0, yaw))
        for y in np.linspace(0, 5, count + 1):
            tracker.step((10.0, y, np.pi / 2))
        budgets.append(tracker.covariance)
    for budget in budgets:
        np.testing.assert_allclose(budget, budgets[0], rtol=1e-12)
    expected_xy = 0.01 * 15 + 0.0025 * np.pi / 2
    assert budgets[0][0, 0] == pytest.approx(expected_xy)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -0.1])
def test_budget_coefficients_reject_invalid_values(bad):
    with pytest.raises(ValueError):
        OdometryUncertaintyParams(alpha1=bad)


def test_budget_reset_is_rejected_before_frame_is_buffered():
    mapper = TopologicalMapper()
    image, descriptor = np.zeros((2, 2, 3), np.uint8), node(0).visual_features
    mapper.process_frame(image, descriptor, (0.0, 0.0, 0.0), np.eye(3))
    with pytest.raises(ValueError, match="decreased"):
        mapper.process_frame(image, descriptor, (1.0, 0.0, 0.0), np.eye(3) * 0.1)
    assert len(mapper._pending_frames) == 1
    mapper.close()


def test_interval_supports_monotonic_anisotropic_budgets():
    before = np.array([[3.0, 0.2], [0.2, 2.0]])
    increment = np.array([[0.4, 0.1], [0.1, 0.3]])
    np.testing.assert_allclose(
        TopologicalMapper._interval_covariance(before, before + increment), increment
    )


def test_vmf_scoring_does_not_modify_readonly_float64_inputs():
    first, second = np.array([2.0, 0.0, 0.0]), np.array([1.0, 1.0, 0.0])
    first.flags.writeable = second.flags.writeable = False
    assert np.isfinite(vmf_log_overlap_ratio(first, 20.0, second, 10.0))
    np.testing.assert_array_equal(first, [2.0, 0.0, 0.0])
    np.testing.assert_array_equal(second, [1.0, 1.0, 0.0])


def test_streamed_segment_matches_dense_reference_and_preserves_images():
    rng = np.random.default_rng(42)
    descriptors = rng.normal(size=(120, 16)).astype(np.float32)
    descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True)
    expected = fit_von_mises_fisher(descriptors)
    medoid = int(np.argmax(descriptors @ expected.mean_direction))
    mapper = TopologicalMapper()
    buffer = mapper._pending_frames
    for i, descriptor in enumerate(descriptors):
        buffer.append(
            FrameRecord(
                descriptor,
                (float(i), 0.0, 0.0),
                np.eye(3) * i,
                np.full((16, 16, 3), i, dtype=np.uint8),
                frame_index=i,
            )
        )
    assert isinstance(buffer[medoid].image, StoredImage)
    mapper.finalize_nodes()
    actual = mapper.graph.nodes[0]
    np.testing.assert_allclose(actual.visual_features, expected.mean_direction, atol=1e-7)
    assert actual.visual_concentration == pytest.approx(expected.concentration)
    assert actual.pose[0] == medoid
    for image, index in zip(actual.views, dict.fromkeys([medoid, 0, 119]), strict=True):
        np.testing.assert_array_equal(image, np.full((16, 16, 3), index, np.uint8))
    assert actual.extent_covariance[0, 0] == pytest.approx(np.mean((np.arange(120) - medoid) ** 2))
    assert len(buffer) == 0
    mapper.close()


def test_spool_retains_frames_after_delayed_boundary():
    buffer = FrameBuffer()
    for i in range(20):
        buffer.append(
            FrameRecord(
                node(i).visual_features,
                (float(i), 0.0, 0.0),
                np.eye(3),
                np.full((2, 2, 3), i, np.uint8),
                frame_index=i,
            )
        )
    assert [frame.frame_index for frame in buffer.segment(12)] == list(range(13))
    buffer.discard_through(12)
    assert [frame.frame_index for frame in buffer] == list(range(13, 20))
    np.testing.assert_array_equal(buffer[-1].image.read(), np.full((2, 2, 3), 19, np.uint8))
    buffer.close()


@pytest.mark.parametrize("mode", ["both", "geometric"])
def test_geometry_rejection_does_not_compute_appearance(mode):
    mapper = TopologicalMapper(gate_mode=mode)
    for i in range(7):
        mapper.graph.add_node(node(i, x=i * 10))
    mapper._sequence_vmf_evidence = Mock(side_effect=AssertionError("unnecessary visual work"))
    mapper._node_similarity = Mock(side_effect=AssertionError("unnecessary cosine work"))
    assert mapper._find_revisit(node(7, x=1000))[0] is None
    mapper.close()


def test_visual_only_uses_selected_vmf_model():
    mapper = TopologicalMapper(gate_mode="visual", visual_model="vmf")
    for i in range(7):
        mapper.graph.add_node(node(i))
    mapper._sequence_vmf_evidence = lambda candidate, index: 10.0 if index == 2 else 0.0
    mapper._sequence_similarity = Mock(side_effect=AssertionError("wrong evidence model"))
    assert mapper._find_revisit(node(7))[0] == 2
    mapper.close()


def test_cached_scores_match_uncached_scores():
    mapper = TopologicalMapper()
    rng = np.random.default_rng(7)
    nodes = []
    for i in range(10):
        direction = rng.normal(size=384).astype(np.float32)
        direction /= np.linalg.norm(direction)
        views = rng.normal(size=(3, 384)).astype(np.float32)
        nodes.append(
            TopoNode(
                i, (0.0, 0.0, 0.0), direction, visual_concentration=i * 77.0, view_features=views
            )
        )
    for first, second in zip(nodes, reversed(nodes), strict=True):
        assert mapper._cached_vmf_evidence(first, second) == pytest.approx(
            mapper._node_vmf_evidence(first, second), abs=1e-10
        )
        assert mapper._cached_similarity(first, second) == pytest.approx(
            mapper._node_similarity(first, second), abs=1e-7
        )
    mapper.close()


def test_batched_geometry_uses_one_regularization():
    covariance = np.array([[[0.3, 0.1], [0.1, 0.8]], [[2.0, -0.2], [-0.2, 1.0]]])
    delta = np.array([[1.0, 2.0], [3.0, -1.0]])
    d2, nll = gaussian_position_statistics(delta, covariance)
    for i in range(2):
        regularized = covariance[i] + np.eye(2) * 1e-6
        expected = delta[i] @ np.linalg.inv(regularized) @ delta[i]
        assert d2[i] == pytest.approx(expected)
        assert nll[i] == pytest.approx(0.5 * (expected + np.log(np.linalg.det(regularized))))


def test_excluded_candidates_cannot_enter_recall_top_k():
    descriptors, xy = np.eye(3), np.zeros((3, 2))
    result = place_recognition_recall(
        descriptors,
        xy,
        descriptors,
        xy,
        query_index=np.arange(3),
        db_index=np.arange(3),
        exclude_window=5,
    )
    assert result == {"recall_at_1": 0.0, "recall_at_5": 0.0, "n_queries": 3}


def test_descriptor_cache_keys_pixels_and_model_and_recovers_corruption(tmp_path):
    extractor = Mock(device="cpu")
    extractor.extract.return_value = node(0).visual_features
    first = DescriptorCache(extractor, str(tmp_path), {"revision": "a", "preprocessing": 1})
    image = np.zeros((4, 4, 3), np.uint8)
    first.extract(image)
    first.extract(image.copy())
    assert extractor.extract.call_count == 1
    second = DescriptorCache(extractor, str(tmp_path), {"revision": "b", "preprocessing": 1})
    second.extract(image)
    assert extractor.extract.call_count == 2
    image[0, 0] = 1
    first.extract(image)
    assert extractor.extract.call_count == 3
    for path in first.directory.glob("*.npy"):
        path.write_bytes(b"incomplete")
    first.extract(image)
    assert extractor.extract.call_count == 4


def test_compact_map_is_operational_and_counts_mean_descriptor(tmp_path):
    graph = TopoGraph()
    for i in range(2):
        value = node(i)
        value.set_views([np.zeros((2, 2, 3), np.uint8)], value.visual_features[None])
        graph.add_node(value)
    graph.add_edge(0, 1, "loop")
    compact = graph.without_images()
    path = tmp_path / "compact.pkl"
    compact.save(str(path))
    restored = TopoGraph.load(str(path))
    assert restored.edge_types == graph.edge_types
    assert restored.nodes[0].views == []
    np.testing.assert_array_equal(restored.nodes[0].visual_features, graph.nodes[0].visual_features)
    np.testing.assert_array_equal(restored.nodes[0].pose_covariance, graph.nodes[0].pose_covariance)
    footprint = map_footprint(graph)
    assert footprint["descriptor_bytes"] == 2 * 2 * 4 * 4
    assert footprint["compact_pickle_bytes"] == len(
        pickle.dumps(compact, protocol=pickle.HIGHEST_PROTOCOL)
    )
    assert len(graph.nodes[0].views) == 1


def test_completion_rejects_missing_or_modified_artifacts(tmp_path):
    graph = tmp_path / "final_graph.pkl"
    graph.write_bytes(b"graph")
    assert completed_run(tmp_path) is None
    atomic_json(
        tmp_path / "completion.json",
        {
            "fingerprint": "expected",
            "frames": 2,
            "artifacts": {"final_graph.pkl": file_digest(graph)},
        },
    )
    assert completed_run(tmp_path, "expected")
    assert completed_run(tmp_path, "different") is None
    graph.write_bytes(b"truncated")
    assert completed_run(tmp_path, "expected") is None
    assert json.loads((tmp_path / "completion.json").read_text())["frames"] == 2


def test_navigation_requires_current_explicit_confirmation():
    selection = PendingSelection()
    first_id = selection.offer([node(0)])
    second_id = selection.offer([node(1), node(2)])
    with pytest.raises(ValueError):
        selection.confirm(first_id, 0)
    with pytest.raises(ValueError):
        selection.confirm(second_id, 3)
    assert selection.confirm(second_id, 2).node_id == 2
    with pytest.raises(ValueError):
        selection.confirm(second_id, 2)


@pytest.mark.parametrize(
    "covariance", [-np.eye(2), np.full((2, 2), np.nan), np.diag([-1e-6, -1e-6])]
)
def test_invalid_geometric_models_are_rejected(covariance):
    distance, likelihood = gaussian_position_statistics(np.zeros(2), covariance)
    assert np.isinf(distance) and np.isinf(likelihood)


def test_empty_database_and_zero_width_exclusion_follow_miss_protocol():
    features = np.eye(3)
    positions = np.zeros((3, 2))
    empty = place_recognition_recall(features, positions, np.empty((0, 3)), np.empty((0, 2)))
    assert empty == {"recall_at_1": 0.0, "recall_at_5": 0.0, "n_queries": 3}
    singleton = place_recognition_recall(
        features[:1],
        positions[:1],
        features[:1],
        positions[:1],
        query_index=np.array([0]),
        db_index=np.array([0]),
        exclude_window=0,
    )
    assert singleton["recall_at_1"] == 0.0
