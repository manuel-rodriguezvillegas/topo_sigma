"""Observable contracts for the isolated local verification and memory screen."""
import cv2
import numpy as np

from vts_core.topo_graph import TopoGraph, TopoNode
from vts_evaluation.distinctive_memory import (
    LocalMatcher, MatchEvidence, decide_extra, replace_views, select_views,
)


def test_local_matcher_supports_repeated_scene_with_spatial_coverage():
    rng = np.random.default_rng(9)
    image = np.full((300, 400, 3), 235, dtype=np.uint8)
    for x, y in rng.integers([20, 20], [380, 280], size=(100, 2)):
        cv2.circle(image, (int(x), int(y)), 6, (0, 0, 0), 2)
    shifted = cv2.warpAffine(image, np.float32([[1, 0, 12], [0, 1, 7]]), (400, 300))
    matcher = LocalMatcher()
    matcher.prepare('a', image)
    matcher.prepare('b', shifted)
    matcher.prepare('rival', shifted.copy())
    matcher.prepare('blank', np.full_like(image, 235))
    assert matcher.compare('a', 'b').supported
    assert not matcher.compare_competition('a', 'b', ['rival']).supported
    assert not matcher.compare('a', 'blank').supported


def test_local_evidence_requires_unique_supported_candidate():
    strong = MatchEvidence(inliers=25, coverage=.3)
    weak = MatchEvidence(inliers=10, coverage=.3)
    assert decide_extra({1: strong, 2: weak}) == 1
    assert decide_extra({1: strong, 2: strong}) is None
    assert decide_extra({1: weak}) is None


def test_memory_keeps_representative_and_equal_budget_without_changing_nodes():
    vectors = np.eye(6, dtype=np.float32)
    graph = TopoGraph()
    for i in range(2):
        graph.add_node(TopoNode(i, (float(i), 0., 0.), vectors[i],
                                view_features=np.stack((vectors[i], vectors[i + 2], vectors[i + 4]))))
    features = np.vstack((vectors[0], vectors[2], vectors[4],
                          vectors[1], vectors[3], vectors[5]))
    before = [n.view_features.copy() for n in graph.nodes.values()]
    for policy in ('diverse', 'discriminative'):
        rebuilt, selected = replace_views(graph, features, np.array([0, 3]),
                                          np.array([3, 6]), np.array([0, 3]), policy)
        assert selected[0][0] == 0 and selected[1][0] == 3
        assert len(selected[0]) == len(selected[1]) == 3
        assert all(j < 3 for j in selected[0])
        assert all(3 <= j < 6 for j in selected[1])
        assert all(rebuilt.nodes[i].pose == graph.nodes[i].pose for i in graph.nodes)
    assert all(np.array_equal(graph.nodes[i].view_features, before[i]) for i in graph.nodes)


def test_view_selection_is_causal_and_rejects_invalid_inputs():
    vectors = np.eye(5, dtype=np.float32)
    original = select_views(vectors[:3], [], 0, 'discriminative')
    assert original == select_views(vectors[:3], [], 0, 'discriminative')
    try:
        select_views(np.full((3, 5), np.nan), [], 0, 'diverse')
    except ValueError:
        pass
    else:
        raise AssertionError('Nonfinite descriptors must be rejected')
