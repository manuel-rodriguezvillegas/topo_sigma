"""Independent evidence and normalized-distribution regression checks."""
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from vts_core.directional import vmf_mean_length, vmf_symmetric_kl
from vts_core.mapper import TopologicalMapper
from vts_core.topo_graph import LoopClosureEvent, TopoGraph, TopoNode
from vts_evaluation.closure_comparison import replay


def node(i: int, basis: int) -> TopoNode:
    feature = np.eye(8)[basis]
    return TopoNode(i, (0., 0., 0.), feature, visual_concentration=50.,
                    view_features=feature[None], pose_covariance=np.eye(2) * (i + 1),
                    gt_pose=(float(i), 0., 0.))


@pytest.mark.parametrize('kappa', [0., 1e-12, 1., 384., 1e6])
def test_kl_finite_symmetric_and_fixed_concentration_control(kappa: float) -> None:
    a, b = np.eye(384)[:2]
    value = vmf_symmetric_kl(a, kappa, b, kappa)
    assert np.isfinite(value) and value >= 0
    assert value == pytest.approx(kappa * vmf_mean_length(384, kappa))
    assert vmf_symmetric_kl(a, kappa, a, kappa) == 0
    assert vmf_symmetric_kl(a, kappa, b, 10.) == pytest.approx(
        vmf_symmetric_kl(b, 10., a, kappa))


@pytest.mark.parametrize('kappa', [0.01, 1., 100.])
def test_bessel_ratio_matches_three_dimensional_closed_form(kappa: float) -> None:
    assert vmf_mean_length(3, kappa) == pytest.approx(1 / np.tanh(kappa) - 1 / kappa)


def test_short_return_does_not_gain_self_observation_support() -> None:
    engine = TopologicalMapper()
    try:
        engine.observe_node(node(0, 0))
        engine.observe_node(node(1, 1))
        pairs = []
        def evidence(a: TopoNode, b: TopoNode) -> float:
            pairs.append((a.node_id, b.node_id))
            return float(a.visual_features @ b.visual_features)
        assert engine._sequence_score(node(2, 2), 0, evidence) == 0.
        assert (1, 1) not in pairs
        assert all(a != b for a, b in pairs)
        assert (0, 0) not in pairs
    finally:
        engine.close()


def test_pending_confirmation_requires_independent_history_and_expires() -> None:
    for confirm in (False, True):
        engine = TopologicalMapper()
        try:
            for i in range(6):
                engine.graph.add_node(node(i, i))
                if i:
                    engine.graph.add_edge(i - 1, i)
            engine.current_node_id = 5
            event = LoopClosureEvent(5, 0, False, 0., 'deferred_visual_verification', 5)
            engine.graph.loop_events.append(event)
            engine._pending_revisits.append(event)
            with patch.object(engine, '_find_revisit', return_value=(None, None, None, 'rejected_by_gate')):
                engine.observe_node(node(6, 1 if confirm else 7))
                if confirm:
                    assert event.accepted and event.reason == 'accepted_deferred'
                    assert event.decision_node_id == 6
                else:
                    assert not event.accepted
                    engine.observe_node(node(7, 7))
                    assert event.reason == 'rejected_verification_timeout'
            assert not engine._pending_revisits
        finally:
            engine.close()


def test_replay_matches_online_nodes_without_ground_truth_or_source_edges() -> None:
    source = TopoGraph()
    online = TopologicalMapper(visual_model='mean_cosine')
    try:
        for i, basis in enumerate([0, 1, 2, 3, 0, 1, 7]):
            observation = node(i, basis)
            source.add_node(observation)
            online.observe_node(observation)
        online.finish_revisits()
        source.add_edge(0, 3, edge_type='loop')
        result = replay(source, 'mean_cosine')
        assert result.edge_types == online.graph.edge_types
        assert result.loop_events == online.graph.loop_events
        assert all(n.gt_pose is None for n in result.nodes.values())
        fixed = replay(source, 'kl_fixed')
        assert fixed.edge_types == result.edge_types
        assert [e.reason for e in fixed.loop_events] == [e.reason for e in result.loop_events]
        source.nodes = {i: replace(n, gt_pose=(-999., 999., 0.)) for i, n in source.nodes.items()}
        assert replay(source, 'mean_cosine').loop_events == result.loop_events
    finally:
        online.close()


def test_end_of_stream_never_accepts_pending_hypothesis() -> None:
    engine = TopologicalMapper(require_visual_verification=True)
    try:
        for i, basis in enumerate([0, 1, 2]):
            engine.observe_node(node(i, basis))
        assert engine._pending_revisits
        engine.finish_revisits()
        assert engine.graph.loop_events[-1].reason == 'rejected_end_of_stream'
        assert not engine.graph.loop_events[-1].accepted
    finally:
        engine.close()


@pytest.mark.parametrize('policy,expected', [
    ('legacy', [(3, 0), (2, 1), (1, 2)]),
    ('self_only', [(3, 0), (2, 1), (1, 2)]),
    ('disjoint', [(3, 0)]),
])
def test_valid_cross_observation_context_is_preserved(policy: str, expected: list) -> None:
    engine = TopologicalMapper(sequence_policy=policy)
    try:
        for i in range(3):
            engine.observe_node(node(i, i))
        pairs = []
        def evidence(a: TopoNode, b: TopoNode) -> float:
            pairs.append((a.node_id, b.node_id))
            return 0.
        engine._sequence_score(node(3, 3), 0, evidence)
        assert pairs == expected
    finally:
        engine.close()


def test_unique_candidate_verification_is_opt_in() -> None:
    source = TopoGraph()
    for i in range(3):
        source.add_node(node(i, i))
    baseline = replay(source, 'cosine')
    verified = replay(source, 'cosine', require_visual_verification=True)
    assert baseline.loop_events[-1].accepted
    assert not verified.loop_events[-1].accepted


def test_invalid_sequence_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match='sequence_policy'):
        TopologicalMapper(sequence_policy='typo')
