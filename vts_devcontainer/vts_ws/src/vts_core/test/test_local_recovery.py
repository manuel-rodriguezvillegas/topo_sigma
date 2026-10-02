from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from vts_core.local_recovery import RecoveryConfig, local_track
from vts_core.mapper import TopologicalMapper
from vts_core.topo_graph import TopoGraph, TopoNode


def observation(i, xy, basis):
    vector = np.eye(12)[basis]
    return TopoNode(i, (*xy, 0.), vector, view_features=vector[None],
                    pose_covariance=np.eye(2) * i * .01, gt_pose=(*xy, 0.))


def track(reverse=False):
    graph = TopoGraph()
    historical = np.array([[0., 0.], [2., 0.], [2., 2.]])
    live = historical[::-1] if reverse else historical
    live = live @ np.array([[0., -1.], [1., 0.]]).T + [100., -50.]
    for i in range(10):
        xy = historical[i] if i < 3 else live[i - 7] if i >= 7 else [i * 20., 0.]
        basis = i if i < 7 else 9 - i if reverse else i - 7
        graph.add_node(observation(i, xy, basis))
        if i:
            graph.add_edge(i - 1, i)
    return graph


@pytest.mark.parametrize('reverse', [False, True])
def test_local_alignment_is_rigid_frame_invariant_and_bidirectional(reverse):
    graph = track(reverse)
    row = local_track(graph, 9, 0 if reverse else 2, 9, -1 if reverse else 1,
                      TopologicalMapper._node_similarity, RecoveryConfig())
    assert row['passed']
    assert row['max_residual_m'] < 1e-10
    assert not set(row['live_ids']) & set(row['historical_ids'])


def test_scale_error_is_not_fitted_away():
    graph = track()
    for i in range(7, 10):
        n = graph.nodes[i]
        graph.nodes[i] = replace(n, pose=(n.pose[0] * 10, n.pose[1] * 10, 0.))
    row = local_track(graph, 9, 2, 9, 1, TopologicalMapper._node_similarity, RecoveryConfig())
    assert row['reason'] == 'local_geometry_mismatch'
    assert row['visual_passed']


def test_straight_corridor_and_static_camera_are_not_sufficient():
    for step in (0., 2.):
        graph = track()
        for ids, offset in ((range(3), 0.), (range(7, 10), 100.)):
            for k, i in enumerate(ids):
                graph.nodes[i] = replace(graph.nodes[i], pose=(offset + step * k, 0., 0.))
        row = local_track(graph, 9, 2, 9, 1, TopologicalMapper._node_similarity, RecoveryConfig())
        assert row['reason'] == 'unobservable_local_shape'


def test_self_support_and_loop_edges_cannot_verify_a_track():
    graph = track()
    row = local_track(graph, 4, 2, 4, 1, TopologicalMapper._node_similarity, RecoveryConfig())
    assert row['reason'] == 'shared_observations'
    graph.edge_types[(0, 1)] = 'loop'
    row = local_track(graph, 9, 2, 9, 1, TopologicalMapper._node_similarity, RecoveryConfig())
    assert row['reason'] == 'missing_odometry_edge'


def test_uniform_appearance_cannot_confirm_geometry_alone():
    graph = track()
    for i, n in graph.nodes.items():
        graph.nodes[i] = replace(n, visual_features=np.eye(12)[0], view_features=np.eye(12)[0:1])
    row = local_track(graph, 9, 2, 9, 1, TopologicalMapper._node_similarity, RecoveryConfig())
    assert row['reason'] == 'weak_visual_support'


def test_confirmation_uses_only_arrived_observations_and_never_ground_truth():
    source = track()
    engines = [TopologicalMapper(recovery_mode='local') for _ in range(2)]
    try:
        for i, n in source.nodes.items():
            for index, engine in enumerate(engines):
                # Force base rejection to isolate the recovery state machine.
                with patch.object(engine, '_find_revisit', return_value=(None, None, None, 'rejected_by_gate')):
                    engine.observe_node(replace(n, gt_pose=None if index else (-999., 999., 1.)))
            if i == 7:
                assert not engines[0].graph.loop_events[-1].accepted
                assert engines[0].graph.loop_events[-1].reason == 'pending_local_recovery'
            if i == 8:
                assert not next(e for e in engines[0].graph.loop_events if e.query_node_id == 7).accepted
        for engine in engines:
            event = next(e for e in engine.graph.loop_events if e.query_node_id == 7)
            assert event.accepted and event.matched_node_id == 0
            assert event.decision_node_id == 9
            assert event.reason == 'accepted_local_recovery'
            engine.finish_revisits()
            assert not engine._recovery_pending
        assert engines[0].graph.loop_events == engines[1].graph.loop_events
    finally:
        for engine in engines:
            engine.close()


def test_end_of_stream_rejects_pending_and_roundtrip_keeps_diagnostics(tmp_path):
    engine = TopologicalMapper(recovery_mode='local')
    try:
        for i, n in track().nodes.items():
            if i > 7:
                break
            with patch.object(engine, '_find_revisit', return_value=(None, None, None, 'rejected_by_gate')):
                engine.observe_node(n)
        engine.finish_revisits()
        assert engine.graph.loop_events[-1].reason == 'recovery_end_of_stream'
        assert not engine.graph.loop_events[-1].accepted
        path = tmp_path / 'graph.pkl'
        engine.graph.save(str(path))
        assert TopoGraph.load(str(path)).recovery_diagnostics == engine.graph.recovery_diagnostics
    finally:
        engine.close()


def test_recovery_preserves_accepted_base_decisions():
    engine = TopologicalMapper(recovery_mode='local')
    try:
        for i, node in track().nodes.items():
            decision = (0, 0, .9, 'accepted') if i >= 2 else (
                None, None, None, 'rejected_by_gate')
            with patch.object(engine, '_find_revisit', return_value=decision):
                engine.observe_node(node)
        assert all(e.accepted and e.matched_node_id == 0 and e.reason == 'accepted'
                   for e in engine.graph.loop_events if e.query_node_id >= 2)
        assert not engine._recovery_pending
    finally:
        engine.close()


def test_multiple_consistent_places_remain_ambiguous():
    engine = TopologicalMapper(recovery_mode='local', recovery_horizon=0)
    try:
        for i, node in track().nodes.items():
            with (
                patch.object(engine, '_find_revisit', return_value=(
                    None, None, None, 'rejected_by_gate')),
                patch('vts_core.mapper.local_track', side_effect=lambda graph, q, m, c, d, sim, cfg: {
                    'matched_node_id': m, 'passed': True,
                }),
            ):
                engine.observe_node(node)
        event = engine.graph.loop_events[-1]
        assert not event.accepted
        assert event.reason == 'recovery_ambiguous'
        assert not engine._recovery_pending
    finally:
        engine.close()


@pytest.mark.parametrize('kwargs', [{'horizon': 3}, {'top_k': 0}, {'residual_m': float('nan')}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        RecoveryConfig(**kwargs)
