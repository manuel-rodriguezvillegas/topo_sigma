"""Behavioral checks for the isolated frame-based recovery experiment."""
from dataclasses import replace

import numpy as np
import pytest

from vts_evaluation.frame_recovery import FrameConfig, align, recover, select_endpoint
from vts_evaluation.frame_recovery_comparison import CacheOnly


def sequence(reverse: bool = False) -> tuple[np.ndarray, np.ndarray, int]:
    # Historical traversal has twice as many observations as the revisit.
    historical = np.array([[min(i, 6) * .5, max(i - 6, 0) * .5] for i in range(13)])
    indices = np.arange(0, 13, 2)
    if reverse:
        indices = indices[::-1]
    features = np.eye(40)
    features[33:40] = features[indices]
    xy = np.zeros((40, 2))
    xy[:13] = historical
    xy[13:33] = np.column_stack((np.arange(20) + 50., np.zeros(20)))
    rotation = np.array([[0., -1.], [1., 0.]])
    xy[33:40] = historical[indices] @ rotation.T + [100., -50.]
    return features, xy, int(indices[-1])


@pytest.mark.parametrize('reverse', [False, True])
def test_flexible_alignment_recovers_variable_sampling_in_both_directions(reverse: bool):
    features, xy, endpoint = sequence(reverse)
    rows = recover(features, xy, 39, 13, FrameConfig())
    assert select_endpoint(rows) == endpoint
    passed = [row for row in rows if row['passed']]
    assert all(len(set(row['historical'])) == 7 for row in passed)
    assert all(not set(row['historical']) & set(row['live']) for row in passed)
    assert all(row['max_residual_m'] < 1e-10 for row in passed)
    assert select_endpoint(recover(features, xy, 39, 13, FrameConfig(), flexible=False)) is None


def test_future_observations_cannot_change_decision():
    features, xy, _ = sequence()
    expected = recover(features, xy, 39, 13, FrameConfig())
    features = np.vstack([features, np.full((20, 40), np.nan)])
    xy = np.vstack([xy, np.full((20, 2), np.nan)])
    assert recover(features, xy, 39, 13, FrameConfig()) == expected


def test_static_and_aliased_visual_tracks_cannot_confirm():
    features, xy, _ = sequence()
    rows = recover(features, np.zeros_like(xy), 39, 13, FrameConfig())
    assert select_endpoint(rows) is None
    features[:] = features[0]
    assert select_endpoint(recover(features, xy, 39, 13, FrameConfig())) is None


def test_scale_change_is_not_explained_by_rigid_alignment():
    features, xy, _ = sequence()
    xy[33:] *= 10
    rows = recover(features, xy, 39, 13, FrameConfig())
    assert select_endpoint(rows) is None
    assert any(row.get('visual_passed') and row['reason'] == 'local_geometry_mismatch' for row in rows)


def test_ambiguity_is_resolved_before_node_ownership():
    rows = [{'endpoint': 2, 'passed': True, 'matched_node_id': 0},
            {'endpoint': 3, 'passed': True, 'matched_node_id': 0}]
    assert select_endpoint(rows) is None
    rows[1]['matched_node_id'] = 1  # A subdivision does not make this ambiguous anew.
    assert select_endpoint(rows) is None
    rows[1]['endpoint'] = 2  # Two directions of one frame are the same hypothesis.
    assert select_endpoint(rows) == 2


def test_alignment_cannot_reuse_one_frame_for_several_observations():
    assert align(np.zeros((7, 3)), flexible=True, penalty=.03) is None
    cost = np.ones((3, 5))
    cost[np.arange(3), [0, 2, 4]] = 0
    assert align(cost, flexible=True, penalty=.03) == [0, 2, 4]
    assert align(cost, flexible=False, penalty=.03) == [2, 3, 4]


def test_cache_miss_never_triggers_model_inference():
    with pytest.raises(RuntimeError, match='inference is prohibited'):
        CacheOnly().extract(np.zeros((2, 2, 3)))


@pytest.mark.parametrize('field,value', [('pairs', 2), ('top_k', 0), ('spacing_m', float('nan'))])
def test_invalid_configuration(field: str, value: float):
    with pytest.raises(ValueError):
        replace(FrameConfig(), **{field: value})
