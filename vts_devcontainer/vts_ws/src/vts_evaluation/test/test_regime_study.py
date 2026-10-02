"""Behavioral checks for the C1 regime study (drift and cross-session stages)."""
import numpy as np
import pytest

from vts_evaluation.regime_study import (
    DRIFT_LEVELS,
    DriftLevel,
    confidence,
    corrupt_odometry,
    pr_curve,
    segment,
    sequence_matrix,
)


def square_path(step: float = 0.1) -> np.ndarray:
    poses, heading, xy = [], 0.0, np.zeros(2)
    for side in range(4):
        for _ in range(int(2.0 / step)):
            poses.append((*xy, heading))
            xy = xy + step * np.array([np.cos(heading), np.sin(heading)])
        heading += np.pi / 2
    poses.append((*xy, heading))
    out = np.asarray(poses)
    out[:, 2] = (out[:, 2] + np.pi) % (2 * np.pi) - np.pi
    return out


def test_zero_drift_is_identity():
    poses = square_path()
    out = corrupt_odometry(poses, DRIFT_LEVELS[0], np.random.default_rng(0))
    np.testing.assert_array_equal(out, poses)


def test_integration_without_noise_reproduces_trajectory():
    # A level with zero noise but nonzero fields goes through re-integration.
    level = DriftLevel("tiny", 1e-15, 1e-15, 1e-15)
    poses = square_path()
    out = corrupt_odometry(poses, level, np.random.default_rng(0))
    np.testing.assert_allclose(out[:, :2], poses[:, :2], atol=1e-9)


def test_scale_bias_scales_path_length():
    poses = square_path()
    level = DriftLevel("scale", 0.0, 0.1, 0.0)
    rng = np.random.default_rng(3)
    expected_scale = 1.0 + np.random.default_rng(3).normal(0.0, 0.1)
    out = corrupt_odometry(poses, level, rng)
    length = lambda p: np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).sum()
    assert length(out) == pytest.approx(expected_scale * length(poses), rel=1e-9)


def test_drift_grows_with_level():
    poses = square_path()
    errors = []
    for level in DRIFT_LEVELS[1:]:
        final = [np.linalg.norm(corrupt_odometry(poses, level, np.random.default_rng(s))[-1, :2]
                                - poses[-1, :2]) for s in range(30)]
        errors.append(np.median(final))
    assert all(a < b for a, b in zip(errors, errors[1:]))


def test_segmentation_partitions_every_frame_once():
    rng = np.random.default_rng(0)
    places = rng.normal(size=(4, 32))
    descriptors = np.concatenate([p + 0.05 * rng.normal(size=(60, 32)) for p in places])
    descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True)
    segments = segment(descriptors.astype(np.float32), {"window_size": 30, "valley_k": 2.0})
    flat = np.concatenate(segments)
    np.testing.assert_array_equal(flat, np.arange(len(descriptors)))


def test_sequence_matrix_rewards_consistent_reverse_traversal():
    direct = np.full((5, 5), 0.1)
    # Query i matches reference 4 - i (reverse traversal).
    for i in range(5):
        direct[i, 4 - i] = 0.9
    seq = sequence_matrix(direct, context=2)
    assert int(seq[4].argmax()) == 0
    assert seq[4, 0] == pytest.approx(0.9)


def test_confidence_mad_excludes_winner():
    scores = np.array([[0.0, 1.0, 2.0, 10.0]])
    top, z = confidence(scores, "mad_z")
    assert top[0] == 3
    assert z[0] == pytest.approx((10 - 1) / (1.4826 * 1.0))


def test_pr_curve_counts_only_positive_queries_in_recall():
    correct = np.array([True, False, True, False])
    positive = np.array([True, True, True, False])
    statistic = np.array([4.0, 3.0, 2.0, 1.0])
    curve = pr_curve(correct, positive, statistic)
    assert curve["positives"] == 3
    assert curve["recall_at_1"] == pytest.approx(2 / 3)
    # Accepting the first query gives P=1 at R=1/3; first three give P=2/3 at R=2/3.
    assert curve["average_precision"] == pytest.approx(1 / 3 * 1.0 + 1 / 3 * 2 / 3)
    assert curve["recall_at_p90"] == pytest.approx(1 / 3)
