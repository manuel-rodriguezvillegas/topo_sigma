import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "anyloc_published" / "historical_src"))

from vts_core.motion import (  # noqa: E402
    OdometryUncertaintyParams,
    PropagatedOdometryTracker,
    relative_pose_covariance,
)


def _curvy_path(steps: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    turns = 0.15 * np.sin(np.linspace(0.0, 6.0, steps)) + 0.02 * rng.standard_normal(steps)
    lengths = 0.05 + 0.02 * rng.random(steps)
    poses = np.zeros((steps + 1, 3))
    for t in range(steps):
        x, y, heading = poses[t]
        poses[t + 1] = (x + lengths[t] * np.cos(heading), y + lengths[t] * np.sin(heading), heading + turns[t])
    poses[:, 2] = (poses[:, 2] + np.pi) % (2 * np.pi) - np.pi
    return poses


def _cumulative(poses: np.ndarray, params: OdometryUncertaintyParams) -> np.ndarray:
    tracker = PropagatedOdometryTracker(params)
    return np.stack([tracker.step(tuple(pose)) for pose in poses])


def test_straight_line_matches_continuous_random_walk():
    params = OdometryUncertaintyParams(alpha1=0.0, alpha2=0.004, alpha3=0.01, alpha4=0.0)
    length, steps = 20.0, 2000
    poses = np.column_stack([np.linspace(0.0, length, steps + 1), np.zeros(steps + 1), np.zeros(steps + 1)])
    final = _cumulative(poses, params)[-1]
    assert np.isclose(final[0, 0], params.alpha3 * length)
    assert np.isclose(final[1, 1], params.alpha2 * length**3 / 3.0, rtol=2e-3)
    assert np.isclose(final[1, 2], params.alpha2 * length**2 / 2.0, rtol=2e-3)
    assert np.isclose(final[2, 2], params.alpha2 * length)


def test_interval_covariance_equals_propagation_started_at_earlier_pose():
    params = OdometryUncertaintyParams(alpha1=0.02, alpha2=0.003, alpha3=0.008, alpha4=0.002)
    poses = _curvy_path(400)
    cumulative = _cumulative(poses, params)
    later = 390
    earlier = np.array([10, 150, 300])
    from_closed_form = relative_pose_covariance(
        "propagated", tuple(poses[later]), cumulative[later], poses[earlier], cumulative[earlier]
    )
    for k, j in enumerate(earlier):
        restarted = _cumulative(poses[j : later + 1], params)[-1]
        np.testing.assert_allclose(from_closed_form[k], restarted, rtol=1e-9, atol=1e-12)


def test_matches_monte_carlo_dead_reckoning():
    params = OdometryUncertaintyParams(alpha1=2e-4, alpha2=1e-4, alpha3=2e-4, alpha4=5e-5)
    poses = _curvy_path(150, seed=3)
    predicted = _cumulative(poses, params)[-1]

    rng = np.random.default_rng(7)
    samples = 40000
    estimate = np.tile(poses[0], (samples, 1))
    for t in range(len(poses) - 1):
        world = poses[t + 1, :2] - poses[t, :2]
        heading = poses[t, 2]
        body = np.array([np.cos(heading) * world[0] + np.sin(heading) * world[1],
                         -np.sin(heading) * world[0] + np.cos(heading) * world[1]])
        distance = np.hypot(*body)
        turn = (poses[t + 1, 2] - heading + np.pi) % (2 * np.pi) - np.pi
        var_s = params.alpha3 * distance + params.alpha4 * abs(turn)
        var_theta = params.alpha1 * abs(turn) + params.alpha2 * distance
        noisy_body = body[None, :] * (1.0 + rng.normal(0.0, np.sqrt(var_s), samples)[:, None] / distance)
        cos, sin = np.cos(estimate[:, 2]), np.sin(estimate[:, 2])
        estimate[:, 0] += cos * noisy_body[:, 0] - sin * noisy_body[:, 1]
        estimate[:, 1] += sin * noisy_body[:, 0] + cos * noisy_body[:, 1]
        estimate[:, 2] += turn + rng.normal(0.0, np.sqrt(var_theta), samples)
    error = estimate - poses[-1]
    empirical = np.cov(error.T)
    assert np.linalg.norm(empirical - predicted) / np.linalg.norm(predicted) < 0.03
