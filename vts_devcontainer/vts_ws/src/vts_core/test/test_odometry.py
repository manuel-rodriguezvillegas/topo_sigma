import numpy as np
import pytest

from vts_core.odometry import (
    BudgetTracker,
    MotionNoise,
    RelativeUncertainty,
    budget,
    interpolate,
    path_length,
    relative_pose,
)


def test_budget_matches_streaming_tracker(rng):
    poses = np.cumsum(rng.normal(0, 0.1, (200, 3)), axis=0)
    tracker = BudgetTracker()
    streamed = np.stack([tracker.step(p) for p in poses])
    assert np.allclose(budget(poses, MotionNoise()), streamed, rtol=0, atol=1e-12)


def test_budget_independent_of_subdivision():
    coarse = np.array([[0.0, 0, 0], [2.0, 0, 0]])
    fine = np.column_stack([np.linspace(0, 2, 21), np.zeros(21), np.zeros(21)])
    assert np.allclose(budget(coarse, MotionNoise())[-1], budget(fine, MotionNoise())[-1])


def test_interpolate_exact_linear_and_gaps():
    times = np.array([0.0, 1.0, 2.0, 5.0])
    poses = np.array([[0.0, 0, 0], [1, 0, 3.1], [2, 0, -3.1], [3, 0, 0]])
    budgets = np.arange(4)[:, None, None] * np.eye(3)
    pose, cov = interpolate(times, poses, budgets, np.array([1.0, 0.5, 1.5]), 1.0, 1e-9)
    assert np.allclose(pose[0], poses[1]) and np.allclose(cov[1], 0.5 * np.eye(3))
    # Heading interpolates across the +-pi wrap, not through zero.
    assert abs(abs(pose[2, 2]) - np.pi) < 0.1
    with pytest.raises(ValueError):
        interpolate(times, poses, budgets, np.array([3.0]), 1.0, 1e-9)  # 3 s gap
    with pytest.raises(ValueError):
        interpolate(times, poses, budgets, np.array([6.0]), 10.0, 1e-9)  # outside coverage


def test_relative_uncertainty_matches_monte_carlo():
    noise = MotionNoise(0.01, 0.002, 0.004, 0.001)
    steps = np.array([[0.2, 0.0, 0.05]] * 30)  # an arc, local increments
    poses = [np.zeros(3)]
    for u in steps:
        c, s = np.cos(poses[-1][2]), np.sin(poses[-1][2])
        poses.append(poses[-1] + [c * u[0] - s * u[1], s * u[0] + c * u[1], u[2]])
    tracker = RelativeUncertainty(noise)
    for p in poses:
        predicted = tracker.step(p)
    rng = np.random.default_rng(0)
    finals = []
    for _ in range(4000):
        x = np.zeros(3)
        for u in steps:
            vt, vr = noise.variances(np.hypot(u[0], u[1]), u[2])
            n = u + rng.normal(0, [np.sqrt(vt), np.sqrt(vt), np.sqrt(vr)])
            c, s = np.cos(x[2]), np.sin(x[2])
            x = x + [c * n[0] - s * n[1], s * n[0] + c * n[1], n[2]]
        finals.append(x)
    empirical = np.cov(np.asarray(finals).T)
    assert np.all(np.linalg.eigvalsh(predicted) > 0)
    assert np.allclose(predicted, empirical, rtol=0.15, atol=2e-4)
    # Heading error grows lateral uncertainty beyond the additive budget.
    assert predicted[1, 1] > budget(np.asarray(poses), noise)[-1][1, 1]


def test_relative_pose_and_path_length():
    origin = np.array([1.0, 1.0, np.pi / 2])
    assert np.allclose(relative_pose(origin, np.array([1.0, 2.0, np.pi / 2])), [1, 0, 0], atol=1e-12)
    assert np.allclose(path_length(np.array([[0, 0, 0], [3, 4, 0], [3, 5, 1.0]])), [0, 5, 6])
