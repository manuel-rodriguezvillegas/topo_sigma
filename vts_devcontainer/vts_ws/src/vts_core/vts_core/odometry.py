"""Recorded SE(2) odometry: uncertainty budget, interpolation and propagation.

Two uncertainty models coexist on purpose:

``budget``
    The additive path-length budget used by the in-session gate since the
    manuscript (``path_length_budget_v1``). Increments depend on travelled
    distance and absolute turning, never on sample count, and heading error is
    not propagated into position. Interval uncertainty between two frames is the
    difference of their cumulative snapshots. Kept bit-compatible with the
    recorded runs.
``RelativeUncertainty``
    First-order SE(2) propagation with Jacobians, expressed in the frame of a
    reset point (e.g. the last cross-session anchor). Heading uncertainty grows
    lateral position uncertainty, which the budget cannot represent.

Both use the same motion coefficients and are heuristics until calibrated.
The recorded trajectory is never modified.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Pose2D = tuple[float, float, float]


def normalize_angle(angle):
    """Wrap angles (scalar or array) to ``[-pi, pi)``."""
    return (np.asarray(angle, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class MotionNoise:
    """Odometry noise coefficients.

    ``alpha1``/``alpha2``: rotational variance per radian turned / metre
    travelled; ``alpha3``/``alpha4``: translational variance per metre / radian.
    Units rad²/rad, rad²/m, m²/m and m²/rad.
    """

    alpha1: float = 0.025
    alpha2: float = 0.005
    alpha3: float = 0.01
    alpha4: float = 0.0025

    def __post_init__(self) -> None:
        values = np.asarray([self.alpha1, self.alpha2, self.alpha3, self.alpha4])
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError("motion noise coefficients must be finite and nonnegative")

    def variances(self, translation, rotation) -> tuple[np.ndarray, np.ndarray]:
        """Translational and rotational variance of increments (arrays allowed)."""
        rotation = np.abs(rotation)
        return (self.alpha3 * translation + self.alpha4 * rotation,
                self.alpha1 * rotation + self.alpha2 * translation)


def _increments(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    steps = np.diff(poses, axis=0)
    return np.hypot(steps[:, 0], steps[:, 1]), normalize_angle(steps[:, 2])


def budget(poses: np.ndarray, noise: MotionNoise) -> np.ndarray:
    """Cumulative additive budget (N, 3, 3) at every recorded pose."""
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 3 or not np.isfinite(poses).all():
        raise ValueError("poses must be a finite (N, 3) array")
    translation, rotation = _increments(poses)
    var_t, var_r = noise.variances(translation, rotation)
    out = np.zeros((len(poses), 3, 3))
    out[1:, 0, 0] = out[1:, 1, 1] = np.cumsum(var_t)
    out[1:, 2, 2] = np.cumsum(var_r)
    return out


class BudgetTracker:
    """Streaming form of :func:`budget` for live odometry publishers."""

    def __init__(self, noise: MotionNoise | None = None) -> None:
        self._noise = noise or MotionNoise()
        self._previous: np.ndarray | None = None
        self._budget = np.zeros((3, 3))

    def step(self, pose: Pose2D) -> np.ndarray:
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError("pose must contain three finite values")
        if self._previous is not None:
            translation, rotation = _increments(np.stack([self._previous, pose]))
            var_t, var_r = self._noise.variances(translation[0], rotation[0])
            self._budget = self._budget + np.diag([var_t, var_t, var_r])
        self._previous = pose
        return self._budget.copy()


def interval_covariance(earlier: np.ndarray, later: np.ndarray) -> np.ndarray:
    """PSD budget accumulated between two cumulative snapshots (``..., n, n``)."""
    difference = np.asarray(later, dtype=np.float64) - np.asarray(earlier, dtype=np.float64)
    difference = 0.5 * (difference + np.swapaxes(difference, -1, -2))
    if np.min(np.linalg.eigvalsh(difference)) < -1e-9:
        raise ValueError("uncertainty budget decreased; pose marginals are not supported")
    return difference


def bracket(times: np.ndarray, query: np.ndarray, max_gap_s: float, exact_atol: float):
    """Bracketing samples of each query: ``(lo, hi, fraction, valid)``.

    A query within ``exact_atol`` of a sample takes that sample (``fraction``
    0, ``lo == hi``). Otherwise it needs samples on both sides at most
    ``max_gap_s`` apart; ``valid`` is False where that fails.
    """
    times = np.asarray(times, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    last = len(times) - 1
    upper = np.searchsorted(times, query, side="left")
    hi = np.minimum(upper, last)
    exact = (upper <= last) & (np.abs(times[hi] - query) <= exact_atol)
    lo = np.where(exact, hi, np.maximum(upper - 1, 0))
    gap = times[hi] - times[lo]
    covered = (query >= times[0]) & (query <= times[-1])
    valid = covered & (exact | ((upper > 0) & (upper <= last) & (gap > 0.0) & (gap <= max_gap_s)))
    fraction = np.where(exact | ~valid, 0.0, (query - times[lo]) / np.where(gap > 0, gap, 1.0))
    return lo, hi, fraction, valid


def interpolate(times: np.ndarray, poses: np.ndarray, budgets: np.ndarray, query: np.ndarray,
                max_gap_s: float, exact_atol: float) -> tuple[np.ndarray, np.ndarray]:
    """Linear pose and budget interpolation at ``query`` timestamps.

    Raises if any query is outside coverage or across a gap larger than
    ``max_gap_s``: such frames cannot be mapped.
    """
    lo, hi, fraction, valid = bracket(times, query, max_gap_s, exact_atol)
    if not valid.all():
        raise ValueError(f"{int((~valid).sum())} timestamps outside odometry coverage or across a gap "
                         f"above {max_gap_s:.3f}s")
    start, end = poses[lo], poses[hi]
    out_pose = np.empty((len(lo), 3))
    out_pose[:, :2] = start[:, :2] + fraction[:, None] * (end[:, :2] - start[:, :2])
    out_pose[:, 2] = normalize_angle(start[:, 2] + fraction * normalize_angle(end[:, 2] - start[:, 2]))
    cov = (1.0 - fraction)[:, None, None] * budgets[lo] + fraction[:, None, None] * budgets[hi]
    cov = 0.5 * (cov + np.swapaxes(cov, -1, -2))
    exact = lo == hi
    out_pose[exact] = poses[hi[exact]]
    cov[exact] = budgets[hi[exact]]
    return out_pose, cov


def path_length(poses: np.ndarray) -> np.ndarray:
    """Cumulative travelled distance (m) at every pose."""
    steps = np.hypot(*np.diff(np.asarray(poses)[:, :2], axis=0).T)
    return np.r_[0.0, np.cumsum(steps)]


def relative_pose(origin: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """``pose`` expressed in the frame of ``origin`` (broadcasts over rows)."""
    origin, pose = np.asarray(origin, dtype=np.float64), np.asarray(pose, dtype=np.float64)
    c, s = np.cos(origin[..., 2]), np.sin(origin[..., 2])
    dx, dy = pose[..., 0] - origin[..., 0], pose[..., 1] - origin[..., 1]
    return np.stack([c * dx + s * dy, -s * dx + c * dy, normalize_angle(pose[..., 2] - origin[..., 2])], -1)


class RelativeUncertainty:
    """First-order SE(2) covariance of the current pose relative to a reset pose.

    Each recorded increment ``u = (dx, dy, dtheta)`` (in the previous robot
    frame) is treated as noisy with covariance ``diag(vt, vt, vr)`` from
    :class:`MotionNoise`. Composition ``x' = x ⊕ u`` propagates

        Sigma' = F Sigma F^T + G Q G^T,

    with ``F = d(x ⊕ u)/dx`` and ``G = d(x ⊕ u)/du``, all in the reset frame.
    """

    def __init__(self, noise: MotionNoise | None = None) -> None:
        self._noise = noise or MotionNoise()
        self._origin: np.ndarray | None = None
        self._previous: np.ndarray | None = None
        self._relative = np.zeros(3)
        self.covariance = np.zeros((3, 3))

    def reset(self, pose: Pose2D) -> None:
        self._origin = self._previous = np.asarray(pose, dtype=np.float64)
        self._relative = np.zeros(3)
        self.covariance = np.zeros((3, 3))

    def step(self, pose: Pose2D) -> np.ndarray:
        pose = np.asarray(pose, dtype=np.float64)
        if self._previous is None:
            self.reset(pose)
            return self.covariance.copy()
        u = relative_pose(self._previous, pose)
        var_t, var_r = self._noise.variances(np.hypot(u[0], u[1]), u[2])
        theta = self._relative[2]
        c, s = np.cos(theta), np.sin(theta)
        f = np.array([[1.0, 0.0, -s * u[0] - c * u[1]], [0.0, 1.0, c * u[0] - s * u[1]], [0.0, 0.0, 1.0]])
        g = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        self.covariance = f @ self.covariance @ f.T + g @ np.diag([var_t, var_t, var_r]) @ g.T
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        self._relative = relative_pose(self._origin, pose)
        self._previous = pose
        return self.covariance.copy()
