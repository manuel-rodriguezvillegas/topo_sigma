"""Recorded SE(2) motion and a path-length uncertainty budget.

This heuristic budget is not a calibrated pose marginal. Its increments depend
on travelled distance (metres) and absolute turning (radians), not sample count.
Subdividing the same piecewise-linear motion leaves the budget unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Pose2D = tuple[float, float, float]


def normalize_angle(angle: float) -> float:
    """Wrap an angle to `[-pi, pi)`."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass(frozen=True)
class OdometryUncertaintyParams:
    """Motion-dependent wheel-odometry budget coefficients.

    `alpha1` and `alpha2` model rotational variance caused by rotation and
    translation. `alpha3` and `alpha4` model translational variance caused
    by translation and rotation. Units are rad²/rad, rad²/m, m²/m,
    and m²/rad respectively. These coefficients require independent calibration.
    """

    alpha1: float = 0.025
    alpha2: float = 0.005
    alpha3: float = 0.01
    alpha4: float = 0.0025

    def __post_init__(self) -> None:
        values = np.asarray([self.alpha1, self.alpha2, self.alpha3, self.alpha4])
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError("odometry uncertainty coefficients must be nonnegative")


class OdometryUncertaintyTracker:
    """Accumulate an additive uncertainty budget for recorded SE(2) motion.

    The returned matrix is monotonic: subtracting two snapshots gives the
    uncertainty accumulated only over that interval. This is the quantity
    used by revisit gating. It does not propagate heading uncertainty into XY. No noise is
    sampled and the recorded trajectory is never modified.
    """

    def __init__(self, params: OdometryUncertaintyParams | None = None) -> None:
        self._params = params or OdometryUncertaintyParams()
        self._previous_pose: Pose2D | None = None
        self._covariance = np.zeros((3, 3), dtype=np.float64)

    @property
    def covariance(self) -> np.ndarray:
        return self._covariance.copy()

    def step(self, measured_pose: Pose2D) -> np.ndarray:
        """Consume a finite recorded pose and return its cumulative 3x3 budget."""
        if np.shape(measured_pose) != (3,) or not np.isfinite(measured_pose).all():
            raise ValueError("measured_pose must contain three finite values")
        measured_pose = tuple(float(value) for value in measured_pose)
        if self._previous_pose is None:
            self._previous_pose = measured_pose
            return self.covariance

        dx = measured_pose[0] - self._previous_pose[0]
        dy = measured_pose[1] - self._previous_pose[1]
        translation = float(np.hypot(dx, dy))
        rotation = normalize_angle(measured_pose[2] - self._previous_pose[2])
        params = self._params
        variance_rotation = params.alpha1 * abs(rotation) + params.alpha2 * translation
        variance_translation = params.alpha3 * translation + params.alpha4 * abs(rotation)
        increment_covariance = np.diag(
            [variance_translation, variance_translation, variance_rotation]
        )
        self._covariance += increment_covariance
        self._covariance = 0.5 * (self._covariance + self._covariance.T)
        self._previous_pose = measured_pose
        return self.covariance
