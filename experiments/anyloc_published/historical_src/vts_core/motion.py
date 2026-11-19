"""Recorded SE(2) motion and the uncertainty it accumulates.

Two models share the coefficients of :class:`OdometryUncertaintyParams`:

- ``additive``: an isotropic path-length budget that never couples heading and position.
- ``propagated``: the linearized odometry motion model of Thrun, Burgard and Fox,
  Probabilistic Robotics (2005), Sec. 5.4, propagated with its Jacobians. Heading
  noise turns into lateral position error, so the position covariance is an ellipse.

In both, noise variances grow linearly with the motion (a random walk), so the
covariance does not depend on how finely the same path is sampled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Pose2D = tuple[float, float, float]

UNCERTAINTY_MODELS: tuple[str, ...] = ("additive", "propagated")


def normalize_angle(angle: float) -> float:
    """Wrap an angle to `[-pi, pi)`."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass(frozen=True)
class OdometryUncertaintyParams:
    """Odometry noise coefficients.

    Rotation variance is ``alpha1 |dtheta| + alpha2 ds`` and translation variance
    ``alpha3 ds + alpha4 |dtheta|``; units rad²/rad, rad²/m, m²/m and m²/rad.
    """

    alpha1: float = 0.025
    alpha2: float = 0.005
    alpha3: float = 0.01
    alpha4: float = 0.0025

    def __post_init__(self) -> None:
        values = np.asarray(self.as_array())
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError("odometry uncertainty coefficients must be nonnegative")

    def as_array(self) -> np.ndarray:
        return np.array([self.alpha1, self.alpha2, self.alpha3, self.alpha4])


def _increment(previous: Pose2D, current: Pose2D) -> tuple[np.ndarray, float, float]:
    displacement = np.array([current[0] - previous[0], current[1] - previous[1]])
    return displacement, float(np.hypot(*displacement)), normalize_angle(current[2] - previous[2])


def _noise_variances(params: OdometryUncertaintyParams, distance: float, turn: float) -> tuple[float, float]:
    translation = params.alpha3 * distance + params.alpha4 * abs(turn)
    rotation = params.alpha1 * abs(turn) + params.alpha2 * distance
    return translation, rotation


def _check_pose(pose: Pose2D) -> Pose2D:
    if np.shape(pose) != (3,) or not np.isfinite(pose).all():
        raise ValueError("pose must contain three finite values")
    return tuple(float(value) for value in pose)


class OdometryUncertaintyTracker:
    """Cumulative isotropic budget ``diag(u, u, v)`` of the additive model."""

    def __init__(self, params: OdometryUncertaintyParams | None = None) -> None:
        self._params = params or OdometryUncertaintyParams()
        self._previous_pose: Pose2D | None = None
        self._covariance = np.zeros((3, 3), dtype=np.float64)

    @property
    def covariance(self) -> np.ndarray:
        return self._covariance.copy()

    def step(self, measured_pose: Pose2D) -> np.ndarray:
        """Consume a recorded pose and return the cumulative 3x3 budget."""
        measured_pose = _check_pose(measured_pose)
        if self._previous_pose is not None:
            _, distance, turn = _increment(self._previous_pose, measured_pose)
            translation, rotation = _noise_variances(self._params, distance, turn)
            self._covariance += np.diag([translation, translation, rotation])
        self._previous_pose = measured_pose
        return self.covariance


class PropagatedOdometryTracker:
    """Cumulative 3x3 pose covariance of dead reckoning, in the odometry frame.

    Each increment is a translation along its own direction plus a rotation,
    with independent noise. The state Jacobian carries heading error into
    position through the lever arm of the increment.
    """

    def __init__(self, params: OdometryUncertaintyParams | None = None) -> None:
        self._params = params or OdometryUncertaintyParams()
        self._previous_pose: Pose2D | None = None
        self._covariance = np.zeros((3, 3), dtype=np.float64)

    @property
    def covariance(self) -> np.ndarray:
        return self._covariance.copy()

    def step(self, measured_pose: Pose2D) -> np.ndarray:
        """Consume a recorded pose and return the cumulative 3x3 covariance."""
        measured_pose = _check_pose(measured_pose)
        if self._previous_pose is not None:
            displacement, distance, turn = _increment(self._previous_pose, measured_pose)
            heading = self._previous_pose[2]
            direction = displacement / distance if distance > 0.0 else np.array([np.cos(heading), np.sin(heading)])
            state_jacobian = np.eye(3)
            state_jacobian[:2, 2] = _perpendicular(displacement)
            noise_jacobian = np.zeros((3, 2))
            noise_jacobian[:2, 0] = direction
            noise_jacobian[2, 1] = 1.0
            noise = np.diag(_noise_variances(self._params, distance, turn))
            covariance = state_jacobian @ self._covariance @ state_jacobian.T + noise_jacobian @ noise @ noise_jacobian.T
            self._covariance = 0.5 * (covariance + covariance.T)
        self._previous_pose = measured_pose
        return self.covariance


def make_tracker(model: str, params: OdometryUncertaintyParams) -> OdometryUncertaintyTracker | PropagatedOdometryTracker:
    if model == "additive":
        return OdometryUncertaintyTracker(params)
    if model == "propagated":
        return PropagatedOdometryTracker(params)
    raise ValueError(f"uncertainty model must be one of {UNCERTAINTY_MODELS}, got {model!r}")


def _perpendicular(vectors: np.ndarray) -> np.ndarray:
    """Rotate 2-D vectors by +90 degrees: the derivative of R(theta) v with respect to theta."""
    vectors = np.asarray(vectors, dtype=np.float64)
    return np.stack([-vectors[..., 1], vectors[..., 0]], axis=-1)


def relative_pose_covariance(
    model: str,
    later_pose: Pose2D | np.ndarray,
    later_covariance: np.ndarray,
    earlier_poses: np.ndarray,
    earlier_covariances: np.ndarray,
) -> np.ndarray:
    """Covariance (n, 3, 3) of the pose change between two snapshots, from the noise accumulated between them.

    The later pose and covariance may be single or one per earlier pose.

    For the propagated model the cumulative covariances satisfy
    ``C_later = Phi C_earlier Phi^T + Q`` with ``Phi = [[I, J (p_later - p_earlier)], [0, 1]]``,
    so the interval covariance ``Q`` is recovered from the two snapshots.
    """
    later_covariance = np.asarray(later_covariance, dtype=np.float64)
    earlier_covariances = np.asarray(earlier_covariances, dtype=np.float64)
    if model == "additive":
        interval = later_covariance - earlier_covariances
    elif model == "propagated":
        lever = _perpendicular(
            np.asarray(later_pose, dtype=np.float64)[..., :2] - np.asarray(earlier_poses, dtype=np.float64)[..., :2]
        )
        transition = np.broadcast_to(np.eye(3), earlier_covariances.shape).copy()
        transition[:, :2, 2] = lever
        interval = later_covariance - transition @ earlier_covariances @ transition.swapaxes(-1, -2)
    else:
        raise ValueError(f"uncertainty model must be one of {UNCERTAINTY_MODELS}, got {model!r}")
    return 0.5 * (interval + interval.swapaxes(-1, -2))
