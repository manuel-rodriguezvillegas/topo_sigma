"""Matching primitives for visual loop closure and geometric gating.

Metric gating applies a nominal chi-square threshold to Mahalanobis distance.
The mapper supplies an additive uncertainty budget and a co-location floor;
its statistical coverage requires independent calibration. Visual ranking and its MAD-relative acceptance test
live together in :mod:`vts_core.mapper`.
"""

from __future__ import annotations

import numpy as np

# 95th percentile of chi-square with two degrees of freedom.
_CHI2_GATE_2DOF: float = 5.991464547107979


def mahalanobis_gate(delta_xy: np.ndarray, covariance_xy: np.ndarray) -> bool:
    """Chi-square gate (95%, 2 dof) on a position difference.

    Args:
        delta_xy: 2-vector difference between two positions.
        covariance_xy: 2x2 covariance of that difference.

    Returns:
        True if the difference is statistically compatible with zero.
    """
    d2 = mahalanobis_distance_squared(delta_xy, covariance_xy)
    return bool(np.isfinite(d2) and d2 <= _CHI2_GATE_2DOF)


def mahalanobis_distance_squared(delta_xy: np.ndarray, covariance_xy: np.ndarray) -> float:
    """Return the normalized innovation squared for a 2-D displacement."""
    return float(gaussian_position_statistics(delta_xy, covariance_xy)[0])


def gaussian_position_nll(delta_xy: np.ndarray, covariance_xy: np.ndarray) -> float:
    """Position NLL including covariance volume, up to a shared constant."""
    return float(gaussian_position_statistics(delta_xy, covariance_xy)[1])


def gaussian_position_statistics(
    delta_xy: np.ndarray, covariance_xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Batched squared distances and NLLs using one regularization and solve."""
    cov = np.asarray(covariance_xy, dtype=np.float64)
    delta = np.asarray(delta_xy, dtype=np.float64)
    if cov.shape[-2:] != (2, 2) or delta.shape[-1:] != (2,):
        raise ValueError("expected (..., 2, 2) covariance and (..., 2) displacement")
    cov = 0.5 * (cov + np.swapaxes(cov, -1, -2)) + np.eye(2) * 1e-6
    finite = np.isfinite(cov).all(axis=(-2, -1))
    safe_cov = np.where(finite[..., None, None], cov, np.eye(2))
    sign, logdet = np.linalg.slogdet(safe_cov)
    valid = finite & (safe_cov[..., 0, 0] > 0) & (sign > 0)
    safe_cov = np.where(valid[..., None, None], safe_cov, np.eye(2))
    solution = np.linalg.solve(safe_cov, delta[..., None])[..., 0]
    d2 = np.einsum("...i,...i->...", delta, solution)
    valid = valid & np.isfinite(d2) & (d2 >= 0)
    return np.where(valid, d2, np.inf), np.where(valid, 0.5 * (d2 + logdet), np.inf)


def fit_se2(points_a: np.ndarray, points_b: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Least-squares SE(2) fit (Umeyama without scale)."""
    if points_a.shape[0] < 2:
        return None
    centroid_a: np.ndarray = points_a.mean(axis=0)
    centroid_b: np.ndarray = points_b.mean(axis=0)
    centered_a: np.ndarray = points_a - centroid_a
    centered_b: np.ndarray = points_b - centroid_b

    h: np.ndarray = centered_a.T @ centered_b
    u, _, vt = np.linalg.svd(h)
    d: float = float(np.sign(np.linalg.det(vt.T @ u.T)))
    correction: np.ndarray = np.diag([1.0, d])
    rotation: np.ndarray = vt.T @ correction @ u.T
    translation: np.ndarray = centroid_b - rotation @ centroid_a
    return rotation, translation


def median_spacing(points: np.ndarray) -> float:
    """Median nearest-neighbor distance of a point set (data-driven scale).

    Args:
        points: (N, 2) positions.

    Returns:
        Median NN distance, or 1.0 for degenerate inputs.
    """
    n: int = points.shape[0]
    if n < 2:
        return 1.0
    deltas: np.ndarray = points[:, None, :] - points[None, :, :]
    distances: np.ndarray = np.linalg.norm(deltas, axis=2)
    np.fill_diagonal(distances, np.inf)
    nn: np.ndarray = distances.min(axis=1)
    return float(np.median(nn))
