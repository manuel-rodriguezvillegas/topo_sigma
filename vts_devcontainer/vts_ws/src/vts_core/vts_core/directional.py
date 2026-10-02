"""Directional statistics for hyperspherical visual descriptors.

DINO descriptors are L2-normalized, so they live on a unit hypersphere rather
than in unconstrained Euclidean space. This module models the observations of
one topological place with a von Mises--Fisher (vMF) distribution. It has no
learned parameters and uses NumPy and SciPy for numerical calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isfinite, lgamma, log, log1p, pi, sqrt

import numpy as np
from scipy.special import ive

_MIN_NORM: float = 1e-12
_MAX_CONCENTRATION: float = 1e6


@lru_cache(maxsize=4096)
def vmf_mean_length(dimension: int, concentration: float) -> float:
    """A_d(kappa), using scaled Bessel functions and a small-k fallback."""
    if dimension < 3 or not isfinite(concentration) or concentration < 0:
        raise ValueError("expected dimension >= 3 and finite nonnegative concentration")
    if concentration == 0:
        return 0.0
    order = dimension / 2 - 1
    denominator = float(ive(order, concentration))
    if denominator > 0:
        return float(ive(order + 1, concentration) / denominator)
    # For tiny arguments at high order both Bessel functions underflow.
    # Perron's continued fraction is well conditioned in this regime.
    if concentration >= dimension:
        raise ArithmeticError("Bessel ratio unavailable outside small-argument regime")
    tail = 0.0
    for index in range(128, -1, -1):
        tail = concentration**2 / (dimension + 2 * index + tail)
    return tail / concentration


def vmf_symmetric_kl(
    first_direction: np.ndarray, first_concentration: float,
    second_direction: np.ndarray, second_concentration: float,
) -> float:
    """Half the sum of forward/reverse KL, not a calibrated match probability."""
    a = np.asarray(first_direction, dtype=np.float64)
    b = np.asarray(second_direction, dtype=np.float64)
    if a.ndim != 1 or a.shape != b.shape or not np.isfinite([a, b]).all():
        raise ValueError("expected finite direction vectors of equal shape")
    if min(np.linalg.norm(a), np.linalg.norm(b)) <= _MIN_NORM:
        raise ValueError("directions must be non-zero")
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    ra = vmf_mean_length(a.size, first_concentration)
    rb = vmf_mean_length(b.size, second_concentration)
    natural_delta = first_concentration * a - second_concentration * b
    expectation_delta = ra * a - rb * b
    return max(0.0, float(0.5 * natural_delta @ expectation_delta))


@dataclass(frozen=True)
class VonMisesFisher:
    """Maximum-likelihood directional summary of one visual segment."""

    mean_direction: np.ndarray
    concentration: float
    mean_resultant_length: float
    sample_count: int


def fit_von_mises_fisher(descriptors: np.ndarray) -> VonMisesFisher:
    """Fit a vMF distribution to an ``(n, d)`` descriptor matrix.

    The standard high-dimensional approximation

    ``kappa = r * (d - r**2) / (1 - r**2)``

    is used for the concentration MLE. Concentration cannot be inferred from
    one observation, so singleton segments are represented as uninformative
    (``kappa = 0``). The numerical concentration cap affects only descriptors
    that are indistinguishable at floating-point precision.
    """
    samples = np.asarray(descriptors, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] < 2:
        raise ValueError("descriptors must have shape (n >= 1, d >= 2)")

    norms = np.linalg.norm(samples, axis=1, keepdims=True)
    if not np.isfinite(samples).all() or np.any(norms <= _MIN_NORM):
        raise ValueError("descriptors must be non-zero")
    samples = samples / norms

    resultant = samples.sum(axis=0)
    return fit_vmf_resultant(resultant, samples.shape[0], samples[0])


def fit_vmf_resultant(
    resultant: np.ndarray, sample_count: int, fallback: np.ndarray
) -> VonMisesFisher:
    """Fit from a streaming sum of unit descriptors and their count."""
    dimension = resultant.size
    resultant_norm = float(np.linalg.norm(resultant))
    if resultant_norm <= _MIN_NORM:
        mean_direction = fallback.copy()
        mean_resultant_length = 0.0
    else:
        mean_direction = resultant / resultant_norm
        mean_resultant_length = float(np.clip(resultant_norm / sample_count, 0.0, 1.0))

    concentration = 0.0
    if sample_count > 1 and mean_resultant_length > _MIN_NORM:
        r = min(mean_resultant_length, 1.0 - 1e-12)
        concentration = r * (dimension - r * r) / (1.0 - r * r)
        concentration = min(float(concentration), _MAX_CONCENTRATION)

    return VonMisesFisher(
        mean_direction=mean_direction.astype(np.float32),
        concentration=concentration,
        mean_resultant_length=mean_resultant_length,
        sample_count=sample_count,
    )


def _log_vmf_normalizer(dimension: int, concentration: float) -> float:
    """Log normalizer of a vMF density without a SciPy dependency.

    DINOv2-S has dimension 384. For this high-order Bessel function, the
    uniform asymptotic expansion is both stable and accurate. Two correction
    terms are included; the omitted error is of order ``nu**-3`` where
    ``nu = dimension / 2 - 1``. The exact uniform-density limit is used at
    zero concentration.
    """
    if dimension < 3:
        raise ValueError("vMF evidence requires dimension >= 3")
    if not isfinite(concentration) or concentration < 0:
        raise ValueError("concentration must be finite and nonnegative")
    kappa = float(concentration)
    log_uniform = lgamma(0.5 * dimension) - log(2.0) - 0.5 * dimension * log(pi)
    if kappa == 0.0:
        return log_uniform
    if kappa < 1e-3:
        return log_uniform - kappa * kappa / (2.0 * dimension)

    order = 0.5 * dimension - 1.0
    z = kappa / order
    root = sqrt(1.0 + z * z)
    eta = root + log(z) - log1p(root)
    t = 1.0 / root
    u1 = (3.0 * t - 5.0 * t**3) / 24.0
    u2 = (81.0 * t**2 - 462.0 * t**4 + 385.0 * t**6) / 1152.0
    correction = 1.0 + u1 / order + u2 / (order * order)
    log_bessel = (
        -0.5 * log(2.0 * pi * order)
        - 0.25 * log1p(z * z)
        + order * eta
        + log(max(correction, _MIN_NORM))
    )
    return order * log(kappa) - 0.5 * dimension * log(2.0 * pi) - log_bessel


def vmf_log_overlap_ratio(
    first_direction: np.ndarray,
    first_concentration: float,
    second_direction: np.ndarray,
    second_concentration: float,
) -> float:
    """Return vMF distribution overlap relative to a uniform descriptor.

    A score of zero means that at least one place is visually uninformative.
    Positive values indicate more overlap than two uniform directions;
    strongly incompatible concentrated places yield negative values. The
    score is symmetric and contains no calibrated or hand-picked threshold.
    """
    first = np.asarray(first_direction, dtype=np.float64).reshape(-1)
    second = np.asarray(second_direction, dtype=np.float64).reshape(-1)
    if first.shape != second.shape or first.size < 3:
        raise ValueError("directions must have the same dimension >= 3")
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if (
        not np.isfinite(first).all()
        or not np.isfinite(second).all()
        or first_norm <= _MIN_NORM
        or second_norm <= _MIN_NORM
    ):
        raise ValueError("directions must be non-zero")
    first = first / first_norm
    second = second / second_norm

    if not all(
        isfinite(value) and value >= 0 for value in (first_concentration, second_concentration)
    ):
        raise ValueError("concentrations must be finite and nonnegative")
    kappa_a = float(first_concentration)
    kappa_b = float(second_concentration)
    cosine = float(np.clip(first @ second, -1.0, 1.0))
    combined_squared = kappa_a * kappa_a + kappa_b * kappa_b + 2.0 * kappa_a * kappa_b * cosine
    combined = sqrt(max(combined_squared, 0.0))
    dimension = first.size
    log_uniform = _log_vmf_normalizer(dimension, 0.0)
    return float(
        _log_vmf_normalizer(dimension, kappa_a)
        + _log_vmf_normalizer(dimension, kappa_b)
        - _log_vmf_normalizer(dimension, combined)
        - log_uniform
    )
