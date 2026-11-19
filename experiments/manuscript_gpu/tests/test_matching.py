import sys
from pathlib import Path

import numpy as np
from scipy.stats import ncx2

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "anyloc_published" / "historical_src"))

from vts_core.matching import disk_probability  # noqa: E402


def test_isotropic_matches_noncentral_chi_square():
    radius = 2.0
    for sigma in (0.01, 0.3, 1.0, 5.0):
        for distance in (0.0, 1.0, 1.99, 2.01, 4.0):
            got = disk_probability(np.array([distance, 0.0]), sigma**2 * np.eye(2), radius)[0]
            want = ncx2.cdf(radius**2 / sigma**2, df=2, nc=distance**2 / sigma**2)
            assert abs(got - want) < 2e-4, (sigma, distance, got, want)


def test_anisotropic_matches_monte_carlo():
    rng = np.random.default_rng(0)
    rotation = np.array([[np.cos(0.7), -np.sin(0.7)], [np.sin(0.7), np.cos(0.7)]])
    covariance = rotation @ np.diag([0.05**2, 2.5**2]) @ rotation.T
    mean = np.array([1.2, -0.4])
    samples = rng.multivariate_normal(mean, covariance, size=400000)
    want = np.mean(np.hypot(*samples.T) <= 2.0)
    got = disk_probability(mean, covariance, 2.0)[0]
    assert abs(got - want) < 3e-3


def test_wrapped_interval_matches_monte_carlo():
    from vts_core.matching import wrapped_interval_probability

    rng = np.random.default_rng(1)
    for mean, std in ((0.1, 0.05), (2.9, 0.4), (-3.1, 1.5), (0.0, 6.0)):
        samples = (mean + std * rng.standard_normal(400000) + np.pi) % (2 * np.pi) - np.pi
        want = np.mean(np.abs(samples) < np.pi / 4)
        got = wrapped_interval_probability(np.array(mean), np.array(std**2), np.pi / 4)
        assert abs(got - want) < 3e-3, (mean, std, got, want)
