import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from odometry_bias import fit_rate, window_drift  # noqa: E402
from stream import Stream, debiased  # noqa: E402


def _drive(steps: np.ndarray, turns: np.ndarray, heading: float = 0.3) -> np.ndarray:
    """Each step moves along the chord of a constant-curvature arc."""
    poses = [np.array([1.0, -2.0, heading])]
    for length, turn in zip(steps, turns):
        x, y, h = poses[-1]
        chord = h + turn / 2
        poses.append(np.array([x + length * np.cos(chord), y + length * np.sin(chord), h + turn]))
    return np.array(poses)


def test_debiasing_straightens_a_track_bent_by_a_constant_heading_rate() -> None:
    steps = np.full(400, 0.05)
    rate = np.radians(1.5)
    bent = _drive(steps, rate * steps)
    straight = _drive(steps, np.zeros_like(steps))
    corrected = debiased(bent, rate)
    assert np.allclose(corrected[:, 2], straight[:, 2], atol=1e-12)
    assert np.abs(corrected[:, :2] - straight[:, :2]).max() < 1e-3


def test_reversing_drifts_the_heading_the_other_way() -> None:
    steps = np.concatenate([np.full(200, 0.05), np.full(200, -0.05)])
    rate = np.radians(1.0)
    corrected = debiased(_drive(steps, rate * steps), rate)
    assert np.allclose(corrected[:, 2], 0.3, atol=1e-12)


@pytest.mark.parametrize("rate_deg", [-1.2, 0.4, 1.6])
def test_fitted_rate_matches_the_simulated_drift(rate_deg: float) -> None:
    rng = np.random.default_rng(0)
    steps = 0.04 + 0.02 * rng.random(3000)
    truth_turns = 0.05 * np.sin(np.linspace(0.0, 40.0, 3000))
    odometry_turns = truth_turns + np.radians(rate_deg) * steps + rng.normal(0.0, 0.0005, 3000)
    truth = _drive(steps, truth_turns)
    stream = Stream(0, [], _drive(steps, odometry_turns), truth, [])
    assert np.degrees(fit_rate([window_drift(stream)])) == pytest.approx(rate_deg, abs=0.05)
