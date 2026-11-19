import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "anyloc_published" / "historical_src"))

from vts_core.node_detection import ConnectivityMonitor, contiguous_normalized_cut  # noqa: E402


def _two_places(old_frames: int, new_frames: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    old, new = np.eye(64)[0], np.eye(64)[1]
    frames = [old + 0.15 * rng.standard_normal(64) for _ in range(old_frames)]
    frames += [0.3 * old + new + 0.15 * rng.standard_normal(64) for _ in range(new_frames)]
    return np.asarray([f / np.linalg.norm(f) for f in frames], dtype=np.float32)


def _rayleigh_normalized_cut(affinity: np.ndarray, split: int) -> float:
    """Shi and Malik (2000), eq. 7: Ncut as y^T (D - W) y / y^T D y with y = 1_A - (vol A / vol B) 1_B."""
    degrees = affinity.sum(axis=1)
    left = degrees[:split].sum()
    right = degrees[split:].sum()
    y = np.where(np.arange(len(degrees)) < split, 1.0, -left / right)
    return float(y @ (np.diag(degrees) - affinity) @ y / (y @ np.diag(degrees) @ y))


def test_split_recovers_the_place_change_and_minimizes_the_normalized_cut():
    for old_frames, seed in ((7, 0), (15, 1), (24, 2)):
        monitor = ConnectivityMonitor(30)
        for descriptor in _two_places(old_frames, 30 - old_frames, seed):
            monitor.update(descriptor)
        affinity = monitor.affinity.astype(np.float64)
        split = contiguous_normalized_cut(monitor.affinity)
        assert split == old_frames
        assert split == min(range(1, 30), key=lambda s: _rayleigh_normalized_cut(affinity, s))
        assert monitor.frames_after_cut[-1] == 30 - old_frames
