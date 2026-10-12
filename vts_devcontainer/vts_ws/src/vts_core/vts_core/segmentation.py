"""Place segmentation by adaptive valleys of windowed algebraic connectivity.

Each frame's descriptor joins a sliding window of ``window`` frames. Cosine
similarities clamped to ``[0, 1]`` form a non-negative affinity graph whose
symmetric normalized Laplacian has second-smallest eigenvalue lambda_2
(isolated vertices use ``D^{-1/2}_ii = 0`` and ``L_ii = 0``). A segment ends at
each confirmed valley of the lambda_2 series.

:func:`segment` (offline, batched eigendecomposition) and :class:`Segmenter`
(streaming) share every numerical step and give identical boundaries; the
float32 affinity arithmetic matches the recorded manuscript runs bit for bit.
"""

from __future__ import annotations

from collections import deque

import numpy as np

_MAD_TO_STD = 1.4826
_BATCH = 1024


def _weights(previous: np.ndarray, descriptor: np.ndarray) -> np.ndarray:
    """Clamped similarities of a new frame to the previous window frames (float32)."""
    return np.clip(previous @ descriptor, 0.0, 1.0)


def _lambda2(affinity: np.ndarray) -> np.ndarray:
    """lambda_2 of a batch of float32 affinity matrices ``(B, L, L)``."""
    degrees = affinity.sum(axis=2)
    nonzero = degrees > 1e-12
    inv_sqrt = np.zeros(degrees.shape, dtype=np.float64)
    inv_sqrt[nonzero] = 1.0 / np.sqrt(degrees[nonzero])
    size = affinity.shape[-1]
    laplacian = np.eye(size) - inv_sqrt[:, :, None] * affinity * inv_sqrt[:, None, :]
    batch, vertex = np.nonzero(~nonzero)
    laplacian[batch, vertex, vertex] = 0.0
    return np.maximum(np.linalg.eigvalsh(laplacian)[:, 1], 0.0)


def lambda2_series(descriptors: np.ndarray, window: int) -> np.ndarray:
    """lambda_2 after each frame from the second one on (length ``N - 1``)."""
    if not isinstance(window, int) or window < 2:
        raise ValueError("window must be an integer >= 2")
    frames = np.asarray(descriptors, dtype=np.float32)
    n = len(frames)
    if n < 2:
        return np.empty(0)
    # band[b, m]: weight between frame b and frame b - 1 - m, computed exactly
    # as the streaming monitor does when frame b arrives.
    band = np.zeros((n, window - 1), dtype=np.float32)
    for b in range(1, n):
        start = max(0, b - window + 1)
        band[b, : b - start] = _weights(frames[start:b], frames[b])[::-1]
    out = np.empty(n - 1)
    # Growing windows (fewer than ``window`` frames) are evaluated one by one.
    for t in range(1, min(window - 1, n)):
        out[t - 1] = _lambda2(_window_affinity(band, np.array([0]), t + 1))[0]
    rows, cols = np.tril_indices(window, -1)
    for first in range(window - 1, n, _BATCH):
        ends = np.arange(first, min(first + _BATCH, n))
        affinity = np.zeros((len(ends), window, window), dtype=np.float32)
        starts = ends - window + 1
        affinity[:, rows, cols] = band[starts[:, None] + rows, rows - cols - 1]
        affinity[:, cols, rows] = affinity[:, rows, cols]
        out[ends - 1] = _lambda2(affinity)
    return out


def _window_affinity(band: np.ndarray, starts: np.ndarray, size: int) -> np.ndarray:
    rows, cols = np.tril_indices(size, -1)
    affinity = np.zeros((len(starts), size, size), dtype=np.float32)
    affinity[:, rows, cols] = band[starts[:, None] + rows, rows - cols - 1]
    affinity[:, cols, rows] = affinity[:, rows, cols]
    return affinity


class ValleyDetector:
    """Two-extremum hysteresis on lambda_2 with an adaptive prominence.

    Alternates between seeking a maximum and a minimum; a valley is confirmed
    when the signal rises ``delta`` above the running minimum. ``delta`` is
    ``k * 1.4826 * MAD`` of the last ``history`` lambda_2 values (infinite
    before ``warmup`` samples).
    """

    def __init__(self, k: float = 2.0, history: int = 300, warmup: int = 30) -> None:
        if k <= 0.0:
            raise ValueError("k must be positive")
        if history < 2 or not 2 <= warmup <= history:
            raise ValueError("need history >= 2 and warmup in [2, history]")
        self._k = k
        self._values: deque[float] = deque(maxlen=history)
        self._warmup = warmup
        self._look_for_max = True
        self._max_value = float("-inf")
        self._min_value = float("inf")
        self._min_index = 0
        self._index = -1
        self.latencies: list[int] = []

    def _delta(self) -> float:
        if len(self._values) < self._warmup:
            return float("inf")
        values = np.array(self._values, dtype=np.float64)
        mad = float(np.median(np.abs(values - np.median(values))))
        return max(self._k * _MAD_TO_STD * mad, 1e-6)

    def step(self, value: float) -> int | None:
        """Feed one lambda_2 sample; return the index of a confirmed valley."""
        self._index += 1
        self._values.append(value)
        delta = self._delta()
        if self._look_for_max:
            self._max_value = max(self._max_value, value)
            if value < self._max_value - delta:
                self._look_for_max = False
                self._min_value, self._min_index = value, self._index
            return None
        if value < self._min_value:
            self._min_value, self._min_index = value, self._index
        if value > self._min_value + delta:
            self._look_for_max = True
            self._max_value = value
            self.latencies.append(self._index - self._min_index)
            return self._min_index
        return None


def segment(descriptors: np.ndarray, window: int = 30, k: float = 2.0) -> list[tuple[int, int]]:
    """Contiguous segments ``[start, stop)`` covering all frames, in order."""
    detector = ValleyDetector(k)
    segments, start = [], 0
    for value in lambda2_series(descriptors, window):
        valley = detector.step(float(value))
        # lambda_2 sample i is produced by frame i + 1.
        if valley is not None and valley + 1 >= start:
            segments.append((start, valley + 2))
            start = valley + 2
    if start < len(descriptors):
        segments.append((start, len(descriptors)))
    return segments


class Segmenter:
    """Streaming segmentation; boundaries identical to :func:`segment`."""

    def __init__(self, window: int = 30, k: float = 2.0) -> None:
        if not isinstance(window, int) or window < 2:
            raise ValueError("window must be an integer >= 2")
        self._window = window
        self._frames: deque[np.ndarray] = deque(maxlen=window)
        self._band: deque[np.ndarray] = deque(maxlen=window)
        self._detector = ValleyDetector(k)
        self._count = 0
        self._start = 0
        self.lambda2: list[float] = []

    def push(self, descriptor: np.ndarray) -> tuple[int, int] | None:
        """Add one frame; return a completed segment ``[start, stop)`` or None."""
        descriptor = np.asarray(descriptor, dtype=np.float32)
        row = np.zeros(self._window - 1, dtype=np.float32)
        previous = list(self._frames)[-(self._window - 1):]
        if previous:
            row[: len(previous)] = _weights(np.stack(previous), descriptor)[::-1]
        self._frames.append(descriptor)
        self._band.append(row)
        self._count += 1
        if self._count < 2:
            return None
        value = float(_lambda2(_window_affinity(np.stack(self._band), np.array([0]), len(self._band)))[0])
        self.lambda2.append(value)
        valley = self._detector.step(value)
        if valley is None or valley + 1 < self._start:
            return None
        completed = (self._start, valley + 2)
        self._start = valley + 2
        return completed

    def flush(self) -> tuple[int, int] | None:
        """Close the final segment at end of stream."""
        if self._start >= self._count:
            return None
        completed = (self._start, self._count)
        self._start = self._count
        return completed

    @property
    def latencies(self) -> list[int]:
        return self._detector.latencies
