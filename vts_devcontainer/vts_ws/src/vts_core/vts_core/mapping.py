"""The mapping engine shared by offline replays and the ROS node.

:class:`OnlineMapper` consumes frames one at a time (descriptor, odometry pose,
cumulative budget), closes segments with :class:`Segmenter`, summarizes each
as a :class:`Place` and associates it immediately. :func:`map_session` runs
the same steps on a stored session; segmentation is batched there, which is
tested to give identical boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vts_core.association import (
    CalibrationPrior,
    Decision,
    GateAssociator,
    PosteriorAssociator,
    PosteriorConfig,
    run,
)
from vts_core.places import Place, summarize
from vts_core.segmentation import Segmenter, segment


@dataclass(frozen=True)
class MapperConfig:
    window: int = 30
    valley_k: float = 2.0
    method: str = "gate"  # "gate" or "posterior"
    gate: dict = field(default_factory=dict)  # GateAssociator keyword arguments
    posterior: PosteriorConfig = PosteriorConfig()

    @classmethod
    def from_dict(cls, mapping: dict) -> MapperConfig:
        if mapping.get("method", "gate") not in {"gate", "posterior"}:
            raise ValueError("mapping.method must be gate or posterior")
        return cls(int(mapping.get("window", 30)), float(mapping.get("valley_k", 2.0)),
                   mapping.get("method", "gate"), dict(mapping.get("gate", {})),
                   PosteriorConfig(**mapping.get("posterior", {})))


def make_associator(config: MapperConfig, prior: CalibrationPrior | None = None):
    if config.method == "gate":
        return GateAssociator(**config.gate)
    return PosteriorAssociator(config.posterior, prior)


@dataclass
class MapResult:
    places: list[Place]
    decisions: list[Decision]
    segments: list[tuple[int, int]]


def map_session(structure: np.ndarray, poses: np.ndarray, budgets: np.ndarray, config: MapperConfig,
                appearance: np.ndarray | None = None, prior: CalibrationPrior | None = None) -> MapResult:
    """Offline replay of one traversal (identical to streaming it frame by frame)."""
    segments = segment(structure, config.window, config.valley_k)
    places = summarize(segments, structure, poses, budgets, appearance)
    return MapResult(places, run(make_associator(config, prior), places), segments)


class OnlineMapper:
    """Frame-by-frame mapping; keeps only the frames of the open segment."""

    def __init__(self, config: MapperConfig = MapperConfig(), prior: CalibrationPrior | None = None) -> None:
        self.config = config
        self.segmenter = Segmenter(config.window, config.valley_k)
        self.associator = make_associator(config, prior)
        self.places: list[Place] = []
        self.decisions: list[Decision] = []
        self._base = 0
        self._descriptors: list[np.ndarray] = []
        self._poses: list[np.ndarray] = []
        self._budgets: list[np.ndarray] = []

    @property
    def frames(self) -> int:
        return self._base + len(self._descriptors)

    def push(self, descriptor: np.ndarray, pose, budget: np.ndarray) -> Place | None:
        """Add one frame; return the place it completed, if any."""
        descriptor = np.asarray(descriptor, dtype=np.float32)
        pose, budget = np.asarray(pose, dtype=np.float64), np.asarray(budget, dtype=np.float64)
        if descriptor.ndim != 1 or not np.isfinite(descriptor).all() or np.linalg.norm(descriptor) < 1e-12:
            raise ValueError("descriptor must be a finite nonzero vector")
        if pose.shape != (3,) or budget.shape != (3, 3) or not np.isfinite(budget).all():
            raise ValueError("expected a (3,) pose and a (3, 3) cumulative budget")
        self._descriptors.append(descriptor)
        self._poses.append(pose)
        self._budgets.append(budget)
        completed = self.segmenter.push(descriptor)
        return self._close(completed) if completed else None

    def finish(self) -> Place | None:
        """Close the final segment at end of stream."""
        completed = self.segmenter.flush()
        return self._close(completed) if completed else None

    def _close(self, completed: tuple[int, int]) -> Place:
        start, stop = completed
        local = slice(start - self._base, stop - self._base)
        (place,) = summarize([(start, stop)], *self._window(local), offset=start)
        self.places.append(place)
        decision = self.associator.observe(place)
        if decision is not None:
            self.decisions.append(decision)
        drop = stop - self._base
        del self._descriptors[:drop], self._poses[:drop], self._budgets[:drop]
        self._base = stop
        return place

    def _window(self, local: slice):
        return (np.stack(self._descriptors[local]), np.stack(self._poses[local]), np.stack(self._budgets[local]))
