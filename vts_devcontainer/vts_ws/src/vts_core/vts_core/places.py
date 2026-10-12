"""Appearance and odometry summary of one segment (a topological place).

A place keeps its segment's frame range, the medoid frame (closest to the mean
direction; it fixes the place pose), up to three representative views (medoid,
first and last frame) and, for each view, the recorded odometry pose. The mean
direction of the segment's unit descriptors is the appearance summary that
worked best across sessions (docs/APPEARANCE_STUDY_C4_2026-10-02.md).

Ground truth is never part of a place; evaluation looks it up by frame index.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Place:
    start: int
    stop: int
    medoid: int
    view_frames: tuple[int, ...]
    mean: np.ndarray  # float64 unit vector
    views: np.ndarray  # (n_views, d) float64 unit rows
    resultant_length: float
    pose: np.ndarray  # odometry (x, y, theta) of the medoid frame
    budget: np.ndarray  # 2x2 cumulative position budget at the medoid frame
    view_poses: np.ndarray  # (n_views, 3) odometry poses of the view frames

    @property
    def size(self) -> int:
        return self.stop - self.start


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def summarize(segments: list[tuple[int, int]], structure: np.ndarray, poses: np.ndarray,
              budgets: np.ndarray, appearance: np.ndarray | None = None, offset: int = 0) -> list[Place]:
    """Places for ``segments`` (frame indices; the arrays start at frame ``offset``).

    ``structure`` (the segmentation descriptors) fixes medoids and view frames;
    ``appearance`` (default: the same array) provides the mean direction and
    view descriptors, so different encoders face identical places.
    """
    appearance = structure if appearance is None else appearance
    if len(appearance) != len(structure) or len(poses) != len(structure):
        raise ValueError("descriptor and pose arrays must be frame aligned")
    places = []
    for first, last in segments:
        start, stop = first - offset, last - offset
        samples = _unit_rows(structure[start:stop])
        resultant = samples.sum(axis=0)
        norm = float(np.linalg.norm(resultant))
        mean32 = (resultant / norm if norm > 1e-12 else samples[0]).astype(np.float32)
        # First maximizer, as the recorded mapper did.
        medoid = start + int(np.argmax(np.asarray(structure[start:stop], dtype=np.float32) @ mean32))
        views = tuple(dict.fromkeys((medoid, start, stop - 1)))
        if appearance is structure:
            mean, length = (resultant / norm if norm > 1e-12 else samples[0]), norm / (stop - start)
        else:
            other = _unit_rows(appearance[start:stop]).sum(axis=0)
            other_norm = float(np.linalg.norm(other))
            mean, length = other / other_norm, other_norm / (stop - start)
        places.append(Place(
            start=first, stop=last, medoid=medoid + offset, view_frames=tuple(v + offset for v in views),
            mean=mean, views=_unit_rows(appearance[list(views)]),
            resultant_length=float(min(length, 1.0)),
            pose=np.asarray(poses[medoid], dtype=np.float64),
            budget=np.asarray(budgets[medoid][:2, :2], dtype=np.float64),
            view_poses=np.asarray(poses[list(views)], dtype=np.float64),
        ))
    return places


def with_odometry(places: list[Place], poses: np.ndarray, budgets: np.ndarray) -> list[Place]:
    """Same places under different (e.g. perturbed) odometry."""
    return [replace(p, pose=np.asarray(poses[p.medoid], dtype=np.float64),
                    budget=np.asarray(budgets[p.medoid][:2, :2], dtype=np.float64),
                    view_poses=np.asarray(poses[list(p.view_frames)], dtype=np.float64)) for p in places]
