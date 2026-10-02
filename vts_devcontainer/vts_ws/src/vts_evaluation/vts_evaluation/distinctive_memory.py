"""Frozen-node screening of local verification and discriminative view memory.

All decisions are chronological. Ground truth enters evaluation only.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from vts_core.topo_graph import TopoGraph, TopoNode


@dataclass(frozen=True)
class MatchEvidence:
    inliers: int = 0
    coverage: float = 0.0
    raw_matches: int = 0
    distinctive_inliers: int | None = None
    distinctive_coverage: float | None = None

    @property
    def useful_inliers(self) -> int:
        return self.inliers if self.distinctive_inliers is None else self.distinctive_inliers

    @property
    def useful_coverage(self) -> float:
        return self.coverage if self.distinctive_coverage is None else self.distinctive_coverage

    @property
    def score(self) -> float:
        return self.useful_inliers * np.sqrt(self.useful_coverage)

    @property
    def supported(self) -> bool:
        return self.useful_inliers >= 12 and self.useful_coverage >= 0.05


class LocalMatcher:
    """Symmetric SIFT matches and spatially distributed homography inliers."""

    def __init__(self) -> None:
        self.sift = cv2.SIFT_create(nfeatures=1200)
        self.bf = cv2.BFMatcher(cv2.NORM_L2)
        self.features: dict[str, tuple[list[cv2.KeyPoint], np.ndarray | None, tuple[int, int]]] = {}
        self.pairs: dict[tuple[str, str], MatchEvidence] = {}
        self.inlier_matches: dict[tuple[str, str], list[tuple[int, float]]] = {}

    def prepare(self, key: str, image: np.ndarray) -> None:
        if key in self.features:
            return
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        points, descriptors = self.sift.detectAndCompute(gray, None)
        self.features[key] = (points, descriptors, gray.shape)

    def compare(self, first: str, second: str) -> MatchEvidence:
        pair = (first, second)
        if pair in self.pairs:
            return self.pairs[pair]
        first_points, a, shape = self.features[first]
        second_points, b, _ = self.features[second]
        inlier_matches: list[tuple[int, float]] = []
        if a is None or b is None or len(a) < 8 or len(b) < 8:
            result = MatchEvidence()
        else:
            forward = {m.queryIdx: m for m, n in self.bf.knnMatch(a, b, k=2)
                       if m.distance < 0.75 * n.distance}
            reverse = {m.queryIdx: m.trainIdx for m, n in self.bf.knnMatch(b, a, k=2)
                       if m.distance < 0.75 * n.distance}
            matches = [(i, m.trainIdx, m.distance) for i, m in forward.items()
                       if reverse.get(m.trainIdx) == i]
            if len(matches) < 12:
                result = MatchEvidence(raw_matches=len(matches))
            else:
                src = np.float32([first_points[i].pt for i, _, _ in matches])
                dst = np.float32([second_points[j].pt for _, j, _ in matches])
                _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0,
                                             maxIters=2000, confidence=0.995)
                selected = mask.ravel().astype(bool) if mask is not None else np.zeros(len(matches), bool)
                kept = src[selected]
                inlier_matches = [(i, float(distance)) for (i, _, distance), keep in
                                  zip(matches, selected) if keep]
                if len(kept) >= 4:
                    extent = np.ptp(kept, axis=0)
                    coverage = float(extent[0] * extent[1] / (shape[0] * shape[1]))
                else:
                    coverage = 0.0
                result = MatchEvidence(len(kept), coverage, len(matches))
        self.pairs[pair] = result
        self.inlier_matches[pair] = inlier_matches
        return result

    def compare_competition(self, first: str, second: str,
                            competitors: Sequence[str]) -> MatchEvidence:
        """Count only spatial inliers whose descriptor favors this place.

        Rivals are other proposed places from the same query, never future views.
        Missing rival features cannot be treated as negative evidence.
        """
        base = self.compare(first, second)
        accepted = self.inlier_matches[(first, second)]
        if not accepted:
            return replace(base, distinctive_inliers=0, distinctive_coverage=0.0)
        distractors = [self.features[key][1] for key in competitors
                       if self.features[key][1] is not None]
        if not distractors:
            return replace(base, distinctive_inliers=0, distinctive_coverage=0.0)
        descriptors = self.features[first][1]
        queries = np.asarray([descriptors[index] for index, _ in accepted])
        rivals = np.concatenate(distractors)
        nearest = self.bf.match(queries, rivals)
        retained = [index for (index, distance), alternative in zip(accepted, nearest)
                    if distance < 0.8 * alternative.distance]
        points, _, shape = self.features[first]
        xy = np.float32([points[index].pt for index in retained])
        if len(xy) >= 4:
            spread = np.ptp(xy, axis=0)
            coverage = float(spread[0] * spread[1] / (shape[0] * shape[1]))
        else:
            coverage = 0.0
        return replace(base, distinctive_inliers=len(retained),
                       distinctive_coverage=coverage)


def select_views(segment: np.ndarray, previous: Sequence[np.ndarray],
                 representative: int, policy: str, budget: int = 3) -> list[int]:
    """Greedy fixed-budget memory; scores use only past map and current segment."""
    if policy not in {'diverse', 'discriminative'}:
        raise ValueError(f'Unknown memory policy: {policy}')
    vectors = np.asarray(segment, dtype=np.float32)
    if vectors.ndim != 2 or not 0 <= representative < len(vectors):
        raise ValueError('Invalid segment or representative')
    if not np.isfinite(vectors).all():
        raise ValueError('Nonfinite descriptor')
    vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    chosen = [representative]
    budget = min(budget, len(vectors))
    if policy == 'discriminative' and previous:
        others = np.concatenate(previous)
        others /= np.maximum(np.linalg.norm(others, axis=1, keepdims=True), 1e-12)
        # Only visually plausible competitors affect the distinctiveness term.
        strongest = np.sort(vectors @ others.T, axis=1)[:, -min(5, len(others)):]
        alias = np.mean(strongest, axis=1)
        alias = np.maximum(alias, 0)
    else:
        alias = np.zeros(len(vectors))
    while len(chosen) < budget:
        current_coverage = np.max(vectors @ vectors[chosen].T, axis=1)
        candidates = [i for i in range(len(vectors)) if i not in chosen]
        def benefit(index: int) -> float:
            new_coverage = np.maximum(current_coverage, vectors @ vectors[index])
            gain = float(np.mean(new_coverage - current_coverage))
            # Reward a distinctive view only if it also explains current frames.
            return gain + (0.2 * (1 - alias[index]) * float(np.mean(vectors @ vectors[index]))
                           if policy == 'discriminative' else 0.)
        chosen.append(max(candidates, key=lambda i: (benefit(i), -i)))
    return chosen


def replace_views(graph: TopoGraph, features: np.ndarray, starts: np.ndarray,
                  ends: np.ndarray, anchors: np.ndarray, policy: str) -> tuple[TopoGraph, dict[int, list[int]]]:
    """Replace view descriptors causally, preserving all node and pose fields."""
    rebuilt = TopoGraph(frame_id=graph.frame_id)
    selections = {}
    previous: list[np.ndarray] = []
    for node_id, node in graph.nodes.items():
        start, end = int(starts[node_id]), int(ends[node_id])
        # The immediately preceding node is a traversal neighbour, excluded
        # from loop candidates. It should not teach the memory that continuity
        # of one place is a confusing alternative.
        competitors = previous[:-1]
        local = select_views(features[start:end], competitors,
                             int(anchors[node_id]) - start, policy)
        ids = [start + i for i in local]
        selections[node_id] = ids
        views = features[ids].copy()
        rebuilt.add_node(replace(node, view_features=views))
        previous.append(views)
    return rebuilt, selections


def source_images(directory: Path) -> TopoGraph:
    """Load and hash-check the original graph containing representative images."""
    import json
    from vts_core.artifacts import file_digest
    completion = json.loads((directory / 'completion.json').read_text())
    path = directory / 'graph_0.pkl'
    if file_digest(path) != completion['artifacts'][path.name]:
        raise ValueError(f'Full image graph changed: {path}')
    return TopoGraph.load(str(path))


def decide_extra(scores: dict[int, MatchEvidence], ratio: float = 1.5) -> int | None:
    """Unique locally supported candidate with a margin over competitors."""
    if not scores:
        return None
    ranked = sorted(scores, key=lambda k: (-scores[k].score, k))
    best = scores[ranked[0]]
    next_score = scores[ranked[1]].score if len(ranked) > 1 else 0.0
    return ranked[0] if best.supported and best.score >= ratio * max(next_score, 1e-9) else None
