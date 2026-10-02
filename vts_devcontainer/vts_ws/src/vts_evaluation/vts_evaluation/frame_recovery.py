"""Experimental B: causal frame-track matching independent of node boundaries.

This module deliberately has no ground-truth or node input. It is an offline
screening component, not enabled in the production mapper.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vts_core.local_recovery import robust_evidence
from vts_core.matching import fit_se2


@dataclass(frozen=True)
class FrameConfig:
    spacing_m: float = 0.5
    pairs: int = 7
    top_k: int = 5
    residual_m: float = 0.75
    visual_z: float = 2.0
    min_span_m: float = 1.0
    min_turn_deg: float = 15.0
    skip_penalty: float = 0.03

    def __post_init__(self) -> None:
        values = (self.spacing_m, self.residual_m, self.visual_z,
                  self.min_span_m, self.min_turn_deg, self.skip_penalty)
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError('Frame thresholds must be finite and positive')
        if type(self.pairs) is not int or self.pairs < 3 or type(self.top_k) is not int or self.top_k < 1:
            raise ValueError('Require at least three pairs and one proposal')
        if self.min_turn_deg >= 90:
            raise ValueError('Minimum turn must be below 90 degrees')


def select_endpoint(rows: list[dict]) -> int | None:
    """Resolve ambiguity before mapping a frame back to any node partition."""
    endpoints = {row['endpoint'] for row in rows if row['passed']}
    return next(iter(endpoints)) if len(endpoints) == 1 else None


def sample_track(arc: np.ndarray, anchor: int, direction: int, limit: int,
                 count: int, spacing: float) -> list[int]:
    """Return anchor-first indices, spaced by travelled distance, within prefix."""
    ids = [anchor]
    i = anchor + direction
    while 0 <= i < limit and len(ids) < count:
        if abs(arc[i] - arc[ids[-1]]) >= spacing:
            ids.append(i)
        i += direction
    return ids


def align(cost: np.ndarray, flexible: bool, penalty: float) -> list[int] | None:
    """Minimum-cost monotone path ending at last column; one or two frame steps.

    Every query observation has a distinct historical observation. Free starts
    allow different fragment lengths; the endpoint anchors the proposed closure.
    """
    n, m = cost.shape
    dp = np.full((n, m), np.inf)
    back = np.full((n, m), -1, dtype=int)
    dp[0] = cost[0]
    for i in range(1, n):
        for j in range(m):
            for step in ((1, 2) if flexible else (1,)):
                if j >= step:
                    value = dp[i - 1, j - step] + cost[i, j] + penalty * (step - 1)
                    if value < dp[i, j]:
                        dp[i, j], back[i, j] = value, j - step
    j = m - 1
    if not np.isfinite(dp[-1, j]):
        return None
    path = [j]
    for i in range(n - 1, 0, -1):
        j = int(back[i, j])
        path.append(j)
    return path[::-1]


def recover(features: np.ndarray, xy: np.ndarray, query: int, history_limit: int,
            config: FrameConfig, flexible: bool = True) -> list[dict]:
    """Propose and verify historical endpoints using only the arrived prefix."""
    if not 0 <= history_limit <= query < len(features):
        raise ValueError('Invalid causal prefix')
    # Ignore all frames after the query even when an offline caller supplies them.
    features, xy = features[:query + 1], xy[:query + 1]
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    live = sample_track(arc, query, -1, query + 1, config.pairs, config.spacing_m)[::-1]
    if len(live) < config.pairs:
        return [{'reason': 'insufficient_live_track', 'passed': False}]
    limit = min(history_limit, live[0])
    if limit == 0:
        return [{'reason': 'insufficient_history', 'passed': False}]
    scores = features[:limit] @ features[query]
    proposals: list[int] = []
    for index in np.argsort(-scores, kind='stable'):
        index = int(index)
        if all(abs(arc[index] - arc[j]) >= 1.0 for j in proposals):
            proposals.append(index)
            if len(proposals) == config.top_k:
                break
    background = features[live] @ features[:limit].T
    rows = []
    for endpoint in proposals:
        for direction in (-1, 1):
            historical = sample_track(arc, endpoint, direction, limit,
                                      2 * config.pairs - 1, config.spacing_m)[::-1]
            row = {'endpoint': endpoint, 'direction': direction, 'live': live,
                   'passed': False}
            if len(historical) < config.pairs:
                rows.append({**row, 'reason': 'insufficient_history'})
                continue
            similarity = features[live] @ features[historical].T
            path = align(1. - similarity, flexible, config.skip_penalty)
            if path is None:
                rows.append({**row, 'reason': 'alignment_failed'})
                continue
            matched = [historical[j] for j in path]
            values = similarity[np.arange(len(live)), path]
            contrast = [robust_evidence(float(v), np.delete(background[i], matched[i]).tolist())
                        for i, v in enumerate(values)]
            visual_ok = min(values) > 0 and min(contrast) > 0 and np.mean(contrast) >= config.visual_z
            a, b = xy[live], xy[matched]
            rotation, translation = fit_se2(b, a)
            residual = float(np.max(np.linalg.norm(b @ rotation.T + translation - a, axis=1)))
            spans = [float(np.max(np.linalg.norm(t[:, None] - t[None, :], axis=-1))) for t in (a, b)]
            turns = []
            for t in (a, b):
                steps = np.diff(t, axis=0)
                norms = np.linalg.norm(steps, axis=1)
                angles = np.degrees(np.arccos(np.clip(np.sum(steps[:-1] * steps[1:], axis=1)
                                   / np.maximum(norms[:-1] * norms[1:], 1e-12), -1, 1)))
                turns.append(float(np.max(np.minimum(angles, 180 - angles))))
            observable = min(spans) >= config.min_span_m and min(turns) >= config.min_turn_deg
            reason = ('unobservable_local_shape' if not observable else
                      'local_geometry_mismatch' if residual > config.residual_m else
                      'weak_visual_support' if not visual_ok else 'consistent')
            rows.append({**row, 'historical': matched, 'pair_scores': values.tolist(),
                         'contrasts': contrast, 'max_residual_m': residual,
                         'visual_passed': bool(visual_ok), 'spans_m': spans, 'turns_deg': turns,
                         'reason': reason, 'passed': reason == 'consistent'})
    return rows
