"""Experimental recovery from global drift using disjoint local node tracks.

Scores are heuristic evidence, not calibrated probabilities. No ground truth,
source loop edges, or scale fitting enter the verifier.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from vts_core.matching import fit_se2
from vts_core.topo_graph import TopoGraph, TopoNode

RECOVERY_MODES = ("off", "visual_union", "soft_odometry", "local")


@dataclass(frozen=True)
class RecoveryConfig:
    top_k: int = 5
    horizon: int = 2
    residual_m: float = 0.75
    min_span_m: float = 1.0
    min_turn_deg: float = 15.0
    visual_z: float = 2.0

    def __post_init__(self) -> None:
        if type(self.top_k) is not int or self.top_k < 1:
            raise ValueError("recovery top_k must be a positive integer")
        if type(self.horizon) is not int or not 0 <= self.horizon <= 2:
            raise ValueError("recovery horizon must be an integer in [0, 2]")
        values = (self.residual_m, self.min_span_m, self.min_turn_deg, self.visual_z)
        if not np.isfinite(values).all() or min(values) <= 0 or self.min_turn_deg >= 180:
            raise ValueError("recovery thresholds must be finite and positive; turn < 180")


def robust_evidence(value: float, background: list[float]) -> float:
    """Bounded MAD-normalized contrast against an unfiltered background."""
    if not background:
        return -10.0  # No relative distinctiveness evidence.
    median = float(np.median(background))
    scale = 1.4826 * float(np.median(np.abs(np.asarray(background) - median)))
    return float(np.clip((value - median) / max(scale, 1e-6), -10, 10))


def local_track(
    graph: TopoGraph, query_id: int, matched_id: int, current_id: int,
    direction: int, similarity: Callable[[TopoNode, TopoNode], float],
    config: RecoveryConfig,
) -> dict:
    """Test a three-pair window containing the original query, without reusing IDs.

    Future support can be used only after it has arrived. Historical support is
    frozen to the eligible database at hypothesis creation. Each attempt replaces
    the previous window; its evidence is never summed with overlapping attempts.
    """
    if direction not in (-1, 1) or not query_id <= current_id <= query_id + 2:
        raise ValueError("invalid recovery alignment")
    live = list(range(current_id - 2, current_id + 1))
    historical = [matched_id + direction * (i - query_id) for i in live]
    row = {"query_node_id": query_id, "matched_node_id": matched_id,
           "decision_node_id": current_id, "direction": direction,
           "live_ids": live, "historical_ids": historical, "passed": False}
    if any(i not in graph.nodes for i in live + historical) or any(
        i >= query_id - 1 for i in historical
    ):
        return {**row, "reason": "insufficient_history"}
    if set(live) & set(historical):
        return {**row, "reason": "shared_observations"}
    for track in (live, historical):
        for a, b in zip(track, track[1:]):
            key = (min(a, b), max(a, b))
            if b not in graph.adjacency[a] or graph.edge_types.get(key) != "odometry":
                return {**row, "reason": "missing_odometry_edge"}

    pair_scores, contrasts = [], []
    background_ids = [i for i in sorted(graph.nodes) if i < query_id - 1 and i not in live]
    for a, b in zip(live, historical):
        value = similarity(graph.nodes[a], graph.nodes[b])
        background = [similarity(graph.nodes[a], graph.nodes[i])
                      for i in background_ids if i != b]
        pair_scores.append(value)
        contrasts.append(robust_evidence(value, background))
    row.update(pair_scores=pair_scores, pair_contrasts=contrasts,
               visual_evidence=float(np.mean(contrasts)))
    visual_ok = min(pair_scores) > 0 and min(contrasts) > 0 and (
        row["visual_evidence"] >= config.visual_z
    )
    row["visual_passed"] = bool(visual_ok)

    points = [np.asarray([graph.nodes[i].pose[:2] for i in ids])
              for ids in (live, historical)]
    spans, turns, lengths = [], [], []
    for xy in points:
        steps = np.diff(xy, axis=0)
        norms = np.linalg.norm(steps, axis=1)
        lengths.append(norms.tolist())
        spans.append(float(np.max(np.linalg.norm(xy[:, None] - xy[None, :], axis=-1))))
        cosine = float(steps[0] @ steps[1] / max(float(np.prod(norms)), 1e-12))
        turns.append(float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))))
    row.update(spans_m=spans, turns_deg=turns, step_lengths_m=lengths)
    # Straight/retraced tracks cannot disambiguate repeated corridors reliably.
    if min(spans) < config.min_span_m or min(min(x) for x in lengths) < 1e-3 or any(
        min(turn, 180 - turn) < config.min_turn_deg for turn in turns
    ):
        return {**row, "reason": "unobservable_local_shape"}
    rotation, translation = fit_se2(points[1], points[0])
    residuals = np.linalg.norm(points[1] @ rotation.T + translation - points[0], axis=1)
    row.update(max_residual_m=float(residuals.max()),
               rms_residual_m=float(np.sqrt(np.mean(residuals**2))))
    geometry_ok = row["max_residual_m"] <= config.residual_m
    row["geometry_passed"] = bool(geometry_ok)
    reason = "consistent" if geometry_ok and visual_ok else (
        "local_geometry_mismatch" if not geometry_ok else "weak_visual_support")
    return {**row, "passed": bool(geometry_ok and visual_ok), "reason": reason}
