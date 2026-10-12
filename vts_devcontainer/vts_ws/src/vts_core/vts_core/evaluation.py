"""Ground-truth evaluation: opportunities, revisit types, metrics and intervals.

Ground truth enters only here, after every decision is made. A place's ground
truth is that of its medoid frame (the frame that fixed its pose).

Revisit type: an opportunity is *aligned* if some matching place within the
tolerance also has a ground-truth heading within ``aligned_deg``; otherwise it
is *opposed*. A forward camera sees different content on opposed passes, so
only geometry can detect them.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from vts_core.association import EXCLUDED_RECENT, Decision
from vts_core.places import Place


def place_gt(places: Sequence[Place], gt: np.ndarray) -> np.ndarray:
    return np.asarray([gt[p.medoid] for p in places], dtype=np.float64).reshape(-1, 3)


def _heading_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs((b - a + np.pi) % (2 * np.pi) - np.pi)


def revisit_kinds(query_gt: np.ndarray, reference_gt: np.ndarray, tolerance: float, aligned_deg: float,
                  eligible: Sequence[np.ndarray] | None = None) -> list[str | None]:
    """Per query: None (no opportunity), "aligned" or "opposed".

    ``eligible[i]`` restricts the reference places for query ``i`` (in-session:
    earlier places except the excluded recent one); default all.
    """
    kinds = []
    for i, q in enumerate(query_gt):
        ref = reference_gt if eligible is None else reference_gt[eligible[i]]
        if not len(ref):
            kinds.append(None)
            continue
        near = np.linalg.norm(ref[:, :2] - q[:2], axis=1) <= tolerance
        if not near.any():
            kinds.append(None)
            continue
        aligned = near & (_heading_difference(q[2], ref[:, 2]) < np.radians(aligned_deg))
        kinds.append("aligned" if aligned.any() else "opposed")
    return kinds


def in_session_eligible(n: int) -> list[np.ndarray]:
    return [np.arange(0, max(i - EXCLUDED_RECENT, 0)) for i in range(n)]


def loop_counts(decisions: Sequence[Decision], gt_xy: np.ndarray, tolerance: float,
                kinds: Sequence[str | None] | None = None, tau: float | None = None) -> dict:
    """Loop-closure TP/FP/FN as in every previous study.

    A decision is an opportunity when its query lies within ``tolerance`` of an
    eligible earlier place. TP: accepted and within tolerance; FP: accepted and
    not; FN: opportunity without a TP. ``tau`` re-thresholds posterior decisions.
    With ``kinds``, TP and FN are also split by revisit type.
    """
    counts = {"tp": 0, "fp": 0, "fn": 0, "tp_aligned": 0, "fn_aligned": 0, "tp_opposed": 0, "fn_opposed": 0}
    for d in decisions:
        accepted = d.accepted if tau is None else (d.matched is not None and d.posterior >= tau)
        eligible = np.arange(0, max(d.query - EXCLUDED_RECENT, 0))
        opportunity = bool(len(eligible)) and bool(
            (np.linalg.norm(gt_xy[eligible] - gt_xy[d.query], axis=1) <= tolerance).any())
        true = accepted and float(np.linalg.norm(gt_xy[d.matched] - gt_xy[d.query])) <= tolerance
        counts["tp"] += int(true)
        counts["fp"] += int(accepted and not true)
        counts["fn"] += int(opportunity and not true)
        if kinds is not None and kinds[d.query] is not None:
            counts[f"{'tp' if true else 'fn'}_{kinds[d.query]}"] += 1
    return counts


def prf(tp: int, fp: int, fn: int) -> dict:
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


# --------------------------------------------------------------------------- #
# Retrieval-style evaluation of cross-session association
# --------------------------------------------------------------------------- #
def top1_statistic(scores: np.ndarray, kind: str = "margin") -> tuple[np.ndarray, np.ndarray]:
    """Top-1 reference per query and its confidence (raw, margin or mad_z)."""
    top = scores.argmax(axis=1)
    value = scores[np.arange(len(scores)), top]
    if kind == "raw":
        return top, value
    if kind == "margin":
        second = np.sort(scores, axis=1)[:, -2] if scores.shape[1] > 1 else np.full(len(scores), -np.inf)
        return top, value - second
    others = np.where(np.eye(scores.shape[1], dtype=bool)[top], np.nan, scores)
    median = np.nanmedian(others, axis=1)
    mad = 1.4826 * np.nanmedian(np.abs(others - median[:, None]), axis=1)
    return top, (value - median) / np.maximum(mad, 1e-9)


def pr_curve(correct: np.ndarray, positive: np.ndarray, statistic: np.ndarray) -> dict:
    """Accept queries by decreasing statistic; precision over accepted top-1 matches."""
    order = np.argsort(-statistic, kind="stable")
    tp = np.cumsum(correct[order])
    precision = tp / np.arange(1, len(order) + 1)
    positives = int(positive.sum())
    recall = tp / max(positives, 1)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-12), 0.0)

    def recall_at(p: float) -> float:
        ok = precision >= p
        return float(recall[ok].max()) if ok.any() else 0.0

    return {"queries": len(order), "positives": positives,
            "recall_at_1": float(correct.sum() / max(positives, 1)),
            "average_precision": float(np.sum(np.diff(np.r_[0.0, recall]) * precision)),
            "max_f1": float(f1.max()) if len(f1) else 0.0,
            "recall_at_p90": recall_at(0.90), "recall_at_p95": recall_at(0.95)}


def chance_top1(distance: np.ndarray, tolerance: float) -> float:
    """Expected top-1 accuracy of a uniformly random reference place."""
    return float((distance <= tolerance).mean(axis=1).mean())


def bootstrap_interval(values: Sequence[float] | np.ndarray, statistic=np.mean, resamples: int = 10000,
                       seed: int = 0, level: float = 0.95) -> tuple[float, float, float]:
    """Percentile interval over resampled units (e.g. session pairs)."""
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(resamples, len(values)))
    stats = np.apply_along_axis(statistic, 1, values[draws])
    alpha = (1 - level) / 2
    return float(statistic(values)), float(np.quantile(stats, alpha)), float(np.quantile(stats, 1 - alpha))
