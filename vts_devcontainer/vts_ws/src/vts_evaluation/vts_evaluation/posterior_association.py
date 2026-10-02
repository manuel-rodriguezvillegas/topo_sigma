"""C2 candidate: posterior loop association with an explicit new-place hypothesis.

For each emitted node ``i`` the hypotheses are "new place" and "revisit of j"
for every eligible earlier node ``j <= i - 2`` (the mapper's temporal
exclusion). With prior ``pi`` of revisiting, split uniformly over candidates,

    P(j | data) ∝ (pi / n) * LR_geo(j) * LR_app(j),    P(new | data) ∝ 1 - pi.

Geometry. Under "same place" the displacement follows the odometry budget
Gaussian with probability ``1 - eps``; with probability ``eps`` odometry is
grossly inconsistent (slip, reset, kidnapping) and the displacement is uniform
over the explored area ``A``. Under "different place" it is uniform over ``A``:

    LR_geo = (1 - eps) * N(delta; 0, Sigma) * A + eps   (never below eps).

Appearance. A likelihood ratio between two score distributions estimated
online, without labels, from pairs of *earlier* nodes (never the query's own
row). Two calibration sources are implemented:

* ``geometry`` (v2, default): "same place" = past pairs that the odometry
  budget confirms (squared Mahalanobis distance <= 5.991); "different place"
  = past pairs it rules out (> 13.82, the 99.9 % chi-square quantile). The
  geometric labels are noisy but cheap, and they let appearance take over
  when geometry later fails (drift, cuts).
* ``split_half`` (v1, ablation): "same place" = the two halves of each
  completed segment; "different place" = all past pairs two emissions apart.
  Diagnosed as over-optimistic: halves are more similar than real revisits.

Both use robust (median/MAD) Gaussian fits.

View mixture (v4, exploratory). A revisit need not share the camera view: a
forward camera passing in the opposite direction sees different content. With
``view_mixture`` the appearance ratio becomes

    LR_app(s) = rho * LR_view(s) + (1 - rho),

where ``LR_view`` compares a "same view" fit (geometry-confirmed past pairs
whose *odometry* headings differ by less than ``view_heading_deg``) against the
background, and ``rho`` is the Laplace-smoothed fraction of confirmed pairs
that share the view. Opposite passes are penalized at most by ``1 - rho``;
shared views with high similarity can contribute strong evidence. Decisions do
not use the candidate's heading, so a cut that rotates odometry cannot veto.

Both fits share a pooled variance, so the log ratio is linear and
non-decreasing in similarity: a higher similarity never reduces the evidence. Geometry and appearance are assumed
conditionally independent. No ground truth or labels are used; the decision
is to accept the MAP revisit when its posterior is at least ``tau``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vts_core.matching import gaussian_position_statistics
from vts_core.topo_graph import LOOP_TEMPORAL_EXCLUSION_NODES, LoopClosureEvent, TopoGraph, TopoNode

_MIN_HALF_FRAMES = 5
_MIN_SAME_SAMPLES = 3
_MIN_DIFF_SAMPLES = 10
_MIN_AREA_M2 = 25.0
_CONFIRM_D2 = 5.991464547107979  # chi-square 95 %, 2 dof (the mapper's gate)
_EXCLUDE_D2 = 13.815510557964274  # chi-square 99.9 %, 2 dof


@dataclass(frozen=True)
class CalibrationPrior:
    """Label-free encoder calibration pooled from *other* environments.

    Each entry is (median, robust scale, count) of geometry-confirmed same-view
    pairs and geometry-incompatible pairs, plus the confirmed/shared counts that
    estimate rho. Online data are combined with ``weight`` pseudo-samples.
    """

    view: tuple[float, float, int]
    diff: tuple[float, float, int]
    confirmed: int
    shared: int
    weight: float = 10.0


def _shrink(online: list[float], prior: tuple[float, float, int] | None, weight: float) -> tuple[float, float] | None:
    """Combine an online robust fit with ``weight`` pseudo-samples of a prior fit."""
    if prior is None:
        return _robust_gaussian(online) if len(online) >= _MIN_SAME_SAMPLES else None
    mu0, s0, _ = prior
    if len(online) < 2:
        return mu0, s0
    mu1, s1 = _robust_gaussian(online)
    n = float(len(online))
    mu = (weight * mu0 + n * mu1) / (weight + n)
    var = (weight * (s0**2 + (mu0 - mu) ** 2) + n * (s1**2 + (mu1 - mu) ** 2)) / (weight + n)
    return mu, float(np.sqrt(var))


def calibration_samples(nodes: list[TopoNode], places: list["Place"], kind: str,
                        view_heading_deg: float = 45.0, place_sigma: float = 0.5) -> dict:
    """Geometry-labelled score samples of one traversal (no ground truth)."""
    view, diff, confirmed, shared = [], [], 0, 0
    for i in range(len(nodes)):
        eligible = list(range(0, i - LOOP_TEMPORAL_EXCLUSION_NODES))
        if not eligible:
            continue
        delta = np.asarray(nodes[i].pose[:2]) - np.asarray([nodes[j].pose[:2] for j in eligible])
        covariance = np.asarray([nodes[i].pose_covariance - nodes[j].pose_covariance for j in eligible])
        covariance = 0.5 * (covariance + covariance.swapaxes(-1, -2)) + np.eye(2) * place_sigma**2
        d2, _ = gaussian_position_statistics(delta, covariance)
        s = np.asarray([similarity(places[i], places[j], kind) for j in eligible])
        heading = np.asarray([nodes[j].pose[2] for j in eligible])
        dh = np.abs((heading - nodes[i].pose[2] + np.pi) % (2 * np.pi) - np.pi)
        ok = d2 <= _CONFIRM_D2
        view += s[ok & (dh < np.radians(view_heading_deg))].tolist()
        diff += s[d2 > _EXCLUDE_D2].tolist()
        confirmed += int(ok.sum())
        shared += int((ok & (dh < np.radians(view_heading_deg))).sum())
    return {"view": view, "diff": diff, "confirmed": confirmed, "shared": shared}


def pooled_prior(samples: list[dict], weight: float = 10.0) -> CalibrationPrior:
    view = sum((s["view"] for s in samples), [])
    diff = sum((s["diff"] for s in samples), [])
    return CalibrationPrior((*_robust_gaussian(view), len(view)), (*_robust_gaussian(diff), len(diff)),
                            sum(s["confirmed"] for s in samples), sum(s["shared"] for s in samples), weight)


@dataclass(frozen=True)
class PosteriorConfig:
    appearance: str = "mean_cosine"  # or "multiview_cosine"
    calibration: str = "geometry"  # or "split_half"
    eps: float = 0.05
    prior_revisit: float = 0.5
    place_sigma: float = 0.5
    use_geometry: bool = True
    use_appearance: bool = True
    view_mixture: bool = False
    view_heading_deg: float = 45.0
    use_prior: bool = False  # cross-environment calibration prior (v5)


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


@dataclass(frozen=True)
class Place:
    """Appearance summary of a node or half-segment: mean direction and views."""

    mean: np.ndarray
    views: np.ndarray


def place_from_frames(descriptors: np.ndarray, indices: np.ndarray,
                      view_ids: tuple[int, ...] | None = None) -> Place:
    """Mean direction of ``indices``; views at ``view_ids`` (default: medoid, first, last)."""
    samples = _unit(descriptors[indices].astype(np.float64))
    mean = _unit(samples.sum(axis=0))
    if view_ids is None:
        medoid = int(indices[int(np.argmax(samples @ mean))])
        view_ids = tuple(dict.fromkeys((medoid, int(indices[0]), int(indices[-1]))))
    return Place(mean, _unit(descriptors[list(view_ids)].astype(np.float64)))


def similarity(a: Place, b: Place, kind: str) -> float:
    if kind == "mean_cosine":
        return float(a.mean @ b.mean)
    pairwise = a.views @ b.views.T
    return float(0.5 * (pairwise.max(axis=1).mean() + pairwise.max(axis=0).mean()))


def _robust_gaussian(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    scale = 1.4826 * float(np.median(np.abs(array - median)))
    return median, max(scale, 1e-3)


def log_likelihood_ratio(scores: np.ndarray, same: tuple[float, float], diff: tuple[float, float]) -> np.ndarray:
    """Pooled-variance Gaussian log ratio: linear and non-decreasing in the score.

    Unequal-variance ratios are non-monotone and can reward low similarity;
    a pooled variance keeps "more similar" from ever counting as less evidence.
    If the "same place" mean does not exceed the background, appearance is
    treated as uninformative.
    """
    (ms, ss), (md, sd) = same, diff
    if ms <= md:
        return np.zeros_like(scores)
    variance = 0.5 * (ss * ss + sd * sd)
    return (ms - md) / variance * (scores - 0.5 * (ms + md))


def associate(nodes: list[TopoNode], places: list[Place], halves: list[tuple[Place, Place] | None],
              config: PosteriorConfig, details: list | None = None,
              prior: CalibrationPrior | None = None) -> list[dict]:
    """Causal posterior association; returns one decision record per node >= 1.

    If ``details`` is a list, per-candidate evidence is appended to it for
    diagnostics; decisions are unaffected.
    """
    kind = config.appearance
    same_scores: list[float] = []
    diff_scores: list[float] = []
    view_scores: list[float] = []
    confirmed = shared = 0
    view_rad = np.radians(config.view_heading_deg)
    score_cache: dict[tuple[int, int], float] = {}

    def score(a: int, b: int) -> float:
        key = (min(a, b), max(a, b))
        if key not in score_cache:
            score_cache[key] = similarity(places[a], places[b], kind)
        return score_cache[key]

    def displacement(i: int, js: list[int]) -> tuple[np.ndarray, np.ndarray]:
        delta = np.asarray(nodes[i].pose[:2]) - np.asarray([nodes[j].pose[:2] for j in js])
        covariance = np.asarray([nodes[i].pose_covariance - nodes[j].pose_covariance for j in js])
        covariance = 0.5 * (covariance + covariance.swapaxes(-1, -2)) + np.eye(2) * config.place_sigma**2
        return gaussian_position_statistics(delta, covariance)

    records = []
    for i in range(len(nodes)):
        eligible = list(range(0, i - LOOP_TEMPORAL_EXCLUSION_NODES))
        if config.calibration == "split_half" and halves[i] is not None:
            same_scores.append(similarity(*halves[i], kind))
        if i == 0:
            continue
        if not eligible:
            records.append({"query": i, "matched": None, "posterior": 0.0, "new": 1.0})
            continue
        s = np.asarray([score(i, j) for j in eligible])
        d2, nll = displacement(i, eligible)
        log_lr = np.zeros(len(eligible))
        log_app = np.zeros(len(eligible))
        if config.use_appearance and config.view_mixture and config.use_prior:
            if prior is None:
                raise ValueError("use_prior requires a CalibrationPrior")
            weight = prior.weight
            view_fit = _shrink(view_scores, prior.view, weight)
            diff_fit = _shrink(diff_scores, prior.diff, weight)
            rho0 = prior.shared / max(prior.confirmed, 1)
            rho = (shared + weight * rho0) / (confirmed + weight)
            view = log_likelihood_ratio(s, view_fit, diff_fit)
            log_app = np.logaddexp(np.log(rho) + view, np.log1p(-rho))
        elif config.use_appearance and config.view_mixture:
            if len(view_scores) >= _MIN_SAME_SAMPLES and len(diff_scores) >= _MIN_DIFF_SAMPLES:
                rho = (shared + 1.0) / (confirmed + 2.0)
                view = log_likelihood_ratio(s, _robust_gaussian(view_scores), _robust_gaussian(diff_scores))
                log_app = np.logaddexp(np.log(rho) + view, np.log1p(-rho))
        elif config.use_appearance and len(same_scores) >= _MIN_SAME_SAMPLES and len(diff_scores) >= _MIN_DIFF_SAMPLES:
            log_app = log_likelihood_ratio(s, _robust_gaussian(same_scores), _robust_gaussian(diff_scores))
        log_lr += log_app
        if config.use_geometry:
            positions = np.asarray([node.pose[:2] for node in nodes[: i + 1]])
            extent = positions.max(axis=0) - positions.min(axis=0)
            area = max(float(np.prod(extent)), _MIN_AREA_M2)
            # nll = 0.5 * (d2 + log det); density = exp(-nll) / (2 pi).
            log_gauss = -nll - np.log(2.0 * np.pi)
            log_lr += np.logaddexp(np.log1p(-config.eps) + log_gauss + np.log(area), np.log(config.eps))
        if details is not None:
            details.append({"query": i, "eligible": eligible, "score": s, "log_app": log_app,
                            "log_geo": log_lr - log_app, "d2": d2})
        log_prior = np.log(config.prior_revisit / len(eligible))
        log_joint = np.r_[np.log1p(-config.prior_revisit), log_prior + log_lr]
        posterior = np.exp(log_joint - np.logaddexp.reduce(log_joint))
        best = int(np.argmax(posterior[1:]))
        records.append({"query": i, "matched": eligible[best], "posterior": float(posterior[1 + best]),
                        "new": float(posterior[0]), "score": float(s[best])})
        # Only after deciding, the query's pairs become calibration data.
        if config.calibration == "geometry":
            ok = d2 <= _CONFIRM_D2
            same_scores.extend(s[ok].tolist())
            diff_scores.extend(s[d2 > _EXCLUDE_D2].tolist())
            heading = np.asarray([nodes[j].pose[2] for j in eligible])
            dh = np.abs((heading - nodes[i].pose[2] + np.pi) % (2 * np.pi) - np.pi)
            view_scores.extend(s[ok & (dh < view_rad)].tolist())
            confirmed += int(ok.sum())
            shared += int((ok & (dh < view_rad)).sum())
        else:
            diff_scores.extend(s.tolist())
    return records


def decisions_to_graph(nodes: list[TopoNode], records: list[dict], tau: float) -> TopoGraph:
    """Materialize decisions at threshold ``tau`` as a graph scored like the mapper's."""
    graph = TopoGraph()
    for node in nodes:
        graph.add_node(node)
        if node.node_id:
            graph.add_edge(node.node_id - 1, node.node_id)
    for record in records:
        accepted = record["matched"] is not None and record["posterior"] >= tau
        graph.loop_events.append(LoopClosureEvent(
            query_node_id=record["query"], matched_node_id=record["matched"], accepted=accepted,
            similarity=record.get("score"), reason="accepted" if accepted else "rejected_posterior",
            decision_node_id=record["query"]))
        if accepted:
            graph.add_edge(record["matched"], record["query"], edge_type="loop")
    return graph
