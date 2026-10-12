"""Building blocks for associating a new session B with a stored map A.

There is no common odometry frame between sessions. Until an anchor exists,
only appearance can associate places; after a few consistent associations an
SE(2) transform between the frames can be estimated and geometry becomes
informative again (docs/PLAN_B_MULTISESSION.md, section 2).

* :func:`similarity_matrix`: B x A appearance scores (one batched product).
* :func:`distance_sequence`: sequence evidence aligned by travelled distance,
  not by place index (segmentations of two sessions do not coincide), in both
  traversal directions, with bounded jumps.
* :class:`AnchorTracker`: a bounded set of SE(2) hypotheses, each proposed by
  two candidate associations with compatible local shape and supported by
  later candidates consistent with it (in the spirit of PCM).

These are components; how they combine is the method under development and
must be measured stage by stage against the appearance-only baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vts_core.association import CHI2_95_2DOF, PLACE_SIGMA
from vts_core.odometry import RelativeUncertainty
from vts_core.places import Place


# --------------------------------------------------------------------------- #
# Appearance
# --------------------------------------------------------------------------- #
def similarity_matrix(queries: list[Place], references: list[Place], kind: str = "mean_cosine") -> np.ndarray:
    """Cosine of mean directions, or the symmetric multi-view score, for all pairs."""
    if kind == "mean_cosine":
        return np.asarray([p.mean for p in queries]) @ np.asarray([p.mean for p in references]).T
    if kind != "multiview_cosine":
        raise ValueError("kind must be mean_cosine or multiview_cosine")

    def padded(places: list[Place]) -> tuple[np.ndarray, np.ndarray]:
        dim = places[0].views.shape[1]
        views = np.zeros((len(places), 3, dim))
        mask = np.zeros((len(places), 3), dtype=bool)
        for i, p in enumerate(places):
            views[i, : len(p.views)], mask[i, : len(p.views)] = p.views, True
        return views, mask

    vq, mq = padded(queries)
    vr, mr = padded(references)
    pairwise = np.einsum("iad,jbd->ijab", vq, vr)
    valid = mq[:, None, :, None] & mr[None, :, None, :]
    pairwise = np.where(valid, pairwise, -np.inf)
    row = np.where(mq[:, None, :], pairwise.max(axis=3), 0.0).sum(axis=2) / mq.sum(axis=1)[:, None]
    col = np.where(mr[None, :, :], pairwise.max(axis=2), 0.0).sum(axis=2) / mr.sum(axis=1)[None, :]
    return 0.5 * (row + col)


def place_distance(places: list[Place], frame_distance: np.ndarray) -> np.ndarray:
    """Travelled distance (local odometry) at each place's medoid frame."""
    return np.asarray([frame_distance[p.medoid] for p in places])


def distance_sequence(scores: np.ndarray, query_distance: np.ndarray, reference_distance: np.ndarray,
                      length_m: float = 4.0, step_m: float = 1.0, jump_m: float = 1.0) -> np.ndarray:
    """Sequence evidence by travelled distance, causal in the query session.

    For query ``i`` and reference ``j`` the evidence averages ``scores[i, j]``
    with, for ``k = 1..length/step``, the query place travelled ``k * step``
    before ``i`` compared with the best reference place within ``jump_m`` of
    ``k * step`` before (same direction) or after (opposite direction) ``j``.
    Terms beyond either map end are skipped; the stronger direction wins.
    Short-term odometry is reliable, so distance aligns sessions whose
    segmentations differ.
    """
    n_q, n_r = scores.shape
    out = np.empty_like(scores, dtype=np.float64)
    steps = np.arange(1, round(length_m / step_m) + 1) * step_m
    for i in range(n_q):
        # Query places strictly before i, nearest to each target distance.
        history = []
        for offset in steps:
            target = query_distance[i] - offset
            if i == 0 or target < query_distance[0] - jump_m:
                break
            k = int(np.argmin(np.abs(query_distance[:i] - target)))
            if abs(query_distance[k] - target) <= jump_m:
                history.append((k, offset))
        best = np.full(n_r, -np.inf)
        for direction in (1, -1):
            total, count = scores[i].astype(np.float64).copy(), np.ones(n_r)
            for k, offset in history:
                target = reference_distance - direction * offset
                lo = np.searchsorted(reference_distance, target - jump_m, side="left")
                hi = np.searchsorted(reference_distance, target + jump_m, side="right")
                for j in np.nonzero(hi > lo)[0]:
                    total[j] += scores[k, lo[j]:hi[j]].max()
                    count[j] += 1
            best = np.maximum(best, total / count)
        out[i] = best
    return out


# --------------------------------------------------------------------------- #
# SE(2) anchoring
# --------------------------------------------------------------------------- #
def fit_se2(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares rotation and translation with ``target ≈ source R^T + t``."""
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if len(source) < 2:
        raise ValueError("an SE(2) fit needs at least two correspondences")
    cs, ct = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - cs).T @ (target - ct))
    rotation = vt.T @ np.diag([1.0, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
    return rotation, ct - rotation @ cs


@dataclass
class Hypothesis:
    rotation: np.ndarray
    translation: np.ndarray
    support: list[tuple[int, int]]  # (query place, reference place)
    evidence: float

    def apply(self, xy: np.ndarray) -> np.ndarray:
        return np.asarray(xy) @ self.rotation.T + self.translation

    @property
    def queries(self) -> set[int]:
        return {i for i, _ in self.support}


@dataclass
class AnchorConfig:
    min_baseline_m: float = 1.0  # two correspondences closer than this fix no rotation
    shape_sigmas: float = 3.0  # local-shape compatibility of two correspondences
    place_sigma: float = PLACE_SIGMA
    detection: float = 0.6  # P(a consistent candidate | hypothesis true): overlap x retrieval
    history: int = 20  # recent candidate associations kept for proposing pairs
    max_hypotheses: int = 50
    min_support: int = 5  # distinct query places supporting an accepted anchor (dev sweep 2026-10-02)
    min_evidence: float = 6.0  # accumulated log likelihood ratio of an accepted anchor
    margin: float = 6.0  # log evidence over the best incompatible hypothesis (dev sweep 2026-10-02)


@dataclass
class AnchorTracker:
    """Bounded multi-hypothesis estimate of the transform map A <- session B.

    Each query place of B brings its candidate associations ``(j, log_lr)``
    (e.g. appearance top-k). Every hypothesis is tested against them: a *hit*
    (some candidate inside its chi-square gate) or a *miss*. Evidence is a
    sequential log likelihood ratio against chance:

        hit:  log[(d + (1 - d) p) / p] + appearance log LR of the hit
        miss: log(1 - d)

    with ``d`` the probability of a hit when the hypothesis is right and ``p``
    the probability that ``k`` unrelated candidates fall in the gate by chance
    (gate area over map area). Without the miss term, any transform of a
    corridor collects supports from random candidates. New hypotheses come from
    pairs of candidates of different queries whose inter-place distances agree
    in both sessions (in the spirit of PCM).
    """

    query_places: list[Place]
    reference_places: list[Place]
    config: AnchorConfig = field(default_factory=AnchorConfig)
    hypotheses: list[Hypothesis] = field(default_factory=list)
    recent: list[tuple[int, int, float]] = field(default_factory=list)
    anchor: Hypothesis | None = None

    def __post_init__(self) -> None:
        xy = np.asarray([p.pose[:2] for p in self.reference_places])
        self._area = max(float(np.prod(xy.max(axis=0) - xy.min(axis=0))), 25.0)

    def _sigma2(self, i: int, j: int, i_ref: int, j_ref: int) -> float:
        """Isotropic variance of a relative displacement in both sessions."""
        qb = np.abs(self.query_places[i].budget - self.query_places[i_ref].budget)
        rb = np.abs(self.reference_places[j].budget - self.reference_places[j_ref].budget)
        return float(np.trace(qb) / 2 + np.trace(rb) / 2 + 2 * self.config.place_sigma**2)

    def _residual(self, h: Hypothesis, i: int, j: int) -> tuple[float, float]:
        """Squared residual and isotropic variance of candidate (i, j) under ``h``."""
        i_ref, j_ref = min(h.support, key=lambda pair: abs(pair[0] - i))
        residual = h.apply(self.query_places[i].pose[:2]) - self.reference_places[j].pose[:2]
        return float(residual @ residual), self._sigma2(i, j, i_ref, j_ref)

    def _consistent(self, h: Hypothesis, i: int, j: int) -> bool:
        squared, variance = self._residual(h, i, j)
        return squared / variance <= CHI2_95_2DOF

    def _refit(self, h: Hypothesis) -> None:
        source = np.asarray([self.query_places[i].pose[:2] for i, _ in h.support])
        target = np.asarray([self.reference_places[j].pose[:2] for _, j in h.support])
        h.rotation, h.translation = fit_se2(source, target)

    def observe(self, i: int, candidates: list[tuple[int, float]]) -> Hypothesis | None:
        """Test hypotheses with query ``i``'s candidates; return the anchor once accepted."""
        config, d = self.config, self.config.detection
        for h in self.hypotheses:
            hits = [(self._residual(h, i, j), j, lr) for j, lr in candidates
                    if i not in h.queries and self._consistent(h, i, j)]
            if not hits:
                h.evidence += np.log1p(-d)
                continue
            (_, variance), j, lr = min(hits, key=lambda hit: hit[0][0] / hit[0][1])
            chance = 1.0 - (1.0 - min(1.0, np.pi * CHI2_95_2DOF * variance / self._area)) ** len(candidates)
            h.evidence += float(np.log((d + (1.0 - d) * chance) / chance) + lr)
            h.support.append((i, j))
            self._refit(h)
        for j, lr in candidates:
            self._propose(i, j, lr)
        self.recent = (self.recent + [(i, j, float(lr)) for j, lr in candidates])[-config.history:]
        self.hypotheses.sort(key=lambda h: -h.evidence)
        del self.hypotheses[config.max_hypotheses:]
        if self.anchor is None:
            self.anchor = self._accepted()
        return self.anchor

    def _propose(self, i: int, j: int, lr: float) -> None:
        config = self.config
        pi, pj = self.query_places[i].pose[:2], self.reference_places[j].pose[:2]
        for i2, j2, lr2 in self.recent:
            if i2 == i or j2 == j:
                continue
            qi, rj = self.query_places[i2].pose[:2], self.reference_places[j2].pose[:2]
            baseline = float(np.linalg.norm(pi - qi))
            if baseline < config.min_baseline_m:
                continue
            if abs(baseline - float(np.linalg.norm(pj - rj))) > config.shape_sigmas * np.sqrt(self._sigma2(i, j, i2, j2)):
                continue
            rotation, translation = fit_se2(np.stack([qi, pi]), np.stack([rj, pj]))
            pair = Hypothesis(rotation, translation, [(i2, j2), (i, j)], float(lr) + lr2)
            if not any(set(h.support) >= set(pair.support) for h in self.hypotheses):
                self.hypotheses.append(pair)

    def _accepted(self) -> Hypothesis | None:
        config = self.config
        if not self.hypotheses:
            return None
        best = self.hypotheses[0]
        if len(best.queries) < config.min_support or best.evidence < config.min_evidence:
            return None
        # Rivals: hypotheses that place the best's support elsewhere.
        rivals = [h.evidence for h in self.hypotheses[1:]
                  if not all(self._consistent(h, i, j) for i, j in best.support)]
        return best if best.evidence - max(rivals, default=0.0) >= config.margin else None


# --------------------------------------------------------------------------- #
# Online association of a session B with a stored map A
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AppearanceModel:
    """Score distributions of same-place and different-place pairs (robust Gaussians)."""

    same: tuple[float, float]
    diff: tuple[float, float]

    def log_lr(self, scores: np.ndarray) -> np.ndarray:
        from vts_core.association import log_likelihood_ratio

        return log_likelihood_ratio(np.asarray(scores, dtype=np.float64), self.same, self.diff)


@dataclass(frozen=True)
class CrossConfig:
    prior_revisit: float = 0.5
    eps: float = 0.2  # geometric outlier mass after anchoring (dev sweep 2026-10-02; 0.05 hurt Freiburg A)
    top_k: int = 3  # appearance candidates offered to the anchor tracker
    use_geometry: bool = True  # geometry after anchoring
    support_window: int = 10  # most recent supports used to refit the local transform
    uncertainty: str = "propagated"  # SE(2) Jacobians from the last support, or the additive "budget"
    sequence_m: float = 0.0  # > 0: appearance from distance-aligned sequences of this length (causal)
    anchor: AnchorConfig = field(default_factory=AnchorConfig)


@dataclass
class CrossDecision:
    query: int
    matched: int
    posterior: float
    new: float
    score: float
    anchored: bool


class CrossSessionAssociator:
    """Posterior over "revisit of map place j" and "new place" for each B place.

    ``P(j) ∝ (pi / n) LR_app(j) LR_geo(j)``, ``P(new) ∝ 1 - pi``. Before an
    anchor exists ``LR_geo = 1`` (frames unrelated). Once the tracker accepts
    an anchor, the query is predicted in A's frame with the transform refitted
    on the most recent supports (odometry drifts slowly in both sessions) and
    ``LR_geo = (1 - eps) N(r; 0, sigma^2 I) area + eps`` as in-session. Causal:
    a decision uses only earlier places of B and the whole of A.
    """

    def __init__(self, reference: list[Place], appearance: AppearanceModel | None, config: CrossConfig = CrossConfig(),
                 kind: str = "mean_cosine") -> None:
        """``appearance=None`` self-calibrates from this session (see :meth:`_appearance_model`)."""
        self.reference, self.appearance, self.config, self.kind = reference, appearance, config, kind
        self._top_scores: list[float] = []
        self._other_scores: list[float] = []
        self.queries: list[Place] = []
        self.tracker = AnchorTracker(self.queries, reference, config.anchor)
        self.decisions: list[CrossDecision] = []
        self.anchor_query: int | None = None
        xy = np.asarray([p.pose[:2] for p in reference])
        self._xy = xy
        self._area = max(float(np.prod(xy.max(axis=0) - xy.min(axis=0))), 25.0)

    def _local_transform(self, h: Hypothesis, i: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        support = sorted(h.support)[-self.config.support_window:]
        if len(support) >= 2:
            source = np.asarray([self.queries[a].pose[:2] for a, _ in support])
            target = np.asarray([self.reference[b].pose[:2] for _, b in support])
            rotation, translation = fit_se2(source, target)
        else:
            rotation, translation = h.rotation, h.translation
        return rotation, translation, support[-1]

    def _sequence_scores(self, place: Place, scores: np.ndarray) -> np.ndarray:
        """Distance-aligned sequence evidence for the newest query (earlier queries only)."""
        if not hasattr(self, "_rows"):
            self._rows, self._ref_distance = [], np.r_[0.0, np.cumsum(np.linalg.norm(
                np.diff(np.asarray([p.pose[:2] for p in self.reference]), axis=0), axis=1))]
            self._query_distance = [0.0]
        else:
            self._query_distance.append(self._query_distance[-1] + float(
                np.linalg.norm(place.pose[:2] - self.queries[-2].pose[:2])))
        self._rows.append(scores)
        return distance_sequence(np.asarray(self._rows), np.asarray(self._query_distance), self._ref_distance,
                                 length_m=self.config.sequence_m)[-1]

    def _appearance_model(self, scores: np.ndarray) -> AppearanceModel:
        """Label-free, causal calibration from this session's own score statistics.

        "Same place": robust fit to the top-1 score of every query so far (most
        top-1 matches are right; the median resists the rest). "Different":
        robust fit to all other scores. The current query is included, so the
        first decision already has a (crude) model.
        """
        from vts_core.association import robust_gaussian

        if self.appearance is not None:
            return self.appearance
        order = np.argsort(-scores)
        self._top_scores.append(float(scores[order[0]]))
        self._other_scores.extend(scores[order[1:]].tolist())
        return AppearanceModel(robust_gaussian(self._top_scores), robust_gaussian(self._other_scores))

    def observe(self, place: Place) -> CrossDecision:
        config = self.config
        i = len(self.queries)
        self.queries.append(place)
        scores = similarity_matrix([place], self.reference, self.kind)[0]
        if config.sequence_m > 0:
            scores = self._sequence_scores(place, scores)
        model = self._appearance_model(scores)
        log_lr = model.log_lr(scores)
        anchor = self.tracker.anchor
        if anchor is not None and config.use_geometry:
            rotation, translation, (i_ref, j_ref) = self._local_transform(anchor, i)
            predicted = place.pose[:2] @ rotation.T + translation
            squared = np.sum((self._xy - predicted) ** 2, axis=1)
            if config.uncertainty == "propagated":
                # Heading error since the last support grows lateral uncertainty.
                propagation = RelativeUncertainty()
                for q in self.queries[i_ref:i + 1]:
                    covariance = propagation.step(q.pose)
                q_budget = np.trace(covariance[:2, :2]) / 2
            else:
                q_budget = np.trace(np.abs(place.budget - self.queries[i_ref].budget)) / 2
            r_budget = np.asarray([np.trace(np.abs(p.budget - self.reference[j_ref].budget)) / 2
                                   for p in self.reference])
            variance = q_budget + r_budget + 2 * config.anchor.place_sigma**2
            nll = 0.5 * (squared / variance + 2.0 * np.log(variance))
            log_lr = log_lr + np.logaddexp(np.log1p(-config.eps) - nll - np.log(2 * np.pi) + np.log(self._area),
                                           np.log(config.eps))
        log_joint = np.r_[np.log1p(-config.prior_revisit), np.log(config.prior_revisit / len(scores)) + log_lr]
        posterior = np.exp(log_joint - np.logaddexp.reduce(log_joint))
        # Ties (e.g. uninformative appearance) fall back to the raw score.
        best = int(np.lexsort((-scores, -posterior[1:]))[0])
        appearance_lr = model.log_lr(scores)
        top = np.argsort(-scores)[: config.top_k]
        if self.tracker.observe(i, [(int(j), float(appearance_lr[j])) for j in top]) is not None \
                and self.anchor_query is None:
            self.anchor_query = i
        decision = CrossDecision(i, best, float(posterior[1 + best]), float(posterior[0]), float(scores[best]),
                                 anchor is not None)
        self.decisions.append(decision)
        return decision
