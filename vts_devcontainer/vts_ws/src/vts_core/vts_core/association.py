"""In-session place association: each new place is a revisit or a new place.

Two associators share one incremental interface, ``observe(place)``, and one
:class:`Decision` record. Both are causal: a decision uses only earlier places.
Candidates exclude the immediately preceding place (``A -> B -> A`` is
eligible).

:class:`GateAssociator`
    The manuscript system and the baseline to beat: a chi-square gate on the
    odometry budget generates candidates, ranked by Gaussian NLL; a unique
    candidate is accepted, several are disambiguated by a robust visual outlier
    test on bidirectional three-place sequences of multi-view cosine
    similarity. ``use_appearance=False`` is the geometry-only baseline.
    Bit-compatible with the recorded runs (``self_only`` sequences, cosine).
:class:`PosteriorAssociator`
    Posterior over "new place" and "revisit of j" (study C2). Geometry is a
    Gaussian under the budget mixed with a uniform outlier over the explored
    area; appearance is a likelihood ratio calibrated online, without labels,
    from earlier pairs that geometry confirms or rules out, optionally with a
    view mixture and a cross-environment calibration prior.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vts_core.places import Place

CHI2_95_2DOF = 5.991464547107979
CHI2_999_2DOF = 13.815510557964274
PLACE_SIGMA = 0.5
EXCLUDED_RECENT = 1  # the immediately preceding place is never a candidate
_MIN_SAME_SAMPLES = 3
_MIN_DIFF_SAMPLES = 10
_MIN_AREA_M2 = 25.0
_BACKGROUND_TARGET = 5


@dataclass
class Decision:
    """One association decision, made when ``query`` was emitted."""

    query: int
    matched: int | None
    accepted: bool
    score: float | None
    reason: str
    posterior: float | None = None
    new: float | None = None


def gaussian_position_statistics(delta: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Batched squared Mahalanobis distance and NLL ``0.5 (d2 + log det)``."""
    cov = np.asarray(covariance, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    cov = 0.5 * (cov + np.swapaxes(cov, -1, -2)) + np.eye(2) * 1e-6
    finite = np.isfinite(cov).all(axis=(-2, -1))
    safe = np.where(finite[..., None, None], cov, np.eye(2))
    sign, logdet = np.linalg.slogdet(safe)
    valid = finite & (safe[..., 0, 0] > 0) & (sign > 0)
    safe = np.where(valid[..., None, None], safe, np.eye(2))
    solution = np.linalg.solve(safe, delta[..., None])[..., 0]
    d2 = np.einsum("...i,...i->...", delta, solution)
    valid = valid & np.isfinite(d2) & (d2 >= 0)
    return np.where(valid, d2, np.inf), np.where(valid, 0.5 * (d2 + logdet), np.inf)


def interval_position_covariance(query: Place, candidates: list[Place], place_sigma: float,
                                 check: bool = True) -> np.ndarray:
    covariance = query.budget - np.asarray([c.budget for c in candidates])
    covariance = 0.5 * (covariance + covariance.swapaxes(-1, -2))
    if check and np.min(np.linalg.eigvalsh(covariance)) < -1e-9:
        raise ValueError("uncertainty budget decreased between places")
    return covariance + np.eye(2) * place_sigma**2


def multiview_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Symmetric mean of best view matches (reference, one pair)."""
    pairwise = first @ second.T
    return float(0.5 * (pairwise.max(axis=1).mean() + pairwise.max(axis=0).mean()))


class SimilarityRows:
    """Incremental similarity of each new place to all earlier ones (one matmul)."""

    def __init__(self, kind: str, dtype: type = np.float64) -> None:
        if kind not in {"mean_cosine", "multiview_cosine"}:
            raise ValueError("kind must be mean_cosine or multiview_cosine")
        self.kind, self._dtype = kind, dtype
        self._blocks: list[np.ndarray] = []
        self._stacked = np.empty((0, 0), dtype=dtype)
        self._offsets: list[int] = []
        self.rows: list[np.ndarray] = []

    def features(self, place: Place) -> np.ndarray:
        if self.kind == "mean_cosine":
            return np.asarray(place.mean, dtype=self._dtype)[None]
        views = np.asarray(place.views, dtype=self._dtype)
        if self._dtype == np.float64:
            return views
        # The recorded mapper re-normalized float32 views once more.
        return views / np.maximum(np.linalg.norm(views, axis=1, keepdims=True), 1e-12)

    def add(self, place: Place) -> np.ndarray:
        block = self.features(place)
        if self._blocks:
            pairwise = block @ self._stacked.T
            if self.kind == "mean_cosine":
                row = pairwise[0].astype(np.float64)
            else:
                # One matmul; per-place reductions keep the recorded rounding.
                bounds = [*self._offsets, pairwise.shape[1]]
                row = np.asarray([0.5 * (pairwise[:, a:b].max(axis=1).mean() + pairwise[:, a:b].max(axis=0).mean())
                                  for a, b in zip(bounds[:-1], bounds[1:])], dtype=np.float64)
        else:
            row = np.empty(0)
        self._offsets.append(self._stacked.shape[0] if self._blocks else 0)
        self._blocks.append(block)
        self._stacked = np.concatenate(self._blocks)
        self.rows.append(row)
        return row

    def pair(self, first: int, second: int) -> float:
        """Similarity with ``first`` as the left operand, as the recorded mapper computed it."""
        a, b = self._blocks[first], self._blocks[second]
        if self.kind == "mean_cosine":
            return float(a[0] @ b[0])
        return multiview_similarity(a, b)

    def get(self, i: int, j: int) -> float:
        return float(self.rows[i][j]) if j < i else float(self.rows[j][i])


def robust_outlier(scores: np.ndarray, index: int, k: float) -> bool:
    """True if ``scores[index]`` exceeds median + k robust std of the others."""
    if scores.shape[0] < 2:
        return False
    others = np.delete(scores, index)
    median = float(np.median(others))
    mad = float(np.median(np.abs(others - median)))
    # Strict: an indistinctive tie (zero MAD) is rejected.
    return float(scores[index]) > median + k * 1.4826 * mad


class GateAssociator:
    """Chi-square odometry gate with visual disambiguation (the baseline)."""

    def __init__(self, visual_k: float = 2.0, gate_scale: float = 1.0, use_appearance: bool = True,
                 place_sigma: float = PLACE_SIGMA) -> None:
        if not np.isfinite([visual_k, gate_scale]).all() or visual_k < 0 or gate_scale <= 0:
            raise ValueError("visual_k must be >= 0 and gate_scale > 0")
        self.visual_k, self.gate_scale = visual_k, gate_scale
        self.use_appearance, self.place_sigma = use_appearance, place_sigma
        self.places: list[Place] = []
        self.similarity = SimilarityRows("multiview_cosine", np.float32)

    def _sequence_scores(self, query: int) -> np.ndarray:
        """Bidirectional three-place sequence score for every candidate.

        The query is compared with candidate ``j``; the two previous places
        ``q-1, q-2`` with ``j-+1, j-+2`` (both traversal directions). A pair
        whose historical place is the live place itself is skipped; the walk
        stops at the ends of the map. The stronger direction is returned.
        """
        sim = self.similarity
        direct = sim.rows[query][: query - EXCLUDED_RECENT]
        live = [query - 1] + ([query - 2] if query >= 2 else [])
        best = np.full(len(direct), -np.inf)
        for direction in (-1, 1):
            for j in range(len(direct)):
                scores = [float(direct[j])]
                historical = j
                for live_id in live:
                    historical += direction
                    if not 0 <= historical < query:
                        break
                    if historical == live_id:
                        continue
                    scores.append(sim.get(live_id, historical) if historical < live_id
                                  else sim.pair(live_id, historical))
                best[j] = max(best[j], float(np.mean(scores)))
        return best

    def observe(self, place: Place) -> Decision | None:
        query = len(self.places)
        self.places.append(place)
        if self.use_appearance:
            self.similarity.add(place)
        if query == 0:
            return None
        eligible = query - EXCLUDED_RECENT
        if eligible <= 0:
            return Decision(query, None, False, None, "no_eligible_candidate")
        candidates = self.places[:eligible]
        delta = place.pose[:2] - np.asarray([c.pose[:2] for c in candidates])
        d2, nll = gaussian_position_statistics(
            delta, interval_position_covariance(place, candidates, self.place_sigma))
        order = np.argsort(nll, kind="stable")
        gated = [int(i) for i in order if d2[i] <= CHI2_95_2DOF * self.gate_scale]
        if not gated:
            return Decision(query, int(order[0]), False, None, "rejected_by_gate")
        if not self.use_appearance:
            return Decision(query, gated[0], True, None, "accepted")
        scores = self._sequence_scores(query)
        if len(gated) == 1:
            return Decision(query, gated[0], True, float(scores[gated[0]]), "accepted")
        for selected in gated:
            if len(scores) == 1 or self._distinctive(scores, selected, gated):
                return Decision(query, selected, True, float(scores[selected]), "accepted")
        return Decision(query, gated[0], False, float(scores[gated[0]]), "rejected_by_gate")

    def _distinctive(self, scores: np.ndarray, selected: int, gated: list[int]) -> bool:
        """Visual outlier among gated candidates, padded with weaker visual ones.

        A visually stronger place already ruled out by odometry is irrelevant:
        appearance verifies an odometric hypothesis, it does not retrieve one.
        """
        background = list(gated)
        if len(background) < _BACKGROUND_TARGET:
            excluded = set(gated)
            background.extend(int(i) for i in np.argsort(-scores, kind="stable")
                              if scores[i] <= scores[selected] and int(i) not in excluded)
        return robust_outlier(scores[background], background.index(selected), self.visual_k)


# --------------------------------------------------------------------------- #
# Posterior association
# --------------------------------------------------------------------------- #
def robust_gaussian(values: list[float] | np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    return median, max(1.4826 * float(np.median(np.abs(array - median))), 1e-3)


def log_likelihood_ratio(scores: np.ndarray, same: tuple[float, float], diff: tuple[float, float]) -> np.ndarray:
    """Pooled-variance Gaussian log ratio: linear and non-decreasing in the score.

    If the "same place" mean does not exceed the background, appearance is
    uninformative (zero).
    """
    (ms, ss), (md, sd) = same, diff
    if ms <= md:
        return np.zeros_like(scores, dtype=np.float64)
    return (ms - md) / (0.5 * (ss * ss + sd * sd)) * (scores - 0.5 * (ms + md))


def geometric_log_lr(nll: np.ndarray, area: float, eps: float) -> np.ndarray:
    """log[(1 - eps) N(delta) A + eps]: Gaussian with a uniform outlier, >= log eps."""
    log_gauss = -nll - np.log(2.0 * np.pi)  # nll = 0.5 (d2 + log det)
    return np.logaddexp(np.log1p(-eps) + log_gauss + np.log(area), np.log(eps))


@dataclass(frozen=True)
class CalibrationPrior:
    """Label-free appearance calibration pooled from other environments.

    ``view``/``diff``: (median, robust scale, count) of geometry-confirmed
    same-view pairs and geometry-excluded pairs; ``confirmed``/``shared`` estimate
    the fraction of confirmed revisits that share the view. Online samples are
    combined with ``weight`` pseudo-samples.
    """

    view: tuple[float, float, int]
    diff: tuple[float, float, int]
    confirmed: int
    shared: int
    weight: float = 10.0


def shrink(online: list[float], prior: tuple[float, float, int], weight: float) -> tuple[float, float]:
    mu0, s0, _ = prior
    if len(online) < 2:
        return mu0, s0
    mu1, s1 = robust_gaussian(online)
    n = float(len(online))
    mu = (weight * mu0 + n * mu1) / (weight + n)
    var = (weight * (s0**2 + (mu0 - mu) ** 2) + n * (s1**2 + (mu1 - mu) ** 2)) / (weight + n)
    return mu, float(np.sqrt(var))


def _heading_difference(query: Place, candidates: list[Place]) -> np.ndarray:
    heading = np.asarray([c.pose[2] for c in candidates])
    return np.abs((heading - query.pose[2] + np.pi) % (2 * np.pi) - np.pi)


@dataclass
class _Calibration:
    """Geometry-labelled appearance samples (no ground truth)."""

    same: list[float] = field(default_factory=list)
    diff: list[float] = field(default_factory=list)
    view: list[float] = field(default_factory=list)
    confirmed: int = 0
    shared: int = 0

    def add(self, scores: np.ndarray, d2: np.ndarray, heading_difference: np.ndarray, view_rad: float) -> None:
        ok = d2 <= CHI2_95_2DOF
        same_view = ok & (heading_difference < view_rad)
        self.same.extend(scores[ok].tolist())
        self.diff.extend(scores[d2 > CHI2_999_2DOF].tolist())
        self.view.extend(scores[same_view].tolist())
        self.confirmed += int(ok.sum())
        self.shared += int(same_view.sum())


def calibration_samples(places: list[Place], kind: str = "mean_cosine", view_heading_deg: float = 45.0,
                        place_sigma: float = PLACE_SIGMA) -> _Calibration:
    """Calibration samples of one whole traversal, for a cross-environment prior."""
    rows = SimilarityRows(kind)
    calibration = _Calibration()
    for i, place in enumerate(places):
        row = rows.add(place)
        eligible = i - EXCLUDED_RECENT
        if eligible <= 0:
            continue
        candidates = places[:eligible]
        d2, _ = gaussian_position_statistics(
            place.pose[:2] - np.asarray([c.pose[:2] for c in candidates]),
            interval_position_covariance(place, candidates, place_sigma, check=False))
        calibration.add(row[:eligible], d2, _heading_difference(place, candidates), np.radians(view_heading_deg))
    return calibration


def pooled_prior(samples: list[_Calibration], weight: float = 10.0) -> CalibrationPrior:
    view = [v for s in samples for v in s.view]
    diff = [v for s in samples for v in s.diff]
    return CalibrationPrior((*robust_gaussian(view), len(view)), (*robust_gaussian(diff), len(diff)),
                            sum(s.confirmed for s in samples), sum(s.shared for s in samples), weight)


@dataclass(frozen=True)
class PosteriorConfig:
    appearance: str = "mean_cosine"  # or "multiview_cosine"
    eps: float = 0.05
    prior_revisit: float = 0.5
    place_sigma: float = PLACE_SIGMA
    use_geometry: bool = True
    use_appearance: bool = True
    view_mixture: bool = False
    view_heading_deg: float = 45.0
    use_prior: bool = False
    tau: float = 0.5


class PosteriorAssociator:
    """Causal posterior association with an explicit new-place hypothesis.

    With prior ``pi`` of revisiting, split uniformly over the ``n`` candidates,
    ``P(j | data) ∝ (pi / n) LR_geo(j) LR_app(j)`` and ``P(new) ∝ 1 - pi``.
    The MAP revisit is accepted when its posterior is at least ``tau``. A
    query's own pairs join the calibration only after its decision.
    """

    def __init__(self, config: PosteriorConfig = PosteriorConfig(), prior: CalibrationPrior | None = None) -> None:
        if config.use_prior and prior is None:
            raise ValueError("use_prior requires a CalibrationPrior")
        self.config, self.prior = config, prior
        self.places: list[Place] = []
        self.similarity = SimilarityRows(config.appearance)
        self.calibration = _Calibration()
        self.details: list[dict] = []
        self._view_rad = np.radians(config.view_heading_deg)
        self._low = np.full(2, np.inf)
        self._high = np.full(2, -np.inf)

    def _log_appearance(self, scores: np.ndarray) -> np.ndarray:
        config, cal = self.config, self.calibration
        if not config.use_appearance:
            return np.zeros(len(scores))
        if config.view_mixture and config.use_prior:
            weight = self.prior.weight
            rho0 = self.prior.shared / max(self.prior.confirmed, 1)
            rho = (cal.shared + weight * rho0) / (cal.confirmed + weight)
            view = log_likelihood_ratio(scores, shrink(cal.view, self.prior.view, weight),
                                        shrink(cal.diff, self.prior.diff, weight))
            return np.logaddexp(np.log(rho) + view, np.log1p(-rho))
        if config.view_mixture:
            if len(cal.view) < _MIN_SAME_SAMPLES or len(cal.diff) < _MIN_DIFF_SAMPLES:
                return np.zeros(len(scores))
            rho = (cal.shared + 1.0) / (cal.confirmed + 2.0)
            view = log_likelihood_ratio(scores, robust_gaussian(cal.view), robust_gaussian(cal.diff))
            return np.logaddexp(np.log(rho) + view, np.log1p(-rho))
        if len(cal.same) < _MIN_SAME_SAMPLES or len(cal.diff) < _MIN_DIFF_SAMPLES:
            return np.zeros(len(scores))
        return log_likelihood_ratio(scores, robust_gaussian(cal.same), robust_gaussian(cal.diff))

    def observe(self, place: Place) -> Decision | None:
        config = self.config
        query = len(self.places)
        self.places.append(place)
        row = self.similarity.add(place)
        self._low = np.minimum(self._low, place.pose[:2])
        self._high = np.maximum(self._high, place.pose[:2])
        if query == 0:
            return None
        eligible = query - EXCLUDED_RECENT
        if eligible <= 0:
            return Decision(query, None, False, None, "no_eligible_candidate", 0.0, 1.0)
        candidates = self.places[:eligible]
        scores = row[:eligible]
        d2, nll = gaussian_position_statistics(
            place.pose[:2] - np.asarray([c.pose[:2] for c in candidates]),
            interval_position_covariance(place, candidates, config.place_sigma, check=False))
        log_app = self._log_appearance(scores)
        log_lr = np.zeros(eligible) + log_app
        if config.use_geometry:
            area = max(float(np.prod(self._high - self._low)), _MIN_AREA_M2)
            log_lr += geometric_log_lr(nll, area, config.eps)
        self.details.append({"query": query, "score": scores, "log_app": log_app, "log_geo": log_lr - log_app,
                             "d2": d2})
        log_joint = np.r_[np.log1p(-config.prior_revisit), np.log(config.prior_revisit / eligible) + log_lr]
        posterior = np.exp(log_joint - np.logaddexp.reduce(log_joint))
        best = int(np.argmax(posterior[1:]))
        self.calibration.add(scores, d2, _heading_difference(place, candidates), self._view_rad)
        accepted = bool(posterior[1 + best] >= config.tau)
        return Decision(query, best, accepted, float(scores[best]), "accepted" if accepted else "rejected_posterior",
                        float(posterior[1 + best]), float(posterior[0]))


def run(associator, places: list[Place]) -> list[Decision]:
    """Feed places in order; return every decision."""
    return [d for d in map(associator.observe, places) if d is not None]
