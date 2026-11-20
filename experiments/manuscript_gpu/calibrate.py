"""Maximum-likelihood calibration of the odometry noise coefficients on the design traversals.

Frame pairs a fixed travelled distance apart give the drift of the recorded odometry against
ground truth, expressed in the frame of the earlier pose so that the unknown transform between
the odometry and ground-truth frames cancels. The predicted covariance of that drift is linear in
the coefficients, plus an isotropic floor for ground-truth and synchronization error. Only
ground-truth poses enter; loop closures and their labels do not.

Pairs overlap, so the objective is a composite likelihood: its maximizer is consistent but its
curvature overstates precision, which is why intervals come from resampling traversals.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from stream import (CALIBRATION_FILE, DEFAULT_PARAMS, DESIGN, RESULTS, Stream, cumulative_covariances, debiased_stream,
                    load_stream)
from vts_core.matching import _CHI2_GATE_2DOF
from vts_core.motion import OdometryUncertaintyParams, relative_pose_covariance

HORIZONS_M = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
ANCHOR_SPACING_M = 1.0
PLACE_SIGMA_M = 0.5
FREE_COEFFICIENTS = {"additive": (2, 3), "propagated": (0, 1, 2, 3), "propagated_debiased": (0, 1, 2, 3)}
TRACKER = {"additive": "additive", "propagated": "propagated", "propagated_debiased": "propagated"}
LOG_BOUNDS = (-18.0, 3.0)
BOOTSTRAP_RESAMPLES = 200


@dataclass(frozen=True)
class Pairs:
    residual: np.ndarray
    basis: np.ndarray
    horizon: np.ndarray


def _rotation(theta: np.ndarray) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)


def _in_frame(rotation: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    return np.einsum("mji,mj->mi", rotation, vectors)


def build_pairs(stream: Stream, model: str) -> Pairs:
    """Pairs of ``stream`` with the covariance basis of calibration model ``model``."""
    odometry, truth = stream.odometry, stream.ground_truth
    steps = np.hypot(*np.diff(odometry[:, :2], axis=0).T)
    travelled = np.concatenate([[0.0], np.cumsum(steps)])
    anchors = np.unique(np.searchsorted(travelled, np.arange(0.0, travelled[-1], ANCHOR_SPACING_M)))
    earlier, later, horizon = [], [], []
    for distance in HORIZONS_M:
        targets = np.searchsorted(travelled, travelled[anchors] + distance)
        keep = targets < len(odometry)
        earlier.append(anchors[keep])
        later.append(targets[keep])
        horizon.append(np.full(keep.sum(), distance))
    earlier, later, horizon = (np.concatenate(x) for x in (earlier, later, horizon))

    odometry_frame = _rotation(odometry[earlier, 2])
    residual = _in_frame(odometry_frame, odometry[later, :2] - odometry[earlier, :2]) - _in_frame(
        _rotation(truth[earlier, 2]), truth[later, :2] - truth[earlier, :2]
    )
    basis = []
    for k in range(4):
        unit = np.zeros(4)
        unit[k] = 1.0
        cumulative = cumulative_covariances(TRACKER[model], OdometryUncertaintyParams(*unit), odometry)
        world = relative_pose_covariance(TRACKER[model], odometry[later], cumulative[later], odometry[earlier],
                                         cumulative[earlier])[:, :2, :2]
        basis.append(odometry_frame.swapaxes(-1, -2) @ world @ odometry_frame)
    return Pairs(residual, np.stack(basis), horizon)


def _mahalanobis_and_logdet(residual: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, b, d = covariance[:, 0, 0], covariance[:, 0, 1], covariance[:, 1, 1]
    det = a * d - b * b
    x, y = residual[:, 0], residual[:, 1]
    return (d * x * x - 2 * b * x * y + a * y * y) / det, np.log(det)


def predicted_covariance(pairs: Pairs, alpha: np.ndarray, floor_variance: float) -> np.ndarray:
    return np.einsum("k,kmij->mij", alpha, pairs.basis) + floor_variance * np.eye(2)


def mean_nll(pairs: Pairs, alpha: np.ndarray, floor_variance: float) -> float:
    d2, logdet = _mahalanobis_and_logdet(pairs.residual, predicted_covariance(pairs, alpha, floor_variance))
    return float(np.mean(0.5 * (d2 + logdet) + np.log(2 * np.pi)))


@dataclass(frozen=True)
class Fit:
    alpha: np.ndarray
    floor_variance: float
    objective: float


def fit(traversals: list[Pairs], model: str, fixed_alpha: np.ndarray | None = None,
        starts: list[np.ndarray] | None = None) -> Fit:
    """Minimize the traversal-averaged mean NLL; with ``fixed_alpha`` only the floor is fitted."""
    free = () if fixed_alpha is not None else FREE_COEFFICIENTS[model]
    base = DEFAULT_PARAMS.as_array() if fixed_alpha is None else np.asarray(fixed_alpha, dtype=np.float64)

    def unpack(x: np.ndarray) -> tuple[np.ndarray, float]:
        alpha = base.copy()
        alpha[list(free)] = np.exp(x[:-1])
        return alpha, float(np.exp(x[-1]))

    def objective(x: np.ndarray) -> float:
        alpha, floor = unpack(x)
        return float(np.mean([mean_nll(p, alpha, floor) for p in traversals]))

    if starts is None:
        default = np.log(np.append(base[list(free)], 0.05**2))
        starts = [default + np.append(np.full(len(free), shift), 0.0) for shift in (-2.3, 0.0, 2.3)]
    best = min(
        (minimize(objective, x0, method="L-BFGS-B", bounds=[LOG_BOUNDS] * (len(free) + 1)) for x0 in starts),
        key=lambda result: result.fun,
    )
    alpha, floor = unpack(best.x)
    return Fit(alpha, floor, float(best.fun))


def _log_parameters(result: Fit, model: str) -> np.ndarray:
    return np.log(np.append(result.alpha[list(FREE_COEFFICIENTS[model])], result.floor_variance))


def gate_coverage(pairs: list[Pairs], alpha: np.ndarray, floor_variance: float) -> dict[str, float]:
    """Fraction of pairs inside the 95 % chi-square gate, per horizon."""
    inside: dict[float, list[bool]] = {h: [] for h in HORIZONS_M}
    for p in pairs:
        d2, _ = _mahalanobis_and_logdet(p.residual, predicted_covariance(p, alpha, floor_variance))
        for h in HORIZONS_M:
            inside[h].extend((d2[p.horizon == h] <= _CHI2_GATE_2DOF).tolist())
    return {str(h): float(np.mean(v)) for h, v in inside.items() if v}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    names = [s.name for s in DESIGN]
    raw = {s.name: load_stream(s) for s in DESIGN}
    corrected = {s.name: debiased_stream(s, raw[s.name]) for s in DESIGN}
    pairs = {model: {n: build_pairs((corrected if model == "propagated_debiased" else raw)[n], model) for n in names}
             for model in FREE_COEFFICIENTS}

    fits = {model: fit(list(pairs[model].values()), model) for model in FREE_COEFFICIENTS}
    default_floor = fit(list(pairs["additive"].values()), "additive", fixed_alpha=DEFAULT_PARAMS.as_array())

    rng = np.random.default_rng(0)
    intervals = {}
    for model, point in fits.items():
        draws = []
        for _ in range(BOOTSTRAP_RESAMPLES):
            sample = rng.integers(0, len(names), len(names))
            refit = fit([pairs[model][names[i]] for i in sample], model, starts=[_log_parameters(point, model)])
            draws.append(np.append(refit.alpha, np.sqrt(refit.floor_variance)))
        draws = np.asarray(draws)
        intervals[model] = {"lower": np.percentile(draws, 2.5, axis=0).tolist(),
                            "upper": np.percentile(draws, 97.5, axis=0).tolist()}

    leave_one_out: dict[str, dict[str, float]] = {}
    for held in names:
        rest = [n for n in names if n != held]
        iso_default = fit([pairs["additive"][n] for n in rest], "additive", fixed_alpha=DEFAULT_PARAMS.as_array())
        iso = fit([pairs["additive"][n] for n in rest], "additive")
        prop = fit([pairs["propagated"][n] for n in rest], "propagated")
        debiased = fit([pairs["propagated_debiased"][n] for n in rest], "propagated_debiased")
        leave_one_out[held] = {
            "isotropic_default": mean_nll(pairs["additive"][held], iso_default.alpha, iso_default.floor_variance),
            "isotropic_calibrated": mean_nll(pairs["additive"][held], iso.alpha, iso.floor_variance),
            "propagated_calibrated": mean_nll(pairs["propagated"][held], prop.alpha, prop.floor_variance),
            "propagated_debiased": mean_nll(pairs["propagated_debiased"][held], debiased.alpha,
                                            debiased.floor_variance),
        }

    gate_variance = PLACE_SIGMA_M**2 + 1e-6
    coverage = {
        "isotropic_default": gate_coverage(list(pairs["additive"].values()), DEFAULT_PARAMS.as_array(), gate_variance),
        "isotropic_calibrated": gate_coverage(list(pairs["additive"].values()), fits["additive"].alpha, gate_variance),
        "propagated_calibrated": gate_coverage(list(pairs["propagated"].values()), fits["propagated"].alpha, gate_variance),
        "propagated_debiased": gate_coverage(list(pairs["propagated_debiased"].values()),
                                             fits["propagated_debiased"].alpha, gate_variance),
    }
    report = {
        "horizons_m": HORIZONS_M,
        "pairs_per_traversal": {n: int(len(pairs["additive"][n].horizon)) for n in names},
        "default_alpha": DEFAULT_PARAMS.as_array().tolist(),
        "default_alpha_floor_m": float(np.sqrt(default_floor.floor_variance)),
        "models": {
            model: {"alpha": f.alpha.tolist(), "free_coefficients": [i + 1 for i in FREE_COEFFICIENTS[model]],
                    "floor_m": float(np.sqrt(f.floor_variance)), "mean_nll": f.objective,
                    "ci95_alpha_then_floor": intervals[model]}
            for model, f in fits.items()
        },
        "leave_one_traversal_out_nll": leave_one_out,
        "gate_coverage_with_place_sigma": coverage,
    }
    RESULTS.mkdir(exist_ok=True)
    CALIBRATION_FILE.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("models", "default_alpha_floor_m", "gate_coverage_with_place_sigma")}, indent=1))


if __name__ == "__main__":
    main()
