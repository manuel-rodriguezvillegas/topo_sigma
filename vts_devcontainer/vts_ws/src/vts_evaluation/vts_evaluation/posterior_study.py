"""C2 screen: posterior association against the existing gates, across regimes.

Regimes reuse the C1 frame store and segmentation. Drift levels follow
:data:`regime_study.DRIFT_LEVELS`. Cut regimes add one or two unannounced
odometry discontinuities (robot kidnapped or odometry restarted): at random
frames in the middle 60 % of the traversal, every later pose is moved by a
random rigid transform (rotation uniform, translation 3-10 m). The uncertainty
budget is unaware of the cut, as in the drift regimes. Baselines and posterior
variants see identical poses for each (traversal, regime, seed).

The primary configuration is ``posterior`` (mean-direction cosine, geometry-
calibrated appearance, eps = 0.05, prior 0.5) at tau = 0.5; all other variants
are ablations. History: v1 calibrated appearance from segment halves; a
one-traversal smoke test and a GT diagnostic showed it over-optimistic, so v2
switched the calibration source before the full run. v1 is kept as a variant.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np

from vts_core import directional, mapper as mapper_module, matching, metrics, motion
from vts_core.artifacts import atomic_json, file_digest, identity
from vts_core.motion import normalize_angle
from vts_evaluation import posterior_association, regime_study
from vts_evaluation.posterior_association import (
    CalibrationPrior, PosteriorConfig, associate, calibration_samples, decisions_to_graph, place_from_frames,
    pooled_prior,
)
from vts_evaluation.regime_study import (
    CID_CONFIRM_SESSIONS, CONFIRM_SESSIONS, DEFAULT_KNOB, DRIFT_LEVELS, HELDOUT_SESSIONS, MAIN_SESSIONS, METHODS, TOLERANCES, WORKSPACE,
    build_nodes, camera_odometry, corrupt_odometry, freeze_segments, load_session, overlay_appearance,
    prf, replay, score, segment, sweep_for,
)

TAU_SWEEP = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999)
PRIMARY = "posterior"
POSTERIOR_VARIANTS = {
    "posterior": PosteriorConfig(),
    "posterior+multiview": PosteriorConfig(appearance="multiview_cosine"),
    "posterior_eps0.01": PosteriorConfig(eps=0.01),
    "posterior_eps0.2": PosteriorConfig(eps=0.2),
    "posterior_geometry_only": PosteriorConfig(use_appearance=False),
    "posterior_appearance_only": PosteriorConfig(use_geometry=False),
    "posterior_v1_split_half": PosteriorConfig(calibration="split_half"),
    # v4, exploratory: added after the revisit-direction diagnostic on these data.
    "posterior_view": PosteriorConfig(view_mixture=True),
    "posterior_view+multiview": PosteriorConfig(view_mixture=True, appearance="multiview_cosine"),
    # v5, exploratory: label-free calibration prior from the other environments.
    "posterior_view_prior": PosteriorConfig(view_mixture=True, use_prior=True),
}


@dataclass(frozen=True)
class Regime:
    name: str
    drift: int  # index into DRIFT_LEVELS
    cuts: int


REGIMES = (*(Regime(level.name, k, 0) for k, level in enumerate(DRIFT_LEVELS)),
           Regime("cut1", 0, 1), Regime("cut2", 0, 2))


def apply_cuts(poses: np.ndarray, cuts: int, rng: np.random.Generator) -> tuple[np.ndarray, list[int]]:
    out = poses.copy()
    n = len(poses)
    if not cuts:
        return out, []
    frames = sorted(int(c) for c in rng.choice(np.arange(int(0.2 * n), int(0.8 * n)), cuts, replace=False))
    for c in frames:
        phi, direction = rng.uniform(-np.pi, np.pi, size=2)
        translation = rng.uniform(3.0, 10.0) * np.array([np.cos(direction), np.sin(direction)])
        rotation = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        pivot = out[c, :2].copy()
        out[c:, :2] = (out[c:, :2] - pivot) @ rotation.T + pivot + translation
        out[c:, 2] = [normalize_angle(a + phi) for a in out[c:, 2]]
    return out, frames


# Order fixes each session's random stream; new groups are only ever appended.
ALL_SESSIONS = [*MAIN_SESSIONS, *HELDOUT_SESSIONS, *CONFIRM_SESSIONS, *CID_CONFIRM_SESSIONS]


def load_places(name: str, frames_root: str, encoder: str):
    session = load_session(name, Path(frames_root))
    mapping = session.config["mapping"]
    # Segmentation, medoids (node poses) and view frames always come from DINOv3.
    frozen = freeze_segments(session.descriptors, segment(session.descriptors, mapping))
    if encoder == "dinov3":
        appearance = None
        source = session.descriptors
    else:
        from vts_evaluation.vpr_descriptors import load_appearance
        appearance = source = load_appearance(name, Path(frames_root), encoder)
        frozen = overlay_appearance(frozen, appearance)
    places = [place_from_frames(source, s.indices, s.view_indices) for s in frozen]
    halves = [(place_from_frames(source, s.indices[: len(s.indices) // 2]),
               place_from_frames(source, s.indices[len(s.indices) // 2:]))
              if len(s.indices) >= 2 * posterior_association._MIN_HALF_FRAMES else None for s in frozen]
    return session, mapping, frozen, appearance, places, halves


def calibrate(task: tuple[str, str, str]) -> dict:
    """Geometry-labelled calibration samples from the recorded (unperturbed) odometry."""
    name, frames_root, encoder = task
    session, _, frozen, appearance, places, _ = load_places(name, frames_root, encoder)
    poses, covariances = camera_odometry(session, session.odom_poses)
    nodes = build_nodes(session, frozen, poses, covariances, attach_gt=False, appearance=appearance)
    return calibration_samples(nodes, places, POSTERIOR_VARIANTS["posterior_view_prior"].appearance)


def session_priors(sessions: list[str], frames_root: str, encoder: str, pool=None) -> dict[str, CalibrationPrior]:
    """Prior for each session from the *other* main environments only."""
    mapper = pool.map if pool is not None else map
    samples = dict(zip(MAIN_SESSIONS, mapper(calibrate, [(s, frames_root, encoder) for s in MAIN_SESSIONS])))
    return {name: pooled_prior([samples[o] for o in MAIN_SESSIONS if o != name]) for name in sessions}


def run_session(task: tuple[str, str, int, str, CalibrationPrior, tuple[str, ...]]) -> dict:
    name, frames_root, seeds, encoder, prior, regimes = task
    session, mapping, frozen, appearance, places, halves = load_places(name, frames_root, encoder)
    common = {"window_size": int(mapping["window_size"]), "valley_k": float(mapping["valley_k"]),
              "verification_threshold": float(mapping["verification_threshold"]),
              "defer_nodes": int(mapping["defer_nodes"]),
              "fixed_concentration": float(mapping["fixed_concentration"]),
              "naive_threshold": float(mapping["naive_threshold"])}
    session_index = ALL_SESSIONS.index(name)
    rows, cut_log = [], []
    for regime_index, regime in enumerate(REGIMES):
        if regime.name not in regimes:
            continue
        level = DRIFT_LEVELS[regime.drift]
        deterministic = level.name == "L0" and not regime.cuts
        for seed in range(1 if deterministic else seeds):
            rng = np.random.default_rng([seed, regime_index, session_index, 2])
            raw = corrupt_odometry(session.odom_poses, level, rng)
            poses, covariances = camera_odometry(session, raw)
            poses, cut_frames = apply_cuts(poses, regime.cuts, rng)
            if regime.cuts:
                cut_log.append({"regime": regime.name, "seed": seed, "frames": cut_frames,
                                "nodes_after_cut": [int(sum(s.medoid >= c for s in frozen)) for c in cut_frames]})
            nodes = build_nodes(session, frozen, poses, covariances, attach_gt=False, appearance=appearance)

            def record(method: str, knob: str, value: float, graph) -> None:
                for tolerance in TOLERANCES:
                    rows.append({"session": name, "regime": regime.name, "seed": seed, "method": method,
                                 "knob": knob, "value": value, "tolerance_m": tolerance,
                                 **score(graph, session, frozen, tolerance)})

            for method, parameters in METHODS.items():
                knob, values = sweep_for(method)
                for value in values:
                    record(method, knob, value, replay(nodes, {**common, **DEFAULT_KNOB, **parameters, knob: value}))
            for method, config in POSTERIOR_VARIANTS.items():
                decisions = associate(nodes, places, halves, config, prior=prior)
                for tau in TAU_SWEEP:
                    record(method, "tau", tau, decisions_to_graph(nodes, decisions, tau))
        print(f"{name} {regime.name}: done", flush=True)
    return {"rows": rows, "cuts": cut_log}


DEFAULTS = {"visual_outlier_k": 2.0, "geometric_gate_scale": 1.0, "tau": 0.5}


def summarize(rows: list[dict], tolerance: float, sessions: list[str]) -> dict:
    counts: dict = defaultdict(lambda: np.zeros(3, dtype=int))
    for row in rows:
        if row["tolerance_m"] == tolerance and row["session"] in sessions:
            counts[(row["regime"], row["method"], row["value"], row["seed"])] += (row["tp"], row["fp"], row["fn"])
    by: dict = defaultdict(lambda: defaultdict(dict))
    for (regime, method, value, seed), c in counts.items():
        by[regime][method].setdefault(value, {})[seed] = prf(*map(int, c))
    out: dict = {}
    for regime, methods in by.items():
        out[regime] = {}
        for method, values in methods.items():
            knob = "tau" if method.startswith("posterior") else sweep_for(method)[0]
            mean = {v: {m: float(np.mean([s[m] for s in seeds.values()])) for m in ("precision", "recall", "f1")}
                    for v, seeds in values.items()}
            default = values[DEFAULTS[knob]]
            f1 = [s["f1"] for s in default.values()]
            best = max(mean, key=lambda v: mean[v]["f1"])
            p90 = [v for v in mean if mean[v]["precision"] >= 0.9]
            out[regime][method] = {
                "default_f1": float(np.mean(f1)), "default_f1_lo": float(np.percentile(f1, 2.5)),
                "default_f1_hi": float(np.percentile(f1, 97.5)),
                "default_precision": mean[DEFAULTS[knob]]["precision"],
                "default_recall": mean[DEFAULTS[knob]]["recall"],
                "best_knob": best, "best_f1": mean[best]["f1"],
                "recall_at_p90": max((mean[v]["recall"] for v in p90), default=0.0),
                "curve": {str(v): mean[v] for v in sorted(mean)},
                "seed_f1": {str(k): v["f1"] for k, v in default.items()},
            }
    return out


def held_out(rows: list[dict], tolerance: float, train: list[str], test: list[str]) -> dict:
    """Knob chosen on all ``train`` traversals (pooled over seeds), applied unchanged to ``test``."""
    counts: dict = defaultdict(lambda: np.zeros(3, dtype=int))
    for row in rows:
        if row["tolerance_m"] == tolerance:
            counts[(row["regime"], row["method"], row["value"], row["session"] in test, row["seed"])] += (
                row["tp"], row["fp"], row["fn"])
    grouped: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    for (regime, method, value, is_test, seed), c in counts.items():
        grouped[regime][method][value][is_test][seed] = c
    out: dict = {}
    for regime, methods in grouped.items():
        out[regime] = {}
        for method, values in methods.items():
            train_f1 = {v: prf(*map(int, sum(d[False].values(), np.zeros(3, dtype=int))))["f1"]
                        for v, d in values.items()}
            best = max(sorted(values), key=lambda v: train_f1[v])
            stats = {k: prf(*map(int, c)) for k, c in values[best][True].items()}
            f1 = [x["f1"] for x in stats.values()]
            out[regime][method] = {
                "chosen_knob": best, "f1": float(np.mean(f1)), "f1_lo": float(np.percentile(f1, 2.5)),
                "f1_hi": float(np.percentile(f1, 97.5)),
                "precision": float(np.mean([x["precision"] for x in stats.values()])),
                "recall": float(np.mean([x["recall"] for x in stats.values()])),
                "tp": float(np.mean([x["tp"] for x in stats.values()])),
                "fp": float(np.mean([x["fp"] for x in stats.values()])),
                "fn": float(np.mean([x["fn"] for x in stats.values()])),
                "seed_f1": {str(k): v["f1"] for k, v in stats.items()},
            }
    return out


def leave_one_out(rows: list[dict], tolerance: float, sessions: list[str]) -> dict:
    """Knob chosen on the other traversals (pooled over seeds), applied to the held-out one.

    Returns, per regime and method, F1 per seed of the micro-aggregate over
    held-out traversals, plus the knob chosen for each held-out traversal.
    """
    counts: dict = defaultdict(lambda: np.zeros(3, dtype=int))
    for row in rows:
        if row["tolerance_m"] == tolerance and row["session"] in sessions:
            key = (row["regime"], row["method"], row["value"], row["session"], row["seed"])
            counts[key] += (row["tp"], row["fp"], row["fn"])
    index: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for (regime, method, value, session, seed), c in counts.items():
        index[regime][method][value].setdefault(session, {})[seed] = c
    out: dict = {}
    for regime, methods in index.items():
        out[regime] = {}
        for method, values in methods.items():
            sessions = sorted(next(iter(values.values())))
            seeds = sorted(next(iter(next(iter(values.values())).values())))
            chosen, per_seed = {}, defaultdict(lambda: np.zeros(3, dtype=int))
            for held in sessions:
                def train_f1(value: float) -> float:
                    total = sum((values[value][s][k] for s in sessions if s != held for k in seeds),
                                np.zeros(3, dtype=int))
                    return prf(*map(int, total))["f1"]
                best = max(sorted(values), key=train_f1)
                chosen[held] = best
                for k in seeds:
                    per_seed[k] += values[best][held][k]
            stats = [prf(*map(int, c)) for c in per_seed.values()]
            out[regime][method] = {
                "chosen_knob": chosen,
                "f1": float(np.mean([x["f1"] for x in stats])),
                "f1_lo": float(np.percentile([x["f1"] for x in stats], 2.5)),
                "f1_hi": float(np.percentile([x["f1"] for x in stats], 97.5)),
                "precision": float(np.mean([x["precision"] for x in stats])),
                "recall": float(np.mean([x["recall"] for x in stats])),
                "seed_f1": {str(k): prf(*map(int, per_seed[k]))["f1"] for k in per_seed},
            }
    return out


def paired(summary: dict, a: str, b: str) -> dict:
    result = {}
    for regime, methods in summary.items():
        fa, fb = methods[a]["seed_f1"], methods[b]["seed_f1"]
        delta = np.asarray([fa[k] - fb[k] for k in fa])
        result[regime] = {"mean": float(delta.mean()), "lo": float(np.percentile(delta, 2.5)),
                          "hi": float(np.percentile(delta, 97.5)), "wins": float(np.mean(delta > 0))}
    return result


def plot(summary: dict, path: Path, tolerance: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = {"geometry+cosine": "#1f77b4", "geometry_only": "#2ca02c", "visual_only+cosine": "#d62728",
             "posterior": "#000000", "posterior_view": "#9467bd", "posterior_appearance_only": "#ff7f0e"}
    regimes = [r.name for r in REGIMES if r.name in summary]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.6), sharex=True, sharey=True)
    for ax in axes.ravel()[len(regimes):]:
        ax.set_visible(False)
    for ax, regime in zip(axes.ravel(), regimes):
        for method, color in shown.items():
            curve = summary[regime][method]["curve"]
            ax.plot([c["recall"] for c in curve.values()], [c["precision"] for c in curve.values()], "o-",
                    ms=2.5, lw=1.2, color=color, label=method)
            d = summary[regime][method]
            ax.plot(d["default_recall"], d["default_precision"], "*", ms=9, color=color)
        ax.set_title(regime)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("recall")
    for ax in axes[:, 0]:
        ax.set_ylabel("precision")
    axes[0, 0].legend(fontsize=6)
    fig.suptitle(f"Operating curves @ {tolerance:g} m (stars: default / pre-registered operating point)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=Path, default=WORKSPACE / "output/regime_study_v1")
    parser.add_argument("--output", type=Path, default=WORKSPACE / "output/posterior_study_v5")
    parser.add_argument("--sessions", nargs="+", default=ALL_SESSIONS)
    parser.add_argument("--appearance", choices=("dinov3", "megaloc"), default="dinov3",
                        help="descriptor used for association; segmentation always uses DINOv3")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--regimes", nargs="+", default=[r.name for r in REGIMES])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results_path = args.output / "results.json"
    if not args.summarize_only:
        modules = (directional, mapper_module, matching, metrics, motion, posterior_association, regime_study)
        protocol = {
            "study": "posterior_study_v5", "sessions": args.sessions, "appearance": args.appearance, "seeds": args.seeds,
            "regimes": [asdict(r) for r in REGIMES if r.name in args.regimes], "tau_sweep": TAU_SWEEP, "primary": PRIMARY,
            "posterior_variants": {k: asdict(v) for k, v in POSTERIOR_VARIANTS.items()},
            "baselines": METHODS, "frames_root": str(args.frames),
            "sources": {Path(m.__file__).name: file_digest(m.__file__) for m in modules},
            "runner_sha256": file_digest(__file__),
            "packages": {name: version(name) for name in ("numpy", "scipy")},
        }
        with ProcessPoolExecutor(args.workers) as pool:
            priors = session_priors(args.sessions, str(args.frames), args.appearance, pool)
            parts = list(pool.map(run_session, [(s, str(args.frames), args.seeds, args.appearance, priors[s],
                                                 tuple(args.regimes)) for s in args.sessions]))
        rows = [row for part in parts for row in part["rows"]]
        cuts = {s: part["cuts"] for s, part in zip(args.sessions, parts)}
        atomic_json(results_path, {"fingerprint": identity(protocol), "protocol": protocol,
                                   "rows": rows, "cuts": cuts,
                                   "calibration_priors": {k: asdict(v) for k, v in priors.items()}})
    results = json.loads(results_path.read_text())
    report = {"fingerprint": results["fingerprint"]}
    sessions = results["protocol"]["sessions"]
    main_sessions = [s for s in sessions if s in MAIN_SESSIONS]
    test_sessions = [s for s in sessions if s in HELDOUT_SESSIONS or s in CONFIRM_SESSIONS
                     or s in CID_CONFIRM_SESSIONS]
    for tolerance in TOLERANCES:
        summary = summarize(results["rows"], tolerance, main_sessions)
        loo = leave_one_out(results["rows"], tolerance, main_sessions)
        report[str(tolerance)] = {
            "summary": summary,
            "paired_vs": {b: paired(summary, PRIMARY, b)
                          for b in ("geometry+cosine", "geometry_only", "posterior_geometry_only")},
            "prior_paired_vs": {b: paired(summary, "posterior_view_prior", b)
                                for b in ("geometry+cosine", "geometry_only", "posterior_geometry_only",
                                          "posterior_view")},
            "view_paired_vs": {b: paired(summary, "posterior_view", b)
                               for b in ("geometry+cosine", "geometry_only", "posterior_geometry_only", "posterior")},
            "leave_one_out": loo,
            "leave_one_out_paired_vs": {b: paired(loo, PRIMARY, b)
                                        for b in ("geometry+cosine", "geometry_only", "posterior_geometry_only")},
            "leave_one_out_prior_paired_vs": {b: paired(loo, "posterior_view_prior", b)
                                              for b in ("geometry+cosine", "geometry_only",
                                                        "posterior_geometry_only", "posterior_view")},
            "leave_one_out_view_paired_vs": {b: paired(loo, "posterior_view", b)
                                             for b in ("geometry+cosine", "geometry_only",
                                                       "posterior_geometry_only", "posterior")},
        }
        if test_sessions:
            test = held_out(results["rows"], tolerance, main_sessions, test_sessions)
            report[str(tolerance)]["held_out"] = test
            report[str(tolerance)]["held_out_paired_vs"] = {
                b: paired(test, PRIMARY, b) for b in ("geometry+cosine", "geometry_only", "posterior_geometry_only")}
            report[str(tolerance)]["held_out_prior_paired_vs"] = {
                b: paired(test, "posterior_view_prior", b)
                for b in ("geometry+cosine", "geometry_only", "posterior_geometry_only", "posterior_view")}
        plot(summary, args.output / f"curves_{str(tolerance).replace('.', 'p')}m.png", tolerance)
    atomic_json(args.output / "report.json", report)
    print(f"Report: {args.output / 'report.json'}")


if __name__ == "__main__":
    main()
