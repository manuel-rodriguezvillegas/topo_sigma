"""Summaries, uncertainty intervals and figures for the regime study (C1).

Drift: micro-aggregate TP/FP/FN over traversals for each seed, then report the
mean and 2.5/97.5 percentiles over seeds. "Default" uses the configured knob
(visual k = 2, gate scale = 1). "Best knob" picks the knob maximizing mean F1
at each drift level; it is an oracle upper bound, not a deployable setting.

Sessions: per scorer, pool the top-1 decisions of all session pairs and sweep
one global threshold. Intervals come from a bootstrap that resamples queries
within each pair; differences against the reference scorer are paired.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from vts_evaluation.regime_study import DEFAULT_KNOB, METHODS, WORKSPACE, pr_curve, prf, sweep_for

REFERENCE = "multiview_cosine|single|mad_z"


def _interval(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "lo": float(np.percentile(array, 2.5)),
            "hi": float(np.percentile(array, 97.5))}


def summarize_drift(results: dict, tolerance: float) -> dict:
    grouped: dict = defaultdict(lambda: defaultdict(lambda: np.zeros(3, dtype=int)))
    for row in results["rows"]:
        if row["tolerance_m"] != tolerance:
            continue
        key = (row["method"], row["value"], row["level"])
        grouped[key][row["seed"]] += (row["tp"], row["fp"], row["fn"])
    summary: dict = defaultdict(dict)
    for (method, value, level), seeds in grouped.items():
        stats = [prf(*map(int, counts)) for counts in seeds.values()]
        summary[method].setdefault(level, {})[value] = {
            metric: _interval([s[metric] for s in stats]) for metric in ("precision", "recall", "f1", "tp", "fp")
        }
    out = {}
    for method, levels in summary.items():
        knob, _ = sweep_for(method)
        default = DEFAULT_KNOB[knob]
        out[method] = {}
        for level, values in sorted(levels.items()):
            best_value = max(values, key=lambda v: values[v]["f1"]["mean"])
            high_precision = [v for v in values if values[v]["precision"]["mean"] >= 0.9]
            hp_value = max(high_precision, key=lambda v: values[v]["recall"]["mean"]) if high_precision else None
            out[method][level] = {
                "default": values[default], "best_knob": best_value, "best": values[best_value],
                "p90_knob": hp_value, "p90": values[hp_value] if hp_value is not None else None,
                "curve": {str(v): {"precision": values[v]["precision"]["mean"],
                                   "recall": values[v]["recall"]["mean"]} for v in sorted(values)},
            }
    drift = defaultdict(list)
    for row in results["drift"]:
        drift[row["level"]].append(row["added_drift_percent"])
    return {"methods": out, "added_drift_percent": {k: _interval(v) for k, v in sorted(drift.items())}}


def summarize_sessions(results: dict, tolerance: float, boots: int, seed: int = 0) -> dict:
    keys = sorted({key.rsplit("|", 1)[0] for pair in results.values() for key in pair["scorers"]})
    rng = np.random.default_rng(seed)
    pairs = sorted(results)
    sizes = {p: len(next(iter(results[p]["scorers"].values()))["per_query"]["correct"]) for p in pairs}
    samples = [{p: rng.integers(0, sizes[p], sizes[p]) for p in pairs} for _ in range(boots)]

    def pooled(key: str, index: dict | None = None) -> dict:
        arrays = defaultdict(list)
        for p in pairs:
            per_query = results[p]["scorers"][f"{key}|{tolerance}"]["per_query"]
            take = slice(None) if index is None else index[p]
            for name in ("correct", "positive", "statistic"):
                arrays[name].append(np.asarray(per_query[name], dtype=float)[take])
        correct, positive, statistic = (np.concatenate(arrays[n]) for n in ("correct", "positive", "statistic"))
        return pr_curve(correct.astype(bool), positive.astype(bool), statistic)

    point = {key: pooled(key) for key in keys}
    boot = {key: [pooled(key, s) for s in samples] for key in keys}
    out = {}
    for key in keys:
        metrics = {}
        for metric in ("average_precision", "recall_at_1", "recall_at_p90", "max_f1"):
            values = [b[metric] for b in boot[key]]
            deltas = [b[metric] - r[metric] for b, r in zip(boot[key], boot[REFERENCE])]
            metrics[metric] = {"value": point[key][metric],
                               "lo": float(np.percentile(values, 2.5)), "hi": float(np.percentile(values, 97.5)),
                               "delta_vs_reference": point[key][metric] - point[REFERENCE][metric],
                               "delta_lo": float(np.percentile(deltas, 2.5)),
                               "delta_hi": float(np.percentile(deltas, 97.5))}
        out[key] = {"queries": point[key]["queries"], "positives": point[key]["positives"], **metrics}
    per_pair = {p: {key: {m: results[p]["scorers"][f"{key}|{tolerance}"][m]
                          for m in ("recall_at_1", "average_precision", "recall_at_p90")}
                    for key in keys} for p in pairs}
    queries = {p: results[p]["query_nodes"] for p in pairs}
    chance = sum(results[p]["chance_top1"][str(tolerance)] * queries[p] for p in pairs) / sum(queries.values())
    return {"reference": REFERENCE, "chance_top1": chance, "pooled": out, "per_pair": per_pair,
            "relations": {p: results[p]["relation"] for p in pairs}}


def plot(drift: dict | None, sessions_raw: dict | None, sessions: dict | None, figures: Path,
         tolerance: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures.mkdir(parents=True, exist_ok=True)
    colors = {"geometry+cosine": "#1f77b4", "geometry+vmf": "#9467bd", "geometry_only": "#2ca02c",
              "visual_only+cosine": "#d62728", "visual_only+vmf": "#ff7f0e"}
    if drift:
        levels = sorted(drift["added_drift_percent"])
        x = [drift["added_drift_percent"][lv]["mean"] for lv in levels]
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        for method in METHODS:
            for ax, metric in zip(axes, ("f1", "precision", "recall")):
                series = [drift["methods"][method][lv]["default"][metric] for lv in levels]
                ax.plot(x, [s["mean"] for s in series], "o-", color=colors[method], label=method)
                ax.fill_between(x, [s["lo"] for s in series], [s["hi"] for s in series],
                                color=colors[method], alpha=0.15)
                ax.set_xscale("symlog", linthresh=0.5)
                ax.set_xlabel("added odometry drift [% of path]")
                ax.set_title(f"{metric} @ {tolerance:g} m (default knob)")
                ax.grid(alpha=0.3)
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures / "drift_default.png", dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(1, len(levels), figsize=(3.2 * len(levels), 3.2), sharey=True)
        for ax, level in zip(np.atleast_1d(axes), levels):
            for method in METHODS:
                curve = drift["methods"][method][level]["curve"]
                ax.plot([c["recall"] for c in curve.values()], [c["precision"] for c in curve.values()],
                        "o-", ms=3, color=colors[method], label=method)
            ax.set_title(f"{level} (+{drift['added_drift_percent'][level]['mean']:.1f}% drift)")
            ax.set_xlabel("recall")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.3)
        np.atleast_1d(axes)[0].set_ylabel("precision")
        np.atleast_1d(axes)[0].legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(figures / "drift_operating_curves.png", dpi=160)
        plt.close(fig)
    if sessions_raw and sessions:
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        best = sorted(sessions["pooled"], key=lambda k: -sessions["pooled"][k]["average_precision"]["value"])[:6]
        for key in best:
            arrays = defaultdict(list)
            for pair in sessions_raw.values():
                per_query = pair["scorers"][f"{key}|{tolerance}"]["per_query"]
                for name in ("correct", "positive", "statistic"):
                    arrays[name].append(np.asarray(per_query[name], dtype=float))
            correct, positive, statistic = (np.concatenate(arrays[n]) for n in ("correct", "positive", "statistic"))
            order = np.argsort(-statistic, kind="stable")
            tp = np.cumsum(correct[order])
            ax.plot(tp / positive.sum(), tp / np.arange(1, len(order) + 1),
                    label=f"{key} (AP {sessions['pooled'][key]['average_precision']['value']:.2f})")
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.set_title(f"Cross-session association @ {tolerance:g} m")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(figures / "sessions_pr.png", dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=WORKSPACE / "output/regime_study_v1")
    parser.add_argument("--tolerance", type=float, default=2.0)
    parser.add_argument("--boots", type=int, default=2000)
    args = parser.parse_args()
    report: dict = {"tolerance_m": args.tolerance}
    drift_path, sessions_path = args.output / "drift_results.json", args.output / "sessions_results.json"
    drift = sessions = sessions_raw = None
    if drift_path.exists():
        drift_results = json.loads(drift_path.read_text())
        drift = summarize_drift(drift_results["results"], args.tolerance)
        report["drift"] = {"fingerprint": drift_results["fingerprint"], **drift}
    if sessions_path.exists():
        session_results = json.loads(sessions_path.read_text())
        sessions_raw = session_results["results"]
        sessions = summarize_sessions(sessions_raw, args.tolerance, args.boots)
        report["sessions"] = {"fingerprint": session_results["fingerprint"], **sessions}
    tag = f"{args.tolerance:g}m".replace(".", "p")
    (args.output / f"report_{tag}.json").write_text(json.dumps(report, indent=2) + "\n")
    plot(drift, sessions_raw, sessions, args.output / f"figures_{tag}", args.tolerance)
    print(f"Report: {args.output / f'report_{tag}.json'}")


if __name__ == "__main__":
    main()
