"""Collect every manuscript number into results/summary.{md,json}."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from stream import CALIBRATION_FILE, DEV, HELDOUT, RESULTS
from mapper_runs import NODE_SENSITIVITY_FILE, TOLERANCES
from summarize import bootstrap, bootstrap_abs, counts, prf

SPLIT_VARIANTS = [
    "topo", "topo_visual", "topo_visual_cosine",
    "disk_isotropic_default", "disk_isotropic_calibrated",
    "gate_propagated", "gate_propagated_geometry_only", "paper", "paper_geometry_only",
    "threshold", "visual_only",
    "anyloc_vlad_g", "anyloc_vlad_g_visual", "anyloc_gem_g", "anyloc_gem_g_visual",
    "dinov3_s", "dinov3_s_visual",
]
SENSITIVITY = ["valley_k1.0", "valley_k1.5", "topo", "valley_k2.5", "valley_k3.0", "valley_fixed"]
PAIRED_TESTS = [
    ("topo", "paper", 2, "F1"), ("topo", "paper", 0, "P"), ("topo", "paper", 1, "R"),
    ("topo", "disk_isotropic_default", 2, "F1"), ("topo", "disk_isotropic_calibrated", 2, "F1"),
    ("topo", "gate_propagated", 2, "F1"),
    ("topo_visual", "topo", 2, "F1"), ("topo_visual_cosine", "topo", 2, "F1"),
    ("topo", "anyloc_vlad_g", 2, "F1"), ("topo_visual_cosine", "anyloc_vlad_g_visual", 2, "F1"),
    ("topo", "anyloc_gem_g", 2, "F1"), ("topo_visual_cosine", "anyloc_gem_g_visual", 2, "F1"),
    ("topo", "dinov3_s", 2, "F1"), ("topo_visual", "dinov3_s_visual", 2, "F1"),
]


def average_precision(scored: list[tuple[float, bool]], opportunities: int) -> float:
    """Area under the precision-recall curve of the decisions ranked by probability."""
    if not opportunities:
        return 0.0
    ranked = sorted(scored, key=lambda item: -item[0])
    hits = np.cumsum([correct for _, correct in ranked])
    precision = hits / np.arange(1, len(ranked) + 1)
    return float(sum(precision[i] for i, (_, correct) in enumerate(ranked) if correct) / opportunities)


def ap_with_interval(results: dict, seqs, variant: str, tolerance: str = "2.0", resamples: int = 10000):
    names = [s.name for s in seqs if (s.name, variant) in results and results[(s.name, variant)]["scored"][tolerance]]
    if len(names) < len(seqs):
        return None
    scored = {n: results[(n, variant)]["scored"][tolerance] for n in names}
    opportunities = {n: next(m["loop_opportunities"] for m in results[(n, variant)]["metrics"]
                             if str(m["tolerance_m"]) == tolerance) for n in names}

    def pooled(sample: list[str]) -> float:
        return average_precision([d for n in sample for d in scored[n]], sum(opportunities[n] for n in sample))

    rng = np.random.default_rng(0)
    draws = [pooled([names[i] for i in rng.integers(0, len(names), len(names))]) for _ in range(resamples)]
    return {"point": pooled(names), "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}


def load(root: Path, seqs) -> dict:
    results = {}
    for seq in seqs:
        for path in sorted((root / "replay" / seq.name).glob("*/result.json")):
            r = json.loads(path.read_text())
            results[(seq.name, r["variant"])] = r
    return results


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
            *("| " + " | ".join(r) + " |" for r in rows), ""]


def loop_rows(results: dict, seqs, variants: list[str], tolerance: float = 2.0) -> tuple[list[list[str]], dict]:
    rows, data = [], {}
    for v in variants:
        per_seq = [counts(results[(s.name, v)], tolerance) for s in seqs if (s.name, v) in results]
        if len(per_seq) < len(seqs):
            rows.append([v, f"incomplete {len(per_seq)}/{len(seqs)}"] + [""] * 7)
            continue
        total = sum(per_seq)
        p, r, f = prf(total)
        rows.append([v, str(total[4]), str(total[3]), *(str(x) for x in total[:3]), f"{p:.3f}", f"{r:.3f}", f"{f:.3f}"])
        data[v] = {"tp": int(total[0]), "fp": int(total[1]), "fn": int(total[2]), "opportunities": int(total[3]),
                   "nodes": int(total[4]), "precision": p, "recall": r, "f1": f}
    return rows, data


def split_section(results: dict, seqs, title: str, variants: list[str]) -> tuple[list[str], dict]:
    lines, data = [f"## {title}", ""], {}
    header = ["Variant", "Nodes", "Opp.", "TP", "FP", "FN", "P", "R", "F1"]
    for tolerance in TOLERANCES:
        rows, data[str(tolerance)] = loop_rows(results, seqs, variants, tolerance)
        lines += [f"### {tolerance} m", "", *table(header, rows)]
    per_seq = {s.name: {v: counts(results[(s.name, v)], 2.0) for v in variants if (s.name, v) in results} for s in seqs}
    lines += ["### 2 m with 95 % CI (bootstrap over traversals)", ""]
    data["ci_2m"] = {}
    for v in variants:
        res = bootstrap_abs(per_seq, v)
        if res:
            data["ci_2m"][v] = res
            lines.append(f"- {v}: " + ", ".join(
                f"{k[0].upper()} {res[k]['point']:.3f} [{res[k]['ci95'][0]:.3f}; {res[k]['ci95'][1]:.3f}]"
                for k in ("precision", "recall", "f1")))
    lines += ["", "### Average precision at 2 m (decisions ranked by probability)", ""]
    data["ap_2m"] = {}
    for v in variants:
        res = ap_with_interval(results, seqs, v)
        if res:
            data["ap_2m"][v] = res
            lines.append(f"- {v}: AP {res['point']:.3f} [{res['ci95'][0]:.3f}; {res['ci95'][1]:.3f}]")
    lines += ["", "### Paired differences at 2 m (bootstrap over traversals)", ""]
    data["paired_2m"] = {}
    for a, b, stat, label in PAIRED_TESTS:
        res = bootstrap(per_seq, a, b, stat)
        if res:
            data["paired_2m"][f"{label}:{a}-{b}"] = res
            lines.append(f"- {label} {a} − {b}: {res['point']:+.3f} [{res['ci95'][0]:+.3f}; {res['ci95'][1]:+.3f}]")
    lines += ["", "### Aligned / opposed revisits at 2 m", ""]
    rows = []
    for v in variants:
        agg: dict[str, int] = {}
        for s in seqs:
            if (s.name, v) in results:
                for k, x in results[(s.name, v)]["breakdown"]["2.0"].items():
                    agg[k] = agg.get(k, 0) + x
        rows.append([v, *(str(agg.get(k, 0)) for k in ("opp_aligned", "tp_aligned", "opp_opposed", "tp_opposed", "fp"))])
    lines += table(["Variant", "Opp. aligned", "TP aligned", "Opp. opposed", "TP opposed", "FP"], rows)
    lines += ["### Per traversal (topo): nodes, loops, F1 per tolerance, coverage per tolerance, RMSE", ""]
    rows = []
    for s in seqs:
        r = results.get((s.name, "topo"))
        if r is None:
            continue
        m = {x["tolerance_m"]: x for x in r["metrics"]}
        rows.append([s.name, str(r["nodes"]), str(m[2.0]["n_loop_edges"]),
                     *(f"{m[t]['loop_f1']:.3f}" for t in TOLERANCES),
                     *(f"{m[t]['coverage_at_tolerance']:.3f}" for t in TOLERANCES), f"{r['node_placement_rmse_m']:.3f}"])
    lines += table(["Traversal", "Nodes", "Loops", *(f"F1@{t}" for t in TOLERANCES),
                    *(f"C@{t}" for t in TOLERANCES), "RMSE"], rows)
    return lines, data


def calibration_section() -> tuple[list[str], dict]:
    report = json.loads(CALIBRATION_FILE.read_text())
    lines = ["## Odometry calibration (development ground truth only)", "",
             f"Pairs per traversal: {report['pairs_per_traversal']}", ""]
    rows = [["isotropic, original", *(f"{a:.4g}" for a in report["default_alpha"]), f"{report['default_alpha_floor_m']:.3f}", "–"]]
    for model, label in (("additive", "isotropic, calibrated"), ("propagated", "propagated, calibrated")):
        m = report["models"][model]
        ci = m["ci95_alpha_then_floor"]
        cells = [f"{v:.4g} [{lo:.3g}; {hi:.3g}]" if i + 1 in m["free_coefficients"] else f"{v:.4g} (unused)"
                 for i, (v, lo, hi) in enumerate(zip(m["alpha"], ci["lower"], ci["upper"]))]
        rows.append([label, *cells, f"{m['floor_m']:.3f} [{ci['lower'][4]:.3f}; {ci['upper'][4]:.3f}]", f"{m['mean_nll']:.3f}"])
    lines += table(["Model", "α1", "α2", "α3", "α4", "floor [m]", "mean NLL"], rows)
    loto = report["leave_one_traversal_out_nll"]
    names = list(next(iter(loto.values())))
    lines += ["Leave-one-traversal-out mean NLL (lower is better):", ""]
    lines += table(["Traversal", *names], [[t, *(f"{v[n]:.3f}" for n in names)] for t, v in loto.items()]
                   + [["mean", *(f"{np.mean([v[n] for v in loto.values()]):.3f}" for n in names)]])
    cov = report["gate_coverage_with_place_sigma"]
    horizons = list(next(iter(cov.values())))
    lines += ["Fraction of true drifts inside the 95 % gate (with σ_place), by travelled distance [m]:", ""]
    lines += table(["Gate", *horizons], [[g, *(f"{c[h]:.2f}" for h in horizons)] for g, c in cov.items()])
    return lines, report


def sensitivity_section(results: dict) -> tuple[list[str], dict]:
    report = json.loads(NODE_SENSITIVITY_FILE.read_text())
    lines = ["## Node creation sensitivity (development)", "", f"Fixed δ calibrated on {report['calibration_case']}: "
             f"{report['fixed_delta']:.4f}", ""]
    rows_by = {}
    for row in report["rows"]:
        key = (row["method"], row["parameter_value"])
        rows_by.setdefault(key, []).append(row)
    out = []
    for (method, value), rows in rows_by.items():
        rates = np.array([r["nodes_per_1000_frames"] for r in rows])
        labeled = [r for r in rows if r["transition_recall"] is not None]
        out.append([f"{method} {value:.4g}", str(sum(r["nodes"] for r in rows)), f"{rates.mean():.3f}",
                    f"{rates.std() / rates.mean():.3f}",
                    f"{np.mean([r['transition_recall'] for r in labeled]):.3f}",
                    f"{np.mean([r['frame_weighted_node_purity'] for r in labeled]):.3f}",
                    f"{np.mean([r['mean_segments_per_semantic_episode'] for r in labeled]):.3f}"])
    lines += table(["Detector", "Nodes", "ρ", "CV", "R_trans", "Purity", "Seg./episode"], out)
    lines += ["Loop closure F1 of the mapper run with each detector:", ""]
    rows = []
    for v in SENSITIVITY:
        cells = []
        for t in TOLERANCES:
            per = [counts(results[(s.name, v)], t) for s in DEV if (s.name, v) in results]
            cells.append(f"{prf(sum(per))[2]:.3f}" if len(per) == len(DEV) else "–")
        rows.append([v, *cells])
    lines += table(["Detector", *(f"F1@{t}" for t in TOLERANCES)], rows)
    return lines, report


def language_section() -> tuple[list[str], dict]:
    path = RESULTS / "language.json"
    if not path.exists():
        return ["## Language retrieval", "", "missing", ""], {}
    report = json.loads(path.read_text())
    lines = ["## Language retrieval (" + report["semantic_model"] + ")", "",
             "Fields per environment and overall:", "", "```",
             json.dumps({"overall": {k: v for k, v in report["overall"].items() if k != "present_by_category"},
                         "environments": {e: {k: v for k, v in r.items() if k != "present_by_category"}
                                          for e, r in report["environments"].items()},
                         "present_by_category": report["overall"].get("present_by_category")}, indent=1),
             "```", ""]
    return lines, {k: report[k] for k in ("semantic_model", "overall", "environments")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    dev, held = load(args.out, DEV), load(args.out, HELDOUT)
    lines, data = ["# Manuscript results on the GPU", ""], {}
    for name, (section, payload) in {
        "calibration": calibration_section(),
        "node_sensitivity": sensitivity_section(dev),
        "development": split_section(dev, DEV, "Development traversals (7)", SPLIT_VARIANTS),
        "unseen": split_section(held, HELDOUT, "Unseen traversals (12)", SPLIT_VARIANTS),
        "language": language_section(),
    }.items():
        lines += section
        data[name] = payload
    (RESULTS / "summary.md").write_text("\n".join(lines))
    (RESULTS / "summary.json").write_text(json.dumps(data, indent=1, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
