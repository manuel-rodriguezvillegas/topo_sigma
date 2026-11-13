"""Collect replay results into results/summary.{json,md} (small files, committed)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import DEV, HELDOUT, HERE, TOLERANCES

ORDER = ["topo", "topo_gpu", "topo_cosine", "geometry_only", "anyloc_gem_s224",
         "anyloc_gem_g224", "anyloc_gem_g", "anyloc_vlad_g",
         "paired_anyloc_gem_g224", "paired_anyloc_gem_g", "paired_anyloc_vlad_g"]


def load(root: Path, seqs) -> dict:
    out = {}
    for s in seqs:
        for path in sorted((root / "replay" / s.name).glob("*/result.json")):
            r = json.loads(path.read_text())
            out[(s.name, r["variant"])] = r
    return out


def counts(r: dict, tol: float) -> np.ndarray:
    m = next(x for x in r["metrics"] if x["tolerance_m"] == tol)
    return np.array([m["loop_tp"], m["loop_fp"], m["loop_fn"], m["loop_opportunities"], m["n_nodes"]])


def prf(c: np.ndarray) -> tuple[float, float, float]:
    tp, fp, fn = c[:3]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * tp / (2 * tp + fp + fn) if tp else 0.0)


def bootstrap(per_seq: dict[str, dict[str, np.ndarray]], a: str, b: str, stat: int, n=10000, seed=0):
    names = [s for s in per_seq if a in per_seq[s] and b in per_seq[s]]
    if not names:
        return None
    rng = np.random.default_rng(seed)
    A = np.array([per_seq[s][a] for s in names])
    B = np.array([per_seq[s][b] for s in names])
    point = prf(A.sum(0))[stat] - prf(B.sum(0))[stat]
    idx = rng.integers(0, len(names), size=(n, len(names)))
    diffs = np.array([prf(A[i].sum(0))[stat] - prf(B[i].sum(0))[stat] for i in idx])
    return {"point": point, "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "sequences": len(names)}


def section(results: dict, seqs, title: str) -> tuple[list[str], dict]:
    lines, data = [f"## {title}", ""], {}
    variants = [v for v in ORDER if any((s.name, v) in results for s in seqs)]
    for tol in TOLERANCES:
        lines += [f"### {tol} m", "", "| Variant | Nodes | Opp. | TP | FP | FN | P | R | F1 | F1 COLD | F1 CID |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for v in variants:
            rows = {s.name: counts(results[(s.name, v)], tol) for s in seqs if (s.name, v) in results}
            if len(rows) < len(seqs):
                lines.append(f"| {v} | incomplete ({len(rows)}/{len(seqs)}) |||||||||")
                continue
            total = sum(rows.values())
            by = {d: sum(rows[s.name] for s in seqs if s.domain == d) for d in ("COLD", "CID-SIMS")}
            p, r, f = prf(total)
            fc = prf(by["COLD"])[2] if isinstance(by["COLD"], np.ndarray) else float("nan")
            fd = prf(by["CID-SIMS"])[2] if isinstance(by["CID-SIMS"], np.ndarray) else float("nan")
            lines.append(f"| {v} | {total[4]} | {total[3]} | {total[0]} | {total[1]} | {total[2]} | "
                         f"{p:.3f} | {r:.3f} | {f:.3f} | {fc:.3f} | {fd:.3f} |")
            data.setdefault(v, {})[str(tol)] = {"tp": int(total[0]), "fp": int(total[1]), "fn": int(total[2]),
                                               "opportunities": int(total[3]), "nodes": int(total[4]),
                                               "precision": p, "recall": r, "f1": f, "f1_cold": fc, "f1_cid": fd}
        lines.append("")
    lines += ["### Aligned / opposed revisits at 2 m (GT heading, ±45°)", "",
              "| Variant | Opp. aligned | TP aligned | Opp. opposed | TP opposed | FP |", "|---|---:|---:|---:|---:|---:|"]
    for v in variants:
        agg: dict[str, int] = {}
        for s in seqs:
            if (s.name, v) in results:
                for k, x in results[(s.name, v)]["breakdown"]["2.0"].items():
                    agg[k] = agg.get(k, 0) + x
        lines.append(f"| {v} | {agg.get('opp_aligned', 0)} | {agg.get('tp_aligned', 0)} | "
                     f"{agg.get('opp_opposed', 0)} | {agg.get('tp_opposed', 0)} | {agg.get('fp', 0)} |")
        data.setdefault(v, {})["breakdown_2m"] = agg
    per_seq = {s.name: {v: counts(results[(s.name, v)], 2.0) for v in variants if (s.name, v) in results} for s in seqs}
    lines += ["", "### Paired bootstrap over traversals at 2 m (10,000 resamples)", ""]
    tests = [("topo", "geometry_only", 0, "precision"), ("topo", "geometry_only", 2, "F1"),
             ("topo", "anyloc_vlad_g", 2, "F1"), ("topo", "paired_anyloc_vlad_g", 2, "F1"),
             ("topo", "anyloc_gem_g", 2, "F1"), ("topo", "paired_anyloc_gem_g", 2, "F1")]
    data["bootstrap_2m"] = {}
    for a, b, stat, label in tests:
        res = bootstrap(per_seq, a, b, stat)
        if res:
            lines.append(f"- {label} {a} − {b}: {res['point']:+.3f} [{res['ci95'][0]:+.3f}; {res['ci95'][1]:+.3f}]"
                         f" ({res['sequences']} traversals)")
            data["bootstrap_2m"][f"{label}:{a}-{b}"] = res
    lines += ["", "### Per traversal, F1 at 2 m", "", "| Traversal | " + " | ".join(variants) + " |",
              "|---|" + "---:|" * len(variants)]
    for s in seqs:
        cells = [f"{prf(per_seq[s.name][v])[2]:.3f}" if v in per_seq[s.name] else "–" for v in variants]
        lines.append(f"| {s.name} | " + " | ".join(cells) + " |")
    return lines + [""], data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    md = ["# AnyLoc published baseline and held-out evaluation — results", ""]
    summary = {}
    repro = args.out / "reproduction_dev.json"
    if repro.exists():
        md += [f"Reproduction of the manuscript on the 7 development traversals: "
               f"**{json.loads(repro.read_text())['status']}**", ""]
    ident = args.out / "descriptors/identities.json"
    if ident.exists():
        summary["encoders"] = json.loads(ident.read_text())
    for seqs, title, key in ((DEV, "Manuscript traversals (development)", "dev"),
                             (HELDOUT, "Held-out traversals", "heldout")):
        results = load(args.out, seqs)
        if results:
            lines, data = section(results, seqs, title)
            md += lines
            summary[key] = data
    target = HERE / "results"
    target.mkdir(exist_ok=True)
    (target / "summary.md").write_text("\n".join(md) + "\n")
    (target / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
