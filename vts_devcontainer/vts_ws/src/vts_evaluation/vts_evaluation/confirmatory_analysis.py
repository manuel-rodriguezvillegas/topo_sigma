"""Pre-registered analysis of docs/CONFIRMATORY_PROTOCOL_2026-10-02.md.

Knobs are chosen on the development traversals (MAIN_SESSIONS) for each regime
and method, pooled over seeds, then applied unchanged to the test traversals.
The primary endpoint is the L0, 2 m micro-F1 difference between
``posterior_view`` and ``geometry+cosine`` over the test traversals, with a
paired bootstrap over traversals (10,000 resamples).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from vts_core.artifacts import atomic_json
from vts_evaluation.regime_study import CONFIRM_SESSIONS, MAIN_SESSIONS, prf

PRIMARY, REFERENCE = "posterior_view", "geometry+cosine"
SECONDARY = ("posterior_geometry_only", "geometry_only", "posterior")
METHODS = (REFERENCE, "geometry_only", "posterior_geometry_only", "posterior", PRIMARY)


def per_session_counts(rows: list[dict], tolerance: float) -> tuple[dict, dict]:
    """Chosen knob per (regime, method) on dev; mean counts per test session at that knob."""
    dev: dict = defaultdict(lambda: np.zeros(3))
    test: dict = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["tolerance_m"] != tolerance:
            continue
        c = np.array([row["tp"], row["fp"], row["fn"]], dtype=float)
        if row["session"] in MAIN_SESSIONS:
            dev[(row["regime"], row["method"], row["value"])] += c
        elif row["session"] in CONFIRM_SESSIONS:
            test[(row["regime"], row["method"], row["value"])][row["session"]].append(c)
    chosen = {}
    for (regime, method, value), c in dev.items():
        best = chosen.get((regime, method))
        f1 = prf(*map(int, c))["f1"]
        if best is None or f1 > best[1] or (f1 == best[1] and value < best[0]):
            chosen[(regime, method)] = (value, f1)
    counts = {key: {s: np.mean(v, axis=0) for s, v in test[(key[0], key[1], value)].items()}
              for key, (value, _) in chosen.items()}
    return {k: v[0] for k, v in chosen.items()}, counts


def micro_f1(counts: dict, sessions: list[str]) -> float:
    total = sum((counts[s] for s in sessions), np.zeros(3))
    tp, fp, fn = total
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("study", type=Path)
    parser.add_argument("--boots", type=int, default=10000)
    args = parser.parse_args()
    rows = json.loads((args.study / "results.json").read_text())["rows"]
    rng = np.random.default_rng(20261002)
    report: dict = {"protocol": "docs/CONFIRMATORY_PROTOCOL_2026-10-02.md", "results": {}}
    for tolerance in (2.0, 1.0):
        knobs, counts = per_session_counts(rows, tolerance)
        sessions = sorted(next(iter(counts.values())))
        samples = [list(rng.choice(sessions, size=len(sessions), replace=True)) for _ in range(args.boots)]
        by_regime = {}
        for regime in sorted({r for r, _ in counts}):
            entry = {"knobs": {m: knobs[(regime, m)] for m in METHODS},
                     "f1": {m: micro_f1(counts[(regime, m)], sessions) for m in METHODS},
                     "counts": {m: (sum((counts[(regime, m)][s] for s in sessions), np.zeros(3))).tolist()
                                for m in METHODS},
                     "per_session_f1": {m: {s: micro_f1(counts[(regime, m)], [s]) for s in sessions}
                                        for m in METHODS},
                     "differences": {}}
            for other in (REFERENCE, *SECONDARY):
                deltas = np.array([micro_f1(counts[(regime, PRIMARY)], b) - micro_f1(counts[(regime, other)], b)
                                   for b in samples])
                entry["differences"][other] = {
                    "delta": entry["f1"][PRIMARY] - entry["f1"][other],
                    "lo": float(np.percentile(deltas, 2.5)), "hi": float(np.percentile(deltas, 97.5)),
                    "p_le_0": float(np.mean(deltas <= 0))}
            by_regime[regime] = entry
        report["results"][str(tolerance)] = by_regime
    primary = report["results"]["2.0"]["L0"]["differences"][REFERENCE]
    report["primary_endpoint"] = {**primary, "confirmed": primary["lo"] > 0}
    secondary = report["results"]["2.0"]["L0"]["differences"]["posterior_geometry_only"]
    report["h2"] = {**secondary, "confirmed": secondary["lo"] > 0}
    atomic_json(args.study / "confirmatory_analysis.json", report)
    for tol, regimes in report["results"].items():
        for regime, entry in regimes.items():
            print(f"\n[{tol} m, {regime}] F1: " + "  ".join(f"{m}={v:.3f}" for m, v in entry["f1"].items()))
            print("  counts TP/FP/FN: " + "  ".join(f"{m}={tuple(int(round(x)) for x in c)}" for m, c in entry["counts"].items()))
            for other, d in entry["differences"].items():
                print(f"  {PRIMARY} - {other:24s} {d['delta']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]")
    print(f"\nPRIMARY (H1): {report['primary_endpoint']}")
    print(f"H2: {report['h2']}")


if __name__ == "__main__":
    main()
