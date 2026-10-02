"""Pre-registered analysis of docs/CONFIRMATORY_PROTOCOL_CID_2026-10-02.md.

1. Checks that the development rows of ``study`` reproduce ``--reference``
   exactly (same code, seeds and knobs), before any test row is read.
2. Chooses each method's knob on the development traversals (L0, pooled), as in
   :mod:`confirmatory_analysis`.
3. Replays L0 on the test traversals at those knobs and records per-query
   outcomes; the 95 % interval of each micro-F1 difference comes from a paired
   bootstrap of queries stratified by traversal (10,000 resamples), with a
   traversal bootstrap reported as a complement.
4. Applies the futility rule: H2 is inconclusive with < 10 aligned opportunities.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vts_core.artifacts import atomic_json
from vts_evaluation.confirmatory_analysis import per_session_counts
from vts_evaluation.posterior_association import associate, decisions_to_graph, place_from_frames
from vts_evaluation.posterior_study import POSTERIOR_VARIANTS, load_places
from vts_evaluation.regime_study import (
    CID_CONFIRM_SESSIONS, DEFAULT_KNOB, MAIN_SESSIONS, METHODS, build_nodes, camera_odometry, replay,
)
from vts_evaluation.revisit_breakdown import classify

PRIMARY = "posterior_view"
COMPARED = ("posterior_geometry_only", "geometry+cosine", "geometry_only", "posterior")
MIN_ALIGNED = 10


def check_development(study_rows: list[dict], reference_rows: list[dict]) -> dict:
    key = lambda r: (r["session"], r["regime"], r["seed"], r["method"], r["value"], r["tolerance_m"])
    ours = {key(r): (r["tp"], r["fp"], r["fn"]) for r in study_rows if r["session"] in MAIN_SESSIONS}
    ref = {key(r): (r["tp"], r["fp"], r["fn"]) for r in reference_rows
           if r["session"] in MAIN_SESSIONS and key(r) in ours}
    mismatches = [k for k in ref if ref[k] != ours[k]]
    return {"compared_rows": len(ref), "mismatches": len(mismatches), "identical": not mismatches and len(ref) > 0}


def query_outcomes(name: str, frames: str, knobs: dict, tolerance: float) -> dict[str, np.ndarray]:
    session, mapping, frozen, appearance, places, halves = load_places(name, frames, "dinov3")
    poses, covariances = camera_odometry(session, session.odom_poses)
    nodes = build_nodes(session, frozen, poses, covariances, attach_gt=False, appearance=appearance)
    gt = np.asarray([session.gt[s.medoid, :2] for s in frozen])
    kinds = classify(session, frozen, tolerance, np.radians(45.0))
    common = {"window_size": int(mapping["window_size"]), "valley_k": float(mapping["valley_k"]),
              "verification_threshold": float(mapping["verification_threshold"]),
              "defer_nodes": int(mapping["defer_nodes"]),
              "fixed_concentration": float(mapping["fixed_concentration"]),
              "naive_threshold": float(mapping["naive_threshold"])}
    out = {}
    for method in (PRIMARY, *COMPARED):
        knob = knobs[("L0", method)]
        if method.startswith("posterior"):
            graph = decisions_to_graph(nodes, associate(nodes, places, halves, POSTERIOR_VARIANTS[method]), knob)
        else:
            knob_name = "geometric_gate_scale" if METHODS[method]["gate_mode"] == "geometric" else "visual_outlier_k"
            graph = replay(nodes, {**common, **DEFAULT_KNOB, **METHODS[method], knob_name: knob})
        rows = []
        for event in graph.loop_events:
            q = event.query_node_id
            correct = bool(event.accepted and event.matched_node_id is not None
                           and np.linalg.norm(gt[q] - gt[event.matched_node_id]) <= tolerance)
            rows.append((q in kinds, correct, bool(event.accepted and not correct), kinds.get(q) == "aligned"))
        out[method] = np.asarray(rows, dtype=bool)
    return out


def f1(rows: np.ndarray) -> float:
    opportunity, correct, wrong = rows[:, 0], rows[:, 1], rows[:, 2]
    tp, fp, fn = correct.sum(), wrong.sum(), (opportunity & ~correct).sum()
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("study", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--frames", default="output/regime_study_v1")
    parser.add_argument("--boots", type=int, default=10000)
    args = parser.parse_args()
    rows = json.loads((args.study / "results.json").read_text())["rows"]
    development = check_development(rows, json.loads((args.reference / "results.json").read_text())["rows"])
    print("development check:", development, flush=True)
    if not development["identical"]:
        raise SystemExit("Development rows do not reproduce the reference study; stopping before test data.")
    report = {"protocol": "docs/CONFIRMATORY_PROTOCOL_CID_2026-10-02.md", "development_check": development,
              "results": {}}
    rng = np.random.default_rng(20261003)
    tests = [s for s in CID_CONFIRM_SESSIONS if any(r["session"] == s for r in rows)]
    for tolerance in (2.0, 1.0):
        knobs, _ = per_session_counts(rows, tolerance)
        per = {s: query_outcomes(s, args.frames, knobs, tolerance) for s in tests}
        pooled = {m: np.concatenate([per[s][m] for s in tests]) for m in (PRIMARY, *COMPARED)}
        aligned = int((pooled[PRIMARY][:, 0] & pooled[PRIMARY][:, 3]).sum())
        entry = {"knobs": {m: knobs[("L0", m)] for m in (PRIMARY, *COMPARED)},
                 "opportunities": int(pooled[PRIMARY][:, 0].sum()), "aligned_opportunities": aligned,
                 "f1": {m: f1(v) for m, v in pooled.items()},
                 "counts": {m: [int(v[:, 1].sum()), int(v[:, 2].sum()), int((v[:, 0] & ~v[:, 1]).sum())]
                            for m, v in pooled.items()},
                 "aligned_recall": {m: float(v[v[:, 0] & v[:, 3], 1].mean()) if aligned else None
                                    for m, v in pooled.items()},
                 "per_session_f1": {m: {s: f1(per[s][m]) for s in tests} for m in (PRIMARY, *COMPARED)},
                 "differences": {}}
        index = {s: np.arange(len(per[s][PRIMARY])) for s in tests}
        query_samples = [{s: rng.choice(index[s], size=len(index[s])) for s in tests} for _ in range(args.boots)]
        session_samples = [list(rng.choice(tests, size=len(tests))) for _ in range(args.boots)]
        for other in COMPARED:
            dq = [f1(np.concatenate([per[s][PRIMARY][b[s]] for s in tests]))
                  - f1(np.concatenate([per[s][other][b[s]] for s in tests])) for b in query_samples]
            dt = [f1(np.concatenate([per[s][PRIMARY] for s in b])) - f1(np.concatenate([per[s][other] for s in b]))
                  for b in session_samples]
            entry["differences"][other] = {
                "delta": entry["f1"][PRIMARY] - entry["f1"][other],
                "query_lo": float(np.percentile(dq, 2.5)), "query_hi": float(np.percentile(dq, 97.5)),
                "traversal_lo": float(np.percentile(dt, 2.5)), "traversal_hi": float(np.percentile(dt, 97.5))}
        report["results"][str(tolerance)] = entry
    main_entry = report["results"]["2.0"]
    h2 = main_entry["differences"]["posterior_geometry_only"]
    h1 = main_entry["differences"]["geometry+cosine"]
    report["h2"] = {**h2, "confirmed": h2["query_lo"] > 0 and main_entry["aligned_opportunities"] >= MIN_ALIGNED,
                    "futile": main_entry["aligned_opportunities"] < MIN_ALIGNED}
    report["h1"] = {**h1, "confirmed": h1["query_lo"] > 0}
    atomic_json(args.study / "query_bootstrap.json", report)
    for tol, e in report["results"].items():
        print(f"\n[{tol} m] opportunities {e['opportunities']} (aligned {e['aligned_opportunities']})")
        for m in e["f1"]:
            print(f"  {m:24s} F1 {e['f1'][m]:.3f}  TP/FP/FN {e['counts'][m]}  R_aligned {e['aligned_recall'][m]}")
        for o, d in e["differences"].items():
            print(f"  {PRIMARY} - {o:24s} {d['delta']:+.3f}  queries [{d['query_lo']:+.3f},{d['query_hi']:+.3f}]"
                  f"  traversals [{d['traversal_lo']:+.3f},{d['traversal_hi']:+.3f}]")
    print("\nH2:", report["h2"])
    print("H1:", report["h1"])


if __name__ == "__main__":
    main()
