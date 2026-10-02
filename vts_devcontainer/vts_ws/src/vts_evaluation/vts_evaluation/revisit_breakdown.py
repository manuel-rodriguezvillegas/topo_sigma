"""Recall split by revisit type: same viewing direction versus opposite passes.

Position-only ground truth counts a revisit whenever representative positions
are within the tolerance, whatever the camera heading. A forward camera that
passes in the opposite direction observes different scene content, so only
geometry can detect such revisits. This replays the posterior study's exact
poses (same seeds and random streams) at the knobs selected leave-one-traversal-
out, and splits each revisit opportunity by whether an eligible node within the
tolerance also has a GT heading within ``--aligned-deg``. Heading is used only
for this evaluation split.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from vts_core.artifacts import atomic_json
from vts_core.topo_graph import LOOP_TEMPORAL_EXCLUSION_NODES
from vts_evaluation.posterior_association import associate, decisions_to_graph, place_from_frames
from vts_evaluation.posterior_study import (
    ALL_SESSIONS, POSTERIOR_VARIANTS, REGIMES, apply_cuts, posterior_association, session_priors,
)
from vts_evaluation.regime_study import (
    DEFAULT_KNOB, DRIFT_LEVELS, METHODS, WORKSPACE, build_nodes, camera_odometry, corrupt_odometry,
    freeze_segments, load_session, overlay_appearance, replay, segment,
)

COMPARED = ("geometry+cosine", "geometry_only", "posterior", "posterior_geometry_only", "posterior_view",
            "posterior_view_prior")


def classify(session, frozen, tolerance: float, aligned_rad: float) -> dict[int, str]:
    gt = np.asarray([session.gt[s.medoid] for s in frozen])
    kinds = {}
    for i in range(len(frozen)):
        eligible = np.arange(0, max(i - LOOP_TEMPORAL_EXCLUSION_NODES, 0))
        if not len(eligible):
            continue
        near = np.linalg.norm(gt[eligible, :2] - gt[i, :2], axis=1) <= tolerance
        if not near.any():
            continue
        dh = np.abs((gt[eligible, 2] - gt[i, 2] + np.pi) % (2 * np.pi) - np.pi)
        kinds[i] = "aligned" if (near & (dh < aligned_rad)).any() else "opposed"
    return kinds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("study", type=Path, help="posterior_study output directory")
    parser.add_argument("--frames", type=Path, default=WORKSPACE / "output/regime_study_v1")
    parser.add_argument("--tolerance", type=float, default=2.0)
    parser.add_argument("--aligned-deg", type=float, default=45.0)
    parser.add_argument("--regimes", nargs="+", default=["L0", "L4", "cut1", "cut2"])
    args = parser.parse_args()
    results = json.loads((args.study / "results.json").read_text())
    report = json.loads((args.study / "report.json").read_text())[str(args.tolerance)]
    encoder = results["protocol"]["appearance"]
    seeds = results["protocol"]["seeds"]
    sessions = results["protocol"]["sessions"]
    methods = [m for m in COMPARED if m in report["leave_one_out"]["L0"]]
    priors = session_priors(sessions, str(args.frames), encoder) if "posterior_view_prior" in methods else {}
    totals: dict = defaultdict(lambda: defaultdict(lambda: np.zeros(5)))  # tp_al, n_al, tp_op, n_op, fp
    for name in sessions:
        session = load_session(name, args.frames)
        mapping = session.config["mapping"]
        frozen = freeze_segments(session.descriptors, segment(session.descriptors, mapping))
        appearance, source = None, session.descriptors
        if encoder != "dinov3":
            from vts_evaluation.vpr_descriptors import load_appearance
            appearance = source = load_appearance(name, args.frames, encoder)
            frozen = overlay_appearance(frozen, appearance)
        places = [place_from_frames(source, s.indices, s.view_indices) for s in frozen]
        halves = [(place_from_frames(source, s.indices[: len(s.indices) // 2]),
                   place_from_frames(source, s.indices[len(s.indices) // 2:]))
                  if len(s.indices) >= 2 * posterior_association._MIN_HALF_FRAMES else None for s in frozen]
        kinds = classify(session, frozen, args.tolerance, np.radians(args.aligned_deg))
        gt = np.asarray([session.gt[s.medoid, :2] for s in frozen])
        common = {"window_size": int(mapping["window_size"]), "valley_k": float(mapping["valley_k"]),
                  "verification_threshold": float(mapping["verification_threshold"]),
                  "defer_nodes": int(mapping["defer_nodes"]),
                  "fixed_concentration": float(mapping["fixed_concentration"]),
                  "naive_threshold": float(mapping["naive_threshold"])}
        source_key = "held_out" if name not in report["leave_one_out"]["L0"]["posterior"]["chosen_knob"] else None
        for regime_index, regime in enumerate(REGIMES):
            if regime.name not in args.regimes:
                continue
            level = DRIFT_LEVELS[regime.drift]
            deterministic = level.name == "L0" and not regime.cuts
            for seed in range(1 if deterministic else seeds):
                # Identical random stream to posterior_study.run_session.
                rng = np.random.default_rng([seed, regime_index, ALL_SESSIONS.index(name), 2])
                raw = corrupt_odometry(session.odom_poses, level, rng)
                poses, covariances = camera_odometry(session, raw)
                poses, _ = apply_cuts(poses, regime.cuts, rng)
                nodes = build_nodes(session, frozen, poses, covariances, attach_gt=False, appearance=appearance)
                for method in methods:
                    entry = (report["held_out"] if source_key else report["leave_one_out"])[regime.name][method]
                    knob = entry["chosen_knob"] if source_key else entry["chosen_knob"][name]
                    if method.startswith("posterior"):
                        graph = decisions_to_graph(nodes, associate(nodes, places, halves, POSTERIOR_VARIANTS[method], prior=priors.get(name)), knob)
                    else:
                        name_knob = "geometric_gate_scale" if METHODS[method]["gate_mode"] == "geometric" else "visual_outlier_k"
                        graph = replay(nodes, {**common, **DEFAULT_KNOB, **METHODS[method], name_knob: knob})
                    counts = np.zeros(5)
                    for event in graph.loop_events:
                        correct = event.accepted and event.matched_node_id is not None and (
                            np.linalg.norm(gt[event.query_node_id] - gt[event.matched_node_id]) <= args.tolerance)
                        kind = kinds.get(event.query_node_id)
                        if kind == "aligned":
                            counts[0] += correct; counts[1] += 1
                        elif kind == "opposed":
                            counts[2] += correct; counts[3] += 1
                        counts[4] += event.accepted and not correct
                    group = "held_out" if source_key else "main"
                    totals[(group, regime.name)][method] += counts / (1 if deterministic else seeds)
        print(f"{name}: done", flush=True)
    out = {}
    for (group, regime), methods in sorted(totals.items()):
        out.setdefault(group, {})[regime] = {
            m: {"aligned_opportunities": c[1], "aligned_recall": c[0] / max(c[1], 1e-9),
                "opposed_opportunities": c[3], "opposed_recall": c[2] / max(c[3], 1e-9),
                "false_positives": c[4], "precision": (c[0] + c[2]) / max(c[0] + c[2] + c[4], 1e-9)}
            for m, c in methods.items()}
    atomic_json(args.study / f"revisit_breakdown_{args.tolerance:g}m.json",
                {"encoder": encoder, "aligned_deg": args.aligned_deg, "tolerance_m": args.tolerance,
                 "results": out})
    for group, regimes in out.items():
        for regime, methods in regimes.items():
            first = next(iter(methods.values()))
            print(f"\n[{encoder}] {group} {regime}: aligned opp {first['aligned_opportunities']:.0f}, "
                  f"opposed opp {first['opposed_opportunities']:.0f}")
            for m, v in methods.items():
                print(f"  {m:24s} R_aligned {v['aligned_recall']:.3f}  R_opposed {v['opposed_recall']:.3f}  "
                      f"FP {v['false_positives']:.1f}  P {v['precision']:.3f}")


if __name__ == "__main__":
    main()
