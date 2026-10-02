"""Fidelity checks for the regime study against recorded DINOv3 runs.

At drift level L0 the descriptor-only segmentation must reproduce recorded node
boundaries and representative poses, and the replay must reproduce the recorded
``self_only/cosine`` closure decisions of ``closure_comparison_v2``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vts_core.topo_graph import TopoGraph
from vts_evaluation.regime_study import (
    BASE, WORKSPACE, build_nodes, camera_odometry, freeze_segments, load_session, replay, segment,
)


def validate(name: str, root: Path) -> dict:
    session = load_session(name, root)
    mapping = session.config["mapping"]
    frozen = freeze_segments(session.descriptors, segment(session.descriptors, mapping))
    recorded = TopoGraph.load(str(WORKSPACE / "output/dinov3_revisits" / name / "compact_graph.pkl"))
    rec_nodes = [recorded.nodes[i] for i in sorted(recorded.nodes)]
    sizes_match = [len(s.indices) for s in frozen] == [n.visual_sample_count for n in rec_nodes]
    poses, covariances = camera_odometry(session, session.odom_poses)
    nodes = build_nodes(session, frozen, poses, covariances, attach_gt=False)
    pose_error = max(float(np.max(np.abs(np.subtract(a.pose, b.pose)))) for a, b in zip(nodes, rec_nodes)) \
        if len(nodes) == len(rec_nodes) else float("inf")
    cov_error = max(float(np.max(np.abs(a.pose_covariance - b.pose_covariance)))
                    for a, b in zip(nodes, rec_nodes)) if len(nodes) == len(rec_nodes) else float("inf")
    view_error = max(float(np.max(np.abs(a.view_features - b.view_features)))
                     for a, b in zip(nodes, rec_nodes)) if len(nodes) == len(rec_nodes) else float("inf")
    graph = replay(nodes, {"window_size": 30, "valley_k": 2.0, "visual_outlier_k": 2.0,
                           **BASE, "gate_mode": "both", "visual_model": "cosine"})
    summary = json.loads((WORKSPACE / "output/closure_comparison_v2/summary.json").read_text())
    expected = summary["results"][name]["self_only/cosine"]["decisions"]
    got = [(e.query_node_id, e.matched_node_id, e.accepted) for e in graph.loop_events]
    want = [(e["query_node_id"], e["matched_node_id"], e["accepted"]) for e in expected]
    return {"nodes": len(nodes), "recorded_nodes": len(rec_nodes), "sizes_match": sizes_match,
            "max_pose_abs_error": pose_error, "max_cov_abs_error": cov_error,
            "max_view_abs_error": view_error, "decisions_match": got == want,
            "accepted": sum(a for *_, a in got), "expected_accepted": sum(a for *_, a in want)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs="+")
    parser.add_argument("--output", type=Path, default=WORKSPACE / "output/regime_study_v1")
    args = parser.parse_args()
    report = {name: validate(name, args.output) for name in args.sessions}
    print(json.dumps(report, indent=2))
    if not all(r["sizes_match"] and r["decisions_match"] for r in report.values()):
        raise SystemExit("Fidelity check FAILED")


if __name__ == "__main__":
    main()
