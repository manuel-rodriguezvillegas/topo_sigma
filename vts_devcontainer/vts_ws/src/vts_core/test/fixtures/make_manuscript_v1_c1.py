"""Reference decisions of the manuscript-v1 code (tag) on the seven C1 sessions."""
import json
import sys
from pathlib import Path

L = sys.argv[1]
for p in ("vts_evaluation", "vts_core", "vts_players"): sys.path.insert(0, f"{L}/vts_devcontainer/vts_ws/src/{p}")
from vts_evaluation.posterior_study import POSTERIOR_VARIANTS, session_priors

from vts_evaluation import posterior_association as pa
from vts_evaluation import regime_study as rs

root = Path("output/regime_study_v1")
NAMES = {"freiburg_a": "cold_freiburg_a_seq2_night1", "freiburg_ext": "cold_freiburg_b_seq3_sunny1",
         "saarbruecken_a": "cold_saarbruecken_a_seq2_night2", "saarbruecken_ext": "cold_saarbruecken_b_seq4_sunny1",
         "cid_sims_apartment1_1": "cid_apartment1_1", "cid_sims_apartment2_1": "cid_apartment2_1",
         "cid_sims_apartment3_1": "cid_apartment3_1"}
priors = session_priors(list(NAMES), str(root), "dinov3")
out = {"source": "git tag manuscript-v1 (vts_evaluation.regime_study / posterior_association)", "sessions": {}}
for legacy, name in NAMES.items():
    s = rs.load_session(legacy, root)
    frozen = rs.freeze_segments(s.descriptors, rs.segment(s.descriptors, s.config["mapping"]))
    poses, cov = rs.camera_odometry(s, s.odom_poses)
    nodes = rs.build_nodes(s, frozen, poses, cov, attach_gt=False)
    row = {"segments": [[int(f.indices[0]), int(f.indices[-1]) + 1] for f in frozen], "medoids": [f.medoid for f in frozen]}
    for method, mode in (("gate", "both"), ("geometry", "geometric")):
        g = rs.replay(nodes, {"window_size": 30, "valley_k": 2.0, "visual_outlier_k": 2.0, **rs.BASE,
                              "gate_mode": mode, "visual_model": "cosine"})
        row[method] = [[e.query_node_id, e.matched_node_id, bool(e.accepted), e.similarity, e.reason] for e in g.loop_events]
        row[method + "_counts"] = {str(t): rs.score(g, s, frozen, t) for t in (0.5, 1.0, 2.0)}
    places = [pa.place_from_frames(s.descriptors, f.indices, f.view_indices) for f in frozen]
    p = priors[legacy]
    row["prior"] = {"view": list(p.view), "diff": list(p.diff), "confirmed": p.confirmed, "shared": p.shared}
    for variant in ("posterior", "posterior+multiview", "posterior_view", "posterior_view_prior"):
        rec = pa.associate(nodes, places, [None] * len(nodes), POSTERIOR_VARIANTS[variant], prior=p)
        row[variant] = [[r["query"], r["matched"], r["posterior"], r["new"]] for r in rec]
    out["sessions"][name] = row
Path(sys.argv[2]).write_text(json.dumps(out, indent=1) + "\n")
print(sum(sum(d[2] for d in r["gate"]) for r in out["sessions"].values()), "accepted;",
      sum(len(r["medoids"]) for r in out["sessions"].values()), "places")
