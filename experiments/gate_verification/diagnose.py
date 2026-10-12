"""Exploratory diagnostic on development data: verify loop candidates without odometry.

For every place decision of each stored baseline map (``<session>__gate__dinov3``),
the candidates are the K earlier eligible places with the highest multi-view
cosine (no odometry gate), the place the baseline matched, and, flagged as
``oracle`` for analysis only, the ground-truth nearest eligible place within
2 m. Every candidate pair is scored on its view pairs by the geometric
verifiers. Ground truth, odometry and the chi-square statistic are attached
for the summary; none of them selects or scores a pair (except the oracle row).

Outputs, one file per session (resumable): ``records/<session>.jsonl``
(candidate pairs) and ``queries/<session>.json`` (per-query baseline facts).

    python diagnose.py --out OUT [--verifiers lg,m|none] [--k 5] [--sessions ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "vts_devcontainer" / "vts_ws" / "src" / "vts_core"))

from vts_core import evaluation as ev  # noqa: E402
from vts_core.association import (  # noqa: E402
    PLACE_SIGMA,
    SimilarityRows,
    gaussian_position_statistics,
    interval_position_covariance,
)
from vts_core.datasets import data_root  # noqa: E402
from vts_core.graph import TopoGraph  # noqa: E402
from vts_core.odometry import normalize_angle  # noqa: E402
from vts_core.provenance import code_identity, json_safe, load_config, resolve  # noqa: E402
from vts_core.session import open_session  # noqa: E402

ORACLE_RADIUS_M = 2.0


def relative(poses: np.ndarray, a: int, b: int) -> tuple[float, float]:
    """Planar distance and absolute heading change between frames ``a`` and ``b``."""
    return (float(np.linalg.norm(poses[a, :2] - poses[b, :2])),
            float(abs(normalize_angle(poses[b, 2] - poses[a, 2]))))


def candidate_pairs(config: dict, name: str, k: int) -> tuple[list[dict], dict]:
    graph = TopoGraph.load(resolve(config, "output_root") / "maps" / f"{name}__gate__dinov3.npz")
    places = graph.places[name]
    decisions = {d.query: d for d in graph.decisions[name]}
    session = open_session(config, name)
    gt_frames, (odom_frames, _) = session.gt, session.odometry
    descriptors = session.descriptors("dinov3")
    root = data_root(config) / config["datasets"][session.recording.dataset]["subdir"]
    gt = ev.place_gt(places, gt_frames)
    eligible_sets = ev.in_session_eligible(len(places))
    kinds = {t: ev.revisit_kinds(gt, gt, t, config["evaluation"]["aligned_deg"], eligible_sets)
             for t in config["evaluation"]["tolerances_m"]}
    similarity = SimilarityRows("multiview_cosine", np.float32)
    rows = [similarity.add(p) for p in places]

    records, queries = [], {}
    for q, place in enumerate(places):
        eligible = eligible_sets[q]
        decision = decisions.get(q)
        if decision is None:
            continue
        query_row = {"query": q, "eligible": len(eligible), "accepted": decision.accepted,
                     "matched": decision.matched, "reason": decision.reason,
                     "kind": {str(t): kinds[t][q] for t in kinds}}
        if decision.matched is not None:
            query_row["matched_gt_dist"] = float(np.linalg.norm(gt[decision.matched, :2] - gt[q, :2]))
        queries[q] = query_row
        if not len(eligible):
            continue
        scores = rows[q][: len(eligible)]
        covariance = interval_position_covariance(place, [places[j] for j in eligible], PLACE_SIGMA, check=False)
        delta = place.pose[:2] - np.asarray([places[j].pose[:2] for j in eligible])
        d2, _ = gaussian_position_statistics(delta, covariance)
        order = np.argsort(-scores, kind="stable")
        rank = np.empty(len(order), dtype=int)
        rank[order] = np.arange(len(order))
        gt_dist = np.linalg.norm(gt[eligible, :2] - gt[q, :2], axis=1)
        chosen = {int(j): set() for j in order[:k]}
        for j in order[:k]:
            chosen[int(j)].add("topk")
        if decision.matched is not None:
            chosen.setdefault(int(decision.matched), set()).add("baseline")
        if gt_dist.min() <= ORACLE_RADIUS_M:
            chosen.setdefault(int(np.argmin(gt_dist)), set()).add("oracle")
        for j, roles in sorted(chosen.items()):
            candidate = places[j]
            views = []
            for fa in place.view_frames:
                for fb in candidate.view_frames:
                    gd, gy = relative(gt_frames, fa, fb)
                    od, oy = relative(odom_frames, fa, fb)
                    views.append({"frame_a": int(fa), "frame_b": int(fb),
                                  "path_a": str(root / session.image_paths[fa]),
                                  "path_b": str(root / session.image_paths[fb]),
                                  "dino_cos": float(descriptors[fa] @ descriptors[fb]),
                                  "gt_dist": gd, "gt_dyaw_abs": gy, "odom_dist": od, "odom_dyaw_abs": oy})
            views.sort(key=lambda v: -v["dino_cos"])
            records.append({
                "session": name, "dataset": session.recording.dataset,
                "environment": session.recording.environment, "query": q, "candidate": j,
                "roles": sorted(roles), "visual_rank": int(rank[j]), "visual_sim": float(scores[j]),
                "d2": float(d2[j]), "gt_dist": float(gt_dist[j]),
                "gt_dyaw_abs": float(abs(normalize_angle(gt[j, 2] - gt[q, 2]))),
                "baseline_accepted_this": bool(decision.accepted and decision.matched == j),
                "views": views})
    return records, queries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--sessions", nargs="*", default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--verifiers", default="lg,m", help="comma list of lg, m; or none")
    parser.add_argument("--mast3r-repo", type=Path, default=HERE / "third_party" / "mast3r")
    parser.add_argument("--mast3r-views", type=int, default=2, help="view pairs per place pair (best DINO cosine)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = load_config(args.config)
    maps = resolve(config, "output_root") / "maps"
    names = args.sessions or sorted(p.name.split("__")[0] for p in maps.glob("*__gate__dinov3.npz"))
    names = [n for n in names if open_session(config, n).recording.role == "development"]
    wanted = set() if args.verifiers == "none" else set(args.verifiers.split(","))
    verifiers = {}
    if "lg" in wanted:
        from verifiers import LightGlueVerifier
        verifiers["lg"] = LightGlueVerifier(args.device)
    if "m" in wanted:
        from verifiers import Mast3rVerifier
        verifiers["m"] = Mast3rVerifier(args.mast3r_repo, args.device)

    (args.out / "records").mkdir(parents=True, exist_ok=True)
    (args.out / "queries").mkdir(parents=True, exist_ok=True)
    (args.out / "run.json").write_text(json.dumps({
        "code_sha256": code_identity(), "k": args.k, "verifiers": sorted(wanted),
        "mast3r_views": args.mast3r_views, "sessions": names}, indent=2))
    for number, name in enumerate(names, 1):
        target = args.out / "records" / f"{name}.jsonl"
        if target.is_file():
            print(f"[{number}/{len(names)}] {name}: done", flush=True)
            continue
        started = time.perf_counter()
        records, queries = candidate_pairs(config, name, args.k)
        if "lg" in verifiers:
            for record in records:
                for view in record["views"]:
                    view.update(verifiers["lg"].score(view["path_a"], view["path_b"]))
            verifiers["lg"].reset()
        if "m" in verifiers:
            selected = [v for r in records for v in r["views"][: args.mast3r_views]]
            for view, row in zip(selected, verifiers["m"].score_many([(v["path_a"], v["path_b"]) for v in selected])):
                view.update(row)
            verifiers["m"].reset()
        (args.out / "queries" / f"{name}.json").write_text(json.dumps(json_safe(queries)))
        partial = target.with_suffix(".partial")
        partial.write_text("".join(json.dumps(json_safe(r)) + "\n" for r in records))
        partial.replace(target)
        print(f"[{number}/{len(names)}] {name}: {len(queries)} queries, {len(records)} pairs, "
              f"{time.perf_counter() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
