"""Replay the exact manuscript mapper with different descriptors and score it.

Variants (``V`` below). ``seg`` feeds the lambda_2 node detector; ``assoc`` is
stored in the nodes and used for loop-closure evidence.

- full replacement (as in the manuscript AnyLoc table): seg = assoc = encoder.
- paired (new): seg = DINOv2-S (TopoSIGMA's own boundaries), assoc = encoder.
  Every method then faces the same nodes and revisit opportunities, up to the
  representative frame chosen inside each segment.

Reproduction checks on the manuscript sequences (must pass before anything else is read):
- ``topo_mac`` (Mac descriptors, vMF) == submission TopoSIGMA, all radii.
- ``paired_cosine_mac`` (Mac descriptors, cosine) == submission cosine ablation.

Run with BLAS limited to one thread per process (set by run_all.sh).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from common import ALL, DEV, HELDOUT, HERE, MAPPER_OPTIONS, TOLERANCES, frames
from vts_core.mapper import TopologicalMapper
from vts_evaluation.tolerance_sweep import EvaluationCase, evaluate_case

DUMMY_IMAGE = np.zeros((1, 1, 3), dtype=np.uint8)  # views are not scored; avoids decoding images
ENCODERS_G = ["anyloc_gem_g224", "anyloc_gem_g", "anyloc_vlad_g"]


def variants(split: str) -> dict[str, dict]:
    seg_ref = "mac" if split == "dev" else "dinov2_s"
    v = {
        "topo": dict(seg=seg_ref, assoc=seg_ref, visual_model="vmf", gate_mode="both"),
        "topo_cosine": dict(seg=seg_ref, assoc=seg_ref, visual_model="cosine", gate_mode="both"),
        "geometry_only": dict(seg=seg_ref, assoc=seg_ref, visual_model="vmf", gate_mode="geometric"),
        "anyloc_gem_s224": dict(seg="anyloc_gem_s224", assoc="anyloc_gem_s224", visual_model="cosine", gate_mode="both"),
    }
    if split == "dev":
        v["topo_gpu"] = dict(seg="dinov2_s", assoc="dinov2_s", visual_model="vmf", gate_mode="both")
    for enc in ENCODERS_G:
        v[enc] = dict(seg=enc, assoc=enc, visual_model="cosine", gate_mode="both")
        v[f"paired_{enc}"] = dict(seg=seg_ref, assoc=enc, visual_model="cosine", gate_mode="both")
    return v


class _SegmentationOverride:
    """Feeds the connectivity monitor a different descriptor than the one stored in nodes."""

    def __init__(self, inner):
        self.inner, self.next = inner, None

    def update(self, _descriptor):
        return self.inner.update(self.next)


def load_descriptors(root: Path, seq: str, encoder: str) -> tuple[np.ndarray, np.ndarray]:
    if encoder == "mac":
        with np.load(HERE / "reference/dinov2_vits14_mac.npz") as data:
            return data[f"{seq}/descriptors"], data[f"{seq}/frames"]
    path = root / "descriptors" / seq / f"{encoder}.npy"
    return np.load(path), np.load(path.with_suffix(".frames.npy"))


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def breakdown(graph, node_gt: dict[int, tuple], tolerance: float, aligned_deg: float = 45.0) -> dict:
    """Split opportunities and true positives into aligned / opposed revisits (GT heading)."""
    ids = sorted(graph.nodes)
    counts = {"opp_aligned": 0, "opp_opposed": 0, "tp_aligned": 0, "tp_opposed": 0, "fp": 0}
    for event in graph.loop_events:
        q = event.query_node_id
        qx, qy, qa = node_gt[q]
        near = [j for j in ids if j < q - 2 and math.hypot(node_gt[j][0] - qx, node_gt[j][1] - qy) <= tolerance]
        kind = None
        if near:
            aligned = any(abs(wrap(node_gt[j][2] - qa)) < math.radians(aligned_deg) for j in near)
            kind = "aligned" if aligned else "opposed"
            counts[f"opp_{kind}"] += 1
        if event.accepted and event.matched_node_id is not None:
            m = int(event.matched_node_id)
            ok = math.hypot(node_gt[m][0] - qx, node_gt[m][1] - qy) <= tolerance
            if ok:
                counts[f"tp_{kind or 'unmatched'}"] = counts.get(f"tp_{kind or 'unmatched'}", 0) + 1
            else:
                counts["fp"] += 1
    return counts


def run_one(job: tuple[str, str, dict, str]) -> dict:
    seq_name, variant, spec, root = job
    root = Path(root)
    seq = ALL[seq_name]
    target = root / "replay" / seq_name / variant
    result_path = target / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    try:
        seg, seg_frames = load_descriptors(root, seq_name, spec["seg"])
        assoc, assoc_frames = load_descriptors(root, seq_name, spec["assoc"])
        mapper = TopologicalMapper(frame_id=seq_name, **{**MAPPER_OPTIONS, "gate_mode": spec["gate_mode"],
                                                         "visual_model": spec["visual_model"]})
        override = None
        if spec["seg"] != spec["assoc"]:
            override = _SegmentationOverride(mapper._monitor)
            mapper._monitor = override
        count = 0
        for k, (image_path, gt_pose, pose, budget, label) in enumerate(frames(seq)):
            name = os.path.basename(image_path)
            if name != seg_frames[k] or name != assoc_frames[k]:
                raise AssertionError(f"frame {k} mismatch: {name} vs {seg_frames[k]} / {assoc_frames[k]}")
            if override is not None:
                override.next = seg[k].astype(np.float32)
            mapper.process_frame(DUMMY_IMAGE, assoc[k].astype(np.float32), pose, budget, gt_pose, label)
            count += 1
        if count != len(seg):
            raise AssertionError(f"{count} frames streamed, {len(seg)} descriptors")
        mapper.finalize_nodes()
        target.mkdir(parents=True, exist_ok=True)
        graph_path, gt_path = target / "graph.pkl", target / "node_gt.json"
        mapper.graph.save(str(graph_path))
        node_gt = {n.node_id: tuple(n.gt_pose) for n in mapper.graph.nodes.values()}
        gt_path.write_text(json.dumps({str(k): v for k, v in node_gt.items()}))
        rows = evaluate_case(EvaluationCase(seq_name, str(graph_path), str(gt_path), str(seq.trajectory_gt)), TOLERANCES)
        result = {
            "sequence": seq_name, "variant": variant, "split": seq.split, "domain": seq.domain, "spec": spec,
            "frames": count, "nodes": len(mapper.graph.nodes),
            "boundaries": [n.visual_sample_count for n in mapper.graph.nodes.values()],
            "metrics": [{k: r[k] for k in ("tolerance_m", "n_nodes", "n_loop_edges", "loop_opportunities",
                                            "loop_tp", "loop_fp", "loop_fn", "loop_precision", "loop_recall",
                                            "loop_f1", "coverage_at_tolerance")} for r in rows],
            "breakdown": {str(t): breakdown(mapper.graph, node_gt, t) for t in TOLERANCES},
            "loops": [[e.query_node_id, e.matched_node_id] for e in mapper.graph.loop_events if e.accepted],
        }
        mapper.close()
        if spec["assoc"] == "anyloc_vlad_g":
            graph_path.unlink()  # 49k-D views; metrics are kept
        result_path.write_text(json.dumps(result, indent=1))
        print(f"{seq_name:32s} {variant:24s} nodes={result['nodes']:3d} "
              f"2m TP/FP/FN={[(m['loop_tp'], m['loop_fp'], m['loop_fn']) for m in result['metrics'] if m['tolerance_m'] == 2.0][0]}",
              flush=True)
        return result
    except Exception as error:  # report and keep the other jobs running
        return {"sequence": seq_name, "variant": variant, "error": f"{error!r}", "trace": traceback.format_exc()}


def check_reproduction(results: list[dict], sequences: list[str]) -> list[str]:
    reference = json.loads((HERE / "reference/manuscript_metrics.json").read_text())["cases"]
    problems = []
    done = {(r["sequence"], r["variant"]) for r in results}
    for name in sequences:
        for variant in ("topo", "topo_cosine"):
            if (name, variant) not in done:
                problems.append(f"{name} {variant}: no result (see ERROR above)")
    for r in results:
        if r.get("split") != "dev" or r["variant"] not in ("topo", "topo_cosine"):
            continue
        key = r["sequence"] + ("" if r["variant"] == "topo" else "_cosine")
        for got, want in zip(r["metrics"], reference[key]):
            for f in ("n_nodes", "n_loop_edges", "loop_opportunities", "loop_tp", "loop_fp", "loop_fn"):
                if got[f] != want[f]:
                    problems.append(f"{key} @{want['tolerance_m']} m {f}: {got[f]} != {want[f]}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "heldout"], default="dev")
    parser.add_argument("--variants", nargs="*", help="subset of variant names")
    parser.add_argument("--sequences", nargs="*")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    seqs = args.sequences or [s.name for s in (DEV if args.split == "dev" else HELDOUT)]
    specs = variants(args.split)
    chosen = args.variants or list(specs)
    jobs = [(s, v, specs[v], str(args.out)) for s in seqs for v in chosen]
    # Big VLAD jobs last so that the cheap checks finish first.
    jobs.sort(key=lambda j: ("vlad" in j[1], j[1]))
    with ProcessPoolExecutor(args.jobs) as pool:
        results = list(pool.map(run_one, jobs))
    errors = [r for r in results if "error" in r]
    for e in errors:
        print("ERROR", e["sequence"], e["variant"], e["error"])
    if args.split == "dev":
        problems = check_reproduction([r for r in results if "error" not in r], seqs)
        status = "PASS" if not problems else "FAIL"
        print(f"Reproduction of the manuscript (topo, topo_cosine): {status}")
        for p in problems[:40]:
            print("  ", p)
        (args.out / "reproduction_dev.json").write_text(json.dumps({"status": status, "problems": problems}, indent=1))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
