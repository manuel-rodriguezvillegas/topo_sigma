"""Run every mapper configuration of the manuscript on GPU descriptors and score it.

Descriptors come from the AnyLoc runner (``$ANYLOC_OUT/descriptors``). Each job writes
``$OUT/replay/<sequence>/<variant>/result.json``; finished jobs are skipped on a rerun.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from stream import ALL, DEV, HELDOUT, RESULTS, cumulative_covariances, gates, load_stream
from vts_core.matching import VisualLikelihood
from common import breakdown, load_descriptors
from vts_core.mapper import TopologicalMapper
from vts_core.metrics import graph_metrics
from vts_evaluation.ground_truth import load_ground_truth_xy
from vts_evaluation.tolerance_sweep import EvaluationCase, evaluate_case

TOLERANCES = [0.5, 1.0, 2.0, 3.0]
MAIN_GATE = "propagated_calibrated"
DESCRIPTOR = "dinov2_s"
MAPPER_OPTIONS = dict(window_size=30, valley_k=2.0, valley_mode="adaptive", valley_delta=0.1,
                      visual_outlier_k=2.0, naive_threshold=0.7)
NODE_SENSITIVITY_FILE = RESULTS / "node_sensitivity.json"
VISUAL_FILE = RESULTS / "visual_likelihood.json"
PLACE_RADIUS_M = 2.0
DUMMY_IMAGE = np.zeros((1, 1, 3), dtype=np.uint8)


def _spec(gate=MAIN_GATE, encoder=DESCRIPTOR, association="disk", gate_mode="geometric", visual_model="vmf",
          images=False, **mapper) -> dict:
    return dict(gate=gate, encoder=encoder, association=association, gate_mode=gate_mode,
                visual_model=visual_model, images=images, mapper={**MAPPER_OPTIONS, **mapper})


COMPARISON_ENCODERS = ("anyloc_vlad_g", "anyloc_gem_g", "dinov3_s")
RETRIEVAL_SCORE = {"anyloc_vlad_g": "cosine", "anyloc_gem_g": "cosine", "dinov3_s": "vmf"}


def variants(split: str) -> dict[str, dict]:
    """Geometric variants; visual ones, which need ``VISUAL_FILE``, are in :func:`visual_variants`."""
    v = {
        "topo": _spec(images=split == "dev"),
        "disk_isotropic_default": _spec(gate="isotropic_default"),
        "disk_isotropic_calibrated": _spec(gate="isotropic_calibrated"),
        "gate_propagated": _spec(association="gate", gate_mode="both"),
        "gate_propagated_geometry_only": _spec(association="gate"),
        "paper": _spec(gate="isotropic_default", association="gate", gate_mode="both"),
        "paper_geometry_only": _spec(gate="isotropic_default", association="gate"),
        "threshold": _spec(association="gate", gate_mode="threshold"),
        "visual_only": _spec(association="gate", gate_mode="visual"),
    }
    for encoder in COMPARISON_ENCODERS:
        v[encoder] = _spec(encoder=encoder)
    if split == "dev":
        for k in (1.0, 1.5, 2.5, 3.0):
            v[f"valley_k{k}"] = _spec(valley_k=k)
        v["valley_fixed"] = _spec(valley_mode="fixed")
    return v


def visual_variants(split: str) -> dict[str, dict]:
    """Disk association reweighted by visual evidence of aligned revisits."""
    return {
        "topo_visual": _spec(gate_mode="both"),
        "topo_visual_cosine": _spec(gate_mode="both", visual_model="cosine"),
        **{f"{e}_visual": _spec(encoder=e, gate_mode="both", visual_model=RETRIEVAL_SCORE[e])
           for e in COMPARISON_ENCODERS},
    }


def visual_likelihood(spec: dict, seq_name: str, split: str) -> VisualLikelihood | None:
    """Development traversals use the fit that left them out; unseen ones the fit on all seven."""
    if spec["gate_mode"] != "both" or spec["association"] != "disk":
        return None
    fits = json.loads(VISUAL_FILE.read_text())[f"{spec['encoder']}:{spec['visual_model']}"]
    fit = fits["leave_out"][seq_name] if split == "dev" else fits["all"]
    return VisualLikelihood(fit["intercept"], fit["slope"])


class _RecordingMonitor:
    """Records the lambda_2 series of the connectivity monitor."""

    def __init__(self, inner):
        self.inner, self.series = inner, []

    def update(self, descriptor):
        value = self.inner.update(descriptor)
        if value is not None:
            self.series.append(value)
        return value


def scored_decisions(graph, node_gt: dict[int, tuple], tolerance: float) -> list[tuple[float, bool]]:
    """(probability, proposed node within tolerance) of every query that had candidates."""
    return [
        (e.probability, bool(np.hypot(node_gt[e.matched_node_id][0] - node_gt[e.query_node_id][0],
                                      node_gt[e.matched_node_id][1] - node_gt[e.query_node_id][1]) <= tolerance))
        for e in graph.loop_events if e.probability is not None
    ]


def run_one(job: tuple[str, str, dict, str, str]) -> dict:
    seq_name, variant, spec, out, descriptor_root = job
    seq = ALL[seq_name]
    target = Path(out) / "replay" / seq_name / variant
    result_path = target / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    try:
        mapper_options = dict(spec["mapper"])
        if mapper_options["valley_mode"] == "fixed":
            mapper_options["valley_delta"] = json.loads(NODE_SENSITIVITY_FILE.read_text())["fixed_delta"]
        gate = gates()[spec["gate"]]
        stream = load_stream(seq)
        covariances = cumulative_covariances(gate.model, gate.params, stream.odometry)
        descriptors, descriptor_frames = load_descriptors(Path(descriptor_root), seq_name, spec["encoder"])
        names = [os.path.basename(p) for p in stream.image_paths]
        if names != list(descriptor_frames):
            raise AssertionError("descriptor frames do not match the frame stream")

        mapper = TopologicalMapper(
            frame_id=seq_name, gate_mode=spec["gate_mode"], visual_model=spec["visual_model"],
            uncertainty_model=gate.model, association=spec["association"], place_radius=PLACE_RADIUS_M,
            visual_likelihood=visual_likelihood(spec, seq_name, seq.split), **mapper_options,
        )
        monitor = _RecordingMonitor(mapper._monitor)
        mapper._monitor = monitor
        for k, image_path in enumerate(stream.image_paths):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR) if spec["images"] else DUMMY_IMAGE
            mapper.process_frame(image, descriptors[k].astype(np.float32), tuple(stream.odometry[k]), covariances[k],
                                 tuple(stream.ground_truth[k]), stream.labels[k])
        mapper.finalize_nodes()

        target.mkdir(parents=True, exist_ok=True)
        graph_path, gt_path = target / "graph.pkl", target / "node_gt.json"
        mapper.graph.save(str(graph_path))
        node_gt = {n.node_id: tuple(n.gt_pose) for n in mapper.graph.nodes.values()}
        gt_path.write_text(json.dumps({str(k): v for k, v in node_gt.items()}))
        np.save(target / "lambda2.npy", np.asarray(monitor.series))
        rows = evaluate_case(EvaluationCase(seq_name, str(graph_path), str(gt_path), str(seq.trajectory_gt)), TOLERANCES)
        trajectory = load_ground_truth_xy(str(seq.trajectory_gt))
        rmse = graph_metrics(mapper.graph, trajectory,
                             node_gt_xy={k: np.asarray(v[:2]) for k, v in node_gt.items()}).node_placement_rmse
        result = {
            "sequence": seq_name, "variant": variant, "split": seq.split, "domain": seq.domain, "spec": spec,
            "valley_delta": mapper_options["valley_delta"], "frames": len(names), "nodes": len(mapper.graph.nodes),
            "node_placement_rmse_m": float(rmse),
            "metrics": [{k: r[k] for k in ("tolerance_m", "n_nodes", "n_loop_edges", "loop_opportunities", "loop_tp",
                                            "loop_fp", "loop_fn", "loop_precision", "loop_recall", "loop_f1",
                                            "coverage_at_tolerance")} for r in rows],
            "breakdown": {str(t): breakdown(mapper.graph, node_gt, t) for t in TOLERANCES},
            "loops": [[e.query_node_id, e.matched_node_id] for e in mapper.graph.loop_events if e.accepted],
            "scored": {str(t): scored_decisions(mapper.graph, node_gt, t) for t in TOLERANCES},
        }
        mapper.close()
        result_path.write_text(json.dumps(result, indent=1))
        two = next(m for m in result["metrics"] if m["tolerance_m"] == 2.0)
        print(f"{seq_name:30s} {variant:34s} nodes={result['nodes']:3d} "
              f"2m TP/FP/FN={two['loop_tp']}/{two['loop_fp']}/{two['loop_fn']}", flush=True)
        return result
    except Exception as error:
        return {"sequence": seq_name, "variant": variant, "error": repr(error), "trace": traceback.format_exc()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--descriptors", type=Path, required=True, help="AnyLoc runner output directory")
    parser.add_argument("--split", choices=["dev", "heldout"], required=True)
    parser.add_argument("--visual", action="store_true", help="run the visual variants")
    parser.add_argument("--variants", nargs="*")
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("--sequences", nargs="*")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    seqs = args.sequences or [s.name for s in (DEV if args.split == "dev" else HELDOUT)]
    specs = visual_variants(args.split) if args.visual else variants(args.split)
    chosen = [v for v in (args.variants or specs) if v not in args.exclude]
    jobs = [(s, v, specs[v], str(args.out), str(args.descriptors)) for s in seqs for v in chosen]
    jobs.sort(key=lambda j: ("vlad" in j[1], j[1]))
    with ProcessPoolExecutor(args.jobs) as pool:
        results = list(pool.map(run_one, jobs))
    errors = [r for r in results if "error" in r]
    for e in errors:
        print("ERROR", e["sequence"], e["variant"], e["error"], e["trace"], sep="\n")
    if errors:
        raise SystemExit(f"{len(errors)} jobs failed")


if __name__ == "__main__":
    main()
