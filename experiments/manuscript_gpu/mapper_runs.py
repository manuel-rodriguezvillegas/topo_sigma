"""Run every mapper configuration of the manuscript on GPU descriptors and score it.

Descriptors come from the AnyLoc runner (``$ANYLOC_OUT/descriptors``). Each job writes
``$OUT/replay/<sequence>/<variant>/result.json``; finished jobs are skipped on a rerun.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from stream import (ALL, DEV, HELDOUT, RESULTS, Sequence, Stream, cumulative_covariances, debiased_stream,
                    evaluated_trajectory, gates, heading_rate, load_stream, readable_test)
from vts_core.matching import VisualLikelihood
from common import breakdown, load_descriptors
from replay import MANUSCRIPT_LOOP_EXCLUSION_NODES
from vts_core.mapper import TopologicalMapper
from vts_core.metrics import graph_metrics
from vts_evaluation.tolerance_sweep import result_row

TOLERANCES = [0.5, 1.0, 2.0, 3.0]
MAIN_GATE = "propagated_calibrated"
DESCRIPTOR = "dinov3_s"
MAPPER_OPTIONS = dict(window_size=30, valley_k=2.0, valley_mode="adaptive", valley_delta=0.1,
                      visual_outlier_k=2.0, naive_threshold=0.7)
NODE_SENSITIVITY_FILE = RESULTS / "node_sensitivity.json"
VISUAL_FILE = RESULTS / "visual_likelihood.json"
PLACE_RADIUS_M = 2.0
DUMMY_IMAGE = np.zeros((1, 1, 3), dtype=np.uint8)
DESCRIPTOR_ABLATIONS = ("dinov2_s", "anyloc_vlad_g", "anyloc_gem_g")
VISUAL_SCORES = {"dinov3_s": ("cosine", "vmf"), **{e: ("cosine",) for e in DESCRIPTOR_ABLATIONS}}
DEBIASED = "propagated_debiased"
DEBIASED_ABLATIONS = ("anyloc_vlad_g", "anyloc_gem_g", "dinov2_s", "geometry_only", "vmf", "boundary_cut",
                      "gate_propagated")
TEST_VARIANTS = ("topo", "topo_debiased", "anyloc_vlad_g", "anyloc_gem_g", "previous_system", "geometry_only", "vmf",
                 "isotropic_default", "isotropic_calibrated", "gate_propagated", "boundary_cut", "dinov2_s",
                 *(f"{name}_debiased" for name in DEBIASED_ABLATIONS))
TEST_REPRODUCTION_VARIANT = "previous_system_two_node_exclusion"


def _spec(gate=MAIN_GATE, encoder=DESCRIPTOR, association="disk", gate_mode="both", visual_model="cosine",
          images=False, **mapper) -> dict:
    return dict(gate=gate, encoder=encoder, association=association, gate_mode=gate_mode,
                visual_model=visual_model, images=images, mapper={**MAPPER_OPTIONS, **mapper})


def geometric_source(encoder: str) -> str:
    """Variant whose graphs (same nodes, no visual evidence) calibrate the visual ratio of ``encoder``."""
    return "geometry_only" if encoder == DESCRIPTOR else f"{encoder}_geometry"


def variants(split: str) -> dict[str, dict]:
    """Variants that need no visual likelihood ratio; they run first."""
    v = {
        "geometry_only": _spec(gate_mode="geometric"),
        "gate_propagated": _spec(association="gate"),
        "previous_system": _spec(gate="isotropic_default", encoder="dinov2_s", association="gate",
                                 visual_model="vmf"),
        "threshold": _spec(association="gate", gate_mode="threshold"),
        "visual_only": _spec(association="gate", gate_mode="visual"),
    }
    for encoder in DESCRIPTOR_ABLATIONS:
        v[geometric_source(encoder)] = _spec(encoder=encoder, gate_mode="geometric")
    v["geometry_only_debiased"] = _spec(gate=DEBIASED, gate_mode="geometric")
    v["gate_propagated_debiased"] = _spec(gate=DEBIASED, association="gate")
    if split == "test":
        v[TEST_REPRODUCTION_VARIANT] = _spec(gate="isotropic_default", encoder="dinov2_s", association="gate",
                                             visual_model="vmf",
                                             loop_exclusion_nodes=MANUSCRIPT_LOOP_EXCLUSION_NODES)
    return v


def visual_variants(split: str) -> dict[str, dict]:
    """The method and its ablations; they need ``VISUAL_FILE``."""
    v = {
        "topo": _spec(images=split == "dev"),
        "topo_debiased": _spec(gate=DEBIASED),
        "vmf": _spec(visual_model="vmf"),
        "isotropic_default": _spec(gate="isotropic_default"),
        "isotropic_calibrated": _spec(gate="isotropic_calibrated"),
        "boundary_cut": _spec(boundary="cut"),
        **{encoder: _spec(encoder=encoder) for encoder in DESCRIPTOR_ABLATIONS},
        **{f"{encoder}_debiased": _spec(encoder=encoder, gate=DEBIASED) for encoder in DESCRIPTOR_ABLATIONS},
        "vmf_debiased": _spec(visual_model="vmf", gate=DEBIASED),
        "boundary_cut_debiased": _spec(boundary="cut", gate=DEBIASED),
    }
    if split == "dev":
        for k in (1.0, 1.5, 2.5, 3.0):
            v[f"valley_k{k}"] = _spec(valley_k=k)
        v["valley_fixed"] = _spec(valley_mode="fixed")
    return v


def visual_likelihood(spec: dict, seq_name: str, split: str) -> dict | None:
    """Design traversals use the fit that left them out; test traversals the fit on all design traversals."""
    if spec["gate_mode"] != "both" or spec["association"] != "disk":
        return None
    fits = json.loads(VISUAL_FILE.read_text())[f"{spec['encoder']}:{spec['visual_model']}"]
    return fits["all"] if split == "test" else fits["leave_out"][seq_name]


def resolved_inputs(spec: dict, seq_name: str, split: str) -> dict:
    """Everything a run depends on, including values read from calibration files."""
    gate = gates()[spec["gate"]]
    rate = heading_rate(ALL[seq_name]) if gate.debiased else 0.0
    delta = spec["mapper"]["valley_delta"]
    if spec["mapper"]["valley_mode"] == "fixed":
        delta = json.loads(NODE_SENSITIVITY_FILE.read_text())["fixed_delta"]
    fit = visual_likelihood(spec, seq_name, split)
    return {"spec": spec, "uncertainty_model": gate.model, "alpha": gate.params.as_array().tolist(),
            "heading_rate_rad_per_m": rate,
            "valley_delta": delta,
            "visual_likelihood": None if fit is None else {k: fit[k] for k in ("intercept", "slope")}}


def scored_decisions(graph, node_gt: dict[int, tuple], tolerance: float) -> list[tuple[float, bool]]:
    """(probability, proposed node within tolerance) of every query that had candidates."""
    return [
        (e.probability, bool(np.hypot(node_gt[e.matched_node_id][0] - node_gt[e.query_node_id][0],
                                      node_gt[e.matched_node_id][1] - node_gt[e.query_node_id][1]) <= tolerance))
        for e in graph.loop_events if e.probability is not None
    ]


@dataclass
class Replay:
    """A mapper ready to consume ``stream``, with the odometry covariances and descriptors of each frame."""

    stream: Stream
    covariances: np.ndarray
    descriptors: np.ndarray
    mapper: TopologicalMapper


def run_inputs(seq: Sequence, spec: dict, stream: Stream) -> dict:
    return {**resolved_inputs(spec, seq.name, seq.split), "first_frame": stream.first}


def prepare_replay(seq: Sequence, spec: dict, inputs: dict, stream: Stream, descriptor_root: Path) -> Replay:
    gate = gates()[spec["gate"]]
    if gate.debiased:
        stream = debiased_stream(seq, stream)
    descriptors, descriptor_frames = load_descriptors(descriptor_root, seq.name, spec["encoder"])
    descriptors, descriptor_frames = descriptors[stream.first:], descriptor_frames[stream.first:]
    if [os.path.basename(p) for p in stream.image_paths] != list(descriptor_frames):
        raise AssertionError("descriptor frames do not match the frame stream")
    mapper = TopologicalMapper(
        frame_id=seq.name, gate_mode=spec["gate_mode"], visual_model=spec["visual_model"],
        uncertainty_model=gate.model, association=spec["association"], place_radius=PLACE_RADIUS_M,
        visual_likelihood=None if inputs["visual_likelihood"] is None
        else VisualLikelihood(**inputs["visual_likelihood"]),
        **{**spec["mapper"], "valley_delta": inputs["valley_delta"]},
    )
    return Replay(stream, cumulative_covariances(gate.model, gate.params, stream.odometry), descriptors, mapper)


def replay_frames(replay: Replay, images: bool) -> Iterator[int]:
    """Feed every frame to the mapper, yielding its index after each one, then flush the last node."""
    stream = replay.stream
    for k, image_path in enumerate(stream.image_paths):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR) if images else DUMMY_IMAGE
        replay.mapper.process_frame(image, replay.descriptors[k].astype(np.float32), tuple(stream.odometry[k]),
                                    replay.covariances[k], tuple(stream.ground_truth[k]), stream.labels[k])
        yield k
    replay.mapper.finalize_nodes()


def run_one(job: tuple[str, str, dict, str, str]) -> dict:
    seq_name, variant, spec, out, descriptor_root = job
    seq = ALL[seq_name]
    target = Path(out) / "replay" / seq_name / variant
    result_path = target / "result.json"
    try:
        stream = load_stream(seq)
        inputs = run_inputs(seq, spec, stream)
        if result_path.exists():
            previous = json.loads(result_path.read_text())
            if previous.get("inputs") == inputs:
                return previous
        replay = prepare_replay(seq, spec, inputs, stream, Path(descriptor_root))
        for _ in replay_frames(replay, spec["images"]):
            pass
        mapper = replay.mapper

        target.mkdir(parents=True, exist_ok=True)
        graph_path, gt_path = target / "graph.pkl", target / "node_gt.json"
        mapper.graph.save(str(graph_path))
        node_gt = {n.node_id: tuple(n.gt_pose) for n in mapper.graph.nodes.values()}
        gt_path.write_text(json.dumps({str(k): v for k, v in node_gt.items()}))
        np.save(target / "lambda2.npy", np.asarray(mapper._monitor.eigenvalues))
        np.save(target / "frames_after_cut.npy", np.asarray(mapper._monitor.frames_after_cut))
        trajectory = evaluated_trajectory(seq)
        node_gt_xy = {k: np.asarray(v[:2]) for k, v in node_gt.items()}
        metrics = {t: graph_metrics(mapper.graph, trajectory, node_gt_xy=node_gt_xy, spatial_tolerance=t,
                                    loop_tolerance=t) for t in TOLERANCES}
        rows = [result_row(seq_name, t, m) for t, m in metrics.items()]
        rmse = metrics[TOLERANCES[0]].node_placement_rmse
        result = {
            "sequence": seq_name, "variant": variant, "split": seq.split, "domain": seq.domain, "route": seq.route,
            "inputs": inputs, "frames": len(stream.image_paths), "nodes": len(mapper.graph.nodes),
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
    parser.add_argument("--split", choices=["dev", "heldout", "test"], required=True)
    parser.add_argument("--visual", action="store_true", help="run the visual variants (design splits)")
    parser.add_argument("--variants", nargs="*")
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("--sequences", nargs="*")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.split == "test":
        split_seqs = readable_test()
        every = {**variants("test"), **visual_variants("test")}
        specs = {v: every[v] for v in (*TEST_VARIANTS, TEST_REPRODUCTION_VARIANT)}
    else:
        split_seqs = DEV if args.split == "dev" else HELDOUT
        specs = visual_variants(args.split) if args.visual else variants(args.split)
    seqs = args.sequences or [s.name for s in split_seqs]
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
