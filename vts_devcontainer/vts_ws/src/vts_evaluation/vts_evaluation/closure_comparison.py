"""Fast chronological loop-decision study on verified, frozen DINOv3 nodes.

No images, descriptor extraction, ROS playback, segmentation or language runs.
This isolates association changes; it cannot measure end-to-end runtime gains.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path

import numpy as np

from vts_core import directional, mapper, matching, metrics, topo_graph
from vts_core.artifacts import atomic_json, file_digest, identity
from vts_core.mapper import TopologicalMapper, VISUAL_MODELS
from vts_core.metrics import graph_metrics
from vts_core.topo_graph import TopoGraph
from vts_evaluation.summarize_experiments import ENVIRONMENTS


PROFILES = {
    "legacy": {"sequence_policy": "legacy", "require_visual_verification": False},
    "self_only": {"sequence_policy": "self_only", "require_visual_verification": False},
    "verification_only": {"sequence_policy": "legacy", "require_visual_verification": True},
    "both": {"sequence_policy": "self_only", "require_visual_verification": True},
}


def load_verified(directory: Path) -> tuple[TopoGraph, dict]:
    manifest = json.loads((directory / "run_manifest.json").read_text())
    completion = json.loads((directory / "completion.json").read_text())
    payload = dict(manifest)
    fingerprint = payload.pop("fingerprint")
    if identity(payload) != fingerprint or completion["fingerprint"] != fingerprint:
        raise ValueError(f"Invalid source manifest: {directory}")
    graph_path = directory / "compact_graph.pkl"
    if file_digest(graph_path) != completion["artifacts"][graph_path.name]:
        raise ValueError(f"Invalid source graph hash: {directory}")
    if completion["encoder_identity"] != manifest["encoder"]:
        raise ValueError(f"Invalid source encoder identity: {directory}")
    if manifest["encoder"]["model"] != "facebook/dinov3-vits16-pretrain-lvd1689m":
        raise ValueError("This focused study requires the completed DINOv3 node set")
    graph = TopoGraph.load(str(graph_path))
    if sorted(graph.nodes) != list(range(len(graph.nodes))):
        raise ValueError("Replay requires contiguous chronological observation IDs")
    if any(node.gt_pose is None for node in graph.nodes.values()):
        raise ValueError("Representative-frame ground truth is required for evaluation")
    return graph, {
        "run_fingerprint": fingerprint,
        "compact_graph_sha256": completion["artifacts"][graph_path.name],
        "parameters": manifest["parameters"]["mapping"],
    }


def replay(source: TopoGraph, visual_model: str, **parameters: object) -> TopoGraph:
    engine = TopologicalMapper(visual_model=visual_model, **parameters)
    try:
        for node_id in sorted(source.nodes):
            # No source loop edges, future nodes, ground truth or room labels
            # enter the association engine. Descriptors/poses are immutable.
            node = replace(source.nodes[node_id], views=[], view_embeddings=None,
                           gt_pose=None, room_label=None)
            engine.observe_node(node)
        engine.finish_revisits()
        return engine.graph
    finally:
        engine.close()


def score(graph: TopoGraph, source: TopoGraph, tolerance: float) -> dict:
    gt = {i: np.asarray(node.gt_pose[:2]) for i, node in source.nodes.items()}
    result = graph_metrics(graph, np.asarray(list(gt.values())), node_gt_xy=gt,
                           loop_tolerance=tolerance)
    # Do not report trajectory coverage from representative nodes alone.
    return {
        "tolerance_m": tolerance,
        "tp": result.loop_true_positives, "fp": result.loop_false_positives,
        "fn": result.loop_false_negatives, "precision": result.loop_precision,
        "recall": result.loop_recall, "f1": result.loop_f1,
    }


def aggregate(rows: list[dict]) -> dict:
    tp, fp, fn = (sum(row[key] for row in rows) for key in ("tp", "fp", "fn"))
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("output/dinov3_revisits"))
    parser.add_argument("--output", type=Path, default=Path("output/closure_comparison_v2"))
    parser.add_argument("--environments", nargs="+", choices=ENVIRONMENTS, default=ENVIRONMENTS)
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(PROFILES))
    parser.add_argument("--verification-threshold", type=float, default=0.0)
    parser.add_argument("--defer-nodes", type=int, default=2)
    parser.add_argument("--fixed-concentration", type=float, default=384.0)
    args = parser.parse_args()
    inputs = {name: load_verified(args.source / name) for name in args.environments}
    # Recorded cosine is a genuine pre-fix reference, not a reimplementation of it.
    old_cosine = {name: load_verified(args.source / (name + "_cosine"))
                  for name in args.environments}
    for name, (source, metadata) in inputs.items():
        if metadata["parameters"]["visual_model"] != "vmf" or (
            old_cosine[name][1]["parameters"]["visual_model"] != "cosine"
        ):
            raise ValueError("Expected recorded vMF and cosine reference runs")
        old = old_cosine[name][0]
        if sorted(source.nodes) != sorted(old.nodes):
            raise ValueError("Recorded cosine reference uses different node boundaries")
        for i, node in source.nodes.items():
            other = old.nodes[i]
            for field in ("pose", "pose_covariance", "extent_covariance", "visual_features", "view_features",
                          "visual_concentration", "gt_pose"):
                if not np.array_equal(getattr(node, field), getattr(other, field)):
                    raise ValueError(f"Reference differs in {name}/{i}/{field}")
    parameters = {
        "verification_threshold": args.verification_threshold,
        "defer_nodes": args.defer_nodes, "fixed_concentration": args.fixed_concentration,
    }
    protocol = {
        "study": "frozen_node_chronological_closure_replay_v2",
        "source": str(args.source.resolve()),
        "inputs": {name: meta for name, (_, meta) in inputs.items()},
        "old_cosine_inputs": {name: meta for name, (_, meta) in old_cosine.items()},
        "parameters": parameters, "models": list(VISUAL_MODELS),
        "profiles": {name: PROFILES[name] for name in args.profiles},
        "sources": {Path(module.__file__).name: file_digest(module.__file__)
                    for module in (directional, mapper, matching, metrics, topo_graph)},
        "runner_sha256": file_digest(__file__),
        "packages": {name: version(name) for name in ("numpy", "scipy")},
        "tolerances_m": [0.5, 1.0, 2.0, 3.0],
    }
    fingerprint = identity(protocol)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "study_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["fingerprint"] != fingerprint:
        raise ValueError("Output belongs to a different study; choose a new --output directory")
    atomic_json(manifest_path, {"fingerprint": fingerprint, **protocol})
    comparisons = [(f"{profile}/{model}", model, PROFILES[profile])
                   for profile in args.profiles for model in VISUAL_MODELS]
    variants_to_run = [("recorded_vmf", None, {}), ("recorded_cosine", None, {}), *comparisons]
    results = {}
    for name, (source, metadata) in inputs.items():
        variants = {}
        for label, model, profile in variants_to_run:
            started = time.perf_counter()
            graph = (source if label == "recorded_vmf" else old_cosine[name][0]
                     if label == "recorded_cosine" else replay(
                         source, model, visual_outlier_k=metadata["parameters"]["visual_outlier_k"],
                         **parameters, **profile))
            duration = time.perf_counter() - started
            variants[label] = {
                "nodes": len(graph.nodes), "replay_seconds": duration,
                "metrics": [score(graph, source, radius) for radius in protocol["tolerances_m"]],
                "reasons": dict(Counter(event.reason for event in graph.loop_events)),
                "decisions": [asdict(event) for event in graph.loop_events],
                "confirmation_delays_nodes": [event.decision_node_id - event.query_node_id
                    for event in graph.loop_events if event.reason == "accepted_deferred"],
            }
            if model is not None:
                graph.save(str(args.output / f"{name}_{label.replace('/', '_')}.pkl"))
            if label in {"legacy/cosine", "legacy/vmf"}:
                reference = old_cosine[name][0] if model == "cosine" else source
                if graph.edge_types != reference.edge_types:
                    raise ValueError(f"Legacy replay does not reproduce recorded edges: {name}/{model}")
        results[name] = variants
        print(f"Compared five scorers across {len(args.profiles)} policies: {name}", flush=True)
    summary = {}
    for model, _, _ in variants_to_run:
        summary[model] = [
            {"tolerance_m": radius, **aggregate([
                variants[model]["metrics"][index] for variants in results.values()])}
            for index, radius in enumerate(protocol["tolerances_m"])
        ]
    atomic_json(args.output / "summary.json", {
        "fingerprint": fingerprint, "protocol": protocol, "aggregate": summary, "results": results,
    })
    print("\n2 m micro-aggregate: model                 TP  FP  FN      P      R     F1")
    for model, rows in summary.items():
        row = rows[2]
        print(f"{model:38s} {row['tp']:3d} {row['fp']:3d} {row['fn']:3d} "
              f"{row['precision']:.3f}  {row['recall']:.3f}  {row['f1']:.3f}")
    print(f"\nResults: {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
