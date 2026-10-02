"""C1: regimes in which appearance evidence can matter for loop closure.

Three offline stages share one frozen frame store:

``frames``
    Decode each traversal once and store DINO descriptors, ground truth, room
    labels and the *raw* recorded odometry samples. Main traversals must be
    descriptor-cache hits; extra COLD sessions may run inference only when
    ``--allow-inference`` is given.
``drift``
    Controlled odometry degradation. Raw odometry increments are corrupted at
    their native rate, re-integrated, and the uncertainty budget is recomputed
    with the *unchanged* coefficients, as a deployed system would do without
    knowing that its odometry got worse. Segmentation depends only on
    descriptors, so node boundaries and revisit opportunities are identical at
    every drift level; only association changes. Operating curves come from
    sweeping existing mapper knobs, not from new decision rules.
``sessions``
    Cross-session association between two COLD traversals of the same lab.
    Their odometry frames are unrelated, so geometry is uninformative and only
    appearance can associate places. COLD ground truth shares one map frame
    per lab, which is checked when the stage runs.

Ground truth and room labels never enter segmentation or association; they are
attached after decisions are made and used only for scoring.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, replace
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np
import yaml

from vts_core import directional, mapper as mapper_module, matching, metrics, motion, node_detection
from vts_core.artifacts import atomic_json, file_digest, identity
from vts_core.directional import fit_vmf_resultant, vmf_log_overlap_ratio, vmf_symmetric_kl
from vts_core.mapper import TopologicalMapper
from vts_core.metrics import graph_metrics
from vts_core.motion import OdometryUncertaintyParams, OdometryUncertaintyTracker, normalize_angle
from vts_core.node_detection import AdaptiveValleyDetector, ConnectivityMonitor
from vts_core.topo_graph import TopoGraph, TopoNode
from vts_players.cid_sims_data import (
    PlanarPoseStream,
    load_ground_truth,
    load_wheel_odometry,
    timestamped_color_images,
)
from vts_players.cold_odometry import ColdOdometry, load_cold_odometry

WORKSPACE = Path(__file__).resolve().parents[3]
CONFIG_DIR = WORKSPACE / "src/vts_bringup/config"

# Session name -> (configuration file, sequence override or None).
MAIN_SESSIONS: dict[str, tuple[str, str | None]] = {
    "freiburg_a": ("cold_freiburg_a", None),
    "freiburg_ext": ("cold_freiburg_ext", None),
    "saarbruecken_a": ("cold_saarbruecken_a", None),
    "saarbruecken_ext": ("cold_saarbruecken_ext", None),
    "cid_sims_apartment1_1": ("cid_sims_apartment1_1", None),
    "cid_sims_apartment2_1": ("cid_sims_apartment2_1", None),
    "cid_sims_apartment3_1": ("cid_sims_apartment3_1", None),
}
# Environments never used during method development. Thresholds are chosen on
# MAIN_SESSIONS and applied here unchanged.
HELDOUT_SESSIONS: dict[str, tuple[str, str | None]] = {
    "ljubljana_a_seq1": ("cold_ljubljana_a_seq1", None),
    "ljubljana_a_seq2": ("cold_ljubljana_a_seq2", None),
}
# Confirmatory test recordings (docs/CONFIRMATORY_PROTOCOL_2026-10-02.md): for each
# lab, part and route, the first clean, previously unused recording.
CONFIRM_SESSIONS: dict[str, tuple[str, str]] = {
    "confirm_freiburg_a_seq1": ("cold_freiburg_a", "cold-freiburg_part_a_seq1_cloudy2"),
    "confirm_freiburg_a_seq2": ("cold_freiburg_a", "cold-freiburg_part_a_seq2_cloudy2"),
    "confirm_freiburg_b_seq3": ("cold_freiburg_ext", "cold-freiburg_part_b_seq3_cloudy2"),
    "confirm_saarbruecken_a_seq1": ("cold_saarbruecken_a", "cold-saarbruecken_part_a_seq1_cloudy1"),
    "confirm_saarbruecken_a_seq2": ("cold_saarbruecken_a", "cold-saarbruecken_part_a_seq2_cloudy2"),
    "confirm_saarbruecken_b_seq3": ("cold_saarbruecken_ext", "cold-saarbruecken_part_b_seq3_cloudy1"),
    "confirm_saarbruecken_b_seq4": ("cold_saarbruecken_ext", "cold-saarbruecken_part_b_seq4_cloudy2"),
    "confirm_ljubljana_a_seq1": ("cold_ljubljana_a_seq1", "cold-ljubljana_part_a_seq1_night3"),
    "confirm_ljubljana_a_seq2": ("cold_ljubljana_a_seq2", "cold-ljubljana_part_a_seq2_cloudy2"),
}
# Second confirmatory test (docs/CONFIRMATORY_PROTOCOL_CID_2026-10-02.md): the
# second traversal of each CID-SIMS apartment, same adapter configuration.
CID_CONFIRM_SESSIONS: dict[str, tuple[str, str]] = {
    "confirm_cid_apartment1_2": ("cid_sims_apartment1_2", "apartment1_2"),
    "confirm_cid_apartment2_2": ("cid_sims_apartment2_2", "apartment2_2"),
    "confirm_cid_apartment3_2": ("cid_sims_apartment3_2", "apartment3_2"),
}
# Additional COLD traversals of the same labs. The map session is the main
# traversal; the condition names the query's illumination relative to it.
EXTRA_SESSIONS: dict[str, tuple[str, str]] = {
    "freiburg_a@cloudy1": ("cold_freiburg_a", "cold-freiburg_part_a_seq2_cloudy1"),
    "freiburg_a@sunny1": ("cold_freiburg_a", "cold-freiburg_part_a_seq2_sunny1"),
    "freiburg_ext@cloudy1": ("cold_freiburg_ext", "cold-freiburg_part_b_seq3_cloudy1"),
    "saarbruecken_a@cloudy1": ("cold_saarbruecken_a", "cold-saarbruecken_part_a_seq2_cloudy1"),
    "saarbruecken_ext@cloudy1": ("cold_saarbruecken_ext", "cold-saarbruecken_part_b_seq4_cloudy1"),
    "saarbruecken_ext@night1": ("cold_saarbruecken_ext", "cold-saarbruecken_part_b_seq4_night1"),
    "saarbruecken_ext@sunny2": ("cold_saarbruecken_ext", "cold-saarbruecken_part_b_seq4_sunny2"),
}
SESSION_PAIRS: dict[str, tuple[str, str, str]] = {
    # pair -> (map session, query session, illumination relation)
    "freiburg_a:night1->cloudy1": ("freiburg_a", "freiburg_a@cloudy1", "different"),
    "freiburg_a:night1->sunny1": ("freiburg_a", "freiburg_a@sunny1", "different"),
    "freiburg_ext:sunny1->cloudy1": ("freiburg_ext", "freiburg_ext@cloudy1", "different"),
    "saarbruecken_a:night2->cloudy1": ("saarbruecken_a", "saarbruecken_a@cloudy1", "different"),
    "saarbruecken_ext:sunny1->cloudy1": ("saarbruecken_ext", "saarbruecken_ext@cloudy1", "different"),
    "saarbruecken_ext:sunny1->night1": ("saarbruecken_ext", "saarbruecken_ext@night1", "different"),
    "saarbruecken_ext:sunny1->sunny2": ("saarbruecken_ext", "saarbruecken_ext@sunny2", "same"),
}
TOLERANCES = (0.5, 1.0, 2.0, 3.0)


@dataclass(frozen=True)
class DriftLevel:
    """Odometry corruption, applied per raw odometry increment.

    ``heading_walk``: heading random walk, rad per sqrt(metre) travelled.
    ``scale_sigma``: std of a per-run translational scale bias (wheel radius).
    ``turn_sigma``: std of a per-run rotational scale bias (wheel base / gyro).
    """

    name: str
    heading_walk: float
    scale_sigma: float
    turn_sigma: float


DRIFT_LEVELS = (
    DriftLevel("L0", 0.0, 0.0, 0.0),
    DriftLevel("L1", 0.01, 0.005, 0.005),
    DriftLevel("L2", 0.02, 0.01, 0.01),
    DriftLevel("L3", 0.05, 0.02, 0.02),
    DriftLevel("L4", 0.10, 0.03, 0.03),
    DriftLevel("L5", 0.20, 0.05, 0.05),
)

# Methods use only existing mapper modes. Each has one swept knob.
BASE = {"sequence_policy": "self_only", "require_visual_verification": False}
METHODS: dict[str, dict] = {
    "geometry+cosine": {**BASE, "gate_mode": "both", "visual_model": "cosine"},
    "geometry+vmf": {**BASE, "gate_mode": "both", "visual_model": "vmf"},
    "geometry_only": {**BASE, "gate_mode": "geometric", "visual_model": "cosine"},
    "visual_only+cosine": {**BASE, "gate_mode": "visual", "visual_model": "cosine"},
    "visual_only+vmf": {**BASE, "gate_mode": "visual", "visual_model": "vmf"},
}
VISUAL_K_SWEEP = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
GATE_SCALE_SWEEP = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
DEFAULT_KNOB = {"visual_outlier_k": 2.0, "geometric_gate_scale": 1.0}


def sweep_for(method: str) -> tuple[str, tuple[float, ...]]:
    if METHODS[method]["gate_mode"] == "geometric":
        return "geometric_gate_scale", GATE_SCALE_SWEEP
    return "visual_outlier_k", VISUAL_K_SWEEP


# --------------------------------------------------------------------------- #
# Frame store
# --------------------------------------------------------------------------- #
def session_config(session: str) -> tuple[dict, bool, str]:
    config_name, sequence = {**MAIN_SESSIONS, **HELDOUT_SESSIONS, **CONFIRM_SESSIONS, **CID_CONFIRM_SESSIONS,
                             **EXTRA_SESSIONS}[session]
    config = yaml.safe_load((CONFIG_DIR / f"{config_name}.yaml").read_text())
    cid = config_name.startswith("cid_sims_")
    return config, cid, sequence or config["player"]["sequences"][0]


def raw_odometry(config: dict, cid: bool, sequence: str) -> ColdOdometry | PlanarPoseStream:
    player = config["player"]
    dataset = Path(player["dataset_root"]) / sequence
    params = OdometryUncertaintyParams(*player["odometry_uncertainty"])
    if cid:
        return load_wheel_odometry(str(dataset / player["odometry_file"]), params)
    return load_cold_odometry(str(dataset / player["odometry_subpath"]), params)


def camera_frames(config: dict, cid: bool, sequence: str):
    """Yield (image_path, timestamp, gt_pose, label) exactly as the players select them."""
    from vts_players.cold_player_node import load_room_labels, parse_cold_filename

    player = config["player"]
    dataset = Path(player["dataset_root"]) / sequence
    odometry = raw_odometry(config, cid, sequence)
    if cid:
        gt = load_ground_truth(str(dataset / player["ground_truth_file"]))
        images = timestamped_color_images(str(dataset / player["color_subdir"]))
        for timestamp, image_path in images[:: player["frame_stride"]]:
            try:
                gt_pose, _ = gt.at(timestamp, player["max_ground_truth_gap_s"])
                odometry.at(timestamp, player["max_odometry_gap_s"])
            except ValueError:
                continue
            yield image_path, timestamp, gt_pose, None
    else:
        labels = load_room_labels(str(dataset / player["labels_subpath"]))
        for image_path in sorted((dataset / player["images_subdir"]).iterdir()):
            parsed = parse_cold_filename(image_path.name)
            if parsed is None:
                continue
            timestamp, gt_pose = parsed
            odometry.at(timestamp, max_gap_s=player["max_odometry_gap_s"])
            yield str(image_path), timestamp, gt_pose, labels.get(image_path.name)


def build_frame_store(session: str, root: Path, allow_inference: bool) -> dict:
    from vts_core.features import build_extractor

    config, cid, sequence = session_config(session)
    mapping = config["mapping"]
    extractor = build_extractor(
        mapping["dino_model"], device="cpu", cache_dir=str(WORKSPACE / mapping["descriptor_cache_dir"])
    )
    descriptors, timestamps, gts, labels, paths = [], [], [], [], []
    started = time.perf_counter()
    for count, (path, timestamp, gt_pose, label) in enumerate(camera_frames(config, cid, sequence), 1):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read image: {path}")
        descriptors.append(extractor.extract(image))
        if extractor.misses and not allow_inference:
            raise RuntimeError(f"{session}: descriptor cache miss; rerun with --allow-inference")
        timestamps.append(timestamp)
        gts.append(gt_pose)
        labels.append(label)
        paths.append(str(Path(path).relative_to(config["player"]["dataset_root"])))
        if count % 500 == 0:
            rate = count / (time.perf_counter() - started)
            print(f"{session}: {count} frames ({rate:.1f}/s, misses={extractor.misses})", flush=True)
    odometry = raw_odometry(config, cid, sequence)
    output = root / "frames" / f"{session}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        descriptors=np.stack(descriptors).astype(np.float32),
        timestamps=np.asarray(timestamps, dtype=np.float64),
        gt=np.asarray(gts, dtype=np.float64),
        odom_t=odometry.timestamps,
        odom_poses=odometry.poses,
    )
    meta = {
        "session": session, "sequence": sequence, "cid": cid, "frames": len(paths),
        "labels": labels, "image_paths": paths, "encoder": extractor.model_identity,
        "cache_hits": extractor.hits, "cache_misses": extractor.misses,
        "frames_sha256": file_digest(output),
    }
    atomic_json(root / "frames" / f"{session}.json", meta)
    return meta


@dataclass
class Session:
    name: str
    config: dict
    cid: bool
    descriptors: np.ndarray
    timestamps: np.ndarray
    gt: np.ndarray
    odom_t: np.ndarray
    odom_poses: np.ndarray
    labels: list


def load_session(session: str, root: Path) -> Session:
    meta = json.loads((root / "frames" / f"{session}.json").read_text())
    path = root / "frames" / f"{session}.npz"
    if file_digest(path) != meta["frames_sha256"]:
        raise ValueError(f"Frame store hash mismatch: {path}")
    data = np.load(path)
    config, cid, _ = session_config(session)
    return Session(session, config, cid, data["descriptors"], data["timestamps"], data["gt"],
                   data["odom_t"], data["odom_poses"], meta["labels"])


# --------------------------------------------------------------------------- #
# Odometry corruption and per-frame interpolation
# --------------------------------------------------------------------------- #
def corrupt_odometry(poses: np.ndarray, level: DriftLevel, rng: np.random.Generator) -> np.ndarray:
    """Re-integrate raw SE(2) increments with scale, turn and heading-walk errors."""
    if level.heading_walk == level.scale_sigma == level.turn_sigma == 0.0:
        return poses.copy()
    scale = 1.0 + rng.normal(0.0, level.scale_sigma)
    turn = 1.0 + rng.normal(0.0, level.turn_sigma)
    out = np.empty_like(poses)
    out[0] = poses[0]
    heading = poses[0, 2]
    for k in range(1, len(poses)):
        previous, current = poses[k - 1], poses[k]
        c, s = np.cos(previous[2]), np.sin(previous[2])
        dx, dy = current[:2] - previous[:2]
        local = np.array([c * dx + s * dy, -s * dx + c * dy]) * scale
        distance = float(np.hypot(*local))
        dtheta = normalize_angle(float(current[2] - previous[2])) * turn
        if level.heading_walk:
            dtheta += rng.normal(0.0, level.heading_walk * np.sqrt(distance))
        c, s = np.cos(heading), np.sin(heading)
        out[k, 0] = out[k - 1, 0] + c * local[0] - s * local[1]
        out[k, 1] = out[k - 1, 1] + s * local[0] + c * local[1]
        heading = heading + dtheta
        out[k, 2] = normalize_angle(heading)
    return out


def camera_odometry(session: Session, raw_poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recompute the budget on (possibly corrupted) raw poses; interpolate at frames."""
    player = session.config["player"]
    tracker = OdometryUncertaintyTracker(OdometryUncertaintyParams(*player["odometry_uncertainty"]))
    covariances = np.stack([tracker.step(tuple(float(v) for v in pose)) for pose in raw_poses])
    if session.cid:
        stream = PlanarPoseStream(session.odom_t, raw_poses, covariances)
        sample = [stream.at(t, player["max_odometry_gap_s"]) for t in session.timestamps]
    else:
        stream = ColdOdometry(session.odom_t, raw_poses, covariances)
        sample = [stream.at(t, max_gap_s=player["max_odometry_gap_s"]) for t in session.timestamps]
    return np.asarray([p for p, _ in sample]), np.stack([c for _, c in sample])


def drift_summary(original: np.ndarray, corrupted: np.ndarray) -> dict:
    steps = np.linalg.norm(np.diff(original[:, :2], axis=0), axis=1)
    path = float(steps.sum())
    error = np.linalg.norm(corrupted[:, :2] - original[:, :2], axis=1)
    return {"path_m": path, "final_added_error_m": float(error[-1]),
            "max_added_error_m": float(error.max()),
            "added_drift_percent": 100.0 * float(error[-1]) / max(path, 1e-9)}


# --------------------------------------------------------------------------- #
# Segmentation (descriptor-only) and node construction (mirrors the mapper)
# --------------------------------------------------------------------------- #
def segment(descriptors: np.ndarray, mapping: dict) -> list[np.ndarray]:
    if mapping.get("valley_mode", "adaptive") != "adaptive":
        raise ValueError("The regime study uses the adaptive detector")
    monitor = ConnectivityMonitor(int(mapping["window_size"]))
    detector = AdaptiveValleyDetector(k=float(mapping["valley_k"]))
    segments, start = [], 0
    for index, descriptor in enumerate(descriptors):
        lambda_2 = monitor.update(descriptor.astype(np.float32))
        if lambda_2 is None:
            continue
        valley = detector.step(lambda_2)
        if valley is None:
            continue
        boundary = valley + 1
        if boundary >= start:
            segments.append(np.arange(start, boundary + 1))
            start = boundary + 1
    if start < len(descriptors):
        segments.append(np.arange(start, len(descriptors)))
    return segments


@dataclass(frozen=True)
class FrozenSegment:
    """Odometry-independent part of a node."""

    indices: np.ndarray
    medoid: int
    view_indices: tuple[int, ...]
    mean_direction: np.ndarray
    concentration: float
    resultant_length: float


def freeze_segments(descriptors: np.ndarray, segments: list[np.ndarray]) -> list[FrozenSegment]:
    frozen = []
    for indices in segments:
        samples = descriptors[indices].astype(np.float64)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        model = fit_vmf_resultant(samples.sum(axis=0), len(indices), samples[0])
        mean = model.mean_direction
        # max() keeps the first maximizer, as in TopologicalMapper._segment_medoid.
        medoid = int(max(indices, key=lambda i: float(descriptors[i] @ mean)))
        views: list[int] = []
        for i in (medoid, int(indices[0]), int(indices[-1])):
            if i not in views:
                views.append(i)
        frozen.append(FrozenSegment(indices, medoid, tuple(views), mean,
                                    model.concentration, model.mean_resultant_length))
    return frozen


def overlay_appearance(frozen: list[FrozenSegment], appearance: np.ndarray) -> list[FrozenSegment]:
    """Same frames, medoid and views; mean direction and concentration from ``appearance``."""
    out = []
    for seg in frozen:
        samples = appearance[seg.indices].astype(np.float64)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        model = fit_vmf_resultant(samples.sum(axis=0), len(seg.indices), samples[0])
        out.append(replace(seg, mean_direction=model.mean_direction, concentration=model.concentration,
                           resultant_length=model.mean_resultant_length))
    return out


def build_nodes(session: Session, frozen: list[FrozenSegment], poses: np.ndarray,
                covariances: np.ndarray, attach_gt: bool,
                appearance: np.ndarray | None = None) -> list[TopoNode]:
    """Nodes for association. ``appearance`` replaces view descriptors (same frames)."""
    views_source = session.descriptors if appearance is None else appearance
    nodes = []
    for node_id, seg in enumerate(frozen):
        rep = seg.medoid
        offsets = poses[seg.indices, :2] - poses[rep, :2]
        node = TopoNode(
            node_id=node_id,
            pose=tuple(float(v) for v in poses[rep]),
            visual_features=seg.mean_direction,
            visual_concentration=seg.concentration,
            visual_resultant_length=seg.resultant_length,
            visual_sample_count=len(seg.indices),
            pose_covariance=covariances[rep][:2, :2].copy(),
            extent_covariance=offsets.T @ offsets / len(seg.indices),
            gt_pose=tuple(float(v) for v in session.gt[rep]) if attach_gt else None,
        )
        views = views_source[list(seg.view_indices)].astype(np.float64)
        node.view_features = (views / np.linalg.norm(views, axis=1, keepdims=True)).astype(np.float32)
        nodes.append(node)
    return nodes


def replay(nodes: list[TopoNode], parameters: dict) -> TopoGraph:
    engine = TopologicalMapper(**parameters)
    try:
        for node in nodes:
            engine.observe_node(replace(node))
        engine.finish_revisits()
        return engine.graph
    finally:
        engine.close()


def score(graph: TopoGraph, session: Session, frozen: list[FrozenSegment], tolerance: float) -> dict:
    gt = {i: session.gt[seg.medoid, :2] for i, seg in enumerate(frozen)}
    result = graph_metrics(graph, session.gt[:, :2], node_gt_xy=gt, loop_tolerance=tolerance)
    return {"tp": result.loop_true_positives, "fp": result.loop_false_positives,
            "fn": result.loop_false_negatives}


def prf(tp: int, fp: int, fn: int) -> dict:
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


# --------------------------------------------------------------------------- #
# Stage: drift
# --------------------------------------------------------------------------- #
def run_drift(root: Path, sessions: list[str], seeds: int, levels: list[DriftLevel]) -> dict:
    rows, drift_rows, segmentation = [], [], {}
    for name in sessions:
        session = load_session(name, root)
        mapping = session.config["mapping"]
        frozen = freeze_segments(session.descriptors, segment(session.descriptors, mapping))
        segmentation[name] = {"nodes": len(frozen), "sizes": [len(s.indices) for s in frozen],
                              "medoids": [s.medoid for s in frozen]}
        common = {"window_size": int(mapping["window_size"]), "valley_k": float(mapping["valley_k"]),
                  "verification_threshold": float(mapping["verification_threshold"]),
                  "defer_nodes": int(mapping["defer_nodes"]),
                  "fixed_concentration": float(mapping["fixed_concentration"]),
                  "naive_threshold": float(mapping["naive_threshold"])}
        for level in levels:
            for seed in range(seeds if level.name != "L0" else 1):
                rng = np.random.default_rng([seed, DRIFT_LEVELS.index(level), sessions.index(name)])
                raw = corrupt_odometry(session.odom_poses, level, rng)
                poses, covariances = camera_odometry(session, raw)
                if level.name == "L0":
                    reference = poses
                drift_rows.append({"session": name, "level": level.name, "seed": seed,
                                   **drift_summary(reference, poses)})
                nodes = build_nodes(session, frozen, poses, covariances, attach_gt=False)
                for method, parameters in METHODS.items():
                    knob, values = sweep_for(method)
                    for value in values:
                        graph = replay(nodes, {**common, **DEFAULT_KNOB, **parameters, knob: value})
                        for tolerance in TOLERANCES:
                            rows.append({"session": name, "level": level.name, "seed": seed,
                                         "method": method, "knob": knob, "value": value,
                                         "tolerance_m": tolerance,
                                         **score(graph, session, frozen, tolerance)})
            print(f"drift {name} {level.name}: done", flush=True)
    return {"rows": rows, "drift": drift_rows, "segmentation": segmentation}


# --------------------------------------------------------------------------- #
# Stage: sessions (appearance-only cross-session association)
# --------------------------------------------------------------------------- #
def _views(node: TopoNode) -> np.ndarray:
    return np.asarray(node.view_features, dtype=np.float64)


SCORERS = {
    "multiview_cosine": lambda a, b: float(
        0.5 * ((_views(a) @ _views(b).T).max(1).mean() + (_views(a) @ _views(b).T).max(0).mean())),
    "mean_cosine": lambda a, b: float(np.asarray(a.visual_features, np.float64)
                                      @ np.asarray(b.visual_features, np.float64)),
    "vmf_overlap": lambda a, b: vmf_log_overlap_ratio(
        a.visual_features, a.visual_concentration, b.visual_features, b.visual_concentration),
    "neg_kl_estimated": lambda a, b: -vmf_symmetric_kl(
        a.visual_features, a.visual_concentration, b.visual_features, b.visual_concentration),
}


def score_matrix(queries: list[TopoNode], references: list[TopoNode], scorer) -> np.ndarray:
    return np.asarray([[scorer(q, r) for r in references] for q in queries])


def sequence_matrix(direct: np.ndarray, context: int = 2) -> np.ndarray:
    """Bidirectional sequence evidence: query i-k against reference j-+k, k<=context."""
    n_q, n_r = direct.shape
    best = np.full_like(direct, -np.inf)
    for direction in (-1, 1):
        total = direct.copy()
        count = np.ones_like(direct)
        for k in range(1, context + 1):
            shifted = np.full_like(direct, np.nan)
            q_index = np.arange(k, n_q)
            for j in range(n_r):
                r = j - direction * k
                if 0 <= r < n_r:
                    shifted[q_index, j] = direct[q_index - k, r]
            valid = ~np.isnan(shifted)
            total[valid] += shifted[valid]
            count[valid] += 1
        best = np.maximum(best, total / count)
    return best


def confidence(scores: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Top-1 reference per query and its decision statistic."""
    top = scores.argmax(axis=1)
    value = scores[np.arange(len(scores)), top]
    if kind == "raw":
        return top, value
    others = np.where(np.eye(scores.shape[1], dtype=bool)[top], np.nan, scores)
    median = np.nanmedian(others, axis=1)
    mad = 1.4826 * np.nanmedian(np.abs(others - median[:, None]), axis=1)
    if kind == "mad_z":
        return top, (value - median) / np.maximum(mad, 1e-9)
    second = np.sort(scores, axis=1)[:, -2]
    return top, value - second  # "margin"


def pr_curve(correct: np.ndarray, positive: np.ndarray, statistic: np.ndarray) -> dict:
    """Accept queries in decreasing statistic order; precision over accepted top-1 matches."""
    order = np.argsort(-statistic, kind="stable")
    tp = np.cumsum(correct[order])
    accepted = np.arange(1, len(order) + 1)
    n_pos = int(positive.sum())
    precision = tp / accepted
    recall = tp / max(n_pos, 1)
    # Average precision over the recall steps (step-wise integral).
    ap = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    f1 = np.where(precision + recall > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-12), 0)

    def recall_at(p: float) -> float:
        ok = precision >= p
        return float(recall[ok].max()) if ok.any() else 0.0

    return {"queries": int(len(order)), "positives": n_pos,
            "recall_at_1": float(correct.sum() / max(n_pos, 1)),
            "average_precision": ap, "max_f1": float(f1.max()) if len(f1) else 0.0,
            "recall_at_p90": recall_at(0.90), "recall_at_p95": recall_at(0.95),
            "recall_at_p99": recall_at(0.99)}


def run_sessions(root: Path, pairs: list[str]) -> dict:
    results = {}
    for pair in pairs:
        map_name, query_name, relation = SESSION_PAIRS[pair]
        map_session, query_session = load_session(map_name, root), load_session(query_name, root)
        # Shared GT map frame: every query trajectory sample must be near the map traversal.
        nearest = np.min(np.linalg.norm(
            query_session.gt[:, None, :2] - map_session.gt[None, ::3, :2], axis=2), axis=1)
        if np.median(nearest) > 0.5:
            raise ValueError(f"{pair}: ground truth does not share a map frame")
        mapping = map_session.config["mapping"]
        built = {}
        for role, session in (("map", map_session), ("query", query_session)):
            frozen = freeze_segments(session.descriptors, segment(session.descriptors, mapping))
            poses, covariances = camera_odometry(session, session.odom_poses)
            built[role] = (frozen, build_nodes(session, frozen, poses, covariances, attach_gt=False))
        map_gt = np.asarray([map_session.gt[s.medoid, :2] for s in built["map"][0]])
        query_gt = np.asarray([query_session.gt[s.medoid, :2] for s in built["query"][0]])
        distance = np.linalg.norm(query_gt[:, None] - map_gt[None], axis=2)
        pair_result = {"relation": relation, "map_nodes": len(map_gt), "query_nodes": len(query_gt),
                       "gt_nearest_median_m": float(np.median(nearest)),
                       # Expected top-1 accuracy of a uniformly random map node.
                       "chance_top1": {str(t): float((distance <= t).mean(axis=1).mean()) for t in TOLERANCES},
                       "scorers": {}}
        for scorer_name, scorer in SCORERS.items():
            direct = score_matrix(built["query"][1], built["map"][1], scorer)
            for evidence, matrix in (("single", direct), ("sequence", sequence_matrix(direct))):
                for kind in ("raw", "margin", "mad_z"):
                    top, statistic = confidence(matrix, kind)
                    for tolerance in TOLERANCES:
                        positive = distance.min(axis=1) <= tolerance
                        correct = distance[np.arange(len(top)), top] <= tolerance
                        key = f"{scorer_name}|{evidence}|{kind}|{tolerance}"
                        pair_result["scorers"][key] = {
                            **pr_curve(correct, positive, statistic),
                            "per_query": {"correct": correct.astype(int).tolist(),
                                          "positive": positive.astype(int).tolist(),
                                          "statistic": statistic.tolist()},
                        }
        results[pair] = pair_result
        print(f"sessions {pair}: {len(query_gt)} queries vs {len(map_gt)} map nodes", flush=True)
    return results


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def protocol(args: argparse.Namespace) -> dict:
    modules = (directional, mapper_module, matching, metrics, motion, node_detection)
    return {
        "study": "regime_study_v1", "stage": args.stage,
        "sources": {Path(m.__file__).name: file_digest(m.__file__) for m in modules},
        "runner_sha256": file_digest(__file__),
        "packages": {name: version(name) for name in ("numpy", "scipy")},
        "drift_levels": [vars(level) for level in DRIFT_LEVELS],
        "methods": METHODS, "visual_k_sweep": VISUAL_K_SWEEP, "gate_scale_sweep": GATE_SCALE_SWEEP,
        "tolerances_m": TOLERANCES, "seeds": args.seeds,
        "frames_root": str(args.frames or args.output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("frames", "drift", "sessions"))
    parser.add_argument("--output", type=Path, default=WORKSPACE / "output/regime_study_v1")
    parser.add_argument("--frames", type=Path, default=None,
                        help="frame-store root (default: --output)")
    parser.add_argument("--sessions", nargs="+", default=None)
    parser.add_argument("--pairs", nargs="+", default=list(SESSION_PAIRS))
    parser.add_argument("--levels", nargs="+", default=[level.name for level in DRIFT_LEVELS])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--allow-inference", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frames_root = args.frames or args.output
    if args.stage == "frames":
        for name in args.sessions or [*MAIN_SESSIONS, *EXTRA_SESSIONS]:
            if (args.output / "frames" / f"{name}.json").exists():
                print(f"frames {name}: exists, skipped", flush=True)
                continue
            meta = build_frame_store(name, args.output,
                                     args.allow_inference or name in EXTRA_SESSIONS or name in HELDOUT_SESSIONS
                                     or name in CONFIRM_SESSIONS or name in CID_CONFIRM_SESSIONS)
            print(f"frames {name}: {meta['frames']} frames, misses={meta['cache_misses']}", flush=True)
        return
    meta = {"fingerprint": None, **protocol(args)}
    if args.stage == "drift":
        levels = [level for level in DRIFT_LEVELS if level.name in args.levels]
        result = run_drift(frames_root, args.sessions or list(MAIN_SESSIONS), args.seeds, levels)
    else:
        result = run_sessions(frames_root, args.pairs)
    meta["fingerprint"] = identity({k: v for k, v in meta.items() if k != "fingerprint"})
    atomic_json(args.output / f"{args.stage}_results.json", {**meta, "results": result})
    print(f"Results: {args.output / f'{args.stage}_results.json'}")


if __name__ == "__main__":
    main()
