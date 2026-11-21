"""Shared definitions: sequences, frame stream, manuscript mapper settings.

The frame stream reproduces the submission players without ROS: same images,
odometry budgets, ground truth and labels, in the same order. ``historical_src``
holds the submission mapper; its default options still reproduce the submission
exactly (checked by the reproduction step of ``replay.py``).
"""

from __future__ import annotations

import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "historical_src"))

from vts_core.motion import OdometryUncertaintyParams  # noqa: E402
from vts_players.cid_sims_data import (  # noqa: E402
    load_ground_truth,
    load_wheel_odometry,
    timestamped_color_images,
)
from vts_players.cold_odometry import load_cold_odometry  # noqa: E402

REPO = HERE.parents[1]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", REPO / "encoder"))
UNCERTAINTY = (0.025, 0.005, 0.01, 0.0025)
TOLERANCES = [0.5, 1.0, 2.0, 3.0]

# Effective mapping configuration of the submission runs (output/submission/*/effective_config.yaml).
MAPPER_OPTIONS = dict(
    window_size=30,
    valley_k=2.0,
    valley_mode="adaptive",
    valley_delta=0.1,
    visual_outlier_k=2.0,
    gate_mode="both",
    naive_threshold=0.7,
)


@dataclass(frozen=True)
class Sequence:
    name: str
    dataset: str  # "cold" | "cid"
    recording: str
    split: str  # "dev" | "heldout" | "test"

    @property
    def path(self) -> Path:
        return DATA_ROOT / "seq_data" / self.recording if self.dataset == "cold" else DATA_ROOT / self.recording

    @property
    def domain(self) -> str:
        return "COLD" if self.dataset == "cold" else "CID-SIMS"

    @property
    def trajectory_gt(self) -> Path:
        return self.path / "std_cam" if self.dataset == "cold" else self.path / "groundtruth.txt"

    @property
    def route(self) -> str:
        """Recordings of one route (COLD lab/part/sequence, CID-SIMS area) share it; resampling units."""
        return self.recording.rsplit("_", 1)[0]


DEV = [
    Sequence("freiburg_a", "cold", "cold-freiburg_part_a_seq2_night1", "dev"),
    Sequence("freiburg_ext", "cold", "cold-freiburg_part_b_seq3_sunny1", "dev"),
    Sequence("saarbruecken_a", "cold", "cold-saarbruecken_part_a_seq2_night2", "dev"),
    Sequence("saarbruecken_ext", "cold", "cold-saarbruecken_part_b_seq4_sunny1", "dev"),
    Sequence("cid_sims_apartment1_1", "cid", "apartment1_1", "dev"),
    Sequence("cid_sims_apartment2_1", "cid", "apartment2_1", "dev"),
    Sequence("cid_sims_apartment3_1", "cid", "apartment3_1", "dev"),
]

# Held-out traversals, fixed before any method was run on them.
HELDOUT = [
    Sequence("heldout_freiburg_a_seq1", "cold", "cold-freiburg_part_a_seq1_cloudy2", "heldout"),
    Sequence("heldout_freiburg_a_seq2", "cold", "cold-freiburg_part_a_seq2_cloudy2", "heldout"),
    Sequence("heldout_freiburg_b_seq3", "cold", "cold-freiburg_part_b_seq3_cloudy2", "heldout"),
    Sequence("heldout_saarbruecken_a_seq1", "cold", "cold-saarbruecken_part_a_seq1_cloudy1", "heldout"),
    Sequence("heldout_saarbruecken_a_seq2", "cold", "cold-saarbruecken_part_a_seq2_cloudy2", "heldout"),
    Sequence("heldout_saarbruecken_b_seq3", "cold", "cold-saarbruecken_part_b_seq3_cloudy1", "heldout"),
    Sequence("heldout_saarbruecken_b_seq4", "cold", "cold-saarbruecken_part_b_seq4_cloudy2", "heldout"),
    Sequence("heldout_ljubljana_a_seq1", "cold", "cold-ljubljana_part_a_seq1_night3", "heldout"),
    Sequence("heldout_ljubljana_a_seq2", "cold", "cold-ljubljana_part_a_seq2_cloudy2", "heldout"),
    Sequence("heldout_cid_apartment1_2", "cid", "apartment1_2", "heldout"),
    Sequence("heldout_cid_apartment2_2", "cid", "apartment2_2", "heldout"),
    Sequence("heldout_cid_apartment3_2", "cid", "apartment3_2", "heldout"),
]

TEST_LIST = HERE / "test_traversals.tsv"


def _test_traversals() -> list[Sequence]:
    """Final test traversals, never used to design or tune the method; downloaded by fetch_test_set.sh."""
    rows = [line.split("\t") for line in TEST_LIST.read_text().splitlines() if line and not line.startswith("#")]
    return [Sequence(name, dataset, recording, "test") for name, dataset, recording, _ in rows]


TEST = _test_traversals()
ALL = {s.name: s for s in DEV + HELDOUT + TEST}

_COLD_NAME = re.compile(r"t(?P<t>\d+\.\d+)_x(?P<x>-?\d+\.?\d*)_y(?P<y>-?\d+\.?\d*)_a(?P<a>-?\d+\.?\d*)")
_COLD_CLASSES = ("CR", "2PO", "RL", "TL", "TR", "LO", "1PO", "KT", "CNR", "PA", "LAB", "ST")


def _canonical_label(raw: str) -> str:
    """Exact copy of vts_players.cold_player_node.canonical_label (that module imports ROS)."""
    raw = raw.strip()
    for class_name in _COLD_CLASSES:
        if class_name in raw:
            return class_name
    return raw.split("-")[0].rstrip("0123456789")


def _room_labels(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    labels = {}
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            labels[os.path.basename(parts[0])] = _canonical_label(parts[1])
    return labels


_REFERENCE_FRAMES: dict[str, set[str]] = {}


def reference_frames(seq: Sequence) -> set[str] | None:
    """Frame names the manuscript actually processed (development traversals only).

    The Mac copy of cold-freiburg_part_a_seq2_night1 lacks one image
    (t1152908172.545660..., listed in places.lst), so the manuscript used 2,875 of
    the 2,876 published frames. A fresh download has it; it is skipped here so the
    stream is identical to the manuscript's.
    """
    if seq.split != "dev":
        return None
    if seq.name not in _REFERENCE_FRAMES:
        with np.load(HERE / "reference/dinov2_vits14_mac.npz") as data:
            _REFERENCE_FRAMES[seq.name] = set(data[f"{seq.name}/frames"].tolist())
    return _REFERENCE_FRAMES[seq.name]


def frames(seq: Sequence) -> Iterator[tuple[str, tuple, tuple, np.ndarray, str | None]]:
    """Yield (image_path, gt_pose, odom_pose, budget, room_label) as the submission players did."""
    keep = reference_frames(seq)
    for item in _frames(seq):
        if keep is None or os.path.basename(item[0]) in keep:
            yield item


def _frames(seq: Sequence) -> Iterator[tuple[str, tuple, tuple, np.ndarray, str | None]]:
    uncertainty = OdometryUncertaintyParams(*UNCERTAINTY)
    if seq.dataset == "cid":
        gt = load_ground_truth(str(seq.path / "groundtruth.txt"))
        odom = load_wheel_odometry(str(seq.path / "odom.txt"), uncertainty)
        for timestamp, image_path in timestamped_color_images(str(seq.path / "color"))[::6]:
            try:
                gt_pose, _ = gt.at(timestamp, 0.1)
                pose, budget = odom.at(timestamp, 0.2)
            except ValueError:
                continue
            yield image_path, gt_pose, pose, budget, None
    else:
        odom = load_cold_odometry(str(seq.path / "odom_scans" / "odom.tdf"), uncertainty)
        labels = _room_labels(seq.path / "localization" / "places.lst")
        for image_path in sorted((seq.path / "std_cam").iterdir()):
            match = _COLD_NAME.search(image_path.name)
            if match is None:
                continue
            gt_pose = (float(match["x"]), float(match["y"]), float(match["a"]))
            pose, budget = odom.at(float(match["t"]), max_gap_s=1.0)
            yield str(image_path), gt_pose, pose, budget, labels.get(image_path.name)


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
        near = [j for j in ids if j < q - graph.loop_exclusion_nodes
                and math.hypot(node_gt[j][0] - qx, node_gt[j][1] - qy) <= tolerance]
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
