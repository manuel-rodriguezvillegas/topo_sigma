"""Dataset adapters: COLD and CID-SIMS recordings to frame-aligned arrays.

Each adapter selects camera frames exactly as the dataset players do and
returns their timestamps, ground truth (evaluation only), optional place
labels, image paths and the *raw* recorded odometry. A recording whose frames
cannot all be synchronized with odometry (gaps, non-monotonic stamps, partial
coverage) raises :class:`DataQualityError`; it is not silently repaired.

The catalogue names every recording and assigns its role. Test recordings stay
untouched until a written confirmatory protocol exists; loading them requires
an explicit flag.
"""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vts_core.odometry import bracket, interpolate

COLD_FILENAME = re.compile(r"t(?P<t>\d+\.\d+)_x(?P<x>-?\d+\.?\d*)_y(?P<y>-?\d+\.?\d*)_a(?P<a>-?\d+\.?\d*)")
COLD_DIRECTORY = re.compile(r"cold-(?P<lab>[a-z]+)_part_(?P<part>[a-z])_(?P<route>seq\d+)_(?P<condition>[a-z]+\d+)$")
CID_DIRECTORY = re.compile(r"(?P<environment>[a-z_]+?\d*)_(?P<traversal>\d+)$")
# Canonical COLD place classes; the first contained in a raw label wins.
COLD_CLASSES = ("CR", "2PO", "RL", "TL", "TR", "LO", "1PO", "KT", "CNR", "PA", "LAB", "ST")


class DataQualityError(ValueError):
    """The recording cannot be synchronized without repairs."""


@dataclass(frozen=True)
class Recording:
    name: str  # session name, e.g. cold_freiburg_a_seq2_night1, cid_apartment1_1
    dataset: str  # "cold" or "cid_sims"
    directory: str  # relative to the dataset root
    environment: str  # sessions of one environment share a ground-truth frame
    route: str
    condition: str
    role: str  # "development" or "test"


@dataclass
class RawFrames:
    timestamps: np.ndarray
    gt: np.ndarray  # (N, 3) ground truth x, y, theta (evaluation only)
    labels: list[str | None]
    image_paths: list[str]  # relative to the dataset root
    odom_t: np.ndarray
    odom_poses: np.ndarray  # raw recorded odometry (M, 3)


def data_root(config: dict) -> Path:
    return Path(os.environ.get("TOPOSIGMA_DATA_ROOT", config["data_root"]))


def dataset_root(config: dict, dataset: str) -> Path:
    return data_root(config) / config["datasets"][dataset]["subdir"]


# --------------------------------------------------------------------------- #
# Catalogue
# --------------------------------------------------------------------------- #
def _role(name: str, config: dict) -> str:
    return "test" if any(fnmatch.fnmatch(name, p) for p in config["test_sessions"]) else "development"


def recording_for(dataset: str, directory: str, config: dict) -> Recording:
    if dataset == "cold":
        match = COLD_DIRECTORY.match(directory)
        if match is None:
            raise ValueError(f"not a COLD recording directory: {directory}")
        lab, part, route, condition = match.group("lab", "part", "route", "condition")
        name = f"cold_{lab}_{part}_{route}_{condition}"
        return Recording(name, dataset, directory, f"cold_{lab}_{part}", route, condition, _role(name, config))
    match = CID_DIRECTORY.match(directory)
    if match is None:
        raise ValueError(f"not a CID-SIMS traversal directory: {directory}")
    environment, traversal = match.group("environment", "traversal")
    name = f"cid_{directory}"
    return Recording(name, dataset, directory, f"cid_{environment}", environment, traversal, _role(name, config))


def catalogue(config: dict) -> dict[str, Recording]:
    """Every recording present under the dataset roots, by session name."""
    found = {}
    for dataset in ("cold", "cid_sims"):
        root = dataset_root(config, dataset)
        if not root.is_dir():
            continue
        for path in sorted(root.iterdir()):
            if not path.is_dir():
                continue
            try:
                recording = recording_for(dataset, path.name, config)
            except ValueError:
                continue
            if dataset == "cid_sims" and not (path / config["datasets"]["cid_sims"]["odometry_file"]).is_file():
                continue
            found[recording.name] = recording
    return found


# --------------------------------------------------------------------------- #
# COLD
# --------------------------------------------------------------------------- #
def canonical_cold_label(raw: str) -> str:
    raw = raw.strip()
    for name in COLD_CLASSES:
        if name in raw:
            return name
    return raw.split("-")[0].rstrip("0123456789")


def read_cold_odometry(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read ``odom.tdf``: seconds/microseconds at columns 3-4, x, y, theta at 8, 9, 11.

    Some logs reset the microsecond field without incrementing seconds; only that
    exact pattern is repaired (as the next second).
    """
    times, poses, previous = [], [], None
    with open(path, encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 12:
                raise DataQualityError(f"{path}:{number}: expected at least 12 fields")
            seconds, micro = int(fields[3]), int(fields[4])
            if previous is not None and seconds == previous[0] and micro < previous[1]:
                seconds += 1
            previous = (int(fields[3]), micro)
            times.append(seconds + 1e-6 * micro)
            poses.append((float(fields[8]), float(fields[9]), float(fields[11])))
    times = np.asarray(times, dtype=np.float64)
    if len(times) < 2 or np.any(np.diff(times) <= 0.0):
        raise DataQualityError(f"{path}: odometry timestamps must be strictly increasing")
    return times, np.asarray(poses, dtype=np.float64)


def read_cold(root: Path, recording: Recording, settings: dict) -> RawFrames:
    directory = root / recording.directory
    labels_file = directory / settings["labels_subpath"]
    raw_labels = {}
    if labels_file.is_file():
        for line in labels_file.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                raw_labels[os.path.basename(parts[0])] = canonical_cold_label(parts[1])
    times, gts, labels, paths = [], [], [], []
    for image in sorted((directory / settings["images_subdir"]).iterdir()):
        match = COLD_FILENAME.search(image.name)
        if match is None:
            continue
        times.append(float(match.group("t")))
        gts.append([float(match.group(k)) for k in "xya"])
        labels.append(raw_labels.get(image.name))
        paths.append(str(image.relative_to(root)))
    odom_t, odom_poses = read_cold_odometry(directory / settings["odometry_subpath"])
    timestamps = np.asarray(times, dtype=np.float64)
    try:
        interpolate(odom_t, odom_poses, np.zeros((len(odom_t), 3, 3)), timestamps,
                    settings["max_odometry_gap_s"], settings["exact_atol_s"])
    except ValueError as error:
        raise DataQualityError(f"{recording.name}: {error}") from error
    return RawFrames(timestamps, np.asarray(gts, dtype=np.float64), labels, paths, odom_t, odom_poses)


# --------------------------------------------------------------------------- #
# CID-SIMS
# --------------------------------------------------------------------------- #
def quaternion_yaw(x, y, z, w):
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def read_pose_rows(path: Path, minimum_fields: int) -> tuple[np.ndarray, np.ndarray]:
    """``timestamp px py pz qx qy qz qw ...`` rows as planar poses."""
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    if any(len(row) < minimum_fields for row in rows):
        raise DataQualityError(f"{path}: expected at least {minimum_fields} fields per row")
    values = np.asarray([[float(v) for v in row[:8]] for row in rows])
    times = values[:, 0]
    if len(times) < 2 or np.any(np.diff(times) <= 0.0):
        raise DataQualityError(f"{path}: timestamps must be strictly increasing")
    yaw = np.asarray([float(quaternion_yaw(*q)) for q in values[:, 4:8]])
    return times, np.column_stack([values[:, 1], values[:, 2], yaw])


def read_cid(root: Path, recording: Recording, settings: dict) -> RawFrames:
    directory = root / recording.directory
    gt_t, gt_poses = read_pose_rows(directory / settings["ground_truth_file"], 8)
    odom_t, odom_poses = read_pose_rows(directory / settings["odometry_file"], 14)
    images = []
    for path in (directory / settings["color_subdir"]).glob("*.png"):
        try:
            images.append((float(path.stem), path))
        except ValueError:
            continue
    images.sort(key=lambda item: item[0])
    atol = settings["exact_atol_s"]
    selected = images[:: settings["frame_stride"]]
    query = np.asarray([t for t, _ in selected])
    ok = (bracket(gt_t, query, settings["max_ground_truth_gap_s"], atol)[3]
          & bracket(odom_t, query, settings["max_odometry_gap_s"], atol)[3])
    kept = [item for item, keep in zip(selected, ok) if keep]
    if not kept:
        raise DataQualityError(f"{recording.name}: no synchronizable frames")
    timestamps = np.asarray([t for t, _ in kept])
    gt, _ = interpolate(gt_t, gt_poses, np.zeros((len(gt_t), 3, 3)), timestamps,
                        settings["max_ground_truth_gap_s"], atol)
    return RawFrames(timestamps, gt, [None] * len(kept), [str(p.relative_to(root)) for _, p in kept],
                     odom_t, odom_poses)


def read_recording(recording: Recording, config: dict) -> RawFrames:
    settings = config["datasets"][recording.dataset]
    reader = read_cold if recording.dataset == "cold" else read_cid
    return reader(dataset_root(config, recording.dataset), recording, settings)
