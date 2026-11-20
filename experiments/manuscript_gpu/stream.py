"""Frame streams of the manuscript traversals and the odometry gates evaluated on them."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent / "anyloc_published"))

from common import ALL, DEV, HELDOUT, TEST, Sequence, frames  # noqa: E402,F401
from vts_core.motion import OdometryUncertaintyParams, make_tracker  # noqa: E402
from vts_evaluation.ground_truth import load_ground_truth_xy  # noqa: E402

RESULTS = HERE / "results"
DESIGN = DEV + HELDOUT
CALIBRATION_FILE = RESULTS / "calibration.json"
TEST_INVENTORY_FILE = RESULTS / "test_inventory.json"
DEFAULT_PARAMS = OdometryUncertaintyParams()


@dataclass(frozen=True)
class Stream:
    """Frames of ``frames(seq)`` from index ``first`` on; descriptors of the full stream are sliced with it."""

    first: int
    image_paths: list[str]
    odometry: np.ndarray
    ground_truth: np.ndarray
    labels: list[str | None]


def ground_truth_start(rows: np.ndarray) -> int:
    """Last index of the leading run of identical rows: before it the localization had not started."""
    changed = np.flatnonzero(np.any(rows[1:] != rows[0], axis=1))
    return int(changed[0]) if changed.size else 0


def load_stream(seq: Sequence) -> Stream:
    paths, odometry, ground_truth, labels = [], [], [], []
    for image_path, gt_pose, pose, _, label in frames(seq):
        paths.append(image_path)
        odometry.append(pose)
        ground_truth.append(gt_pose)
        labels.append(label)
    truth = np.asarray(ground_truth, dtype=np.float64)
    first = ground_truth_start(truth)
    return Stream(first, paths[first:], np.asarray(odometry, dtype=np.float64)[first:], truth[first:], labels[first:])


def evaluated_trajectory(seq: Sequence) -> np.ndarray:
    """Ground-truth xy used for coverage, trimmed by the same rule as the frame stream."""
    xy = load_ground_truth_xy(str(seq.trajectory_gt))
    return xy[ground_truth_start(xy):]


def cumulative_covariances(model: str, params: OdometryUncertaintyParams, odometry: np.ndarray) -> np.ndarray:
    tracker = make_tracker(model, params)
    return np.stack([tracker.step(tuple(pose)) for pose in odometry])


@dataclass(frozen=True)
class Gate:
    name: str
    model: str
    params: OdometryUncertaintyParams


def gates() -> dict[str, Gate]:
    """The isotropic gate with its original coefficients and both calibrated gates."""
    fitted = json.loads(CALIBRATION_FILE.read_text())["models"]
    return {
        "isotropic_default": Gate("isotropic_default", "additive", DEFAULT_PARAMS),
        "isotropic_calibrated": Gate("isotropic_calibrated", "additive",
                                     OdometryUncertaintyParams(*fitted["additive"]["alpha"])),
        "propagated_calibrated": Gate("propagated_calibrated", "propagated",
                                      OdometryUncertaintyParams(*fitted["propagated"]["alpha"])),
    }


def readable_test() -> list[Sequence]:
    """Test traversals that the inventory found readable, in list order."""
    inventory = json.loads(TEST_INVENTORY_FILE.read_text())
    return [seq for seq in TEST if "frames" in inventory[seq.name]]
