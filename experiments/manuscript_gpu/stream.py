"""Frame streams of the manuscript traversals and the odometry gates evaluated on them."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, replace
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
BIAS_FILE = RESULTS / "odometry_bias.json"
DEFAULT_PARAMS = OdometryUncertaintyParams()


@dataclass(frozen=True)
class Stream:
    """Frames of ``frames(seq)`` from index ``first`` on; descriptors of the full stream are sliced with it."""

    first: int
    image_paths: list[str]
    odometry: np.ndarray
    ground_truth: np.ndarray
    labels: list[str | None]


def signed_step_lengths(odometry: np.ndarray) -> np.ndarray:
    """Length of each odometry step, negative when the robot reverses."""
    displacement = np.diff(odometry[:, :2], axis=0)
    heading = odometry[:-1, 2]
    forward = displacement[:, 0] * np.cos(heading) + displacement[:, 1] * np.sin(heading)
    return np.sign(forward) * np.hypot(*displacement.T)


def platform(seq: Sequence) -> str:
    """Robot that recorded the traversal: one per COLD lab, one for CID-SIMS."""
    return "cid-sims" if seq.dataset == "cid" else seq.recording.removeprefix("cold-").split("_")[0]


def debiased(odometry: np.ndarray, heading_rate: float) -> np.ndarray:
    """Odometry with a heading drift of ``heading_rate`` rad per signed metre removed, re-integrated from the first pose."""
    steps = signed_step_lengths(odometry)
    correction = -heading_rate * np.concatenate([[0.0], np.cumsum(steps)])
    midpoint = (correction[:-1] + correction[1:]) / 2
    c, s = np.cos(midpoint), np.sin(midpoint)
    d = np.diff(odometry[:, :2], axis=0)
    rotated = np.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]], axis=1)
    xy = odometry[0, :2] + np.concatenate([np.zeros((1, 2)), np.cumsum(rotated, axis=0)])
    heading = (odometry[:, 2] + correction + np.pi) % (2 * np.pi) - np.pi
    return np.column_stack([xy, heading])


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
    debiased: bool = False


def gates() -> dict[str, Gate]:
    """The isotropic gate with its original coefficients and the calibrated gates."""
    fitted = json.loads(CALIBRATION_FILE.read_text())["models"]
    return {
        "isotropic_default": Gate("isotropic_default", "additive", DEFAULT_PARAMS),
        "isotropic_calibrated": Gate("isotropic_calibrated", "additive",
                                     OdometryUncertaintyParams(*fitted["additive"]["alpha"])),
        "propagated_calibrated": Gate("propagated_calibrated", "propagated",
                                      OdometryUncertaintyParams(*fitted["propagated"]["alpha"])),
        "propagated_debiased": Gate("propagated_debiased", "propagated",
                                    OdometryUncertaintyParams(*fitted["propagated_debiased"]["alpha"]), debiased=True),
    }


def heading_rate(seq: Sequence) -> float:
    """Heading drift of the robot of ``seq`` in rad per signed metre, fitted on the design traversals."""
    return float(np.radians(json.loads(BIAS_FILE.read_text())[platform(seq)]["rate_deg_per_m"]))


def debiased_stream(seq: Sequence, stream: Stream) -> Stream:
    return replace(stream, odometry=debiased(stream.odometry, heading_rate(seq)))

def readable_test() -> list[Sequence]:
    """Test traversals that the inventory found readable, in list order."""
    inventory = json.loads(TEST_INVENTORY_FILE.read_text())
    return [seq for seq in TEST if "frames" in inventory[seq.name]]
