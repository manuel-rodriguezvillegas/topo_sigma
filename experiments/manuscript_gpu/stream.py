"""Frame streams of the manuscript traversals and the odometry gates evaluated on them."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent / "anyloc_published"))

from common import ALL, DEV, HELDOUT, Sequence, frames  # noqa: E402,F401
from vts_core.motion import OdometryUncertaintyParams, make_tracker  # noqa: E402

RESULTS = HERE / "results"
CALIBRATION_FILE = RESULTS / "calibration.json"
DEFAULT_PARAMS = OdometryUncertaintyParams()


@dataclass(frozen=True)
class Stream:
    image_paths: list[str]
    odometry: np.ndarray
    ground_truth: np.ndarray
    labels: list[str | None]


def load_stream(seq: Sequence) -> Stream:
    paths, odometry, ground_truth, labels = [], [], [], []
    for image_path, gt_pose, pose, _, label in frames(seq):
        paths.append(image_path)
        odometry.append(pose)
        ground_truth.append(gt_pose)
        labels.append(label)
    return Stream(paths, np.asarray(odometry, dtype=np.float64), np.asarray(ground_truth, dtype=np.float64), labels)


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
