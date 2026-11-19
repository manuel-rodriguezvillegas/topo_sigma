"""Node-creation sensitivity of the development traversals, for each descriptor and boundary rule.

lambda_2 and the window cuts are recomputed from the stored descriptors, so no replay is needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from stream import DEV, load_stream
from common import load_descriptors
from mapper_runs import NODE_SENSITIVITY_FILE
from vts_core.node_detection import ConnectivityMonitor
from vts_evaluation.node_segmentation import aggregate_results, run_experiment_on_signals

ENCODERS = ("dinov2_s", "dinov3_s")
K_VALUES = [1.0, 1.5, 2.0, 2.5, 3.0]
FINAL_K = 2.0
WINDOW = 30
CALIBRATION_CASE = "freiburg_a"
FIXED_DELTA_SOURCE = ("dinov3_s", "minimum")


def connectivity(descriptors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    monitor = ConnectivityMonitor(WINDOW)
    for descriptor in descriptors:
        monitor.update(descriptor)
    return np.asarray(monitor.eigenvalues), np.asarray(monitor.frames_after_cut)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptors", type=Path, required=True, help="AnyLoc runner output directory")
    args = parser.parse_args()
    labels = {seq.name: load_stream(seq).labels if seq.dataset == "cold" else None for seq in DEV}
    for name, frame_labels in labels.items():
        if frame_labels is not None and any(label is None for label in frame_labels):
            raise ValueError(f"{name}: frames without a room label")
    detectors, fixed_delta = {}, None
    for encoder in ENCODERS:
        signals = {seq.name: connectivity(load_descriptors(args.descriptors, seq.name, encoder)[0]) for seq in DEV}
        for boundary in ("minimum", "cut"):
            loaded = {name: (signal, labels[name], cuts if boundary == "cut" else None)
                      for name, (signal, cuts) in signals.items()}
            calibration, rows = run_experiment_on_signals(
                loaded, CALIBRATION_CASE, K_VALUES, FINAL_K, history=300, warmup=30,
                tolerance_frames=15, frame_rate_hz=5.0,
            )
            detectors[f"{encoder}/{boundary}"] = {"summary": aggregate_results(rows, FINAL_K), "rows": rows}
            if (encoder, boundary) == FIXED_DELTA_SOURCE:
                fixed_delta = calibration.delta
    report = {"fixed_delta": fixed_delta, "calibration_case": CALIBRATION_CASE, "detectors": detectors}
    NODE_SENSITIVITY_FILE.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
