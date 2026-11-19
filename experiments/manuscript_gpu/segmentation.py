"""Node-creation sensitivity on the lambda_2 traces recorded by the development replays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from stream import DEV, load_stream
from mapper_runs import NODE_SENSITIVITY_FILE
from vts_evaluation.node_segmentation import aggregate_results, run_experiment_on_signals

K_VALUES = [1.0, 1.5, 2.0, 2.5, 3.0]
FINAL_K = 2.0
CALIBRATION_CASE = "freiburg_a"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    loaded = {}
    for seq in DEV:
        signal = np.load(args.out / "replay" / seq.name / "topo" / "lambda2.npy")
        labels = load_stream(seq).labels if seq.dataset == "cold" else None
        if labels is not None and any(label is None for label in labels):
            raise ValueError(f"{seq.name}: frames without a room label")
        loaded[seq.name] = (signal, labels)
    calibration, rows = run_experiment_on_signals(
        loaded, CALIBRATION_CASE, K_VALUES, FINAL_K, history=300, warmup=30, tolerance_frames=15, frame_rate_hz=5.0
    )
    report = {"fixed_delta": calibration.delta, "calibration_case": CALIBRATION_CASE,
              "summary": aggregate_results(rows, FINAL_K), "rows": rows}
    NODE_SENSITIVITY_FILE.write_text(json.dumps(report, indent=1))
    print(json.dumps(report["summary"], indent=1))


if __name__ == "__main__":
    main()
