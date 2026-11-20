"""Systematic heading drift of the wheel odometry, one rate per robot, fitted on the design traversals.

Unequal wheel radii turn a differential-drive robot by a fixed angle per metre driven (Borenstein and
Feng, 1996, UMBmark). The rate of a robot is the least-squares slope of the heading error accumulated
over windows of fixed travelled distance against the signed distance driven in them, pooled over its
design traversals. Only ground-truth poses enter.
"""

from __future__ import annotations

import json

import numpy as np

from stream import BIAS_FILE, DESIGN, Stream, load_stream, platform, signed_step_lengths

WINDOW_M = 8.0
ANCHOR_SPACING_M = 1.0


def window_drift(stream: Stream) -> tuple[np.ndarray, np.ndarray]:
    """Signed distance driven and odometry heading error over windows of ``WINDOW_M`` travelled metres."""
    steps = signed_step_lengths(stream.odometry)
    travelled = np.concatenate([[0.0], np.cumsum(np.abs(steps))])
    signed = np.concatenate([[0.0], np.cumsum(steps)])
    start = np.unique(np.searchsorted(travelled, np.arange(0.0, travelled[-1] - WINDOW_M, ANCHOR_SPACING_M)))
    end = np.searchsorted(travelled, travelled[start] + WINDOW_M)
    keep = end < len(travelled)
    start, end = start[keep], end[keep]
    turn = np.diff(np.unwrap(stream.odometry[:, 2]))
    truth = np.diff(np.unwrap(stream.ground_truth[:, 2]))
    accumulated = np.concatenate([[0.0], np.cumsum(turn - truth)])
    return signed[end] - signed[start], accumulated[end] - accumulated[start]


def fit_rate(windows: list[tuple[np.ndarray, np.ndarray]]) -> float:
    distance = np.concatenate([w[0] for w in windows])
    error = np.concatenate([w[1] for w in windows])
    return float(np.sum(distance * error) / np.sum(distance**2))


def residual_rms_deg(windows: list[tuple[np.ndarray, np.ndarray]], rate: float) -> float:
    return float(np.degrees(np.sqrt(np.mean(np.concatenate([w[1] - rate * w[0] for w in windows]) ** 2))))


def main() -> None:
    windows = {seq.name: window_drift(load_stream(seq)) for seq in DESIGN}
    report = {}
    for robot in sorted({platform(seq) for seq in DESIGN}):
        names = [seq.name for seq in DESIGN if platform(seq) == robot]
        rate = fit_rate([windows[n] for n in names])
        report[robot] = {
            "rate_deg_per_m": float(np.degrees(rate)),
            "per_traversal_deg_per_m": {n: float(np.degrees(fit_rate([windows[n]]))) for n in names},
            "leave_one_out_deg_per_m": {n: float(np.degrees(fit_rate([windows[m] for m in names if m != n])))
                                        for n in names},
            f"heading_error_rms_over_{WINDOW_M:g}m_deg": {
                "raw": residual_rms_deg([windows[n] for n in names], 0.0),
                "corrected": residual_rms_deg([windows[n] for n in names], rate)},
        }
    BIAS_FILE.write_text(json.dumps(report, indent=1))
    for robot, entry in report.items():
        print(robot, f"{entry['rate_deg_per_m']:.3f} deg/m", entry[f"heading_error_rms_over_{WINDOW_M:g}m_deg"])


if __name__ == "__main__":
    main()
