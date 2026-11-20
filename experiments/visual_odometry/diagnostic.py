"""How much heading and position drift a visual heading could remove, on the design traversals only.

Each consecutive frame pair gives a visual heading change from the essential matrix of its matches.
Three heading sources are compared against ground truth over a travelled distance, as in the
odometry calibration: wheels, vision (wheels where vision has no estimate), and their per-step
inverse-variance fusion. Positions are rebuilt from the wheel step lengths and each heading source.

The camera intrinsics are not published, so each dataset gets one focal length, chosen with the
heading sign on a grid against ground truth over short windows; the principal point is the image
centre. The visual step variance is a robust per-dataset estimate from the same windows.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parents[0] / "manuscript_gpu"))
from stream import CALIBRATION_FILE, DESIGN, Sequence, Stream, load_stream, signed_step_lengths  # noqa: E402

RESULTS = HERE / "results"
IMAGE_SIZE = (640, 480)
FOCAL_GRID = np.geomspace(300.0, 1200.0, 25)
MIN_MATCHES = 15
WINDOW_STEPS = 5
SEARCH_STRIDE = 4
GATE_SIGMAS = 3.0
HORIZONS_M = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
ANCHOR_SPACING_M = 1.0


def wrap(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def visual_yaw(points0: np.ndarray, points1: np.ndarray, focal: float) -> tuple[float, int]:
    """Rotation of the camera about its y axis (OpenCV frame, y down) between two views, and the inlier count."""
    if len(points0) < MIN_MATCHES:
        return float("nan"), 0
    camera = np.array([[focal, 0.0, IMAGE_SIZE[0] / 2], [0.0, focal, IMAGE_SIZE[1] / 2], [0.0, 0.0, 1.0]])
    p0, p1 = points0.astype(np.float64), points1.astype(np.float64)
    essential, mask = cv2.findEssentialMat(p0, p1, camera, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if essential is None or mask is None or mask.sum() < MIN_MATCHES:
        return float("nan"), 0
    inliers, rotation, _, _ = cv2.recoverPose(essential[:3], p0, p1, camera, mask=mask)
    return float(np.arctan2(rotation[0, 2], rotation[2, 2])), int(inliers)


@dataclass(frozen=True)
class Traversal:
    seq: Sequence
    stream: Stream
    matches: dict[str, np.ndarray]

    @property
    def wheel_turn(self) -> np.ndarray:
        return wrap(np.diff(self.stream.odometry[:, 2]))

    @property
    def truth_turn(self) -> np.ndarray:
        return wrap(np.diff(self.stream.ground_truth[:, 2]))

    @property
    def step_length(self) -> np.ndarray:
        return signed_step_lengths(self.stream.odometry)

    def pair(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = self.matches["offsets"][k], self.matches["offsets"][k + 1]
        return self.matches["points0"][lo:hi], self.matches["points1"][lo:hi]

    def visual_turns(self, focal: float, steps: np.ndarray) -> np.ndarray:
        return np.array([visual_yaw(*self.pair(k), focal)[0] for k in steps])


def window_residuals(turns: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Heading error summed over consecutive windows of steps that all have an estimate."""
    count = len(turns) // WINDOW_STEPS
    t = turns[: count * WINDOW_STEPS].reshape(count, WINDOW_STEPS)
    g = truth[: count * WINDOW_STEPS].reshape(count, WINDOW_STEPS)
    keep = np.isfinite(t).all(axis=1)
    return wrap(t[keep].sum(axis=1) - g[keep].sum(axis=1))


@dataclass(frozen=True)
class CameraFit:
    focal: float
    sign: float
    step_variance: float


def fit_camera(traversals: list[Traversal]) -> CameraFit:
    """Focal length and heading sign with the smallest median windowed error, then a robust step variance."""
    def subset(t: Traversal) -> np.ndarray:
        steps = np.arange(len(t.truth_turn))
        return steps[(steps // WINDOW_STEPS) % SEARCH_STRIDE == 0]

    best = None
    for focal in FOCAL_GRID:
        turns = {t.seq.name: t.visual_turns(focal, subset(t)) for t in traversals}
        for sign in (1.0, -1.0):
            residual = np.concatenate([window_residuals(sign * turns[t.seq.name], t.truth_turn[subset(t)])
                                       for t in traversals])
            score = float(np.median(np.abs(residual)))
            if best is None or score < best[0]:
                best = (score, focal, sign, residual)
    _, focal, sign, residual = best
    mad_sigma = 1.4826 * np.median(np.abs(residual - np.median(residual)))
    return CameraFit(float(focal), sign, float(mad_sigma**2 / WINDOW_STEPS))


def wheel_variance(t: Traversal, alpha: np.ndarray) -> np.ndarray:
    return alpha[0] * np.abs(t.wheel_turn) + alpha[1] * np.abs(t.step_length)


def heading_sources(t: Traversal, camera: CameraFit, alpha: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    wheel = t.wheel_turn
    visual = camera.sign * t.visual_turns(camera.focal, np.arange(len(wheel)))
    available = np.isfinite(visual)
    var_w, var_v = wheel_variance(t, alpha), camera.step_variance
    consistent = available & ((visual - wheel) ** 2 <= GATE_SIGMAS**2 * (var_w + var_v))
    gain = var_w / (var_w + var_v)
    fused = np.where(consistent, wheel + gain * (visual - wheel), wheel)
    sources = {"wheels": wheel, "vision": np.where(available, visual, wheel), "fused": fused}
    usage = {"steps": int(len(wheel)), "visual_available": float(available.mean()),
             "visual_fused": float(consistent.mean())}
    return sources, usage


def rebuild(t: Traversal, turns: np.ndarray) -> np.ndarray:
    """Planar track from the wheel step lengths and a heading increment per step."""
    heading = t.stream.odometry[0, 2] + np.concatenate([[0.0], np.cumsum(turns)])
    direction = heading[:-1] + turns / 2
    steps = t.step_length[:, None] * np.stack([np.cos(direction), np.sin(direction)], axis=1)
    xy = t.stream.odometry[0, :2] + np.concatenate([np.zeros((1, 2)), np.cumsum(steps, axis=0)])
    return np.column_stack([xy, heading])


def horizon_pairs(t: Traversal) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    travelled = np.concatenate([[0.0], np.cumsum(np.abs(t.step_length))])
    anchors = np.unique(np.searchsorted(travelled, np.arange(0.0, travelled[-1], ANCHOR_SPACING_M)))
    earlier, later, horizon = [], [], []
    for distance in HORIZONS_M:
        targets = np.searchsorted(travelled, travelled[anchors] + distance)
        keep = targets < len(travelled)
        earlier.append(anchors[keep])
        later.append(targets[keep])
        horizon.append(np.full(keep.sum(), distance))
    return tuple(np.concatenate(x) for x in (earlier, later, horizon))


def drift(track: np.ndarray, truth: np.ndarray, earlier: np.ndarray, later: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Heading and position error of the relative motion, in the frame of the earlier pose."""
    def relative(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c, s = np.cos(poses[earlier, 2]), np.sin(poses[earlier, 2])
        d = poses[later, :2] - poses[earlier, :2]
        return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1), poses[later, 2] - poses[earlier, 2]

    track_xy, track_turn = relative(track)
    truth_xy, truth_turn = relative(truth)
    return wrap(track_turn - truth_turn), np.linalg.norm(track_xy - truth_xy, axis=1)


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2))) if len(values) else float("nan")


def summarize(errors: dict[str, dict[str, list]], horizons: list[np.ndarray]) -> dict:
    horizon = np.concatenate(horizons)
    return {source: {str(h): {"heading_rms_deg": float(np.degrees(rms(np.concatenate(e["heading"])[horizon == h]))),
                              "position_rms_m": rms(np.concatenate(e["position"])[horizon == h])}
                     for h in HORIZONS_M}
            for source, e in errors.items()}


def evaluate(traversals: list[Traversal], camera: CameraFit, alpha: np.ndarray) -> dict:
    errors = {name: {"heading": [], "position": []} for name in ("wheels", "vision", "fused")}
    horizons, usage, per_traversal = [], {}, {}
    for t in traversals:
        sources, usage[t.seq.name] = heading_sources(t, camera, alpha)
        earlier, later, horizon = horizon_pairs(t)
        horizons.append(horizon)
        per_traversal[t.seq.name] = {}
        for name, turns in sources.items():
            heading, position = drift(rebuild(t, turns), t.stream.ground_truth, earlier, later)
            errors[name]["heading"].append(heading)
            errors[name]["position"].append(position)
            at32 = horizon == 32.0
            per_traversal[t.seq.name][name] = rms(position[at32])
    return {"camera": camera.__dict__, "usage": usage, "pooled": summarize(errors, horizons),
            "position_rms_at_32m": per_traversal}


def markdown(report: dict) -> str:
    lines = ["# Visual heading diagnostic (design traversals)", ""]
    for dataset, r in report.items():
        c = r["camera"]
        used = np.mean([u["visual_fused"] for u in r["usage"].values()])
        available = np.mean([u["visual_available"] for u in r["usage"].values()])
        lines += [f"## {dataset}", "",
                  f"Focal {c['focal']:.0f} px, sign {c['sign']:+.0f}, visual step σ "
                  f"{np.degrees(np.sqrt(c['step_variance'])):.2f}°. Visual estimate on {available:.0%} of steps, "
                  f"fused on {used:.0%}.", "",
                  "| Horizon [m] | " + " | ".join(f"{s} heading [°]" for s in r["pooled"]) + " | "
                  + " | ".join(f"{s} position [m]" for s in r["pooled"]) + " |",
                  "|" + "---|" * (1 + 2 * len(r["pooled"]))]
        for h in HORIZONS_M:
            cells = [f"{r['pooled'][s][str(h)]['heading_rms_deg']:.2f}" for s in r["pooled"]]
            cells += [f"{r['pooled'][s][str(h)]['position_rms_m']:.2f}" for s in r["pooled"]]
            lines.append(f"| {h:g} | " + " | ".join(cells) + " |")
        lines += ["", "Position RMS at 32 m per traversal [m]:", "",
                  "| Traversal | wheels | vision | fused |", "|---|---|---|---|"]
        lines += [f"| {n} | " + " | ".join(f"{v[s]:.2f}" for s in ("wheels", "vision", "fused")) + " |"
                  for n, v in r["position_rms_at_32m"].items()]
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=Path, required=True)
    args = parser.parse_args()
    cv2.setRNGSeed(0)
    alpha = np.asarray(json.loads(CALIBRATION_FILE.read_text())["models"]["propagated"]["alpha"])
    traversals = [Traversal(seq, load_stream(seq), dict(np.load(args.matches / f"{seq.name}.npz"))) for seq in DESIGN]
    report = {}
    for dataset in ("cold", "cid"):
        group = [t for t in traversals if t.seq.dataset == dataset]
        camera = fit_camera(group)
        print(dataset, camera, flush=True)
        report[dataset] = evaluate(group, camera, alpha)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "vo_diagnostic.json").write_text(json.dumps(report, indent=1))
    (RESULTS / "vo_diagnostic.md").write_text(markdown(report))
    print(markdown(report))


if __name__ == "__main__":
    main()
