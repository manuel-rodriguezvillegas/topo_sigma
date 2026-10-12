"""Odometry-independent geometric verifiers for a pair of images.

``LightGlueVerifier``: SuperPoint + LightGlue matches, fundamental-matrix
inliers (MAGSAC, no intrinsics needed).
``Mast3rVerifier``: MASt3R (metric checkpoint) reciprocal descriptor matches,
fundamental inliers, and a metric relative pose obtained without intrinsics:
the second image's pointmap is predicted in both camera frames (pairs (a, b)
and (b, a)) and aligned with a trimmed Kabsch fit.

Camera convention (OpenCV): x right, y down, z forward. The planar relative
heading is the rotation about the camera y axis; its sign is not compared with
odometry (only magnitudes are), so the convention cannot bias the summary.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np


def fundamental_inliers(p0: np.ndarray, p1: np.ndarray) -> int:
    import cv2

    if len(p0) < 8:
        return 0
    _, mask = cv2.findFundamentalMat(p0.astype(np.float64), p1.astype(np.float64), cv2.USAC_MAGSAC, 1.0,
                                     0.999, 10000)
    return 0 if mask is None else int(mask.sum())


class LightGlueVerifier:
    name = "lg"

    def __init__(self, device: str = "cuda", max_keypoints: int = 2048) -> None:
        import torch
        from lightglue import LightGlue, SuperPoint

        self.torch, self.device = torch, device
        self.extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
        self.matcher = LightGlue(features="superpoint").eval().to(device)
        self.cache: dict[str, dict] = {}

    def reset(self) -> None:
        self.cache.clear()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def _features(self, path: str) -> dict:
        if path not in self.cache:
            from lightglue.utils import load_image

            with self.torch.inference_mode():
                self.cache[path] = self.extractor.extract(load_image(path).to(self.device))
        return self.cache[path]

    def score(self, path_a: str, path_b: str) -> dict:
        from lightglue.utils import rbd

        with self.torch.inference_mode():
            fa, fb = self._features(path_a), self._features(path_b)
            out = rbd(self.matcher({"image0": fa, "image1": fb}))
            matches = out["matches"].cpu().numpy()
            ka, kb = rbd(fa)["keypoints"].cpu().numpy(), rbd(fb)["keypoints"].cpu().numpy()
        p0, p1 = ka[matches[:, 0]], kb[matches[:, 1]]
        return {"lg_matches": int(len(matches)), "lg_f_inliers": fundamental_inliers(p0, p1)}


def kabsch_trimmed(source: np.ndarray, target: np.ndarray, keep: float = 0.8, rounds: int = 3):
    """Rigid ``target ~ R source + t`` with iterative trimming of the worst residuals."""
    index = np.arange(len(source))
    for _ in range(rounds + 1):
        s, t = source[index], target[index]
        cs, ct = s.mean(axis=0), t.mean(axis=0)
        u, _, vt = np.linalg.svd((s - cs).T @ (t - ct))
        d = np.sign(np.linalg.det(vt.T @ u.T))
        rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
        translation = ct - rotation @ cs
        residual = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
        index = np.argsort(residual)[: max(int(keep * len(source)), 3)]
    return rotation, translation, float(np.median(residual[index]))


class Mast3rVerifier:
    name = "m"
    CHECKPOINT = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"

    def __init__(self, repo: Path, device: str = "cuda", batch_size: int = 8, size: int = 512) -> None:
        import torch

        sys.path.insert(0, str(repo))
        import mast3r.utils.path_to_dust3r  # noqa: F401  (puts dust3r on sys.path)
        from dust3r.inference import inference
        from dust3r.utils.image import load_images
        from mast3r.fast_nn import fast_reciprocal_NNs
        from mast3r.model import AsymmetricMASt3R

        self.torch, self.device, self.batch_size, self.size = torch, device, batch_size, size
        self._inference, self._load, self._nn = inference, load_images, fast_reciprocal_NNs
        self.model = AsymmetricMASt3R.from_pretrained(self.CHECKPOINT).to(device).eval()
        self.cache: dict[str, dict] = {}
        self.rng = np.random.default_rng(0)

    def reset(self) -> None:
        self.cache.clear()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def _image(self, path: str, idx: int) -> dict:
        if path not in self.cache:
            self.cache[path] = self._load([path], size=self.size, verbose=False)[0]
        image = dict(self.cache[path])
        image["idx"], image["instance"] = idx, str(idx)
        return image

    def score_many(self, pairs: list[tuple[str, str]]) -> list[dict]:
        results = []
        for first in range(0, len(pairs), self.batch_size):
            chunk = pairs[first:first + self.batch_size]
            try:
                results.extend(self._score_chunk(chunk))
            except Exception as error:  # one bad image must not stop a multi-hour run
                results.extend({"m_error": repr(error)[:300]} for _ in chunk)
        return results

    def _score_chunk(self, chunk: list[tuple[str, str]]) -> list[dict]:
        torch = self.torch
        started = time.perf_counter()
        views = []
        for k, (a, b) in enumerate(chunk):
            ia, ib = self._image(a, 2 * k), self._image(b, 2 * k + 1)
            views += [(ia, ib), (ib, ia)]
        # Mixed image sizes cannot share a batch; dust3r then returns per-pair lists.
        shapes = {tuple(int(x) for x in view["true_shape"].reshape(-1)) for pair in views for view in pair}
        batch = 2 * len(chunk) if len(shapes) == 1 else 1
        with torch.inference_mode():
            out = self._inference(views, self.model, self.device, batch_size=batch, verbose=False)
        elapsed = (time.perf_counter() - started) / len(chunk)
        pred1, pred2, view1 = out["pred1"], out["pred2"], out["view1"]
        results = []
        for k in range(len(chunk)):
            fwd, bwd = 2 * k, 2 * k + 1
            h, w = (int(v) for v in view1["true_shape"][fwd])
            xy0, xy1 = self._nn(pred1["desc"][fwd].to(self.device), pred2["desc"][fwd].to(self.device),
                                subsample_or_initxy1=8, device=self.device, dist="dot", block_size=2**13)
            xy0, xy1 = (np.asarray(x.cpu() if hasattr(x, "cpu") else x) for x in (xy0, xy1))
            border = ((xy0[:, 0] >= 3) & (xy0[:, 0] < w - 3) & (xy0[:, 1] >= 3) & (xy0[:, 1] < h - 3)
                      & (xy1[:, 0] >= 3) & (xy1[:, 0] < w - 3) & (xy1[:, 1] >= 3) & (xy1[:, 1] < h - 3))
            xy0, xy1 = xy0[border], xy1[border]
            # Image b in camera a (forward pass) and in camera b (backward pass), same pixel grid.
            in_a = pred2["pts3d_in_other_view"][fwd].reshape(-1, 3).float().cpu().numpy()
            in_b = pred1["pts3d"][bwd].reshape(-1, 3).float().cpu().numpy()
            conf = np.minimum(pred2["conf"][fwd].reshape(-1).float().cpu().numpy(),
                              pred1["conf"][bwd].reshape(-1).float().cpu().numpy())
            good = np.flatnonzero((conf >= np.quantile(conf, 0.5)) & np.isfinite(in_a).all(1) & np.isfinite(in_b).all(1))
            if len(good) > 20000:
                good = self.rng.choice(good, 20000, replace=False)
            row = {"m_matches": int(len(xy0)), "m_f_inliers": fundamental_inliers(xy0, xy1),
                   "m_conf_median": float(np.median(conf)), "m_seconds": elapsed}
            if len(good) >= 50:
                rotation, translation, residual = kabsch_trimmed(in_b[good], in_a[good])
                forward = rotation @ np.array([0.0, 0.0, 1.0])
                row.update({"m_yaw_abs": float(abs(np.arctan2(forward[0], forward[2]))),
                            "m_dist": float(np.hypot(translation[0], translation[2])),
                            "m_dy": float(translation[1]), "m_align_residual": residual,
                            "m_scene_depth": float(np.median(in_a[good][:, 2]))})
            results.append(row)
        return results
