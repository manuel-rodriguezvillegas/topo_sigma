"""Appearance-only descriptor stores from a VPR-trained model (C4 screen).

The frame store of :mod:`regime_study` fixes frames, segmentation and node
poses with DINOv3. This module adds a second, frame-aligned descriptor array
used only for loop association, so both encoders face identical nodes and
revisit opportunities.

MegaLoc (Berton & Masone, 2025): DINOv2-B/14 backbone with SALAD-style optimal
transport aggregation, 8448-D L2-normalized output. Code is loaded from the
official repository at a pinned commit; weights are the official safetensors
file, whose SHA-256 is recorded. Preprocessing follows the authors' evaluation
example: RGB, ImageNet normalization, resize to 322x322.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as tf

from vts_core.artifacts import atomic_json, file_digest
from vts_evaluation.regime_study import WORKSPACE, session_config

MEGALOC_REPO = "gmberton/MegaLoc"
MEGALOC_COMMIT = "5fe0dd697c4a70ba3e23607f6716ab3c606b16db"
MEGALOC_WEIGHTS = ("gberton/MegaLoc", "model.safetensors")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MegaLocExtractor:
    def __init__(self, threads: int = 8) -> None:
        from huggingface_hub import hf_hub_download

        torch.set_num_threads(threads)
        self.model = torch.hub.load(f"{MEGALOC_REPO}:{MEGALOC_COMMIT}", "get_trained_model",
                                    trust_repo=True).eval()
        weights = hf_hub_download(*MEGALOC_WEIGHTS)
        digest = hashlib.sha256()
        with open(weights, "rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        self.model_identity = {
            "model": "megaloc", "repo": MEGALOC_REPO, "commit": MEGALOC_COMMIT,
            "weights": "/".join(MEGALOC_WEIGHTS), "weights_sha256": digest.hexdigest(),
            "preprocessing": {"color": "RGB", "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
                              "resize": [322, 322], "antialias": True},
            "dimension": int(self.model.feat_dim), "torch": torch.__version__, "device": "cpu",
        }

    @torch.inference_mode()
    def extract(self, images_bgr: list[np.ndarray]) -> np.ndarray:
        batch = []
        for image in images_bgr:
            tensor = tf.to_tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            tensor = tf.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
            batch.append(tf.resize(tensor, [322, 322], antialias=True))
        output = self.model(torch.stack(batch)).numpy().astype(np.float32)
        if not np.isfinite(output).all() or np.any(np.abs(np.linalg.norm(output, axis=1) - 1) > 1e-3):
            raise ValueError("MegaLoc produced an invalid normalized descriptor")
        return output


def build_store(session: str, root: Path, extractor: MegaLocExtractor, batch_size: int) -> dict:
    base = json.loads((root / "frames" / f"{session}.json").read_text())
    config, _, _ = session_config(session)
    dataset_root = Path(config["player"]["dataset_root"])
    descriptors = []
    started = time.perf_counter()
    paths = base["image_paths"]
    for start in range(0, len(paths), batch_size):
        images = [cv2.imread(str(dataset_root / p), cv2.IMREAD_COLOR) for p in paths[start:start + batch_size]]
        if any(image is None for image in images):
            raise ValueError(f"{session}: unreadable image near frame {start}")
        descriptors.append(extractor.extract(images))
        done = start + len(images)
        if done % (batch_size * 32) < batch_size:
            print(f"{session}: {done}/{len(paths)} ({done / (time.perf_counter() - started):.2f}/s)", flush=True)
    output = root / "frames" / f"{session}__megaloc.npz"
    np.savez_compressed(output, descriptors=np.concatenate(descriptors))
    meta = {"session": session, "frames": len(paths), "encoder": extractor.model_identity,
            "base_frames_sha256": base["frames_sha256"], "frames_sha256": file_digest(output)}
    atomic_json(root / "frames" / f"{session}__megaloc.json", meta)
    return meta


def load_appearance(session: str, root: Path, encoder: str) -> np.ndarray:
    meta = json.loads((root / "frames" / f"{session}__{encoder}.json").read_text())
    base = json.loads((root / "frames" / f"{session}.json").read_text())
    path = root / "frames" / f"{session}__{encoder}.npz"
    if meta["base_frames_sha256"] != base["frames_sha256"] or file_digest(path) != meta["frames_sha256"]:
        raise ValueError(f"{session}: appearance store does not match its frame store")
    descriptors = np.load(path)["descriptors"]
    if len(descriptors) != base["frames"]:
        raise ValueError(f"{session}: appearance store has the wrong number of frames")
    return descriptors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sessions", nargs="+")
    parser.add_argument("--frames", type=Path, default=WORKSPACE / "output/regime_study_v1")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    extractor = MegaLocExtractor()
    print(json.dumps(extractor.model_identity), flush=True)
    for session in args.sessions:
        if (args.frames / "frames" / f"{session}__megaloc.json").exists():
            print(f"{session}: exists, skipped", flush=True)
            continue
        meta = build_store(session, args.frames, extractor, args.batch_size)
        print(f"{session}: {meta['frames']} frames done", flush=True)


if __name__ == "__main__":
    main()
