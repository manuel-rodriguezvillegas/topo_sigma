"""Frozen image encoders and per-session descriptor extraction.

No training, projection head or dataset-specific state: descriptors come from
pinned public weights. ``dinov3`` (ViT-S/16 CLS token) drives segmentation and
is the default appearance model; ``megaloc`` (VPR-trained) and ``anyloc``
(training-free DINOv2 value-facet GeM) are optional appearance baselines.
Torch is imported only here, so the rest of the package runs without it.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

DINOV3_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"
DINOV3_REVISION = "114c1379950215c8b35dfcd4e90a5c251dde0d32"
DINOV2_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
MEGALOC_REPO, MEGALOC_COMMIT = "gmberton/MegaLoc", "5fe0dd697c4a70ba3e23607f6716ab3c606b16db"
MEGALOC_WEIGHTS = ("gberton/MegaLoc", "model.safetensors")
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def _sha256(path: str | Path) -> str:
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resolve_device(requested: str = "auto"):
    """CUDA, then Apple MPS (native macOS only), then CPU."""
    import torch

    choice = requested.strip().lower()
    if choice not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError(f"unsupported device {requested!r}")
    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "mps" if mps else "cpu"
    elif (choice == "cuda" and not torch.cuda.is_available()) or (choice == "mps" and not mps):
        raise RuntimeError(f"device {choice!r} is not available to PyTorch")
    return torch.device(choice)


class Encoder:
    """Maps a batch of BGR images to L2-normalized float32 descriptors."""

    name: str
    identity: dict

    def encode(self, images_bgr: list[np.ndarray]) -> np.ndarray:
        raise NotImplementedError


class DinoV3(Encoder):
    name = "dinov3"

    def __init__(self, device: str = "auto") -> None:
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoImageProcessor, AutoModel

        self._torch, self._device = torch, resolve_device(device)
        try:
            self._processor = AutoImageProcessor.from_pretrained(DINOV3_MODEL, revision=DINOV3_REVISION)
            self._model = AutoModel.from_pretrained(DINOV3_MODEL, revision=DINOV3_REVISION).to(self._device).eval()
        except OSError as error:
            raise RuntimeError(f"Cannot load {DINOV3_MODEL}: request access on Hugging Face and run "
                               "`hf auth login` (or set HF_TOKEN), or cache the pinned revision.") from error
        checkpoint = hf_hub_download(DINOV3_MODEL, "model.safetensors", revision=DINOV3_REVISION,
                                     local_files_only=True)
        self.identity = {"model": DINOV3_MODEL, "revision": DINOV3_REVISION, "output": "pooler_cls",
                         "checkpoint_sha256": _sha256(checkpoint), "processor": self._processor.to_dict(),
                         "torch": torch.__version__, "device": str(self._device)}

    def encode(self, images_bgr: list[np.ndarray]) -> np.ndarray:
        from PIL import Image

        torch = self._torch
        with torch.inference_mode():
            pil = [Image.fromarray(np.ascontiguousarray(image[:, :, ::-1])) for image in images_bgr]
            inputs = {k: v.to(self._device) for k, v in self._processor(images=pil, return_tensors="pt").items()}
            features = self._model(**inputs).pooler_output  # CLS token, registers excluded
            return torch.nn.functional.normalize(features, dim=-1).cpu().numpy().astype(np.float32)


def _imagenet_batch(images_bgr: list[np.ndarray], size: int):
    import cv2
    import torch
    import torchvision.transforms.functional as tf

    batch = []
    for image in images_bgr:
        tensor = tf.normalize(tf.to_tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), IMAGENET_MEAN, IMAGENET_STD)
        batch.append(tf.resize(tensor, [size, size], antialias=True))
    return torch.stack(batch)


class MegaLoc(Encoder):
    """MegaLoc (Berton & Masone, 2025): DINOv2-B/14 + SALAD-style aggregation, 8448-D."""

    name = "megaloc"

    def __init__(self, device: str = "cpu") -> None:
        import torch
        from huggingface_hub import hf_hub_download

        self._torch, self._device = torch, resolve_device(device)
        self._model = torch.hub.load(f"{MEGALOC_REPO}:{MEGALOC_COMMIT}", "get_trained_model",
                                     trust_repo=True).to(self._device).eval()
        self.identity = {"model": "megaloc", "repo": MEGALOC_REPO, "commit": MEGALOC_COMMIT,
                         "weights_sha256": _sha256(hf_hub_download(*MEGALOC_WEIGHTS)),
                         "preprocessing": {"color": "RGB", "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
                                           "resize": [322, 322], "antialias": True},
                         "torch": torch.__version__, "device": str(self._device)}

    def encode(self, images_bgr: list[np.ndarray]) -> np.ndarray:
        with self._torch.inference_mode():
            return self._model(_imagenet_batch(images_bgr, 322).to(self._device)).cpu().numpy().astype(np.float32)


class AnyLocGeM(Encoder):
    """Training-free AnyLoc-GeM: DINOv2 value-facet patches, signed GeM (p = 3).

    The published configuration is ViT-G/14 block 31; smaller backbones must be
    reported as a scaled AnyLoc-GeM.
    """

    name = "anyloc"

    def __init__(self, model_name: str = "dinov2_vitg14", layer: int = 31, device: str = "auto") -> None:
        import torch

        self._torch, self._device = torch, resolve_device(device)
        self._model = torch.hub.load(f"facebookresearch/dinov2:{DINOV2_REVISION}", model_name,
                                     trust_repo=True).to(self._device).eval()
        if not 0 <= layer < len(self._model.blocks):
            raise ValueError(f"layer {layer} is invalid for {model_name}")
        self._captured = None
        self._model.blocks[layer].attn.qkv.register_forward_hook(lambda _m, _i, out: setattr(self, "_captured", out))
        self.identity = {"model": model_name, "revision": DINOV2_REVISION, "layer": layer, "gem_power": 3.0,
                         "torch": torch.__version__, "device": str(self._device)}

    def encode(self, images_bgr: list[np.ndarray]) -> np.ndarray:
        torch = self._torch
        with torch.inference_mode():
            self._model(_imagenet_batch(images_bgr, 224).to(self._device))
            qkv = self._captured[:, 1:, :]  # [batch, patches, 3 * dim]
            dim = qkv.shape[-1] // 3
            local = torch.nn.functional.normalize(qkv[..., 2 * dim:], dim=-1)
            moment = (local.sign() * local.abs().pow(3.0)).mean(dim=1)
            pooled = moment.sign() * moment.abs().pow(1.0 / 3.0)
            return torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy().astype(np.float32)


ENCODERS: dict[str, Callable[..., Encoder]] = {"dinov3": DinoV3, "megaloc": MegaLoc, "anyloc": AnyLocGeM}


def build_encoder(name: str, device: str = "auto") -> Encoder:
    if name not in ENCODERS:
        raise ValueError(f"unknown encoder {name!r}; choose from {sorted(ENCODERS)}")
    return ENCODERS[name](device=device)


def encode_paths(encoder: Encoder, paths: list[Path], batch_size: int = 8, label: str = "") -> np.ndarray:
    """Encode images from disk in batches, logging throughput."""
    import cv2

    out, started = [], time.perf_counter()
    for first in range(0, len(paths), batch_size):
        images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths[first:first + batch_size]]
        if any(image is None for image in images):
            raise ValueError(f"{label}: unreadable image near frame {first}")
        out.append(encoder.encode(images))
        done = first + len(images)
        if done % (batch_size * 50) < batch_size or done == len(paths):
            print(f"{label}: {done}/{len(paths)} frames ({done / (time.perf_counter() - started):.1f}/s)",
                  flush=True)
    descriptors = np.concatenate(out)
    if not np.isfinite(descriptors).all() or np.max(np.abs(np.linalg.norm(descriptors, axis=1) - 1)) > 1e-3:
        raise ValueError(f"{label}: encoder produced invalid descriptors")
    return descriptors
