"""Frozen visual foundation-model feature extraction.

The mapper deliberately uses an unmodified DINOv3 Small backbone by default. There is no
training, projection head, checkpoint selection, or dataset-specific state in
the mapping pipeline; evaluation sequences therefore never enter a learning
or model-selection path.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms

from vts_core.artifacts import file_digest
from vts_core.descriptor_cache import DescriptorCache

DINO_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINOV3_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"
DINOV3_REVISION = "114c1379950215c8b35dfcd4e90a5c251dde0d32"
DEFAULT_DINO_MODEL = "dinov3_vits16"


class FeatureExtractor(Protocol):
    """Maps a BGR image to an L2-normalized descriptor."""

    model_identity: dict[str, object]

    @property
    def device(self) -> str:
        """Execution device used by the underlying model."""
        raise NotImplementedError

    def extract(self, image_bgr: np.ndarray) -> np.ndarray:
        raise NotImplementedError


_IMAGENET_TRANSFORM: Callable[[Image.Image], torch.Tensor] = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _to_pil(image_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(image_bgr[:, :, ::-1].copy())


def resolve_torch_device(requested: str = "auto") -> torch.device:
    """Resolve an inference device, preferring CUDA and then Apple MPS.

    MPS is usable only when PyTorch runs natively on macOS. Linux containers
    on Docker Desktop do not expose Apple's Metal device, so they correctly
    fall back to CPU.
    """
    choice = requested.strip().lower()
    supported = {"auto", "cuda", "mps", "cpu"}
    if choice not in supported:
        raise ValueError(
            f"Unsupported DINO device {requested!r}; choose one of {sorted(supported)}"
        )

    cuda_available = torch.cuda.is_available()
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend and mps_backend.is_available())

    if choice == "auto":
        choice = "cuda" if cuda_available else "mps" if mps_available else "cpu"
    elif choice == "cuda" and not cuda_available:
        raise RuntimeError("DINO device 'cuda' was requested, but CUDA is not available to PyTorch")
    elif choice == "mps" and not mps_available:
        raise RuntimeError("DINO device 'mps' was requested, but MPS is not available to PyTorch")
    return torch.device(choice)


class FrozenDinoV3Extractor(FeatureExtractor):
    """Official DINOv3 ViT-S/16 CLS descriptor, with pinned weights and processor."""

    def __init__(self, device: str = "auto") -> None:
        from huggingface_hub import hf_hub_download
        from transformers import AutoImageProcessor, AutoModel

        self._device = resolve_torch_device(device)
        try:
            self._processor = AutoImageProcessor.from_pretrained(
                DINOV3_MODEL, revision=DINOV3_REVISION
            )
            self._model = AutoModel.from_pretrained(
                DINOV3_MODEL, revision=DINOV3_REVISION
            ).to(self._device).eval()
        except OSError as exc:
            raise RuntimeError(
                "Could not load DINOv3 Small. Request access to "
                f"https://huggingface.co/{DINOV3_MODEL}, then authenticate with "
                "`hf auth login` in the execution environment (or set HF_TOKEN). "
                "For offline runs, cache this pinned revision first."
            ) from exc
        self.checkpoint = Path(hf_hub_download(
            DINOV3_MODEL, "model.safetensors", revision=DINOV3_REVISION,
            local_files_only=True,
        ))

    @property
    def device(self) -> str:
        return str(self._device)

    @torch.no_grad()
    def extract(self, image_bgr: np.ndarray) -> np.ndarray:
        inputs = self._processor(images=_to_pil(image_bgr), return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        # Pooler output is the CLS token, excluding the four register tokens.
        features = self._model(**inputs).pooler_output
        return F.normalize(features, dim=-1)[0].cpu().numpy().astype(np.float32)


class FrozenDinoV2Extractor(FeatureExtractor):
    """Stock DINOv2 ViT-S/14 loaded from the official Torch Hub repository."""

    def __init__(self, model_name: str = "dinov2_vits14", device: str = "auto") -> None:
        self._device = resolve_torch_device(device)
        self._model = torch.hub.load(
            f"facebookresearch/dinov2:{DINO_REVISION}", model_name, trust_repo=True
        )
        self._model.to(self._device).eval()

    @property
    def device(self) -> str:
        return str(self._device)

    @torch.no_grad()
    def extract(self, image_bgr: np.ndarray) -> np.ndarray:
        tensor = _IMAGENET_TRANSFORM(_to_pil(image_bgr)).unsqueeze(0)
        features = self._model(tensor.to(self._device))
        vector = features.reshape(-1).cpu().numpy().astype(np.float32)
        return vector / max(float(np.linalg.norm(vector)), 1e-12)


def generalized_mean(features: torch.Tensor, power: float = 3.0) -> torch.Tensor:
    """Pool local descriptors with the signed generalized mean.

    DINO attention features contain negative values. The signed formulation
    is real-valued for the AnyLoc exponent (p=3), unlike a fractional
    ``torch.pow`` applied directly to negative values.
    """
    if features.ndim != 3 or features.shape[1] == 0:
        raise ValueError("Expected local descriptors with shape [B, patches, D]")
    if not np.isfinite(power) or power <= 0:
        raise ValueError("GeM power must be positive")
    moment = features.sign() * features.abs().pow(power)
    pooled = moment.mean(dim=1)
    return pooled.sign() * pooled.abs().pow(1.0 / power)


class AnyLocGeMExtractor(FeatureExtractor):
    """Training-free AnyLoc-GeM descriptor from DINOv2 value facets.

    The published AnyLoc configuration uses DINOv2 ViT-G/14, block 31,
    value-facet patch descriptors and fixed GeM power p=3. Smaller DINOv2
    models can be selected for a computationally matched experiment; in that
    case ``layer`` must identify a valid block and the result must be reported
    as a scaled AnyLoc-GeM configuration.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitg14",
        layer: int = 31,
        device: str = "auto",
        power: float = 3.0,
    ) -> None:
        self._device = resolve_torch_device(device)
        self._model = torch.hub.load(
            f"facebookresearch/dinov2:{DINO_REVISION}", model_name, trust_repo=True
        )
        self._model.to(self._device).eval()
        if not 0 <= layer < len(self._model.blocks):
            raise ValueError(
                f"DINO layer {layer} is invalid for {model_name}; "
                f"expected 0..{len(self._model.blocks) - 1}"
            )
        self._power = power
        self._hook_output: torch.Tensor | None = None
        self._hook_handle = self._model.blocks[layer].attn.qkv.register_forward_hook(
            self._capture_qkv
        )

    def _capture_qkv(
        self,
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self._hook_output = output

    @property
    def device(self) -> str:
        return str(self._device)

    @torch.no_grad()
    def extract(self, image_bgr: np.ndarray) -> np.ndarray:
        tensor = _IMAGENET_TRANSFORM(_to_pil(image_bgr)).unsqueeze(0)
        self._hook_output = None
        self._model(tensor.to(self._device))
        if self._hook_output is None:
            raise RuntimeError("DINOv2 attention hook produced no descriptors")

        # qkv output: [batch, CLS + patches, 3 * embedding_dimension].
        qkv = self._hook_output[:, 1:, :]
        dimension = qkv.shape[-1] // 3
        local = F.normalize(qkv[..., 2 * dimension :], dim=-1)
        pooled = F.normalize(generalized_mean(local, self._power), dim=-1)
        return pooled[0].cpu().numpy().astype(np.float32)

    def __del__(self) -> None:
        handle = getattr(self, "_hook_handle", None)
        if handle is not None:
            handle.remove()


def build_extractor(
    model_name: str = DEFAULT_DINO_MODEL,
    device: str = "auto",
    backend: str = "dino_cls",
    layer: int = -1,
    cache_dir: str | None = None,
) -> FeatureExtractor:
    """Build a frozen global descriptor extractor."""
    is_dinov3 = model_name in {DEFAULT_DINO_MODEL, DINOV3_MODEL}
    if backend == "dino_cls":
        extractor = (
            FrozenDinoV3Extractor(device=device)
            if is_dinov3 else FrozenDinoV2Extractor(model_name, device=device)
        )
    elif backend == "anyloc_gem":
        if layer < 0:
            raise ValueError("anyloc_gem requires an explicit DINO block index")
        if not model_name.startswith("dinov2_"):
            raise ValueError("anyloc_gem requires a DINOv2 backbone and its value-facet layout")
        extractor = AnyLocGeMExtractor(model_name, layer=layer, device=device)
    else:
        raise ValueError(f"Unknown feature backend {backend!r}")
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / f"{model_name}_pretrain.pth"
    if is_dinov3:
        checkpoint = extractor.checkpoint
    model_identity = {
        "model": DINOV3_MODEL if is_dinov3 else model_name,
        "revision": DINOV3_REVISION if is_dinov3 else DINO_REVISION,
        "backend": backend,
        "layer": layer,
        "gem_power": 3.0,
        "device": extractor.device,
        "dimension": int(
            extractor._model.config.hidden_size if is_dinov3 else extractor._model.embed_dim
        ),
        "preprocessing_source_sha256": file_digest(__file__),
        "checkpoint_sha256": file_digest(checkpoint) if checkpoint.is_file() else None,
        "packages": {name: version(name) for name in ("torch", "torchvision", "numpy", "pillow")},
    }
    if is_dinov3:
        model_identity["processor"] = extractor._processor.to_dict()
        model_identity["packages"]["transformers"] = version("transformers")
    extractor.model_identity = model_identity
    return DescriptorCache(extractor, cache_dir, model_identity) if cache_dir else extractor
