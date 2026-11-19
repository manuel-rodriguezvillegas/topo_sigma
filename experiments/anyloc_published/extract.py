"""GPU descriptor extraction for the manuscript sequences and the held-out set.

Encoders (all frozen, no fitting on these data):

- ``dinov2_s``        DINOv2 ViT-S/14 CLS, 224x224 (the TopoSIGMA descriptor;
                      historical extractor code).
- ``anyloc_gem_s224`` AnyLoc-GeM adaptation of the manuscript: ViT-S/14 block 11
                      value facet, GeM p=3, 224x224 (historical extractor code).
- ``anyloc_gem_g224`` Same, with the published ViT-G/14 block 31 (backbone only).
- ``anyloc_gem_g``    Published AnyLoc-GeM: ViT-G/14 block 31 value facet at native
                      resolution (centre crop to a multiple of 14, no resize), GeM p=3.
- ``anyloc_vlad_g``   Published AnyLoc-VLAD-DINOv2: same features, hard VLAD with the
                      released 32-cluster *indoor* vocabulary, intra-normalised (49,152-D).
- ``dinov3_s``        DINOv3 ViT-S/16 CLS, pinned revision; the encoder of the current
                      ``vts_core`` (gated weights: needs ``hf auth login``). Not in the default list.

``anyloc_gem_g`` and ``anyloc_vlad_g`` share one forward pass. Outputs go to
``$OUT/descriptors/<sequence>/<encoder>.npy`` with the frame names; finished
files are skipped, so the script resumes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from common import ALL, frames
from vts_core.features import DINO_REVISION, AnyLocGeMExtractor, FrozenDinoV2Extractor, generalized_mean

CURRENT_DESCRIPTORS = (Path(__file__).resolve().parents[2]
                       / "vts_devcontainer/vts_ws/src/vts_core/vts_core/descriptors.py")
VOCAB_URL = "https://github.com/AnyLoc/DINO/releases/download/v1/dinov2_vitg14_l31_value_c32_indoor_c_centers.pt"
NATIVE = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_image(path: str) -> np.ndarray:
    import cv2

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read {path}")
    return image


class AnyLocViTG:
    """ViT-G/14 block-31 value facet at native resolution -> (GeM, VLAD-indoor)."""

    def __init__(self, device: str) -> None:
        self.device = device
        model = torch.hub.load(f"facebookresearch/dinov2:{DINO_REVISION}", "dinov2_vitg14", trust_repo=True)
        model.blocks = model.blocks[:32]  # blocks after 31 do not affect the hooked qkv output
        self.model = model.to(device).eval()
        self._qkv = None
        self.model.blocks[31].attn.qkv.register_forward_hook(lambda _m, _i, out: setattr(self, "_qkv", out))
        vocab_path = Path(torch.hub.get_dir()) / "checkpoints/anyloc_files" / VOCAB_URL.rsplit("/", 1)[-1]
        if not vocab_path.exists():
            vocab_path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(VOCAB_URL, str(vocab_path))
        centers = torch.load(vocab_path, map_location="cpu")
        if isinstance(centers, dict):  # be tolerant to a wrapped tensor
            centers = next(v for v in centers.values() if torch.is_tensor(v))
        assert centers.shape == (32, 1536), centers.shape
        self.centers = centers.float().to(device)
        self.centers_unit = F.normalize(self.centers, dim=-1)
        self.vocab_sha = __import__("hashlib").sha256(vocab_path.read_bytes()).hexdigest()

    @torch.no_grad()
    def __call__(self, images_bgr: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        tensors = []
        for image in images_bgr:
            t = NATIVE(Image.fromarray(image[:, :, ::-1].copy()))
            h, w = (t.shape[1] // 14) * 14, (t.shape[2] // 14) * 14
            tensors.append(transforms.functional.center_crop(t, [h, w]))
        batch = torch.stack(tensors).to(self.device)
        self._qkv = None
        self.model(batch)
        dim = self._qkv.shape[-1] // 3
        local = F.normalize(self._qkv[:, 1:, 2 * dim:].float(), dim=-1)  # [B, P, 1536]
        gem = F.normalize(generalized_mean(local, 3.0), dim=-1)
        # Hard VLAD as in AnyLoc: cosine assignment, residual sums, intra-norm, global L2.
        labels = (local @ self.centers_unit.T).argmax(dim=-1)  # [B, P]
        one_hot = F.one_hot(labels, num_classes=self.centers.shape[0]).float()  # [B, P, C]
        sums = one_hot.transpose(1, 2) @ local  # [B, C, D]
        counts = one_hot.sum(dim=1).unsqueeze(-1)  # [B, C, 1]
        residuals = sums - counts * self.centers.unsqueeze(0)
        vlad = F.normalize(F.normalize(residuals, dim=-1).flatten(1), dim=-1)
        return gem.cpu().numpy().astype(np.float32), vlad.cpu().numpy().astype(np.float16)


def current_descriptors():
    """The self-contained descriptor module of the current ``vts_core``, loaded beside the historical one."""
    spec = importlib.util.spec_from_file_location("current_vts_descriptors", CURRENT_DESCRIPTORS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+", default=list(ALL))
    parser.add_argument("--encoders", nargs="+",
                        default=["dinov2_s", "anyloc_gem_s224", "anyloc_gem_g224", "anyloc_gem_g", "anyloc_vlad_g"])
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False  # keep fp32 semantics close to the CPU reference
    torch.backends.cudnn.allow_tf32 = False

    single = {
        "dinov2_s": lambda: FrozenDinoV2Extractor("dinov2_vits14", device=device),
        "anyloc_gem_s224": lambda: AnyLocGeMExtractor("dinov2_vits14", layer=11, device=device),
        "anyloc_gem_g224": lambda: AnyLocGeMExtractor("dinov2_vitg14", layer=31, device=device),
    }
    identities = {}
    for name in [e for e in args.encoders if e in single]:
        extractor = single[name]()
        identities[name] = {"code": "historical vts_core.features", "device": device}
        for seq_name in args.sequences:
            target = args.out / "descriptors" / seq_name / f"{name}.npy"
            if target.exists():
                continue
            start, names, vectors = time.time(), [], []
            for image_path, *_ in frames(ALL[seq_name]):
                vectors.append(extractor.extract(load_image(image_path)))
                names.append(os.path.basename(image_path))
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(target.with_suffix(".frames.npy"), np.array(names))
            np.save(target, np.stack(vectors).astype(np.float32))
            print(f"{name} {seq_name}: {len(names)} frames, {time.time() - start:.0f} s", flush=True)
        del extractor
        torch.cuda.empty_cache()

    if "dinov3_s" in args.encoders:
        descriptors = current_descriptors()
        encoder = descriptors.DinoV3(device)
        identities["dinov3_s"] = encoder.identity
        for seq_name in args.sequences:
            target = args.out / "descriptors" / seq_name / "dinov3_s.npy"
            if target.exists():
                continue
            paths = [Path(p) for p, *_ in frames(ALL[seq_name])]
            vectors = descriptors.encode_paths(encoder, paths, batch_size=args.batch, label=f"dinov3_s {seq_name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(target.with_suffix(".frames.npy"), np.array([p.name for p in paths]))
            np.save(target, vectors)
        del encoder
        torch.cuda.empty_cache()

    if {"anyloc_gem_g", "anyloc_vlad_g"} & set(args.encoders):
        vitg = AnyLocViTG(device)
        identities["anyloc_vitg_native"] = {"vocabulary": VOCAB_URL, "vocabulary_sha256": vitg.vocab_sha,
                                            "dinov2_revision": DINO_REVISION, "layer": 31, "facet": "value"}
        for seq_name in args.sequences:
            gem_t = args.out / "descriptors" / seq_name / "anyloc_gem_g.npy"
            vlad_t = args.out / "descriptors" / seq_name / "anyloc_vlad_g.npy"
            if gem_t.exists() and vlad_t.exists():
                continue
            start, names, gems, vlads = time.time(), [], [], []
            stream = [p for p, *_ in frames(ALL[seq_name])]
            for k in range(0, len(stream), args.batch):
                paths = stream[k:k + args.batch]
                gem, vlad = vitg([load_image(p) for p in paths])
                gems.append(gem)
                vlads.append(vlad)
                names += [os.path.basename(p) for p in paths]
            gem_t.parent.mkdir(parents=True, exist_ok=True)
            for t in (gem_t, vlad_t):
                np.save(t.with_suffix(".frames.npy"), np.array(names))
            np.save(gem_t, np.concatenate(gems))
            np.save(vlad_t, np.concatenate(vlads))
            print(f"anyloc_vitg {seq_name}: {len(names)} frames, {time.time() - start:.0f} s", flush=True)
    (args.out / "descriptors").mkdir(parents=True, exist_ok=True)
    path = args.out / "descriptors" / "identities.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps(previous | identities, indent=2, default=str))


if __name__ == "__main__":
    main()
