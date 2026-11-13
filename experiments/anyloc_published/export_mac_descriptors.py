"""Mac only: export the exact manuscript DINOv2-S descriptors in frame order.

Reads the descriptor cache that produced the submission (keyed by decoded image
bytes and encoder identity) and writes ``reference/dinov2_vits14_mac.npz``.
Every frame must be a cache hit, which also proves that the decoded images and
frame order are identical to the submission run. The GPU machine replays the
manuscript mapper from these vectors, so segmentation does not depend on
CPU/GPU numerical differences.

    uv run --with "numpy<2" --with opencv-python-headless python export_mac_descriptors.py
"""

from __future__ import annotations

import json

import cv2
import numpy as np

from common import DEV, HERE, REPO, frames
from vts_core.descriptor_cache import DescriptorCache

OUTPUT = REPO / "vts_devcontainer/vts_ws/output"
IDENTITY = json.loads((OUTPUT / "dino_v2_v3_controlled_2026_09_27/offline_summary.json").read_text())[
    "protocol"]["encoder_identities"]["v2"]


class CacheOnly:
    device = "cpu"

    def extract(self, _image):
        raise RuntimeError("cache miss: decoded image or frame stream differs from the submission run")


def main() -> None:
    cache = DescriptorCache(CacheOnly(), str(OUTPUT / "cache/descriptors"), IDENTITY)
    arrays: dict[str, np.ndarray] = {}
    for seq in DEV:
        names, vectors = [], []
        for image_path, *_ in frames(seq):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            vectors.append(cache.extract(image))
            names.append(image_path.rsplit("/", 1)[-1])
        arrays[f"{seq.name}/descriptors"] = np.stack(vectors).astype(np.float32)
        arrays[f"{seq.name}/frames"] = np.array(names)
        print(f"{seq.name}: {len(names)} frames", flush=True)
    target = HERE / "reference/dinov2_vits14_mac.npz"
    np.savez_compressed(target, **arrays)
    print(f"hits={cache.hits} misses={cache.misses} -> {target}")


if __name__ == "__main__":
    main()
