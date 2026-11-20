"""SuperPoint + LightGlue matches between consecutive frames of the design traversals.

Writes ``<out>/<sequence>.npz``: the matches of the pair (k, k + 1) of the frame stream are
``points0[offsets[k]:offsets[k + 1]]`` and ``points1[...]``, in pixels.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "manuscript_gpu"))
from stream import DESIGN, load_stream  # noqa: E402

MAX_KEYPOINTS = 2048


class Matcher:
    def __init__(self, device: str) -> None:
        from lightglue import LightGlue, SuperPoint

        self.device = device
        self.extractor = SuperPoint(max_num_keypoints=MAX_KEYPOINTS).eval().to(device)
        self.matcher = LightGlue(features="superpoint").eval().to(device)

    @torch.inference_mode()
    def features(self, path: str) -> dict:
        from lightglue.utils import load_image

        return self.extractor.extract(load_image(path).to(self.device))

    @torch.inference_mode()
    def match(self, first: dict, second: dict) -> tuple[np.ndarray, np.ndarray]:
        from lightglue.utils import rbd

        pairs = rbd(self.matcher({"image0": first, "image1": second}))["matches"].cpu().numpy()
        k0, k1 = rbd(first)["keypoints"].cpu().numpy(), rbd(second)["keypoints"].cpu().numpy()
        return k0[pairs[:, 0]], k1[pairs[:, 1]]


def match_stream(matcher: Matcher, paths: list[str]) -> dict[str, np.ndarray]:
    offsets, points0, points1 = [0], [], []
    previous = matcher.features(paths[0])
    for path in paths[1:]:
        current = matcher.features(path)
        p0, p1 = matcher.match(previous, current)
        points0.append(p0)
        points1.append(p1)
        offsets.append(offsets[-1] + len(p0))
        previous = current
    return {"offsets": np.asarray(offsets), "points0": np.concatenate(points0).astype(np.float32),
            "points1": np.concatenate(points1).astype(np.float32)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    matcher = Matcher("cuda" if torch.cuda.is_available() else "cpu")
    for seq in DESIGN:
        target = args.out / f"{seq.name}.npz"
        if target.exists():
            continue
        stream = load_stream(seq)
        partial = args.out / f"{seq.name}.partial.npz"
        np.savez(partial, **match_stream(matcher, stream.image_paths))
        partial.rename(target)
        print(f"{seq.name}: {len(stream.image_paths) - 1} pairs", flush=True)


if __name__ == "__main__":
    main()
