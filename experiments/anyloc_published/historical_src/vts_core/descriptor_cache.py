"""Reusable frozen descriptors keyed by exact decoded image and encoder identity."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np

from vts_core.artifacts import atomic_bytes, identity


class DescriptorCache:
    def __init__(self, extractor, directory: str, model_identity: dict):
        self.extractor = extractor
        self.model_identity = model_identity
        self.directory = Path(directory) / identity(model_identity)
        self.hits = 0
        self.misses = 0

    @property
    def device(self) -> str:
        return self.extractor.device

    def _valid(self, vector: np.ndarray) -> bool:
        return (
            vector.dtype == np.float32
            and vector.ndim == 1
            and vector.size >= 3
            and vector.size == self.model_identity.get("dimension", vector.size)
            and np.isfinite(vector).all()
            and abs(np.linalg.norm(vector) - 1) < 1e-4
        )

    def extract(self, image_bgr: np.ndarray) -> np.ndarray:
        image = np.ascontiguousarray(image_bgr)
        digest = hashlib.sha256(identity([image.shape, image.dtype.str]).encode())
        digest.update(memoryview(image))
        path = self.directory / (digest.hexdigest() + ".npy")
        try:
            vector = np.load(path, allow_pickle=False)
            if self._valid(vector):
                self.hits += 1
                return vector
        except (OSError, ValueError, EOFError):
            pass
        vector = self.extractor.extract(image)
        if not self._valid(vector):
            raise ValueError("encoder produced an invalid normalized float32 descriptor")
        payload = io.BytesIO()
        np.save(payload, vector, allow_pickle=False)
        atomic_bytes(path, payload.getvalue())
        self.misses += 1
        return vector
