"""Canonical per-session store: frames, raw odometry, ground truth, descriptors.

Layout of ``<store_root>/<session>/``::

    manifest.json   recording, adapter settings, frame count, SHA-256 of every
                    file, encoder identities and provenance
    frames.npz      timestamps, gt (N, 3), raw odometry (odom_t, odom_poses),
                    labels and image paths (relative to the dataset root)
    <encoder>.npy   (N, D) float32 unit descriptors, frame aligned

One store replaces thousands of per-image cache files. Every file is verified
against the manifest when read. Ground truth and labels are evaluation-only.
"""

from __future__ import annotations

import io
import json
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from vts_core.datasets import RawFrames, Recording, catalogue, read_recording, recording_for
from vts_core.odometry import MotionNoise, budget, interpolate
from vts_core.provenance import atomic_bytes, atomic_json, code_identity, file_digest, resolve

STORE_VERSION = 1


class TestDataError(PermissionError):
    """A test session was requested without explicit permission."""

    __test__ = False  # not a pytest test class


@dataclass
class Session:
    name: str
    recording: Recording
    manifest: dict
    timestamps: np.ndarray
    gt: np.ndarray
    labels: list[str | None]
    image_paths: list[str]
    odom_t: np.ndarray
    odom_poses: np.ndarray
    directory: Path

    def __len__(self) -> int:
        return len(self.timestamps)

    @property
    def encoders(self) -> list[str]:
        return sorted(self.manifest["encoders"])

    def descriptors(self, encoder: str = "dinov3") -> np.ndarray:
        if encoder not in self.manifest["encoders"]:
            raise KeyError(f"{self.name}: no {encoder} descriptors (have {self.encoders})")
        path = self.directory / f"{encoder}.npy"
        if file_digest(path) != self.manifest["files"][path.name]:
            raise ValueError(f"{path}: hash does not match the manifest")
        array = np.load(path, allow_pickle=False)
        if array.shape[0] != len(self) or array.dtype != np.float32:
            raise ValueError(f"{path}: expected ({len(self)}, D) float32")
        return array

    def camera_odometry(self, odom_poses: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Odometry pose and cumulative budget at every frame.

        ``odom_poses`` replaces the recorded poses (e.g. a perturbed copy); the
        budget is always recomputed from the poses actually used.
        """
        adapter = self.manifest["adapter"]
        raw = self.odom_poses if odom_poses is None else np.asarray(odom_poses, dtype=np.float64)
        budgets = budget(raw, MotionNoise(*adapter["motion_noise"]))
        return interpolate(self.odom_t, raw, budgets, self.timestamps,
                           adapter["max_odometry_gap_s"], adapter["exact_atol_s"])

    @cached_property
    def odometry(self) -> tuple[np.ndarray, np.ndarray]:
        return self.camera_odometry()


def store_directory(config: dict, name: str) -> Path:
    return resolve(config, "store_root") / name


def _frames_bytes(raw: RawFrames) -> bytes:
    payload = io.BytesIO()
    np.savez_compressed(
        payload, timestamps=raw.timestamps, gt=raw.gt, odom_t=raw.odom_t, odom_poses=raw.odom_poses,
        labels=np.asarray(["" if label is None else label for label in raw.labels], dtype=str),
        image_paths=np.asarray(raw.image_paths, dtype=str))
    return payload.getvalue()


def write_session(config: dict, recording: Recording, raw: RawFrames, source: dict) -> Path:
    """Create (or replace) a store with frames only; descriptors are added later."""
    directory = store_directory(config, recording.name)
    atomic_bytes(directory / "frames.npz", _frames_bytes(raw))
    manifest = {
        "store_version": STORE_VERSION, "session": recording.name, "recording": asdict(recording),
        "adapter": config["datasets"][recording.dataset], "frames": len(raw.timestamps),
        "files": {"frames.npz": file_digest(directory / "frames.npz")}, "encoders": {},
        "source": source, "code_sha256": code_identity(),
    }
    atomic_json(directory / "manifest.json", manifest)
    return directory


def add_descriptors(directory: Path, encoder: str, descriptors: np.ndarray, encoder_identity: dict) -> None:
    manifest = json.loads((directory / "manifest.json").read_text())
    descriptors = np.ascontiguousarray(descriptors, dtype=np.float32)
    if descriptors.ndim != 2 or len(descriptors) != manifest["frames"] or not np.isfinite(descriptors).all():
        raise ValueError(f"{directory.name}: descriptors must be finite ({manifest['frames']}, D)")
    if np.max(np.abs(np.linalg.norm(descriptors, axis=1) - 1.0)) > 1e-3:
        raise ValueError(f"{directory.name}: descriptors must be L2-normalized")
    payload = io.BytesIO()
    np.save(payload, descriptors, allow_pickle=False)
    atomic_bytes(directory / f"{encoder}.npy", payload.getvalue())
    manifest["files"][f"{encoder}.npy"] = file_digest(directory / f"{encoder}.npy")
    manifest["encoders"][encoder] = {**encoder_identity, "dimension": int(descriptors.shape[1])}
    atomic_json(directory / "manifest.json", manifest)


def open_session(config: dict, name: str, allow_test: bool = False) -> Session:
    directory = store_directory(config, name)
    manifest = json.loads((directory / "manifest.json").read_text())
    recording = Recording(**manifest["recording"])
    if recording.role == "test" and not allow_test:
        raise TestDataError(f"{name} is test data; it needs a written protocol and --allow-test")
    if file_digest(directory / "frames.npz") != manifest["files"]["frames.npz"]:
        raise ValueError(f"{directory}/frames.npz: hash does not match the manifest")
    data = np.load(directory / "frames.npz", allow_pickle=False)
    labels = [label or None for label in data["labels"].tolist()]
    return Session(name, recording, manifest, data["timestamps"], data["gt"], labels,
                   data["image_paths"].tolist(), data["odom_t"], data["odom_poses"], directory)


def stored_sessions(config: dict, role: str | None = "development") -> list[str]:
    root = resolve(config, "store_root")
    names = []
    for manifest in sorted(root.glob("*/manifest.json")):
        recorded = json.loads(manifest.read_text())["recording"]
        if role is None or recorded["role"] == role:
            names.append(recorded["name"])
    return names


def build_frames(config: dict, name: str, allow_test: bool = False) -> Path:
    """Read a raw recording into a new store (frames and odometry only)."""
    recording = catalogue(config)[name]
    if recording.role == "test" and not allow_test:
        raise TestDataError(f"{name} is test data; it needs a written protocol and --allow-test")
    return write_session(config, recording, read_recording(recording, config), {"kind": "raw"})


def import_legacy(config: dict, frames_dir: Path, legacy_name: str, appearance: tuple[str, ...] = ("megaloc",),
                  check_raw: bool = True) -> str:
    """Convert a C1 frame store (``<legacy>.npz/.json``) without recomputing descriptors.

    With ``check_raw`` the frames are re-read from the raw dataset and must
    match the legacy arrays exactly, so imported and newly built stores are
    interchangeable.
    """
    meta = json.loads((frames_dir / f"{legacy_name}.json").read_text())
    if file_digest(frames_dir / f"{legacy_name}.npz") != meta["frames_sha256"]:
        raise ValueError(f"{legacy_name}: legacy store hash mismatch")
    data = np.load(frames_dir / f"{legacy_name}.npz")
    dataset = "cid_sims" if meta["cid"] else "cold"
    recording = recording_for(dataset, meta["sequence"], config)
    if recording.role == "test":
        raise TestDataError(f"{recording.name} is test data; legacy import refused")
    raw = RawFrames(data["timestamps"], data["gt"], meta["labels"], meta["image_paths"], data["odom_t"],
                    data["odom_poses"])
    max_difference = None
    if check_raw:
        # Libm differences across platforms (container vs host) move headings
        # by an ulp; anything beyond 1e-12 is a real mismatch.
        fresh = read_recording(recording, config)
        max_difference = 0.0
        for field in ("timestamps", "gt", "odom_t", "odom_poses"):
            a, b = getattr(fresh, field), getattr(raw, field)
            if a.shape != b.shape or not np.allclose(a, b, rtol=0.0, atol=1e-12):
                raise ValueError(f"{legacy_name}: raw {field} differ from the legacy store")
            max_difference = max(max_difference, float(np.max(np.abs(a - b), initial=0.0)))
        if fresh.labels != raw.labels or fresh.image_paths != raw.image_paths:
            raise ValueError(f"{legacy_name}: raw labels or image paths differ from the legacy store")
    directory = write_session(config, recording, raw, {"kind": "legacy_c1", "legacy_name": legacy_name,
                                                       "legacy_sha256": meta["frames_sha256"],
                                                       "raw_max_abs_difference": max_difference})
    add_descriptors(directory, "dinov3", data["descriptors"], {**meta["encoder"], "source": "legacy_c1"})
    for encoder in appearance:
        extra = frames_dir / f"{legacy_name}__{encoder}.json"
        if extra.is_file():
            extra_meta = json.loads(extra.read_text())
            if extra_meta["base_frames_sha256"] != meta["frames_sha256"]:
                raise ValueError(f"{legacy_name}: {encoder} store does not match its frame store")
            array = np.load(frames_dir / f"{legacy_name}__{encoder}.npz")["descriptors"]
            add_descriptors(directory, encoder, array, {**extra_meta["encoder"], "source": "legacy_c1"})
    return recording.name
