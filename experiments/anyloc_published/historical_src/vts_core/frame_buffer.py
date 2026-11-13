"""Disk-backed pending segments; exact images and medoids without growing RAM."""

from __future__ import annotations

import pickle
import sqlite3
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from tempfile import TemporaryDirectory

import numpy as np

from vts_core.topo_graph import Pose2D


@dataclass
class StoredImage:
    buffer: FrameBuffer
    index: int

    def read(self) -> np.ndarray:
        row = self.buffer._db.execute(
            "SELECT image FROM frames WHERE id = ?", (self.index,)
        ).fetchone()
        return pickle.loads(row[0])


@dataclass
class FrameRecord:
    descriptor: np.ndarray
    pose: Pose2D
    covariance: np.ndarray
    image: np.ndarray | StoredImage
    gt_pose: Pose2D | None = None
    room_label: str | None = None
    frame_index: int = 0


class FrameSegment(Sequence[FrameRecord]):
    def __init__(self, buffer: FrameBuffer, boundary: int):
        self.buffer = buffer
        self.boundary = boundary

    def __len__(self) -> int:
        return self.buffer._db.execute(
            "SELECT count(*) FROM frames WHERE id <= ?", (self.boundary,)
        ).fetchone()[0]

    def __iter__(self) -> Iterator[FrameRecord]:
        for (metadata,) in self.buffer._db.execute(
            "SELECT metadata FROM frames WHERE id <= ? ORDER BY id", (self.boundary,)
        ):
            yield self.buffer._decode(metadata)

    def __getitem__(self, index: int) -> FrameRecord:
        if index < 0:
            index += len(self)
        if index < 0:
            raise IndexError(index)
        row = self.buffer._db.execute(
            "SELECT metadata FROM frames WHERE id <= ? ORDER BY id LIMIT 1 OFFSET ?",
            (self.boundary, index),
        ).fetchone()
        if row is None:
            raise IndexError(index)
        return self.buffer._decode(row[0])


class FrameBuffer(FrameSegment):
    def __init__(self):
        self._directory = TemporaryDirectory(prefix="toposigma-frames-")
        self._db = sqlite3.connect(self._directory.name + "/frames.sqlite")
        self._db.execute("PRAGMA journal_mode=OFF")
        self._db.execute("PRAGMA synchronous=OFF")
        self._db.execute("PRAGMA cache_size=-2048")
        self._db.execute("CREATE TABLE frames(id INTEGER PRIMARY KEY, metadata BLOB, image BLOB)")
        super().__init__(self, 2**63 - 1)

    def append(self, frame: FrameRecord) -> None:
        metadata = (
            frame.descriptor,
            frame.pose,
            frame.covariance,
            frame.gt_pose,
            frame.room_label,
            frame.frame_index,
        )
        self._db.execute(
            "INSERT INTO frames VALUES (?, ?, ?)",
            (
                frame.frame_index,
                pickle.dumps(metadata, protocol=5),
                pickle.dumps(frame.image, protocol=5),
            ),
        )
        self._db.commit()

    def _decode(self, metadata: bytes) -> FrameRecord:
        descriptor, pose, covariance, gt, label, index = pickle.loads(metadata)
        return FrameRecord(descriptor, pose, covariance, StoredImage(self, index), gt, label, index)

    def segment(self, boundary: int | None = None) -> FrameSegment:
        return FrameSegment(self, self.boundary if boundary is None else boundary)

    def discard_through(self, boundary: int) -> None:
        self._db.execute("DELETE FROM frames WHERE id <= ?", (boundary,))
        self._db.commit()

    def close(self) -> None:
        self._db.close()
        self._directory.cleanup()

    def __del__(self) -> None:
        try:
            self.close()
        except (AttributeError, sqlite3.Error):
            pass
