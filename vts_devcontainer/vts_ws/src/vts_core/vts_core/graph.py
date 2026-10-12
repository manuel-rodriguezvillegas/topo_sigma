"""Topological graph over places of one or several sessions, and its storage.

Nodes are places identified by ``(session, index)``; each keeps the pose of
its own session's odometry frame, never corrected by place recognition.
Edges are ``odometry`` (consecutive places of a session), ``loop`` (accepted
in-session revisit) or ``merge`` (place of one session associated with a place
of another). Graphs are saved as a single ``.npz`` without pickled objects.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

from vts_core.association import Decision
from vts_core.places import Place
from vts_core.provenance import atomic_bytes, json_safe

EDGE_TYPES = ("odometry", "loop", "merge")


class TopoGraph:
    def __init__(self) -> None:
        self.sessions: list[str] = []
        self.places: dict[str, list[Place]] = {}
        self.edges: dict[tuple[tuple[str, int], tuple[str, int]], str] = {}
        self.decisions: dict[str, list[Decision]] = {}

    def add_session(self, name: str, places: list[Place], decisions: list[Decision] = ()) -> None:
        if name in self.places:
            raise ValueError(f"session {name} already in the graph")
        self.sessions.append(name)
        self.places[name] = list(places)
        self.decisions[name] = list(decisions)
        for index in range(1, len(places)):
            self.add_edge((name, index - 1), (name, index), "odometry")
        for decision in decisions:
            if decision.accepted:
                self.add_edge((name, decision.matched), (name, decision.query), "loop")

    def add_edge(self, a: tuple[str, int], b: tuple[str, int], kind: str) -> None:
        if kind not in EDGE_TYPES:
            raise ValueError(f"edge type must be one of {EDGE_TYPES}")
        for session, index in (a, b):
            if not 0 <= index < len(self.places.get(session, ())):
                raise KeyError(f"unknown node {(session, index)}")
        if a != b:
            self.edges[tuple(sorted((a, b)))] = kind

    def nodes(self) -> list[tuple[str, int]]:
        return [(s, i) for s in self.sessions for i in range(len(self.places[s]))]

    def neighbours(self, node: tuple[str, int]) -> list[tuple[str, int]]:
        return [b if a == node else a for a, b in self.edges if node in (a, b)]

    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        arrays: dict[str, np.ndarray] = {}
        for k, name in enumerate(self.sessions):
            places = self.places[name]
            dim = places[0].mean.size if places else 0
            views = np.zeros((len(places), 3, dim))
            view_poses = np.zeros((len(places), 3, 3))
            view_frames = np.full((len(places), 3), -1)
            for i, p in enumerate(places):
                views[i, : len(p.views)] = p.views
                view_poses[i, : len(p.views)] = p.view_poses
                view_frames[i, : len(p.views)] = p.view_frames
            arrays.update({
                f"s{k}_range": np.asarray([[p.start, p.stop, p.medoid] for p in places], dtype=np.int64).reshape(-1, 3),
                f"s{k}_view_frames": view_frames, f"s{k}_views": views, f"s{k}_view_poses": view_poses,
                f"s{k}_mean": np.asarray([p.mean for p in places]).reshape(len(places), dim),
                f"s{k}_resultant": np.asarray([p.resultant_length for p in places]),
                f"s{k}_pose": np.asarray([p.pose for p in places]).reshape(-1, 3),
                f"s{k}_budget": np.asarray([p.budget for p in places]).reshape(-1, 2, 2),
            })
        meta = {
            "sessions": self.sessions,
            "edges": [[a[0], a[1], b[0], b[1], kind] for (a, b), kind in sorted(self.edges.items())],
            "decisions": {s: [vars(d) for d in ds] for s, ds in self.decisions.items()},
        }
        arrays["meta"] = np.asarray(json.dumps(json_safe(meta)))
        payload = io.BytesIO()
        np.savez_compressed(payload, **arrays)
        atomic_bytes(path, payload.getvalue())

    @classmethod
    def load(cls, path: str | Path) -> TopoGraph:
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        graph = cls()
        for k, name in enumerate(meta["sessions"]):
            places = []
            for i, (start, stop, medoid) in enumerate(data[f"s{k}_range"]):
                frames = tuple(int(f) for f in data[f"s{k}_view_frames"][i] if f >= 0)
                places.append(Place(int(start), int(stop), int(medoid), frames, data[f"s{k}_mean"][i],
                                    data[f"s{k}_views"][i, : len(frames)], float(data[f"s{k}_resultant"][i]),
                                    data[f"s{k}_pose"][i], data[f"s{k}_budget"][i],
                                    data[f"s{k}_view_poses"][i, : len(frames)]))
            graph.sessions.append(name)
            graph.places[name] = places
            graph.decisions[name] = [Decision(**d) for d in meta["decisions"].get(name, [])]
        for sa, ia, sb, ib, kind in meta["edges"]:
            graph.edges[((sa, ia), (sb, ib))] = kind
        return graph
