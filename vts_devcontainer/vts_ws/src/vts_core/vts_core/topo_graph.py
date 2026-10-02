"""Unified topological graph data model.

This single representation replaces the previous duality between the
edge-list produced by the graph builder and the ``Graph`` class expected by
downstream modules. It is a plain, picklable structure with
no ROS, torch, or model dependencies, so mapping, language and evaluation
code can load it safely.
"""

from __future__ import annotations

import os
import pickle
import tempfile
from dataclasses import dataclass, field, replace

import numpy as np

Pose2D = tuple[float, float, float]
# Exclude only the immediately preceding emitted node: A -> B -> A is eligible.
# Shared by mapping and evaluation.
LOOP_TEMPORAL_EXCLUSION_NODES: int = 1


@dataclass
class LoopClosureEvent:
    """One loop-closure decision made at a newly emitted node.

    Ground truth is intentionally absent: evaluation joins ``query_node_id``
    and ``matched_node_id`` with node ground truth after mapping has finished.
    """

    query_node_id: int
    matched_node_id: int | None
    accepted: bool
    similarity: float | None
    reason: str
    decision_node_id: int | None = None


@dataclass
class TopoNode:
    """A node of the topological map.

    Attributes:
        node_id: Unique identifier within one graph.
        pose: (x, y, theta) in the odometry frame of the run that created it.
        visual_features: L2-normalized place-recognition descriptor.
        visual_concentration: Concentration of the vMF distribution fitted to
            every descriptor in the node segment. Zero means that directional
            confidence could not be estimated.
        visual_resultant_length: Mean resultant length in [0, 1], retained as
            an interpretable, dimension-independent concentration diagnostic.
        visual_sample_count: Number of segment descriptors used by the fit.
        view_features: Up to ``MAX_VIEWS`` L2-normalized visual descriptors,
            one per representative view. This small matrix supports robust
            multi-view place matching without retaining every frame.
        views: Up to ``MAX_VIEWS`` representative BGR images of the place.
            Multiple raw views replace the old SIFT-stitched panoramas, which
            frequently degenerated and polluted both the descriptors and the
            CLIP semantics.
        view_embeddings: One semantic (CLIP/SigLIP) embedding per view,
            shape (n_views, d). Filled lazily by the semantic encoder.
        pose_covariance: 2x2 covariance of (x, y) accumulated from odometry
            at creation time. Used for probabilistic revisit gating instead
            of fixed metric thresholds.
        extent_covariance: 2x2 spatial covariance of the camera poses assigned
            to this place. It represents how much two valid observations of
            the same topological node may differ in position.
        room_label: Optional ground-truth room label (evaluation only).
        gt_pose: Optional ground-truth (x, y, theta) of the *representative*
            frame that fixed this node's pose (the segment medoid), attached
            for evaluation only and NEVER used by the mapping algorithm. It
            must reference the same frame as ``pose`` so that placement error
            measures odometric distortion rather than segmentation choice.
    """

    node_id: int
    pose: Pose2D
    visual_features: np.ndarray
    visual_concentration: float = 0.0
    visual_resultant_length: float = 0.0
    visual_sample_count: int = 0
    views: list[np.ndarray] = field(default_factory=list)
    view_features: np.ndarray | None = None
    view_embeddings: np.ndarray | None = None
    semantic_model_identity: str | None = None
    pose_covariance: np.ndarray = field(default_factory=lambda: np.zeros((2, 2), dtype=np.float64))
    extent_covariance: np.ndarray = field(
        default_factory=lambda: np.zeros((2, 2), dtype=np.float64)
    )
    room_label: str | None = None
    gt_pose: Pose2D | None = None

    MAX_VIEWS: int = 3

    def set_views(
        self,
        images: list[np.ndarray],
        visual_features: np.ndarray | None = None,
    ) -> None:
        """Replace representative views and their optional visual descriptors.

        The visual descriptors are deliberately separate from
        ``view_embeddings``: the former are DINO place features, while the
        latter are semantic embeddings populated lazily by the language
        module.
        """
        kept_images = [image for image in images if image is not None][: self.MAX_VIEWS]
        self.views = kept_images
        self.view_embeddings = None
        if visual_features is None:
            self.view_features = None
            return
        features = np.asarray(visual_features, dtype=np.float32)
        if features.ndim != 2 or features.shape[0] != len(kept_images):
            raise ValueError("visual_features must have shape (number of views, dimension)")
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        self.view_features = features / np.maximum(norms, 1e-12)

    def add_view(self, image: np.ndarray) -> None:
        """Add a representative view, keeping at most ``MAX_VIEWS`` diverse ones.

        Keeps the first, the most recent, and the middle view, which gives a
        cheap diversity heuristic without storing every frame.

        Args:
            image: BGR image to add.
        """
        if image is None:
            return
        # This legacy convenience method cannot maintain descriptor/image
        # alignment. Call set_views() when visual descriptors are available.
        self.view_features = None
        self.view_embeddings = None
        if len(self.views) < self.MAX_VIEWS:
            self.views.append(image)
            return
        # first / middle / last policy: replace the last slot, promote the
        # previous last to the middle slot.
        self.views[1] = self.views[2]
        self.views[2] = image


class TopoGraph:
    """Undirected topological graph with dict-based adjacency.

    Attributes:
        nodes: Mapping node_id -> TopoNode.
        adjacency: Mapping node_id -> set of neighbor node_ids.
        frame_id: Identifier of the run / coordinate frame this graph lives in.
    """

    def __init__(self, frame_id: str = "map") -> None:
        self.nodes: dict[int, TopoNode] = {}
        self.adjacency: dict[int, set[int]] = {}
        self.edge_types: dict[tuple[int, int], str] = {}
        self.loop_events: list[LoopClosureEvent] = []
        self.recovery_diagnostics: list[dict] = []
        self.frame_id: str = frame_id
        self._next_id: int = 0

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def new_node_id(self) -> int:
        """Return a fresh node id."""
        node_id: int = self._next_id
        self._next_id += 1
        return node_id

    def add_node(self, node: TopoNode) -> None:
        """Insert a node (id must be unique)."""
        if node.node_id in self.nodes:
            raise ValueError(f"Duplicate node id {node.node_id}")
        self.nodes[node.node_id] = node
        self.adjacency.setdefault(node.node_id, set())
        self._next_id = max(self._next_id, node.node_id + 1)

    def add_edge(self, id_a: int, id_b: int, edge_type: str = "odometry") -> None:
        """Insert an undirected edge between two existing nodes."""
        if edge_type not in {"odometry", "loop"}:
            raise ValueError("edge_type must be odometry or loop")
        if id_a == id_b:
            return
        if id_a not in self.nodes or id_b not in self.nodes:
            raise KeyError(f"Edge ({id_a}, {id_b}) references unknown node")
        self.adjacency[id_a].add(id_b)
        self.adjacency[id_b].add(id_a)
        self.edge_types[(min(id_a, id_b), max(id_a, id_b))] = edge_type

    def remove_edge(self, id_a: int, id_b: int) -> None:
        """Remove an undirected edge if it exists."""
        self.adjacency.get(id_a, set()).discard(id_b)
        self.adjacency.get(id_b, set()).discard(id_a)
        self.edge_types.pop((min(id_a, id_b), max(id_a, id_b)), None)

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def edges(self) -> list[tuple[int, int]]:
        """Return each undirected edge exactly once as (min_id, max_id)."""
        seen: set[tuple[int, int]] = set()
        for node_id, neighbors in self.adjacency.items():
            for neighbor_id in neighbors:
                edge: tuple[int, int] = (
                    min(node_id, neighbor_id),
                    max(node_id, neighbor_id),
                )
                seen.add(edge)
        return sorted(seen)

    def positions(self) -> np.ndarray:
        """Return an (N, 2) array of node (x, y) positions, sorted by id."""
        ids: list[int] = sorted(self.nodes)
        return np.array([self.nodes[i].pose[:2] for i in ids], dtype=np.float64).reshape(-1, 2)

    def feature_matrix(self) -> tuple[list[int], np.ndarray]:
        """Return (sorted node ids, stacked L2-normalized descriptors)."""
        ids: list[int] = sorted(self.nodes)
        if not ids:
            return [], np.empty((0, 0), dtype=np.float32)
        features: np.ndarray = np.stack([self.nodes[i].visual_features for i in ids]).astype(
            np.float32
        )
        norms: np.ndarray = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / np.maximum(norms, 1e-12)
        return ids, features

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def without_images(self) -> TopoGraph:
        """A reloadable map retaining all matching, pose and topology state."""
        compact = TopoGraph(self.frame_id)
        for node in self.nodes.values():
            compact.add_node(replace(node, views=[]))
        for a, b in self.edges():
            compact.add_edge(a, b, self.edge_types.get((a, b), "odometry"))
        compact.loop_events = list(self.loop_events)
        compact.recovery_diagnostics = list(self.recovery_diagnostics)
        return compact

    def save(self, path: str) -> None:
        """Atomically publish a trusted local pickle."""
        parent = os.path.dirname(os.path.abspath(path))
        fd, temporary = tempfile.mkstemp(dir=parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                pickle.dump(self, stream, protocol=pickle.HIGHEST_PROTOCOL)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def load(path: str) -> TopoGraph:
        """Load a pickled graph from disk."""
        with open(path, "rb") as f:
            graph: TopoGraph = pickle.load(f)
        if not isinstance(graph, TopoGraph):
            raise TypeError(f"File {path} does not contain a TopoGraph")
        # Drop obsolete auxiliary attributes from trusted historical maps.
        for name in set(vars(graph)) - set(vars(TopoGraph())):
            delattr(graph, name)
        if not hasattr(graph, "edge_types"):
            graph.edge_types = {}
        if not hasattr(graph, "recovery_diagnostics"):
            graph.recovery_diagnostics = []
        if not hasattr(graph, "loop_events"):
            graph.loop_events = []
        for node in graph.nodes.values():
            if not hasattr(node, "extent_covariance"):
                node.extent_covariance = np.zeros((2, 2), dtype=np.float64)
            if not hasattr(node, "view_features"):
                node.view_features = None
            if not hasattr(node, "visual_concentration"):
                node.visual_concentration = 0.0
            if not hasattr(node, "visual_resultant_length"):
                node.visual_resultant_length = 0.0
            if not hasattr(node, "visual_sample_count"):
                node.visual_sample_count = 0
        return graph
