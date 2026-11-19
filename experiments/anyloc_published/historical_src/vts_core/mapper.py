"""Dataset-agnostic online visual topological mapper.

Consumes synchronized (image, pose, covariance) samples and incrementally
builds a :class:`TopoGraph`. The covariance is the cumulative odometry
covariance of :mod:`vts_core.motion` for the chosen ``uncertainty_model``.

Design (deliberately simplified to a single, reliable run):

- Node creation: a node is emitted retrospectively at each confirmed valley of
  the visual stream's windowed algebraic connectivity. Frames received during
  the detector's confirmation latency are carried into the following segment,
  and the final segment is flushed explicitly at end-of-run.
- Revisit / loop closure: interval-relative odometry uncertainty generates and
  ranks non-local candidates first. A unique compatible place is accepted
  directly; when odometry is ambiguous, a bidirectional three-node sequence
  compares training-free von Mises--Fisher place distributions traversed in
  either direction.
- Every emitted node keeps its recorded odometry pose. Sequential and loop
  edges describe topology; place recognition never changes those poses.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import NamedTuple

import numpy as np

from vts_core.directional import _log_vmf_normalizer, fit_vmf_resultant, vmf_log_overlap_ratio
from vts_core.frame_buffer import FrameBuffer, FrameRecord, StoredImage
from vts_core.matching import (
    _CHI2_GATE_2DOF,
    VisualLikelihood,
    disk_probability,
    gaussian_position_statistics,
    wrapped_interval_probability,
)
from vts_core.motion import UNCERTAINTY_MODELS, normalize_angle, relative_pose_covariance
from vts_core.node_detection import (
    AdaptiveValleyDetector,
    ConnectivityMonitor,
    FixedValleyDetector,
)
from vts_core.topo_graph import (
    LOOP_TEMPORAL_EXCLUSION_NODES,
    LoopClosureEvent,
    Pose2D,
    TopoGraph,
    TopoNode,
)

_PLACE_LOCALIZATION_SIGMA: float = 0.5

# Visual revisit gate. A loop closure is accepted only if the candidate's vMF
# overlap evidence is a robust *outlier* among candidates that remain
# plausible after the independent context/geometric checks: at least the
# median plus k robust standard deviations
# (k * 1.4826 * MAD), with k the default below (tunable per run via the
# ``visual_outlier_k`` constructor argument). This is the same MAD-relative
# criterion used by the valley detector, applied to visual
# similarity. It is what stops the "pollution snowball": a place that is
# similar to *many* nodes (e.g. a generic corridor) produces no clear outlier
# and is rejected. LOWER k -> easier loop closure, RAISE it -> stricter.
_VISUAL_OUTLIER_K: float = 2.0
# Below this many nodes there is not enough evidence to judge an outlier;
# stay conservative and do not close loops yet.
_MIN_NODES_FOR_REVISIT: int = 5
# Existing nodes this close in temporal order are traversal neighbours, not
# loop-closure candidates. This structural exclusion adds no metric threshold.

# Revisit gate ablation modes. "both" is the full system (visual outlier test
# AND geometric gate must agree); "visual" and "geometric" disable one gate
# each; "threshold" replaces the whole dual gate with a naive absolute
# cosine-similarity threshold — the baseline the dual gate is argued against.
GATE_MODES: tuple[str, ...] = ("both", "visual", "geometric", "threshold")
# "gate": chi-square gate and visual outlier test. "disk": accept the most probable
# earlier place when the probability of being within ``place_radius`` of it exceeds
# _REVISIT_DECISION, optionally reweighted by visual evidence.
ASSOCIATIONS: tuple[str, ...] = ("gate", "disk")
# Bayes decision for equal costs of a false and a missed closure, the weighting of F1.
_REVISIT_DECISION: float = 0.5
# Heading difference below which a forward camera sees the same scene again.
_ALIGNED_HEADING: float = np.pi / 4
VISUAL_MODELS: tuple[str, ...] = ("vmf", "cosine")
VALLEY_MODES: tuple[str, ...] = ("adaptive", "fixed")
# Where a confirmed valley splits the stream: at the newest frame of the window of least
# lambda_2 ("minimum"), or at that window's least-normalized-cut split ("cut").
BOUNDARIES: tuple[str, ...] = ("minimum", "cut")
# Absolute cosine-similarity threshold used by the "threshold" ablation mode.
_NAIVE_THRESHOLD: float = 0.7


class RevisitDecision(NamedTuple):
    accepted_id: int | None
    proposed_id: int | None
    similarity: float | None
    probability: float | None
    reason: str


class TopologicalMapper:
    """Incremental topological mapper for a single run."""

    def __init__(
        self,
        window_size: int = 30,
        frame_id: str = "run",
        valley_k: float = 1.5,
        valley_mode: str = "adaptive",
        valley_delta: float = 0.1,
        visual_outlier_k: float = _VISUAL_OUTLIER_K,
        gate_mode: str = "both",
        naive_threshold: float = _NAIVE_THRESHOLD,
        visual_model: str = "vmf",
        uncertainty_model: str = "additive",
        association: str = "gate",
        place_radius: float = 2.0,
        visual_likelihood: VisualLikelihood | None = None,
        boundary: str = "minimum",
    ) -> None:
        """
        Args:
            window_size: Sliding window size for the connectivity monitor.
                Larger = smoother lambda_2 = fewer, coarser nodes.
            frame_id: Identifier of this run's odometry frame.
            valley_k: Sensitivity of node creation. The valley detector fires
                when lambda_2 rises ``valley_k`` robust deviations above its
                running minimum, so a LARGER value yields FEWER, more separated
                nodes (raise it if nodes cluster too densely).
            valley_mode: ``adaptive`` for MAD-scaled prominence or ``fixed``
                for the controlled absolute-prominence baseline.
            valley_delta: Absolute prominence used only in ``fixed`` mode.
            visual_outlier_k: Strictness of the visual revisit gate (robust
                MAD multiplier). LOWER it to close loops more aggressively, RAISE for
                stricter loop closures. See ``_VISUAL_OUTLIER_K``.
            gate_mode: Revisit-gate ablation mode, one of :data:`GATE_MODES`.
                ``both`` (default) is the odometry-first dual gate;
                ``visual`` / ``geometric`` keep only one gate; ``threshold``
                is the naive absolute-similarity baseline.
            naive_threshold: Cosine-similarity threshold used when
                ``gate_mode == "threshold"``.
            visual_model: Evidence model used to resolve ambiguous geometric
                candidates: ``vmf`` for the final mapper or ``cosine`` for
                controlled external descriptor baselines.
            uncertainty_model: One of :data:`vts_core.motion.UNCERTAINTY_MODELS`;
                it must match the tracker that produced the input covariances.
            association: One of :data:`ASSOCIATIONS`.
            place_radius: Distance in metres within which two nodes are the same place
                (``disk`` association).
            visual_likelihood: Visual likelihood ratio used by ``disk`` association
                with ``gate_mode="both"``.
            boundary: One of :data:`BOUNDARIES`.
        """
        if gate_mode not in GATE_MODES:
            raise ValueError(f"gate_mode must be one of {GATE_MODES}, got {gate_mode!r}")
        if visual_model not in VISUAL_MODELS:
            raise ValueError(f"visual_model must be one of {VISUAL_MODELS}, got {visual_model!r}")
        if boundary not in BOUNDARIES:
            raise ValueError(f"boundary must be one of {BOUNDARIES}, got {boundary!r}")
        if association not in ASSOCIATIONS:
            raise ValueError(f"association must be one of {ASSOCIATIONS}, got {association!r}")
        if association == "disk" and gate_mode not in {"geometric", "both"}:
            raise ValueError("disk association supports gate_mode 'geometric' or 'both'")
        if association == "disk" and gate_mode == "both" and visual_likelihood is None:
            raise ValueError("disk association with visual evidence needs a visual_likelihood")
        if not place_radius > 0.0:
            raise ValueError("place_radius must be positive")
        if uncertainty_model not in UNCERTAINTY_MODELS:
            raise ValueError(f"uncertainty_model must be one of {UNCERTAINTY_MODELS}, got {uncertainty_model!r}")
        if valley_mode not in VALLEY_MODES:
            raise ValueError(f"valley_mode must be one of {VALLEY_MODES}, got {valley_mode!r}")
        numeric = [valley_k, valley_delta, visual_outlier_k, naive_threshold]
        if not np.isfinite(numeric).all() or min(valley_k, valley_delta, visual_outlier_k) < 0:
            raise ValueError("gate and detector parameters must be finite and nonnegative")
        self.graph: TopoGraph = TopoGraph(frame_id=frame_id)
        self.current_node_id: int | None = None
        self._visual_outlier_k: float = visual_outlier_k
        self._gate_mode: str = gate_mode
        self._naive_threshold: float = naive_threshold
        self._visual_model = visual_model
        self._uncertainty_model = uncertainty_model
        self._association = association
        self._place_radius = place_radius
        self._visual_likelihood = visual_likelihood
        self._boundary = boundary

        self._monitor: ConnectivityMonitor = ConnectivityMonitor(window_size)
        self._detector: AdaptiveValleyDetector | FixedValleyDetector = (
            AdaptiveValleyDetector(k=valley_k)
            if valley_mode == "adaptive"
            else FixedValleyDetector(delta=valley_delta)
        )
        self._frame_count: int = 0
        self._pending_frames = FrameBuffer()
        self._visual_cache: dict[int, tuple] = {}

        # Wall-clock instrumentation for map maintenance and loop search.
        self._update_time_s: float = 0.0
        self._revisit_search_time_s: float = 0.0
        self._revisit_search_count: int = 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def process_frame(
        self,
        image_bgr: np.ndarray,
        descriptor: np.ndarray,
        pose: Pose2D,
        covariance: np.ndarray,
        gt_pose: Pose2D | None = None,
        room_label: str | None = None,
    ) -> int | None:
        """Process one synchronized sample.

        Thin timing wrapper around :meth:`_process_frame_impl` so the full
        pipeline cost can be reported separately from descriptor extraction.

        Args:
            image_bgr: Camera image.
            descriptor: L2-normalized place descriptor of the image.
            pose: Odometry pose (x, y, theta) at capture time.
            covariance: Cumulative 3x3 odometry covariance at capture time.
            gt_pose: Optional ground-truth pose at capture time, carried
                through for evaluation only (never used to map). When given,
                the node inherits the ground-truth pose of its representative
                (medoid) frame, so placement error is measured against the
                exact frame that fixed the node's pose.
            room_label: Optional evaluation-only place label for this frame.

        Returns:
            The id of a node created/updated by this frame, or None.
        """
        start: float = time.perf_counter()
        try:
            return self._process_frame_impl(
                image_bgr, descriptor, pose, covariance, gt_pose, room_label
            )
        finally:
            self._update_time_s += time.perf_counter() - start

    def _process_frame_impl(
        self,
        image_bgr: np.ndarray,
        descriptor: np.ndarray,
        pose: Pose2D,
        covariance: np.ndarray,
        gt_pose: Pose2D | None = None,
        room_label: str | None = None,
    ) -> int | None:
        descriptor = np.asarray(descriptor)
        covariance = np.asarray(covariance, dtype=np.float64)
        if (
            descriptor.ndim != 1
            or descriptor.size < 2
            or not np.isfinite(descriptor).all()
            or np.linalg.norm(descriptor) < 1e-12
        ):
            raise ValueError("descriptor must be a finite nonzero vector of dimension >= 2")
        if np.shape(pose) != (3,) or not np.isfinite(pose).all():
            raise ValueError("pose must contain three finite values")
        if (
            covariance.shape != (3, 3)
            or not np.isfinite(covariance).all()
            or not np.allclose(covariance, covariance.T, atol=1e-9)
        ):
            raise ValueError("uncertainty budget must be finite symmetric 3x3")
        record: FrameRecord = FrameRecord(
            descriptor=descriptor.astype(np.float32),
            pose=pose,
            covariance=covariance.astype(np.float64).copy(),
            image=image_bgr,
            gt_pose=gt_pose,
            room_label=room_label,
            frame_index=self._frame_count,
        )
        self._pending_frames.append(record)
        self._frame_count += 1
        lambda_2: float | None = self._monitor.update(record.descriptor)
        if lambda_2 is None:
            return None

        valley_index: int | None = self._detector.step(lambda_2)
        if valley_index is None:
            return None

        # Connectivity sample zero is produced by camera frame one, so the
        # detector's lambda-series index maps to camera index + 1.
        boundary_frame_index = valley_index + 1
        if self._boundary == "cut":
            boundary_frame_index -= self._monitor.frames_after_cut[valley_index]
        segment = self._pending_frames.segment(boundary_frame_index)
        if not segment:
            return None
        result = self._create_node(segment)
        self._pending_frames.discard_through(boundary_frame_index)
        return result

    def finalize_nodes(self) -> int | None:
        """Flush the exact final segment once; include its cost in mapper time."""
        if not self._pending_frames:
            return None
        start = time.perf_counter()
        try:
            result = self._create_node(self._pending_frames.segment())
            self._pending_frames.discard_through(self._pending_frames.boundary)
            return result
        finally:
            self._update_time_s += time.perf_counter() - start

    def finalize(self) -> int | None:
        return self.finalize_nodes()

    def close(self) -> None:
        """Release the temporary frame spool after saving or on failure."""
        self._pending_frames.close()

    def performance_stats(self) -> dict[str, float]:
        """Computational-cost summary for map maintenance.

        Times are wall-clock and exclude descriptor extraction (the encoder's
        cost, incurred by the caller). ``map_update_time_ms`` is the mean
        per-frame map-maintenance cost; ``loop_closure_time_ms`` is the mean
        cost of one revisit (loop-closure) search at node creation.
        """
        frames: int = max(self._frame_count, 1)
        searches: int = max(self._revisit_search_count, 1)
        latencies = self._detector.latencies
        return {
            "frames": float(self._frame_count),
            "n_nodes": float(len(self.graph.nodes)),
            "n_edges": float(len(self.graph.edges())),
            "map_update_time_ms": 1000.0 * self._update_time_s / frames,
            "loop_closure_time_ms": (1000.0 * self._revisit_search_time_s / searches),
            "total_map_time_s": self._update_time_s,
            "detector_warmup_samples": float(self._detector.warmup),
            "mean_detection_latency_samples": (float(np.mean(latencies)) if latencies else 0.0),
            "max_detection_latency_samples": (float(np.max(latencies)) if latencies else 0.0),
        }

    # ------------------------------------------------------------------ #
    # Edge bookkeeping
    # ------------------------------------------------------------------ #
    def _segment_medoid(
        self, segment: Sequence[FrameRecord], mean_descriptor: np.ndarray
    ) -> FrameRecord:
        """Frame of the current node segment closest to its mean descriptor."""
        return max(segment, key=lambda frame: float(frame.descriptor @ mean_descriptor))

    def _create_node(self, segment: Sequence[FrameRecord]) -> int:
        """Create one node from a completed, non-empty visual segment."""
        if not segment:
            raise ValueError("cannot create a node from an empty segment")
        first = np.asarray(segment[0].descriptor, dtype=np.float64)
        fallback = first / np.linalg.norm(first)
        resultant = np.zeros_like(first)
        for frame in segment:
            sample = np.asarray(frame.descriptor, dtype=np.float64)
            resultant += sample / np.linalg.norm(sample)
        directional_model = fit_vmf_resultant(resultant, len(segment), fallback)
        mean_descriptor = directional_model.mean_direction
        representative = self._segment_medoid(segment, mean_descriptor)
        extent_covariance = np.zeros((2, 2))
        for frame in segment:
            offset = np.asarray(frame.pose[:2]) - representative.pose[:2]
            extent_covariance += np.outer(offset, offset)
        extent_covariance /= len(segment)

        candidate: TopoNode = TopoNode(
            node_id=self.graph.new_node_id(),
            pose=representative.pose,
            visual_features=mean_descriptor,
            visual_concentration=directional_model.concentration,
            visual_resultant_length=directional_model.mean_resultant_length,
            visual_sample_count=directional_model.sample_count,
            pose_covariance=representative.covariance.copy(),
            extent_covariance=extent_covariance,
            # Evaluation-only: pin the node's GT to the SAME frame whose pose
            # became the node pose, so placement error reflects odometric
            # distortion, not the medoid/segment-boundary frame mismatch.
            gt_pose=representative.gt_pose,
            room_label=representative.room_label,
        )
        # Medoid + segment endpoints provide viewpoint diversity at a fixed,
        # tiny storage cost. Deduplicate frames for very short segments.
        selected_frames: list[FrameRecord] = []
        selected_indices: set[int] = set()
        for frame in (representative, segment[0], segment[-1]):
            if frame.frame_index not in selected_indices:
                selected_frames.append(frame)
                selected_indices.add(frame.frame_index)
        candidate.set_views(
            [
                frame.image.read() if isinstance(frame.image, StoredImage) else frame.image
                for frame in selected_frames
            ],
            np.stack([frame.descriptor for frame in selected_frames]),
        )

        if self.current_node_id is None:
            self.graph.add_node(candidate)
            self.current_node_id = candidate.node_id
            return candidate.node_id

        search_start: float = time.perf_counter()
        decision = self._find_revisit(candidate)
        self._revisit_search_time_s += time.perf_counter() - search_start
        self._revisit_search_count += 1
        self.graph.add_node(candidate)
        self.graph.add_edge(self.current_node_id, candidate.node_id)
        self.graph.loop_events.append(
            LoopClosureEvent(
                query_node_id=candidate.node_id,
                matched_node_id=decision.proposed_id,
                accepted=decision.accepted_id is not None,
                similarity=decision.similarity,
                reason=decision.reason,
                probability=decision.probability,
            )
        )
        if decision.accepted_id is not None:
            self.graph.add_edge(decision.accepted_id, candidate.node_id, edge_type="loop")
        self.current_node_id = candidate.node_id
        return candidate.node_id

    def _visual_outlier(self, similarities: np.ndarray, match_index: int) -> bool:
        """True if the match is a robust visual outlier in its evidence pool.

        Rejects matches that are not clearly distinctive — the core defence
        against perceptual-aliasing welds and descriptor-pollution snowballs.
        The strictness is ``self._visual_outlier_k`` (configurable per run).
        """
        # Search-level warm-up already guarantees enough map evidence. After
        # structural/geometric filtering, two surviving scores are sufficient
        # to assess whether the selected place is visually distinctive.
        if similarities.shape[0] < 2:
            return False
        match_sim: float = float(similarities[match_index])
        others: np.ndarray = np.delete(similarities, match_index)
        median: float = float(np.median(others))
        mad: float = float(np.median(np.abs(others - median)))
        threshold: float = median + self._visual_outlier_k * 1.4826 * mad
        # Strict inequality makes the zero-MAD/all-equal case deterministic:
        # an indistinctive tie is rejected rather than accepted.
        return match_sim > threshold

    def _distinctive_within_geometry(
        self,
        similarities: np.ndarray,
        selected: int,
        geometric_indices: list[int],
    ) -> bool:
        """Verify appearance without letting impossible aliases veto a loop.

        The decisive pool contains every geometrically compatible candidate.
        If that pool is too small for a robust statistic, it is supplemented
        only with weaker visual candidates. A visually stronger node already
        disproved by odometry is deliberately irrelevant: appearance verifies
        an odometric hypothesis; it does not globally retrieve the hypothesis.
        """
        background = list(geometric_indices)
        excluded = set(geometric_indices)
        if len(background) < _MIN_NODES_FOR_REVISIT:
            visual_order = np.argsort(-similarities, kind="stable")
            background.extend(
                int(index)
                for index in visual_order
                if (similarities[index] <= similarities[selected] and int(index) not in excluded)
            )
        pool_scores = similarities[background]
        return self._visual_outlier(pool_scores, background.index(selected))

    @staticmethod
    def _view_feature_matrix(node: TopoNode) -> np.ndarray:
        """Return normalized per-view features, with legacy mean fallback."""
        features = getattr(node, "view_features", None)
        if features is None or np.asarray(features).size == 0:
            features = np.asarray(node.visual_features, dtype=np.float32)[None, :]
        matrix = np.asarray(features, dtype=np.float32)
        return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)

    @classmethod
    def _node_similarity(cls, first: TopoNode, second: TopoNode) -> float:
        """Symmetric multi-view similarity between two topological places.

        Averaging each view's best counterpart rewards agreement across the
        stored viewpoints and avoids letting one accidental image pair decide
        a loop closure.
        """
        first_features = cls._view_feature_matrix(first)
        second_features = cls._view_feature_matrix(second)
        pairwise = first_features @ second_features.T
        return float(0.5 * (np.mean(np.max(pairwise, axis=1)) + np.mean(np.max(pairwise, axis=0))))

    @staticmethod
    def _node_vmf_evidence(first: TopoNode, second: TopoNode) -> float:
        """Information-geometric visual evidence between two places."""
        return vmf_log_overlap_ratio(
            first.visual_features,
            first.visual_concentration,
            second.visual_features,
            second.visual_concentration,
        )

    def _cached_node(self, node: TopoNode) -> tuple:
        """Cache immutable emitted-node directions and normalization constants."""
        key = (
            id(node),
            id(node.visual_features),
            id(node.view_features),
            node.visual_concentration,
        )
        cached = self._visual_cache.get(node.node_id)
        if cached is None or cached[0] != key:
            direction = np.asarray(node.visual_features, dtype=np.float64)
            direction = direction / np.linalg.norm(direction)
            kappa = max(float(node.visual_concentration), 0.0)
            cached = (
                key,
                self._view_feature_matrix(node),
                direction,
                kappa,
                _log_vmf_normalizer(direction.size, kappa),
                _log_vmf_normalizer(direction.size, 0.0),
            )
            self._visual_cache[node.node_id] = cached
        return cached

    def _cached_similarity(self, first: TopoNode, second: TopoNode) -> float:
        a, b = self._cached_node(first), self._cached_node(second)
        pairwise = a[1] @ b[1].T
        return float(0.5 * (pairwise.max(axis=1).mean() + pairwise.max(axis=0).mean()))

    def _cached_vmf_evidence(self, first: TopoNode, second: TopoNode) -> float:
        a, b = self._cached_node(first), self._cached_node(second)
        cosine = float(np.clip(a[2] @ b[2], -1.0, 1.0))
        combined = np.sqrt(max(a[3] ** 2 + b[3] ** 2 + 2 * a[3] * b[3] * cosine, 0.0))
        return float(a[4] + b[4] - _log_vmf_normalizer(a[2].size, combined) - a[5])

    def _odometry_step(self, node_id: int, direction: int) -> int | None:
        """Return an adjacent sequential node in ``direction`` if it exists.

        Node identifiers follow emission order. Checking the stored edge type
        prevents a loop edge from ever becoming visual sequence context.
        """
        neighbor = node_id + direction
        if neighbor not in self.graph.nodes:
            return None
        key = (min(node_id, neighbor), max(node_id, neighbor))
        if neighbor not in self.graph.adjacency.get(node_id, set()):
            return None
        if self.graph.edge_types.get(key, "odometry") != "odometry":
            return None
        return neighbor

    def _sequence_score(
        self,
        candidate: TopoNode,
        matched_id: int,
        scorer: Callable[[TopoNode, TopoNode], float],
    ) -> float:
        """Agreement of recent observations with a historical place.

        The candidate itself is compared with ``matched_id``. Up to two recent
        live nodes are then compared with historical sequential neighbors.
        Both historical directions are evaluated and the stronger mean is
        returned. Consequently, ``room -> door -> corridor`` can match the
        reverse traversal ``corridor -> door -> room`` without requiring the
        two door-facing images to be near-identical.
        """
        direct = scorer(candidate, self.graph.nodes[matched_id])
        if self.current_node_id is None:
            return direct

        live_context: list[int] = [self.current_node_id]
        cursor = self.current_node_id
        for _ in range(1):
            previous = self._odometry_step(cursor, -1)
            if previous is None:
                break
            live_context.append(previous)
            cursor = previous

        directional_scores: list[float] = []
        for direction in (-1, 1):
            scores = [direct]
            historical = matched_id
            for live_id in live_context:
                historical = self._odometry_step(historical, direction)
                if historical is None:
                    break
                scores.append(scorer(self.graph.nodes[live_id], self.graph.nodes[historical]))
            directional_scores.append(float(np.mean(scores)))
        return max(directional_scores)

    def _sequence_similarity(self, candidate: TopoNode, matched_id: int) -> float:
        """Bidirectional sequence score using multi-view cosine similarity."""
        return self._sequence_score(candidate, matched_id, self._cached_similarity)

    def _sequence_vmf_evidence(self, candidate: TopoNode, matched_id: int) -> float:
        """Bidirectional sequence score using vMF overlap evidence."""
        return self._sequence_score(candidate, matched_id, self._cached_vmf_evidence)

    def _disk_revisit(self, candidate: TopoNode, eligible_ids: list[int]) -> RevisitDecision:
        """Most probable earlier place, given the odometry covariance accumulated since each was seen."""
        nodes = [self.graph.nodes[node_id] for node_id in eligible_ids]
        poses = np.asarray([node.pose for node in nodes])
        relative = relative_pose_covariance(
            self._uncertainty_model,
            candidate.pose,
            candidate.pose_covariance,
            poses,
            np.asarray([node.pose_covariance for node in nodes]),
        )
        displacement = np.asarray(candidate.pose[:2]) - poses[:, :2]
        probability = disk_probability(displacement, relative[:, :2, :2], self._place_radius)
        scores = None
        if self._gate_mode == "both":
            heading_change = np.asarray([normalize_angle(candidate.pose[2] - pose[2]) for pose in poses])
            aligned = wrapped_interval_probability(heading_change, relative[:, 2, 2], _ALIGNED_HEADING)
            similarity = self._cached_vmf_evidence if self._visual_model == "vmf" else self._cached_similarity
            scores = np.asarray([similarity(candidate, node) for node in nodes])
            ratio = aligned * self._visual_likelihood.ratio(scores) + (1.0 - aligned)
            prior = np.clip(probability, 1e-12, 1.0 - 1e-12)
            odds = prior / (1.0 - prior) * ratio
            probability = odds / (1.0 + odds)
        best = int(np.argmax(probability))
        accepted = bool(probability[best] > _REVISIT_DECISION)
        return RevisitDecision(
            eligible_ids[best] if accepted else None,
            eligible_ids[best],
            None if scores is None else float(scores[best]),
            float(probability[best]),
            "accepted" if accepted else "rejected_by_probability",
        )

    def _find_revisit(self, candidate: TopoNode) -> RevisitDecision:
        """Find a revisit with the configured association.

        With ``gate`` association, the full method intentionally gives the two signals asymmetric jobs:
        odometry generates and ranks every statistically compatible place. A
        sole compatible place needs no appearance tie-break; when several
        remain, visual sequence evidence disambiguates them in odometric order.
        Appearance therefore cannot suppress an unambiguous geometric revisit
        or promote a geometrically implausible alias.
        """
        ids = sorted(self.graph.nodes)
        eligible_count = len(ids) - LOOP_TEMPORAL_EXCLUSION_NODES
        if self._association == "disk":
            if eligible_count < 1:
                return RevisitDecision(None, None, None, None, "no_candidates")
            return self._disk_revisit(candidate, ids[:eligible_count])
        if eligible_count < _MIN_NODES_FOR_REVISIT:
            return RevisitDecision(None, None, None, None, "no_distinctive_match")

        eligible_ids = ids[:eligible_count]
        scorer = (
            self._sequence_vmf_evidence
            if self._visual_model == "vmf"
            else self._sequence_similarity
        )

        if self._gate_mode in {"threshold", "visual"}:
            scores = np.asarray(
                [
                    self._cached_similarity(candidate, self.graph.nodes[node_id])
                    if self._gate_mode == "threshold"
                    else scorer(candidate, node_id)
                    for node_id in eligible_ids
                ]
            )
            selected = int(np.argmax(scores))
            proposed_id = eligible_ids[selected]
            accepted = (
                scores[selected] >= self._naive_threshold
                if self._gate_mode == "threshold"
                else self._visual_outlier(scores, selected)
            )
            return RevisitDecision(
                proposed_id if accepted else None,
                proposed_id,
                float(scores[selected]),
                None,
                "accepted" if accepted else "rejected_by_gate",
            )

        nodes = [self.graph.nodes[node_id] for node_id in eligible_ids]
        poses = np.asarray([node.pose for node in nodes])
        deltas = np.asarray(candidate.pose[:2]) - poses[:, :2]
        covariance = relative_pose_covariance(
            self._uncertainty_model,
            candidate.pose,
            candidate.pose_covariance,
            poses,
            np.asarray([node.pose_covariance for node in nodes]),
        )[:, :2, :2]
        if np.min(np.linalg.eigvalsh(covariance)) < -1e-9:
            raise ValueError("odometry covariance between nodes is not positive semidefinite")
        covariance += np.eye(2) * _PLACE_LOCALIZATION_SIGMA**2
        d2, nll = gaussian_position_statistics(deltas, covariance)
        order = np.argsort(nll, kind="stable")
        geometric_indices = [int(i) for i in order if d2[i] <= _CHI2_GATE_2DOF]
        proposed = geometric_indices[0] if geometric_indices else int(order[0])
        proposed_id = eligible_ids[proposed]
        if not geometric_indices:
            return RevisitDecision(None, proposed_id, None, None, "rejected_by_gate")
        if self._gate_mode == "geometric":
            return RevisitDecision(proposed_id, proposed_id, None, None, "accepted")
        if len(geometric_indices) == 1:
            return RevisitDecision(proposed_id, proposed_id, scorer(candidate, proposed_id), None, "accepted")

        # Preserve the complete weaker-score background for ambiguous geometry.
        scores = np.asarray([scorer(candidate, node_id) for node_id in eligible_ids])
        for selected in geometric_indices:
            if self._distinctive_within_geometry(scores, selected, geometric_indices):
                accepted_id = eligible_ids[selected]
                return RevisitDecision(accepted_id, accepted_id, float(scores[selected]), None, "accepted")
        return RevisitDecision(None, proposed_id, float(scores[proposed]), None, "rejected_by_gate")
