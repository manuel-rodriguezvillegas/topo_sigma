"""Dataset-agnostic online visual topological mapper.

Consumes synchronized (image, pose, covariance) samples — wherever they come
from a compatible additive-budget adapter — and incrementally builds
a :class:`TopoGraph`.

Design (deliberately simplified to a single, reliable run):

- Node creation: a node is emitted retrospectively at each confirmed valley of
  the visual stream's windowed algebraic connectivity. Frames received during
  the detector's confirmation latency are carried into the following segment,
  and the final segment is flushed explicitly at end-of-run.
- Revisit / loop closure: interval-relative odometry uncertainty generates and
  ranks non-local candidates first. Strict context verification is optional.
  Experimental local recovery can retain visually retrieved candidates outside
  that gate and verify disjoint trajectory fragments after a bounded delay.
- Every emitted node keeps its recorded odometry pose. Sequential and loop
  edges describe topology; place recognition never changes those poses.
- Input covariance fields must carry the bundled players' monotonic additive
  uncertainty budget. Ordinary pose marginals are not supported.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import numpy as np

from vts_core.directional import (
    _log_vmf_normalizer,
    fit_vmf_resultant,
    vmf_log_overlap_ratio,
    vmf_symmetric_kl,
)
from vts_core.frame_buffer import FrameBuffer, FrameRecord, StoredImage
from vts_core.local_recovery import RECOVERY_MODES, RecoveryConfig, local_track, robust_evidence
from vts_core.matching import _CHI2_GATE_2DOF, gaussian_position_statistics, mahalanobis_gate
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
# Small geometric pools can use weaker visual scores as additional background.
# This is a scoring heuristic, never a minimum map size for revisits.
_VISUAL_BACKGROUND_TARGET: int = 5
# Existing nodes this close in temporal order are traversal neighbours, not
# loop-closure candidates. This structural exclusion adds no metric threshold.

# Revisit gate ablation modes. "both" is the full system (visual outlier test
# AND geometric gate must agree); "visual" and "geometric" disable one gate
# each; "threshold" replaces the whole dual gate with a naive absolute
# cosine-similarity threshold — the baseline the dual gate is argued against.
GATE_MODES: tuple[str, ...] = ("both", "visual", "geometric", "threshold")
VISUAL_MODELS: tuple[str, ...] = ("cosine", "mean_cosine", "vmf", "kl_fixed", "kl_estimated")
VALLEY_MODES: tuple[str, ...] = ("adaptive", "fixed")
# Absolute cosine-similarity threshold used by the "threshold" ablation mode.
_NAIVE_THRESHOLD: float = 0.7


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
        visual_model: str = "cosine",
        verification_threshold: float = 0.0,
        defer_nodes: int = 2,
        fixed_concentration: float = 384.0,
        sequence_policy: str = "self_only",
        require_visual_verification: bool = False,
        recovery_mode: str = "off",
        recovery_top_k: int = 5,
        recovery_horizon: int = 2,
        recovery_residual_m: float = 0.75,
        recovery_min_span_m: float = 1.0,
        recovery_min_turn_deg: float = 15.0,
        recovery_visual_z: float = 2.0,
        geometric_gate_scale: float = 1.0,
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
            visual_model: Appearance score, defaulting to multi-view cosine.
                Distribution comparisons are experimental alternatives.
            verification_threshold: Shared multi-view cosine floor for verification.
            defer_nodes: Maximum subsequent observations allowed for confirmation.
            fixed_concentration: Shared concentration for the fixed-KL control.
            sequence_policy: Reject self-pairs by default; legacy and disjoint
                policies exist for controlled comparisons.
            require_visual_verification: Experimental strict verification for
                every geometric hypothesis, including unique candidates.
            recovery_mode: Experimental additive recovery, disabled by default.
                ``local`` verifies local tracks; ``visual_union`` and
                ``soft_odometry`` are controls without track verification.
            recovery_top_k: Maximum global visual proposals per rejected query.
            recovery_horizon: Maximum subsequent nodes to wait, from zero to two.
            recovery_residual_m: Maximum rigid local alignment residual in metres.
            recovery_min_span_m: Minimum span of each three-node track.
            recovery_min_turn_deg: Minimum departure from a straight or retraced track.
            recovery_visual_z: Minimum mean robust visual contrast, not a probability.
            geometric_gate_scale: Multiplier of the squared-distance gate for
                the expanded-gate control; one preserves the original gate.
        """
        if gate_mode not in GATE_MODES:
            raise ValueError(f"gate_mode must be one of {GATE_MODES}, got {gate_mode!r}")
        if visual_model not in VISUAL_MODELS:
            raise ValueError(f"visual_model must be one of {VISUAL_MODELS}, got {visual_model!r}")
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
        if not np.isfinite(verification_threshold) or not -1 <= verification_threshold <= 1:
            raise ValueError("verification_threshold must be in [-1, 1]")
        if not isinstance(defer_nodes, int) or defer_nodes < 0:
            raise ValueError("defer_nodes must be a nonnegative integer")
        if not np.isfinite(fixed_concentration) or fixed_concentration <= 0:
            raise ValueError("fixed_concentration must be finite and positive")
        if sequence_policy not in {"legacy", "self_only", "disjoint"}:
            raise ValueError("unknown sequence_policy")
        self._sequence_policy = sequence_policy
        self._require_visual_verification = require_visual_verification
        self._visual_model = visual_model
        self._verification_threshold = verification_threshold
        self._defer_nodes = defer_nodes
        self._fixed_concentration = fixed_concentration
        self._pending_revisits: list[LoopClosureEvent] = []
        if recovery_mode not in RECOVERY_MODES:
            raise ValueError(f"recovery_mode must be one of {RECOVERY_MODES}")
        if recovery_mode != "off" and (require_visual_verification or gate_mode != "both"):
            raise ValueError("recovery requires gate_mode=both and strict verification disabled")
        if not np.isfinite(geometric_gate_scale) or geometric_gate_scale <= 0:
            raise ValueError("geometric_gate_scale must be finite and positive")
        self._geometric_gate_scale = geometric_gate_scale
        self._recovery_mode = recovery_mode
        self._recovery_config = RecoveryConfig(
            recovery_top_k, recovery_horizon, recovery_residual_m,
            recovery_min_span_m, recovery_min_turn_deg, recovery_visual_z,
        )
        self._recovery_pending: list[tuple[LoopClosureEvent, tuple[int, ...]]] = []

        self._monitor: ConnectivityMonitor = ConnectivityMonitor(window_size)
        self._detector: AdaptiveValleyDetector | FixedValleyDetector = (
            AdaptiveValleyDetector(k=valley_k)
            if valley_mode == "adaptive"
            else FixedValleyDetector(delta=valley_delta)
        )
        self._frame_count: int = 0
        self._pending_frames = FrameBuffer()
        self._previous_budget: np.ndarray | None = None
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
            covariance: Monotonic additive 3x3 uncertainty budget at capture time.
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
        self._interval_covariance(
            np.zeros((3, 3)) if self._previous_budget is None else self._previous_budget, covariance
        )
        self._previous_budget = covariance.copy()
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
        segment = self._pending_frames.segment(boundary_frame_index)
        if not segment:
            return None
        result = self._create_node(segment)
        self._pending_frames.discard_through(boundary_frame_index)
        return result

    def finalize_nodes(self) -> int | None:
        """Flush the exact final segment once; include its cost in mapper time."""
        if not self._pending_frames:
            self.finish_revisits()
            return None
        start = time.perf_counter()
        try:
            result = self._create_node(self._pending_frames.segment())
            self._pending_frames.discard_through(self._pending_frames.boundary)
            self.finish_revisits()
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
    @staticmethod
    def _interval_covariance(earlier: np.ndarray, later: np.ndarray) -> np.ndarray:
        """PSD uncertainty accumulated between two cumulative snapshots."""
        difference = np.asarray(later, dtype=np.float64) - np.asarray(earlier, dtype=np.float64)
        difference = 0.5 * (difference + difference.T)
        eigenvalues, eigenvectors = np.linalg.eigh(difference)
        if np.min(eigenvalues) < -1e-9:
            raise ValueError("uncertainty budget decreased; pose marginals are not supported")
        # Clip roundoff only, never covariance resets.
        return (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T

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
            pose_covariance=representative.covariance[:2, :2].copy(),
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

        return self.observe_node(candidate)

    def observe_node(self, candidate: TopoNode) -> int:
        """Emit one immutable segment; also used for chronological closure-only replay."""
        if candidate.node_id != len(self.graph.nodes):
            raise ValueError("nodes must be emitted once in contiguous chronological order")
        if self.current_node_id is None:
            self.graph.add_node(candidate)
            self.current_node_id = candidate.node_id
            return candidate.node_id

        search_start: float = time.perf_counter()
        revisited_id, proposed_id, similarity, reason = self._find_revisit(candidate)
        self._revisit_search_count += 1
        self.graph.add_node(candidate)
        self.graph.add_edge(self.current_node_id, candidate.node_id)
        self.graph.loop_events.append(
            LoopClosureEvent(
                query_node_id=candidate.node_id,
                matched_node_id=proposed_id,
                accepted=revisited_id is not None,
                similarity=similarity,
                reason=reason,
                decision_node_id=candidate.node_id,
            )
        )
        if revisited_id is not None:
            self.graph.add_edge(revisited_id, candidate.node_id, edge_type="loop")
        self._confirm_pending(candidate)
        if self._recovery_mode != "off":
            self._recover_revisit(candidate, self.graph.loop_events[-1])
        self._revisit_search_time_s += time.perf_counter() - search_start
        if reason == "deferred_visual_verification":
            self._pending_revisits.append(self.graph.loop_events[-1])
        self.current_node_id = candidate.node_id
        return candidate.node_id

    def finish_revisits(self) -> None:
        """An unresolved hypothesis at end-of-stream is rejected, never auto-accepted."""
        for event in self._pending_revisits:
            event.reason = "rejected_end_of_stream"
            event.decision_node_id = self.current_node_id
        self._pending_revisits.clear()
        for event, _ in self._recovery_pending:
            event.reason = "recovery_end_of_stream"
            event.decision_node_id = self.current_node_id
        self._recovery_pending.clear()

    def _confirm_pending(self, candidate: TopoNode) -> None:
        remaining: list[LoopClosureEvent] = []
        for event in self._pending_revisits:
            query_id, matched_id = event.query_node_id, event.matched_node_id
            age = candidate.node_id - query_id
            confirmed = False
            # No observation used in the original live window can become
            # historical evidence. Future observations are also excluded.
            blocked = set(range(max(0, query_id - 2), candidate.node_id + 1))
            for direction in (-1, 1):
                historical = matched_id
                for _ in range(age):
                    historical = self._odometry_step(historical, direction)
                    if historical is None or historical in blocked:
                        break
                if historical is None or historical in blocked:
                    continue
                matched = self.graph.nodes[historical]
                delta = np.asarray(candidate.pose[:2]) - matched.pose[:2]
                if not self._passes_gates(delta, self._revisit_covariance(candidate, matched)):
                    continue
                pool = [i for i in sorted(self.graph.nodes) if i not in blocked and i != matched_id]
                scores = np.asarray([self._pair_evidence(candidate, self.graph.nodes[i]) for i in pool])
                distinctive = len(pool) == 1 or self._visual_outlier(scores, pool.index(historical))
                if distinctive and self._pair_informative(candidate, matched) and (
                    self._cached_similarity(candidate, matched) > self._verification_threshold
                ):
                    confirmed = True
                    break
            if confirmed:
                event.accepted = True
                event.reason = "accepted_deferred"
                event.decision_node_id = candidate.node_id
                self.graph.add_edge(query_id, matched_id, edge_type="loop")
            elif age >= self._defer_nodes:
                event.reason = "rejected_verification_timeout"
                event.decision_node_id = candidate.node_id
            else:
                remaining.append(event)
        self._pending_revisits = remaining

    def _recover_revisit(self, candidate: TopoNode, event: LoopClosureEvent) -> None:
        """Additive experimental recovery: never replace an accepted base decision."""
        config = self._recovery_config
        if not event.accepted:
            ids = [i for i in sorted(self.graph.nodes) if i < candidate.node_id - 1]
            scores = {i: self._cached_similarity(candidate, self.graph.nodes[i]) for i in ids}
            ranked = sorted(ids, key=lambda i: (-scores[i], i))[:config.top_k]
            outside = tuple(i for i in ranked if not self._passes_gates(
                np.asarray(candidate.pose[:2]) - self.graph.nodes[i].pose[:2],
                self._revisit_covariance(candidate, self.graph.nodes[i]),
            ))
            if outside and self._recovery_mode == "local":
                event.reason = "pending_local_recovery"
                self._recovery_pending.append((event, outside))
            elif outside:
                # Controls use the same top-k proposal pool, without local verification.
                evidence = {i: robust_evidence(scores[i], [scores[j] for j in ids if j != i])
                            for i in outside}
                eligible = [i for i in outside if evidence[i] >= config.visual_z and scores[i] > 0]
                if eligible:
                    if self._recovery_mode == "soft_odometry":
                        nll = {i: float(gaussian_position_statistics(
                            np.asarray(candidate.pose[:2]) - self.graph.nodes[i].pose[:2],
                            self._revisit_covariance(candidate, self.graph.nodes[i]),
                        )[1]) for i in outside}
                        selected = max(eligible, key=lambda i: evidence[i] - robust_evidence(
                            nll[i], [nll[j] for j in outside if j != i]
                        ))
                    else:
                        selected = max(eligible, key=lambda i: scores[i])
                    event.accepted, event.matched_node_id = True, selected
                    event.similarity, event.reason = scores[selected], "accepted_" + self._recovery_mode
                    self.graph.add_edge(candidate.node_id, selected, edge_type="loop")
            self.graph.recovery_diagnostics.append({
                "query_node_id": candidate.node_id, "decision_node_id": candidate.node_id,
                "reason": "proposal", "top_visual_ids": ranked, "outside_gate_ids": list(outside),
            })
        remaining = []
        for hypothesis, matches in self._recovery_pending:
            checks = [local_track(self.graph, hypothesis.query_node_id, matched_id,
                                  candidate.node_id, direction, self._cached_similarity, config)
                      for matched_id in matches for direction in (-1, 1)]
            self.graph.recovery_diagnostics.extend(checks)
            # Distinct places, not two orientations of one place, compete.
            passed = {row["matched_node_id"] for row in checks if row["passed"]}
            if len(passed) == 1:
                selected = next(iter(passed))
                hypothesis.accepted, hypothesis.matched_node_id = True, selected
                hypothesis.reason = "accepted_local_recovery"
                hypothesis.similarity = self._cached_similarity(
                    self.graph.nodes[hypothesis.query_node_id], self.graph.nodes[selected])
                hypothesis.decision_node_id = candidate.node_id
                self.graph.add_edge(hypothesis.query_node_id, selected, edge_type="loop")
            elif candidate.node_id - hypothesis.query_node_id >= config.horizon:
                hypothesis.reason = "recovery_ambiguous" if len(passed) > 1 else "recovery_timeout"
                hypothesis.decision_node_id = candidate.node_id
            else:
                remaining.append((hypothesis, matches))
        self._recovery_pending = remaining

    def _visual_outlier(self, similarities: np.ndarray, match_index: int) -> bool:
        """True if the match is a robust visual outlier in its evidence pool.

        Rejects matches that are not clearly distinctive — the core defence
        against perceptual-aliasing welds and descriptor-pollution snowballs.
        The strictness is ``self._visual_outlier_k`` (configurable per run).
        """
        # A relative outlier needs a comparison. A singleton cannot establish
        # visual distinctiveness; this does not prevent geometric/threshold matching.
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

    def _passes_gates(
        self,
        delta_xy: np.ndarray,
        combined_cov: np.ndarray,
    ) -> bool:
        """Single uncertainty-aware chi-square gate for co-location."""
        return mahalanobis_gate(delta_xy, combined_cov)

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
        if len(background) < _VISUAL_BACKGROUND_TARGET:
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
                if self._sequence_policy == "disjoint" and (
                    historical in {candidate.node_id, *live_context} or live_id == matched_id
                ):
                    break
                if self._sequence_policy == "self_only" and historical == live_id:
                    # Ignore this pair, but keep walking: later pairs can be valid.
                    continue
                scores.append(scorer(self.graph.nodes[live_id], self.graph.nodes[historical]))
            directional_scores.append(float(np.mean(scores)))
        return max(directional_scores)

    def _sequence_similarity(self, candidate: TopoNode, matched_id: int) -> float:
        """Bidirectional sequence score using multi-view cosine similarity."""
        return self._sequence_score(candidate, matched_id, self._cached_similarity)

    def _sequence_vmf_evidence(self, candidate: TopoNode, matched_id: int) -> float:
        """Bidirectional sequence score using vMF overlap evidence."""
        return self._sequence_score(candidate, matched_id, self._cached_vmf_evidence)

    def _pair_informative(self, first: TopoNode, second: TopoNode) -> bool:
        if self._visual_model in {"vmf", "kl_estimated"}:
            return first.visual_concentration > 0 and second.visual_concentration > 0
        return True

    def _pair_evidence(self, first: TopoNode, second: TopoNode) -> float:
        if self._visual_model == "cosine":
            return self._cached_similarity(first, second)
        if self._visual_model == "vmf":
            return self._cached_vmf_evidence(first, second)
        a, b = self._cached_node(first), self._cached_node(second)
        if self._visual_model == "mean_cosine":
            return float(np.clip(a[2] @ b[2], -1.0, 1.0))
        ka, kb = (self._fixed_concentration,) * 2 if self._visual_model == "kl_fixed" else (a[3], b[3])
        # Higher is better for every evidence model; KL itself is nonnegative.
        return -vmf_symmetric_kl(a[2], ka, b[2], kb)

    def _sequence_evidence(self, candidate: TopoNode, matched_id: int) -> float:
        if self._visual_model == "vmf":
            return self._sequence_vmf_evidence(candidate, matched_id)
        return self._sequence_score(candidate, matched_id, self._pair_evidence)

    def _revisit_covariance(self, candidate: TopoNode, matched: TopoNode) -> np.ndarray:
        """Uncertainty of the relative odometric displacement.

        A node's spatial extent describes the region it covers; it is not
        measurement uncertainty. Mixing the two made long corridor segments
        increasingly easy to accept as co-located. The cumulative covariance
        snapshots already encode uncertainty at the current and revisited
        observations through their interval-relative difference.
        """
        return (
            self._interval_covariance(matched.pose_covariance, candidate.pose_covariance)
            + np.eye(2) * _PLACE_LOCALIZATION_SIGMA**2
        )

    def _find_revisit(
        self, candidate: TopoNode
    ) -> tuple[int | None, int | None, float | None, str]:
        """Find a geometrically plausible revisit with explicit visual verification."""
        ids = sorted(self.graph.nodes)
        eligible_count = len(ids) - LOOP_TEMPORAL_EXCLUSION_NODES
        if eligible_count <= 0:
            return None, None, None, "no_eligible_candidate"

        eligible_ids = ids[:eligible_count]
        scorer = self._sequence_evidence

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
            return (
                proposed_id if accepted else None,
                proposed_id,
                float(scores[selected]),
                "accepted" if accepted else "rejected_by_gate",
            )

        nodes = [self.graph.nodes[node_id] for node_id in eligible_ids]
        deltas = np.asarray(candidate.pose[:2]) - np.asarray([node.pose[:2] for node in nodes])
        budgets = np.asarray([node.pose_covariance for node in nodes])
        covariance = candidate.pose_covariance - budgets
        covariance = 0.5 * (covariance + covariance.swapaxes(-1, -2))
        if np.min(np.linalg.eigvalsh(covariance)) < -1e-9:
            raise ValueError("uncertainty budget decreased between nodes")
        covariance += np.eye(2) * _PLACE_LOCALIZATION_SIGMA**2
        d2, nll = gaussian_position_statistics(deltas, covariance)
        order = np.argsort(nll, kind="stable")
        geometric_indices = [int(i) for i in order if d2[i] <= _CHI2_GATE_2DOF * self._geometric_gate_scale]
        proposed = geometric_indices[0] if geometric_indices else int(order[0])
        proposed_id = eligible_ids[proposed]
        if not geometric_indices:
            return None, proposed_id, None, "rejected_by_gate"
        if self._gate_mode == "geometric":
            return proposed_id, proposed_id, None, "accepted"
        if not self._require_visual_verification and len(geometric_indices) == 1:
            return proposed_id, proposed_id, float(scorer(candidate, proposed_id)), "accepted"
        # Every strict-mode closure needs appearance evidence, including a
        # sole geometric candidate. Score background independently of the winner.
        scores = np.asarray([scorer(candidate, node_id) for node_id in eligible_ids])
        for selected in geometric_indices:
            matched = nodes[selected]
            distinctive = len(scores) == 1 or self._distinctive_within_geometry(
                scores, selected, geometric_indices
            )
            verified = not self._require_visual_verification or (
                self._pair_informative(candidate, matched)
                and self._sequence_similarity(candidate, matched.node_id) > self._verification_threshold
            )
            if distinctive and verified:
                accepted_id = eligible_ids[selected]
                return accepted_id, accepted_id, float(scores[selected]), "accepted"
        if not self._require_visual_verification:
            return None, proposed_id, float(scores[proposed]), "rejected_by_gate"
        # Keep one explicit hypothesis, for bounded future confirmation.
        selected = max(geometric_indices, key=lambda i: scores[i])
        reason = "deferred_visual_verification" if self._defer_nodes else "rejected_visual_verification"
        return None, eligible_ids[selected], float(scores[selected]), reason
