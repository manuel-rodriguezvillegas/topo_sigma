"""Behavioral checks for the C2 posterior associator."""
import numpy as np
import pytest

from vts_core.topo_graph import TopoNode
from vts_evaluation.posterior_association import (
    Place, PosteriorConfig, associate, decisions_to_graph, log_likelihood_ratio,
)
from vts_evaluation.posterior_study import apply_cuts


def corridor(n: int = 12, revisit_of: dict[int, int] | None = None, dim: int = 16, seed: int = 0):
    """Nodes along a line; queries in ``revisit_of`` return to an earlier pose and appearance."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, dim))
    nodes, places = [], []
    for i in range(n):
        source = (revisit_of or {}).get(i, i)
        direction = base[source] + (0.05 * rng.normal(size=dim) if source != i else 0)
        direction /= np.linalg.norm(direction)
        x = 2.0 * source
        nodes.append(TopoNode(node_id=i, pose=(x, 0.0, 0.0), visual_features=direction.astype(np.float32),
                              pose_covariance=np.eye(2) * 0.01 * i))
        places.append(Place(direction, direction[None, :]))
    return nodes, places, [None] * n


def test_log_ratio_is_non_decreasing_and_uninformative_when_means_invert():
    s = np.linspace(-1, 1, 101)
    assert np.all(np.diff(log_likelihood_ratio(s, (0.7, 0.05), (0.5, 0.2))) >= 0)
    assert np.all(log_likelihood_ratio(s, (0.4, 0.1), (0.5, 0.1)) == 0)


def test_decisions_are_causal():
    nodes, places, halves = corridor(12, {10: 2})
    full = associate(nodes, places, halves, PosteriorConfig())
    prefix = associate(nodes[:11], places[:11], halves[:11], PosteriorConfig())
    assert full[: len(prefix)] == prefix


def test_geometric_revisit_wins_and_no_revisit_prefers_new_place():
    nodes, places, halves = corridor(12, {10: 2})
    records = {r["query"]: r for r in associate(nodes, places, halves, PosteriorConfig(use_appearance=False))}
    assert records[10]["matched"] == 2
    assert records[10]["posterior"] > records[10]["new"]
    assert records[6]["new"] > records[6]["posterior"]


def test_eps_floor_bounds_geometric_veto():
    nodes, places, halves = corridor(12, {10: 2})
    far = nodes[10]
    nodes[10] = TopoNode(node_id=10, pose=(500.0, 0.0, 0.0), visual_features=far.visual_features,
                         pose_covariance=far.pose_covariance)
    strict = associate(nodes, places, halves, PosteriorConfig(use_appearance=False, eps=1e-6))[9]
    robust = associate(nodes, places, halves, PosteriorConfig(use_appearance=False, eps=0.2))[9]
    assert strict["query"] == robust["query"] == 10
    assert robust["posterior"] > strict["posterior"]


def test_graph_materialization_respects_threshold():
    nodes, places, halves = corridor(12, {10: 2})
    records = associate(nodes, places, halves, PosteriorConfig(use_appearance=False))
    open_graph = decisions_to_graph(nodes, records, tau=0.0)
    closed_graph = decisions_to_graph(nodes, records, tau=1.1)
    assert len(open_graph.loop_events) == len(closed_graph.loop_events) == len(nodes) - 1
    assert not any(e.accepted for e in closed_graph.loop_events)
    assert sum(t == "loop" for t in open_graph.edge_types.values()) > 0
    assert sum(t == "odometry" for t in closed_graph.edge_types.values()) == len(nodes) - 1


def test_cuts_are_rigid_after_the_cut_frame():
    poses = np.column_stack([np.arange(100.0), np.zeros(100), np.zeros(100)])
    out, frames = apply_cuts(poses, 1, np.random.default_rng(0))
    c = frames[0]
    np.testing.assert_array_equal(out[:c], poses[:c])
    steps = lambda p: np.linalg.norm(np.diff(p[c:, :2], axis=0), axis=1)
    np.testing.assert_allclose(steps(out), steps(poses))
    assert np.linalg.norm(out[c, :2] - poses[c, :2]) >= 3.0 - 1e-9


def test_view_mixture_decisions_are_well_formed():
    nodes, places, halves = corridor(14, {10: 2, 12: 4})
    plain = associate(nodes, places, halves, PosteriorConfig(use_appearance=False))
    mixed = associate(nodes, places, halves, PosteriorConfig(view_mixture=True))
    # Decisions remain causal and well formed with the mixture enabled.
    assert [r["query"] for r in plain] == [r["query"] for r in mixed]
    assert all(0.0 <= r["posterior"] <= 1.0 for r in mixed)
