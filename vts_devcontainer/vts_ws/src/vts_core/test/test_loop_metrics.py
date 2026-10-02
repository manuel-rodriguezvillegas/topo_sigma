import unittest

import numpy as np

from vts_core.metrics import graph_metrics
from vts_core.topo_graph import LoopClosureEvent, TopoGraph, TopoNode


def _node(node_id: int, x: float, y: float = 0.0) -> TopoNode:
    return TopoNode(
        node_id=node_id,
        pose=(x, y, 0.0),
        visual_features=np.array([1.0, 0.0], dtype=np.float32),
        room_label="room-a",
    )


class LoopMetricTest(unittest.TestCase):
    def test_short_return_is_an_evaluable_loop(self) -> None:
        graph = TopoGraph()
        positions = np.array([[0.0, 0.0], [10.0, 0.0], [0.1, 0.0]])
        for i, (x, y) in enumerate(positions):
            graph.add_node(_node(i, x, y))
        graph.add_edge(0, 1)
        graph.add_edge(1, 2)
        graph.loop_events.append(LoopClosureEvent(2, 0, False, 0.0, "rejected_by_gate"))
        gt = dict(enumerate(positions))
        missed = graph_metrics(graph, positions, node_gt_xy=gt, loop_tolerance=0.5)
        self.assertEqual(missed.loop_false_negatives, 1)
        graph.add_edge(0, 2, edge_type="loop")
        graph.loop_events[-1] = LoopClosureEvent(2, 0, True, 1.0, "accepted")
        closed = graph_metrics(graph, positions, node_gt_xy=gt, loop_tolerance=0.5)
        self.assertEqual(closed.loop_true_positives, 1)
        self.assertEqual(closed.loop_false_negatives, 0)

    def test_sequential_edges_do_not_enter_closure_precision(self) -> None:
        graph = TopoGraph()
        positions = ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (0.1, 0.0))
        for node_id, (x, y) in enumerate(positions):
            graph.add_node(_node(node_id, x, y))
        for a, b in ((0, 1), (1, 2), (2, 3)):
            graph.add_edge(a, b)
            graph.add_edge(a, b, edge_type="odometry")
        graph.add_edge(0, 3)
        graph.add_edge(0, 3, edge_type="loop")
        graph.loop_events.append(LoopClosureEvent(3, 0, True, 0.9, "accepted"))
        node_gt = {
            node_id: np.array(position, dtype=np.float64)
            for node_id, position in enumerate(positions)
        }

        result = graph_metrics(
            graph,
            np.array(positions, dtype=np.float64),
            node_gt_xy=node_gt,
            loop_tolerance=0.5,
        )

        self.assertEqual(result.n_sequential_edges, 3)
        self.assertEqual(result.n_loop_edges, 1)
        self.assertEqual(result.loop_true_positives, 1)
        self.assertEqual(result.loop_false_positives, 0)
        self.assertEqual(result.loop_false_negatives, 0)
        self.assertEqual(result.loop_precision, 1.0)
        self.assertEqual(result.loop_recall, 1.0)
        self.assertEqual(result.semantic_loop_precision, 1.0)
        self.assertEqual(result.semantic_loop_shortcuts, 0)


if __name__ == "__main__":
    unittest.main()
