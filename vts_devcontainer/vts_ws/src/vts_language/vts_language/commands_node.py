"""Language suggestions and explicit user-confirmed navigation targets.

Manual mode writes ranked suggestions only. In topic mode, send a text query,
then JSON {"query_id": "<returned id>", "confirm": <candidate node id>} on
/language/command to select a destination. Relative margins cannot establish
that a requested destination exists; they never publish targets automatically.
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import suppress

import cv2
import rclpy
from geometry_msgs.msg import Pose2D
from PIL import Image
from rclpy.node import Node
from std_msgs.msg import String

from vts_core.provenance import atomic_json
from vts_language.retrieval import MapPlace, PlaceRetriever, SemanticEncoder, load_map
from vts_language.selection import PendingSelection


class CommandsNode(Node):
    """Maps natural-language commands to topological-map targets."""

    def __init__(self) -> None:
        super().__init__("commands")

        self.declare_parameter("graph_path", "output/ros/graph_0.npz")
        self.declare_parameter("semantic_model", "openai/clip-vit-base-patch32")
        self.declare_parameter("semantic_revision", "")
        self.declare_parameter("mode", "manual")
        self.declare_parameter("query_sentence", "corridor")
        self.declare_parameter("top_k", 3)
        self.declare_parameter("output_dir", "output/commands")

        graph_path: str = self.get_parameter("graph_path").get_parameter_value().string_value
        model_name: str = self.get_parameter("semantic_model").get_parameter_value().string_value
        self.mode: str = self.get_parameter("mode").get_parameter_value().string_value
        self._query_sentence: str = (
            self.get_parameter("query_sentence").get_parameter_value().string_value
        )
        self._top_k: int = self.get_parameter("top_k").get_parameter_value().integer_value
        self._output_dir: str = self.get_parameter("output_dir").get_parameter_value().string_value
        os.makedirs(self._output_dir, exist_ok=True)

        self._target_pub = self.create_publisher(Pose2D, "/language/navigation_target", 10)
        self._result_pub = self.create_publisher(String, "/language/result", 10)

        places = load_map(graph_path)
        encoder: SemanticEncoder = SemanticEncoder(
            model_name, str(self.get_parameter("semantic_revision").value) or None
        )
        self._selection = PendingSelection()
        self._retriever: PlaceRetriever = PlaceRetriever(encoder, places)
        self.get_logger().info(
            f"Loaded graph '{graph_path}' ({len(places)} places) with model '{model_name}'."
        )

        if self.mode == "topic":
            self.create_subscription(String, "/language/command", self._on_command, 10)
            self.get_logger().info(
                "Topic mode: publish a query on /language/command (std_msgs/String)."
            )

    # ------------------------------------------------------------------ #
    def _on_command(self, msg: String) -> None:
        query: str = msg.data.strip()
        if not query:
            return
        if query.startswith("{"):
            try:
                request = json.loads(query)
                selected = self._selection.confirm(request["query_id"], request["confirm"])
            except (ValueError, KeyError, TypeError) as error:
                self.get_logger().error(str(error))
                return
            target = Pose2D(
                x=float(selected.pose[0]), y=float(selected.pose[1]), theta=float(selected.pose[2])
            )
            self._target_pub.publish(target)
            report = {
                "status": "confirmed",
                "query_id": request["query_id"],
                "node_id": selected.key[1],
            }
            self._result_pub.publish(String(data=json.dumps(report)))
            atomic_json(os.path.join(self._output_dir, "result.json"), report)
        else:
            self.answer(query)

    def answer(self, sentence: str) -> dict[str, object]:
        """Answer one query: publish, log, and persist the report to disk."""
        ranked, distinctive = self._retriever.query(sentence, top_k=self._top_k)

        report: dict[str, object] = {
            "query": sentence,
            "query_id": self._selection.offer([node for node, _, _ in ranked]),
            "status": "needs_confirmation" if ranked else "no_candidates",
            "destination_presence": "unknown",
            "distinctive_margin": distinctive,
            "candidates": [
                {
                    "node_id": node.key[1],
                    "posterior": round(posterior, 4),
                    "score": round(score, 4),
                    "pose": [float(v) for v in node.pose],
                }
                for node, posterior, score in ranked
            ],
        }

        out: String = String()
        out.data = json.dumps(report)
        self._result_pub.publish(out)
        self.get_logger().info(json.dumps(report, indent=2))

        atomic_json(os.path.join(self._output_dir, "result.json"), report)
        for rank, (node, _, _) in enumerate(ranked, start=1):
            self._save_views(node, rank)

        return report

    def _save_views(self, node: MapPlace, rank: int) -> None:
        for view_index, view in enumerate(node.views):
            path: str = os.path.join(
                self._output_dir,
                f"rank{rank}_node{node.key[1]}_v{view_index}.pdf",
            )
            rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb).save(path, "PDF", resolution=150.0)


def _flush(node: CommandsNode, seconds: float = 1.0) -> None:
    """Spin briefly so queued publications are delivered before exit."""
    deadline: float = time.time() + seconds
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    with suppress(KeyboardInterrupt):
        node: CommandsNode = CommandsNode()
        if node.mode == "manual":
            node.answer(node._query_sentence)
            _flush(node)
        elif node.mode == "topic":
            rclpy.spin(node)
        else:
            node.get_logger().error(f"Invalid mode: {node.mode}")
        node.destroy_node()
    rclpy.try_shutdown()
    sys.exit(0)


if __name__ == "__main__":
    main()
