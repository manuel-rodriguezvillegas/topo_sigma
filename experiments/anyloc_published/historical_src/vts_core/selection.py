"""Explicit confirmation of a current language suggestion before navigation."""

from __future__ import annotations

from uuid import uuid4

from vts_core.topo_graph import TopoNode


class PendingSelection:
    def __init__(self) -> None:
        self.query_id: str | None = None
        self._nodes: dict[int, TopoNode] = {}

    def offer(self, nodes: list[TopoNode]) -> str:
        self.query_id = uuid4().hex
        self._nodes = {node.node_id: node for node in nodes}
        return self.query_id

    def confirm(self, query_id: str, node_id: int) -> TopoNode:
        if (
            not isinstance(node_id, int)
            or isinstance(node_id, bool)
            or query_id != self.query_id
            or node_id not in self._nodes
        ):
            raise ValueError("confirmation must name a candidate from the current query_id")
        node = self._nodes[node_id]
        self.query_id = None
        self._nodes = {}
        return node
