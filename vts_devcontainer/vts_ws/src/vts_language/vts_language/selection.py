"""Explicit confirmation of a current language suggestion before navigation."""

from __future__ import annotations

from uuid import uuid4

from vts_language.retrieval import MapPlace


class PendingSelection:
    def __init__(self) -> None:
        self.query_id: str | None = None
        self._nodes: dict[int, MapPlace] = {}

    def offer(self, places: list[MapPlace]) -> str:
        """Candidates are confirmed by their rank-independent place index."""
        self.query_id = uuid4().hex
        self._nodes = {place.key[1]: place for place in places}
        return self.query_id

    def confirm(self, query_id: str, node_id: int) -> MapPlace:
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
