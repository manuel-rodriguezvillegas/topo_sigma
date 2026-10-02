import numpy as np
import pytest

from vts_core.retrieval import PlaceRetriever
from vts_core.topo_graph import TopoGraph, TopoNode


class Encoder:
    identity = "new-model"
    logit_scale = 100.0

    def embed_images(self, images):
        return np.array([[1.0, 0.0, 0.0]], np.float32)

    def embed_text(self, sentence):
        return np.array([-1.0, 0.0, 0.0], np.float32)


def test_singleton_negative_match_is_not_distinctive():
    graph = TopoGraph()
    graph.add_node(TopoNode(0, (0.0, 0.0, 0.0), np.ones(3), views=[np.zeros((2, 2, 3), np.uint8)]))
    retriever = PlaceRetriever(Encoder(), graph)
    ranked, distinctive = retriever.query("an absent destination")
    assert ranked[0][2] == -1
    assert distinctive is False
    with pytest.raises(ValueError, match="top_k"):
        retriever.query("anything", top_k=0)


def test_semantic_cache_reencoded_on_model_change():
    graph = TopoGraph()
    node = TopoNode(
        0,
        (0.0, 0.0, 0.0),
        np.ones(3),
        views=[np.zeros((2, 2, 3), np.uint8)],
        view_embeddings=np.array([[0.0, 1.0, 0.0]]),
        semantic_model_identity="old-model",
    )
    graph.add_node(node)
    PlaceRetriever(Encoder(), graph)
    np.testing.assert_array_equal(node.view_embeddings, [[1.0, 0.0, 0.0]])
    assert node.semantic_model_identity == Encoder.identity


def test_model_change_on_indexed_compact_map_requires_source_images():
    graph = TopoGraph()
    graph.add_node(
        TopoNode(
            0,
            (0.0, 0.0, 0.0),
            np.ones(3),
            view_embeddings=np.ones((1, 3)),
            semantic_model_identity="old",
        )
    )
    with pytest.raises(ValueError, match="no images"):
        PlaceRetriever(Encoder(), graph)
