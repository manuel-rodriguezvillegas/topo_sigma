"""Exploratory language-to-place ranking.

The relative softmax margin describes competition within the map. It neither
estimates destination presence nor authorizes a navigation target.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from vts_core.graph import TopoGraph
from vts_core.provenance import file_digest, identity

CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"

_PROMPT_TEMPLATES: tuple[str, ...] = (
    "a photo of a {}",
    "a photo of a {} in a building",
    "an indoor scene showing a {}",
    "a picture of the {} of an office building",
)

# Minimum top-2 posterior margin for an unambiguous answer. With softmax at
# CLIP's logit scale, 0.1 means "the best node holds at least 10 percentage
# points more probability mass than the runner-up" — a relative,
# per-query heuristic, not a calibrated probability or similarity cutoff.
_MIN_POSTERIOR_MARGIN: float = 0.1

# Imperative command phrasings stripped to recover the bare place description,
# so "take me to the printer area" embeds as "printer area" (the templates
# expect a noun phrase, not a full sentence).
_COMMAND_PREFIXES: tuple[str, ...] = (
    "take me to",
    "bring me to",
    "lead me to",
    "guide me to",
    "navigate to",
    "go to",
    "drive to",
    "move to",
    "head to",
    "walk to",
    "get to",
    "show me",
    "find",
    "locate",
    "where is",
    "where's",
    "where are",
    "i want to go to",
    "can you take me to",
)


def place_phrase(query: str) -> str:
    """Reduce a natural-language command to its place noun phrase."""
    phrase: str = query.strip().lower().rstrip(" ?.!")
    for prefix in _COMMAND_PREFIXES:
        if phrase.startswith(prefix):
            phrase = phrase[len(prefix) :].strip()
            break
    for article in ("the ", "a ", "an "):
        if phrase.startswith(article):
            phrase = phrase[len(article) :]
            break
    return phrase or query.strip().lower()


class SemanticEncoder:
    """Wrapper around a Hugging Face CLIP/SigLIP checkpoint."""

    def __init__(
        self, model_name: str = "openai/clip-vit-base-patch32", revision: str | None = None
    ) -> None:
        if revision is None:
            if model_name != "openai/clip-vit-base-patch32":
                raise ValueError("provide an immutable revision for a nondefault semantic model")
            revision = CLIP_REVISION
        if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise ValueError("semantic revision must be an immutable 40-character commit hash")
        self._device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = AutoModel.from_pretrained(model_name, revision=revision).to(self._device)
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(model_name, revision=revision)
        self.identity = identity(
            {
                "model": model_name,
                "revision": revision,
                "source": file_digest(__file__),
                "transformers": version("transformers"),
                "torch": version("torch"),
                "device": str(self._device),
            }
        )
        scale = getattr(self._model, "logit_scale", None)
        self._logit_scale: float = float(scale.exp().item()) if scale is not None else 100.0

    @property
    def logit_scale(self) -> float:
        """The model's learned similarity temperature."""
        return self._logit_scale

    @staticmethod
    def _as_embedding(output: object) -> torch.Tensor:
        """Extract the projected embedding tensor across transformers versions.

        transformers 5.x returns a model-output object from
        ``get_image_features`` / ``get_text_features`` whose ``pooler_output``
        is already the projected (shared-space) embedding; 4.x returns the
        tensor directly. Handle both.
        """
        if torch.is_tensor(output):
            return output
        for attr in ("image_embeds", "text_embeds", "pooler_output"):
            value = getattr(output, attr, None)
            if value is not None:
                return value
        return output.last_hidden_state[:, 0]  # type: ignore[attr-defined]

    @torch.no_grad()
    def embed_images(self, images_bgr: list[np.ndarray]) -> np.ndarray:
        """Embed BGR images; returns (n, d) L2-normalized float32."""
        pil_images: list[Image.Image] = [
            Image.fromarray(img[:, :, ::-1].copy()) for img in images_bgr
        ]
        inputs = self._processor(images=pil_images, return_tensors="pt").to(self._device)
        features: torch.Tensor = self._as_embedding(self._model.get_image_features(**inputs))
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def embed_text(self, query: str) -> np.ndarray:
        """Embed a text query through the prompt-template ensemble.

        The query is first reduced to its place noun phrase (so a full command
        like "take me to the kitchen" embeds as "kitchen") and then averaged
        over the prompt templates.
        """
        phrase: str = place_phrase(query)
        prompts: list[str] = [t.format(phrase) for t in _PROMPT_TEMPLATES]
        inputs = self._processor(text=prompts, return_tensors="pt", padding=True).to(self._device)
        features: torch.Tensor = self._as_embedding(self._model.get_text_features(**inputs))
        features = features / features.norm(dim=-1, keepdim=True)
        mean: torch.Tensor = features.mean(dim=0)
        mean = mean / mean.norm()
        return mean.cpu().numpy().astype(np.float32)


@dataclass
class MapPlace:
    """A place of a stored map with its representative views (BGR images)."""

    key: tuple[str, int]
    pose: np.ndarray
    views: list[np.ndarray]
    view_embeddings: np.ndarray | None = field(default=None, repr=False)


def load_map(graph_path: str | Path) -> list[MapPlace]:
    """Places of a graph saved by the graph builder, with views from ``<graph>_views/``."""
    import cv2

    graph_path = Path(graph_path)
    graph = TopoGraph.load(graph_path)
    views_dir = graph_path.with_name(graph_path.stem + "_views")
    places = []
    for session, index in graph.nodes():
        place = graph.places[session][index]
        views = [cv2.imread(str(views_dir / f"{index}_{k}.jpg"), cv2.IMREAD_COLOR) for k in range(len(place.view_frames))]
        places.append(MapPlace((session, index), place.pose, [v for v in views if v is not None]))
    return places


class PlaceRetriever:
    """Maps natural-language queries to places of a topological map."""

    def __init__(self, encoder: SemanticEncoder, places: list[MapPlace]) -> None:
        self._encoder = encoder
        self._places = [p for p in places if p.views]
        for place in self._places:
            place.view_embeddings = encoder.embed_images(place.views)

    def query(self, sentence: str, top_k: int = 3) -> tuple[list[tuple[MapPlace, float, float]], bool]:
        """Rank places for a sentence.

        Returns (ranked list of (place, posterior, raw multi-view cosine), distinctive).
        The flag is a relative heuristic (top-2 posterior margin), never evidence
        that the destination exists; ambiguous answers need user confirmation.
        """
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not sentence.strip():
            raise ValueError("query must not be empty")
        if not self._places:
            return [], False
        text = self._encoder.embed_text(sentence)
        scores = np.asarray([float((p.view_embeddings @ text).max()) for p in self._places])
        logits = self._encoder.logit_scale * scores
        posterior = np.exp(logits - logits.max())
        posterior /= posterior.sum()
        order = np.argsort(posterior)[::-1]
        ranked = [(self._places[i], float(posterior[i]), float(scores[i])) for i in order[:top_k]]
        distinctive = len(order) >= 2 and float(posterior[order[0]] - posterior[order[1]]) >= _MIN_POSTERIOR_MARGIN
        return ranked, distinctive
