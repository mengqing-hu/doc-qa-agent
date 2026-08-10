"""Create dense embeddings for document chunks and search queries."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Any

from sentence_transformers import SentenceTransformer

from src.core.config import Config


logger = logging.getLogger(__name__)
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"


class Embedder:
    """Encode chunk text with a Sentence Transformers model."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        model: Any | None = None,
    ) -> None:
        """Load the configured model, or use an injected model for testing."""
        self.config = config if config is not None else Config()
        model_name = str(
            self.config.get(
                "embedding", "model_name", default=DEFAULT_EMBEDDING_MODEL
            )
        )
        self.model_name = model_name
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        self.model = (
            model
            if model is not None
            else SentenceTransformer(model_name, token=hf_token)
        )
        logger.info("Loaded embedding model: %s", model_name)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode a sequence of texts as normalized dense vectors."""
        if not texts:
            return []

        embeddings = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Encode one query using the same embedding space as chunks."""
        embeddings = self.embed_texts([query])
        return embeddings[0]

    def embed_chunks(self, chunks: Sequence[dict[str, Any]]) -> list[list[float]]:
        """Encode the ``text`` field of each nested chunk record."""
        return self.embed_texts([str(chunk.get("text", "")) for chunk in chunks])
