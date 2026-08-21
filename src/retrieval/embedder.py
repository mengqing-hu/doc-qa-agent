"""Create dense embeddings for document chunks and search queries."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Any

from openai import OpenAI, OpenAIError
from sentence_transformers import SentenceTransformer

from src.core.config import Config


logger = logging.getLogger(__name__)
DEFAULT_PROVIDER = "local"
DEFAULT_LOCAL_MODEL_NAME = "BAAI/bge-large-en-v1.5"
DEFAULT_SCADSAI_MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
DEFAULT_SCADSAI_BASE_URL = "https://llm.scads.ai/v1"


class Embedder:
    """Encode chunk text with a local Sentence Transformers model or ScaDS.AI."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        model: Any | None = None,
        client: Any | None = None,
    ) -> None:
        """Load the configured local model, or configure the ScaDS.AI embedding client."""
        self.config = config if config is not None else Config()
        self.provider = str(
            self.config.get("embedding", "provider", default=DEFAULT_PROVIDER)
        ).lower()
        default_model_name = (
            DEFAULT_SCADSAI_MODEL_NAME
            if self.provider == "scadsai"
            else DEFAULT_LOCAL_MODEL_NAME
        )
        self.model_name = str(
            self.config.get("embedding", "model_name", default=default_model_name)
        )

        self.model: Any | None = None
        self.client: Any | None = None

        if self.provider == "local":
            hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
            self.model = (
                model
                if model is not None
                else SentenceTransformer(self.model_name, token=hf_token)
            )
        elif self.provider == "scadsai":
            self.client = client if client is not None else OpenAI(
                api_key=self.config.scadsai_api_key,
                base_url=str(
                    self.config.get(
                        "embedding", "base_url", default=DEFAULT_SCADSAI_BASE_URL
                    )
                ),
            )
        else:
            raise ValueError("embedding.provider must be either 'local' or 'scadsai'")

        logger.info("Configured %s embedding model: %s", self.provider, self.model_name)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode a sequence of texts as normalized dense vectors."""
        if not texts:
            return []

        return (
            self._embed_locally(texts)
            if self.provider == "local"
            else self._embed_with_scadsai(texts)
        )

    def embed_query(self, query: str) -> list[float]:
        """Encode one query using the same embedding space as chunks."""
        return self.embed_texts([query])[0]

    def embed_chunks(self, chunks: Sequence[dict[str, Any]]) -> list[list[float]]:
        """Encode the ``text`` field of each nested chunk record."""
        return self.embed_texts([str(chunk.get("text", "")) for chunk in chunks])

    def _embed_locally(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode texts with the local Sentence Transformers model."""
        if self.model is None:
            raise RuntimeError("Local embedding model is not configured")

        embeddings = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def _embed_with_scadsai(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode texts through ScaDS.AI's OpenAI-compatible embeddings endpoint."""
        if self.client is None:
            raise RuntimeError("ScaDS.AI embedding client is not configured")

        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=list(texts),
            )
        except OpenAIError as error:
            logger.exception(
                "ScaDS.AI embedding request failed for %s", self.model_name
            )
            raise RuntimeError("ScaDS.AI embedding request failed") from error

        if len(response.data) != len(texts):
            raise RuntimeError(
                "ScaDS.AI embedding response did not contain one vector per text"
            )
        ordered = sorted(response.data, key=lambda item: item.index)
        return [_normalize(item.embedding) for item in ordered]


def _normalize(vector: Sequence[float]) -> list[float]:
    """L2-normalize an embedding vector to match the local provider's output."""
    magnitude = sum(value * value for value in vector) ** 0.5
    if magnitude == 0:
        return list(vector)
    return [value / magnitude for value in vector]
