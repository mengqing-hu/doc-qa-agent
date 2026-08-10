"""Rerank fused retrieval candidates with a Cross-Encoder."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Any

from sentence_transformers import CrossEncoder

from src.core.config import Config


logger = logging.getLogger(__name__)
DEFAULT_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_BATCH_SIZE = 16
DEFAULT_TOP_K = 5


class Reranker:
    """Score query-chunk pairs and return the most relevant candidates."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        model: Any | None = None,
    ) -> None:
        """Load the configured Cross-Encoder, or use an injected test model."""
        self.config = config if config is not None else Config()
        self.model_name = str(
            self.config.get("reranking", "model_name", default=DEFAULT_MODEL_NAME)
        )
        self.batch_size = int(
            self.config.get("reranking", "batch_size", default=DEFAULT_BATCH_SIZE)
        )
        if self.batch_size <= 0:
            raise ValueError("reranking.batch_size must be greater than zero")

        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        self.model = (
            model
            if model is not None
            else CrossEncoder(self.model_name, token=hf_token)
        )
        logger.info("Loaded reranking model: %s", self.model_name)

    def rerank(
        self,
        query: str,
        chunks: Sequence[dict[str, Any]],
        *,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return query candidates ranked by descending Cross-Encoder score.

        Args:
            query: User question used to score each candidate.
            chunks: Fused candidates containing a ``text`` field.
            top_k: Optional number of reranked candidates to return.

        Returns:
            Copies of the highest-scoring chunk records with ``rerank_score``.
        """
        result_count = (
            top_k
            if top_k is not None
            else int(
                self.config.get("retrieval", "final_top_k", default=DEFAULT_TOP_K)
            )
        )
        if result_count <= 0:
            raise ValueError("top_k must be greater than zero")
        if not chunks:
            return []

        # Score pairs together because a Cross-Encoder models query-document interaction.
        pairs = [(query, str(chunk.get("text", ""))) for chunk in chunks]
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        if len(scores) != len(chunks):
            raise ValueError("Cross-Encoder returned an unexpected number of scores")

        ranked_chunks = [
            {**chunk, "rerank_score": float(score)}
            for chunk, score in zip(chunks, scores, strict=True)
        ]
        ranked_chunks.sort(
            key=lambda chunk: (-chunk["rerank_score"], str(chunk["chunk_id"]))
        )
        results = ranked_chunks[:result_count]
        logger.info(
            "Reranked %d candidate(s) and returned %d result(s)",
            len(chunks),
            len(results),
        )
        return results
