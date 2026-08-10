"""Build an in-memory BM25 index for parsed document chunks."""

from __future__ import annotations

import logging
import re
from typing import Any

from rank_bm25 import BM25Okapi

from src.core.config import Config


logger = logging.getLogger(__name__)
DEFAULT_TOP_K = 20
TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


class BM25Retriever:
    """Search the same nested chunks used by the vector store."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        config: Config | None = None,
    ) -> None:
        """Build an in-memory BM25 index from chunk text."""
        if not chunks:
            raise ValueError("At least one chunk is required to build a BM25 index")

        self.config = config if config is not None else Config()
        self.chunks = chunks
        tokenized_chunks = [_tokenize(str(chunk.get("text", ""))) for chunk in chunks]
        self.index = BM25Okapi(tokenized_chunks)
        logger.info("Built BM25 index for %d chunk(s)", len(chunks))

    def search(self, query: str, *, top_k: int | None = None) -> list[dict[str, Any]]:
        """Return the highest-scoring chunks for a query."""
        result_count = top_k or int(
            self.config.get("retrieval", "bm25_top_k", default=DEFAULT_TOP_K)
        )
        if result_count <= 0:
            raise ValueError("top_k must be greater than zero")

        scores = self.index.get_scores(_tokenize(query))
        ranked_indices = sorted(
            range(len(scores)), key=lambda index: scores[index], reverse=True
        )[:result_count]
        return [
            {
                **self.chunks[index],
                "score": float(scores[index]),
            }
            for index in ranked_indices
        ]


def _tokenize(text: str) -> list[str]:
    """Normalize text into Unicode word tokens for BM25 matching."""
    return TOKEN_PATTERN.findall(text.lower())
