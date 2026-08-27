"""Fuse dense and BM25 retrieval results with reciprocal rank fusion."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.core.config import Config
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_store import VectorStore


logger = logging.getLogger(__name__)
DEFAULT_TOP_K = 10
DEFAULT_RRF_K = 60


class HybridRetriever:
    """Combine dense and BM25 candidates into one ranked result list."""

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_retriever: BM25Retriever,
        config: Config | None = None,
    ) -> None:
        """Store retrievers and configuration used for candidate fusion."""
        self.config = config if config is not None else Config()
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever

    def search(self, query: str, *, top_k: int | None = None) -> list[dict[str, Any]]:
        """Return RRF-ranked chunks from dense and BM25 retrieval.

        Args:
            query: User question used by both retrieval methods.
            top_k: Optional number of fused results to return.

        Returns:
            Deduplicated chunk records ranked by descending RRF score.
        """
        result_count = (
            top_k
            if top_k is not None
            else int(
                self.config.get(
                    "retrieval", "hybrid_top_k", default=DEFAULT_TOP_K
                )
            )
        )
        if result_count <= 0:
            raise ValueError("top_k must be greater than zero")

        rrf_k = float(
            self.config.get("retrieval", "rrf_k", default=DEFAULT_RRF_K)
        )
        if rrf_k < 0:
            raise ValueError("rrf_k must be greater than or equal to zero")

        dense_top_k = int(self.config.get("retrieval", "dense_top_k", default=20))
        bm25_top_k = int(self.config.get("retrieval", "bm25_top_k", default=20))
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="retrieval") as executor:
            dense_future = executor.submit(self.vector_store.search, query, top_k=dense_top_k)
            bm25_future = executor.submit(self.bm25_retriever.search, query, top_k=bm25_top_k)
            dense_results = dense_future.result()
            bm25_results = bm25_future.result()

        fused_results: dict[str, dict[str, Any]] = {}
        _add_ranked_results(
            fused_results,
            dense_results,
            source="dense",
            metric_name="distance",
            rrf_k=rrf_k,
        )
        _add_ranked_results(
            fused_results,
            bm25_results,
            source="bm25",
            metric_name="score",
            rrf_k=rrf_k,
        )
        ranked_results = sorted(
            fused_results.values(),
            key=lambda result: (-float(result["rrf_score"]), str(result["chunk_id"])),
        )[:result_count]
        logger.info(
            "Fused %d dense and %d BM25 result(s) into %d chunk(s)",
            len(dense_results),
            len(bm25_results),
            len(ranked_results),
        )
        return ranked_results


def _add_ranked_results(
    fused_results: dict[str, dict[str, Any]],
    results: list[dict[str, Any]],
    *,
    source: str,
    metric_name: str,
    rrf_k: float,
) -> None:
    """Merge one ranked result list into the RRF accumulator."""
    for rank, result in enumerate(results, start=1):
        chunk_id = str(result["chunk_id"])
        fused_result = fused_results.setdefault(
            chunk_id,
            {
                "chunk_id": chunk_id,
                "text": str(result.get("text", "")),
                "metadata": dict(result.get("metadata") or {}),
                "rrf_score": 0.0,
                "dense_rank": None,
                "dense_distance": None,
                "bm25_rank": None,
                "bm25_score": None,
            },
        )
        fused_result["rrf_score"] += 1 / (rrf_k + rank)
        fused_result[f"{source}_rank"] = rank
        fused_result[f"{source}_{metric_name}"] = result.get(metric_name)
