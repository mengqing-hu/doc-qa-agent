"""Wrap the existing retrieval pipeline as a ReAct tool."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.agent.state import Evidence
from src.generation.rag_pipeline import RAGPipeline


class VectorRetrieveTool:
    """Query the private document index through the existing RAG pipeline."""

    def __init__(self, pipeline: RAGPipeline) -> None:
        """Store the pipeline used to retrieve and rerank document chunks."""
        self.pipeline = pipeline

    def run(
        self,
        query: str,
        *,
        metadata_filter: Mapping[str, str] | None = None,
    ) -> list[Evidence]:
        """Return chunks adapted into the shared Evidence shape.

        With `metadata_filter`, return every chunk whose metadata matches it
        (no similarity ranking or top-k cap). Otherwise fall back to the
        existing reranked similarity search.
        """
        if metadata_filter:
            vector_store = self.pipeline.hybrid_retriever.vector_store
            chunks = vector_store.get_by_metadata(metadata_filter)
        else:
            chunks = self.pipeline.retrieve(query)
        return [chunk_to_evidence(chunk) for chunk in chunks]


def chunk_to_evidence(chunk: dict[str, Any]) -> Evidence:
    """Adapt one retrieved document chunk into the shared Evidence shape.

    `score` carries the reranker's relevance score when the chunk came from
    the similarity search path; it is absent (None) for metadata-filtered
    chunks, which were never scored or ranked.
    """
    raw_score = chunk.get("rerank_score")
    return {
        "chunk_id": str(chunk.get("chunk_id", "")),
        "text": str(chunk.get("text", "")),
        "origin": "rag",
        "metadata": dict(chunk.get("metadata") or {}),
        "score": float(raw_score) if raw_score is not None else None,
    }
