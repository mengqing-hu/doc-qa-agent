"""Wrap the existing retrieval pipeline as a ReAct tool."""

from __future__ import annotations

from typing import Any

from src.agent.state import Evidence
from src.generation.rag_pipeline import RAGPipeline


class VectorRetrieveTool:
    """Query the private document index through the existing RAG pipeline."""

    def __init__(self, pipeline: RAGPipeline) -> None:
        """Store the pipeline used to retrieve and rerank document chunks."""
        self.pipeline = pipeline

    def run(self, query: str) -> list[Evidence]:
        """Return reranked document chunks adapted into the shared Evidence shape."""
        chunks = self.pipeline.retrieve(query)
        return [chunk_to_evidence(chunk) for chunk in chunks]


def chunk_to_evidence(chunk: dict[str, Any]) -> Evidence:
    """Adapt one retrieved document chunk into the shared Evidence shape."""
    return {
        "chunk_id": str(chunk.get("chunk_id", "")),
        "text": str(chunk.get("text", "")),
        "origin": "rag",
        "metadata": dict(chunk.get("metadata") or {}),
    }
