"""Coordinate hybrid retrieval, reranking, prompt construction, and generation."""

from __future__ import annotations

import logging
from typing import Any

from src.generation.llm import LLM
from src.generation.prompts import PromptBuilder
from src.generation.response import RAGResponse, SourceReference
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker


logger = logging.getLogger(__name__)


class RAGPipeline:
    """Answer questions from retrieved document chunks."""

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: Reranker,
        prompt_builder: PromptBuilder,
        llm: LLM,
    ) -> None:
        """Store the retrieval and generation components used for each question."""
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.prompt_builder = prompt_builder
        self.llm = llm

    def answer(self, query: str) -> RAGResponse:
        """Retrieve evidence, generate an answer, and return its source references."""
        return self.generate(query, self.retrieve(query))

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        """Retrieve and rerank the final evidence chunks for a query."""
        normalized_query = _normalized_query(query)

        hybrid_results = self.hybrid_retriever.search(normalized_query)
        return self.reranker.rerank(normalized_query, hybrid_results)

    def generate(
        self,
        query: str,
        evidence_chunks: list[dict[str, Any]],
        *,
        max_context_characters: int | None = None,
        max_tokens: int | None = None,
    ) -> RAGResponse:
        """Generate a grounded answer from already retrieved evidence chunks."""
        normalized_query = _normalized_query(query)
        prompt = self.prompt_builder.build(
            normalized_query,
            evidence_chunks,
            max_context_characters=max_context_characters,
        )
        answer = self.llm.generate(prompt, max_tokens=max_tokens)
        sources = tuple(_source_reference(chunk) for chunk in evidence_chunks)
        logger.info(
            "Answered question with %d retrieved source(s)",
            len(sources),
        )
        return RAGResponse(answer=answer, sources=sources)


def _normalized_query(query: str) -> str:
    """Validate and normalize a query shared by retrieval and generation."""
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query must not be empty")
    return normalized_query


def _source_reference(chunk: dict[str, Any]) -> SourceReference:
    """Convert one reranked chunk into a public citation record."""
    metadata = chunk.get("metadata") or {}
    return SourceReference(
        chunk_id=str(chunk.get("chunk_id", "unknown")),
        source=str(metadata.get("source", "unknown source")),
        page=metadata.get("page"),
        section_title=metadata.get("section_title"),
    )
