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
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")

        hybrid_results = self.hybrid_retriever.search(normalized_query)
        reranked_results = self.reranker.rerank(normalized_query, hybrid_results)
        prompt = self.prompt_builder.build(normalized_query, reranked_results)
        answer = self.llm.generate(prompt)
        sources = tuple(_source_reference(chunk) for chunk in reranked_results)
        logger.info(
            "Answered question with %d retrieved source(s)",
            len(sources),
        )
        return RAGResponse(answer=answer, sources=sources)


def _source_reference(chunk: dict[str, Any]) -> SourceReference:
    """Convert one reranked chunk into a public citation record."""
    metadata = chunk.get("metadata") or {}
    return SourceReference(
        chunk_id=str(chunk.get("chunk_id", "unknown")),
        source=str(metadata.get("source", "unknown source")),
        page=metadata.get("page"),
        section_title=metadata.get("section_title"),
    )
