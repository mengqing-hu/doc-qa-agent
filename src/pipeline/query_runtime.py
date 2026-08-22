"""Assemble the online RAG pipeline from an existing Chroma collection."""

from __future__ import annotations

from src.core.config import Config
from src.generation.llm import LLM
from src.generation.prompts import PromptBuilder
from src.generation.rag_pipeline import RAGPipeline
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import VectorStore


def build_query_pipeline(config: Config) -> RAGPipeline:
    """Build a query pipeline without parsing or embedding source documents."""
    vector_store = VectorStore(config)
    chunks = vector_store.load_chunks()
    if not chunks:
        raise RuntimeError(
            "The configured Chroma collection is empty. "
            "Build the index with 'python -m scripts.build_index' before querying."
        )

    bm25_retriever = BM25Retriever(chunks, config)
    hybrid_retriever = HybridRetriever(vector_store, bm25_retriever, config)
    return RAGPipeline(
        hybrid_retriever,
        Reranker(config),
        PromptBuilder(config),
        LLM(config),
    )
