"""Assemble the online RAG pipeline from an existing Chroma collection."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from src.core.config import PROJECT_ROOT, Config
from src.generation.llm import LLM
from src.generation.prompts import PromptBuilder
from src.generation.rag_pipeline import RAGPipeline
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import VectorStore


_PIPELINE_CACHE: dict[tuple[str, str, int, int, int], RAGPipeline] = {}
_PIPELINE_CACHE_LOCK = threading.Lock()
_MAX_CACHED_PIPELINES = 4
logger = logging.getLogger(__name__)


def build_query_pipeline(config: Config) -> RAGPipeline:
    """Build or reuse a query pipeline for the configured index and models."""
    cache_key = _pipeline_cache_key(config)
    with _PIPELINE_CACHE_LOCK:
        cached_pipeline = _PIPELINE_CACHE.get(cache_key)
        if cached_pipeline is not None:
            logger.info("Reusing cached query pipeline")
            return cached_pipeline

        pipeline = _build_query_pipeline(config)
        _PIPELINE_CACHE[cache_key] = pipeline
        logger.info("Cached query pipeline with loaded models and index")
        while len(_PIPELINE_CACHE) > _MAX_CACHED_PIPELINES:
            _PIPELINE_CACHE.pop(next(iter(_PIPELINE_CACHE)))
        return pipeline


def clear_query_pipeline_cache() -> None:
    """Release cached models and indexes, primarily for tests or reconfiguration."""
    with _PIPELINE_CACHE_LOCK:
        _PIPELINE_CACHE.clear()


def _build_query_pipeline(config: Config) -> RAGPipeline:
    """Construct a query pipeline without parsing or embedding source documents."""
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


def _pipeline_cache_key(config: Config) -> tuple[str, str, int, int, int]:
    """Return a key that changes when configuration or indexed data changes."""
    config_path = Path(config.config_path).resolve()
    env_path = Path(config.env_path).resolve()
    index_path = Path(
        config.get("vector_store", "persist_dir", default="data/chroma_db")
    )
    if not index_path.is_absolute():
        index_path = PROJECT_ROOT / index_path
    return (
        str(config_path),
        str(env_path),
        _file_mtime_ns(config_path),
        _file_mtime_ns(env_path),
        _directory_mtime_ns(index_path),
    )


def _file_mtime_ns(path: Path) -> int:
    """Return a file modification timestamp, or zero for an absent optional file."""
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        return 0


def _directory_mtime_ns(path: Path) -> int:
    """Return the newest modification time beneath an index directory."""
    if not path.exists():
        return 0
    newest_mtime = _file_mtime_ns(path)
    for child in path.rglob("*"):
        newest_mtime = max(newest_mtime, _file_mtime_ns(child))
    return newest_mtime
