"""Persist and query embedded chunks with ChromaDB."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import chromadb

from src.core.config import PROJECT_ROOT, Config
from src.retrieval.embedder import Embedder


logger = logging.getLogger(__name__)
DEFAULT_COLLECTION_NAME = "doc_chunks"
DEFAULT_PERSIST_DIRECTORY = "data/chroma_db"
DEFAULT_TOP_K = 5


class VectorStore:
    """Store chunk text, embeddings, and metadata in a persistent collection."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        embedder: Embedder | None = None,
    ) -> None:
        """Open the configured Chroma collection."""
        self.config = config if config is not None else Config()
        configured_persist_directory = Path(
            self.config.get(
                "vector_store", "persist_dir", default=DEFAULT_PERSIST_DIRECTORY
            )
        )
        persist_directory = _resolve_persist_directory(configured_persist_directory)
        persist_directory.mkdir(parents=True, exist_ok=True)
        collection_name = str(
            self.config.get(
                "vector_store", "collection_name", default=DEFAULT_COLLECTION_NAME
            )
        )
        self.client = chromadb.PersistentClient(path=str(persist_directory))
        self.collection = self.client.get_or_create_collection(collection_name)
        self.embedder = embedder if embedder is not None else Embedder(self.config)

    def add_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Upsert chunk text, embeddings, and scalar metadata."""
        if not chunks:
            return

        ids = [str(chunk["chunk_id"]) for chunk in chunks]
        documents = [str(chunk.get("text", "")) for chunk in chunks]
        metadatas = [_chroma_metadata(chunk.get("metadata", {})) for chunk in chunks]
        embeddings = [chunk.get("embedding") for chunk in chunks]
        if not all(isinstance(embedding, list) for embedding in embeddings):
            embeddings = self.embedder.embed_chunks(chunks)
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        logger.info(
            "Upserted %d chunk(s) into collection %s",
            len(chunks),
            self.collection.name,
        )

    def replace_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Replace all records in this collection with the supplied chunks."""
        collection_name = self.collection.name
        self.client.delete_collection(collection_name)
        self.collection = self.client.get_or_create_collection(collection_name)
        logger.info("Cleared collection %s before rebuilding", collection_name)
        self.add_chunks(chunks)

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        where: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search chunks by dense similarity with an optional metadata filter."""
        result_count = top_k or int(
            self.config.get("retrieval", "dense_top_k", default=DEFAULT_TOP_K)
        )
        if result_count <= 0:
            raise ValueError("top_k must be greater than zero")

        query_result = self.collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=result_count,
            where=dict(where) if where else None,
            include=["documents", "metadatas", "distances"],
        )
        return _format_query_results(query_result)

    def load_chunks(self) -> list[dict[str, Any]]:
        """Load all indexed chunks so BM25 can be rebuilt without re-ingestion."""
        result = self.collection.get(include=["documents", "metadatas"])
        chunks = _format_collection_records(result)
        logger.info(
            "Loaded %d indexed chunk(s) from collection %s",
            len(chunks),
            self.collection.name,
        )
        return chunks


def _chroma_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only scalar metadata values accepted by Chroma."""
    return {
        key: value
        for key, value in metadata.items()
        if value is not None and isinstance(value, (str, int, float, bool))
    }


def _resolve_persist_directory(configured_directory: Path) -> Path:
    """Resolve relative Chroma paths from the project root."""
    if configured_directory.is_absolute():
        return configured_directory
    return PROJECT_ROOT / configured_directory


def _format_query_results(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Convert Chroma's batched response into nested chunk records."""
    identifiers = (result.get("ids") or [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    return [
        {
            "chunk_id": chunk_id,
            "text": documents[index],
            "metadata": metadatas[index] or {},
            "distance": distances[index] if index < len(distances) else None,
        }
        for index, chunk_id in enumerate(identifiers)
    ]


def _format_collection_records(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Convert a Chroma collection read into chunk records for lexical search."""
    identifiers = result.get("ids") or []
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    return [
        {
            "chunk_id": str(chunk_id),
            "text": str(documents[index]) if index < len(documents) else "",
            "metadata": dict(metadatas[index] or {}) if index < len(metadatas) else {},
        }
        for index, chunk_id in enumerate(identifiers)
    ]
