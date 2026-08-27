"""Persist and query embedded chunks with ChromaDB."""

from __future__ import annotations

import itertools
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
DEFAULT_METADATA_FILTER_LIMIT = 200


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

    def get_by_metadata(
        self,
        where: Mapping[str, Any],
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return every chunk whose metadata exactly matches `where`.

        Unlike `search`, this performs no similarity ranking: it returns all
        matching chunks (up to `limit` per distinct value combination), which
        is what a structural request (e.g. every chunk of one type, or every
        chunk from one or more documents) needs instead of the top-k most
        similar chunks to a query.

        When a value is a list (match any of these), `limit` applies
        separately to each value rather than to the combined query — a
        single shared limit would let one value's matches crowd out
        another's (e.g. a longer document silently squeezing a shorter one
        out of a multi-document filter). Multiple list-valued keys are
        combined as every value combination between them, each queried and
        capped independently.
        """
        if not where:
            raise ValueError("where must not be empty")
        result_limit = limit if limit is not None else DEFAULT_METADATA_FILTER_LIMIT
        if result_limit <= 0:
            raise ValueError("limit must be greater than zero")

        keys = list(where.keys())
        value_options = [
            list(value) if isinstance(value, (list, tuple)) else [value]
            for value in where.values()
        ]
        combinations = [
            dict(zip(keys, combination, strict=True))
            for combination in itertools.product(*value_options)
        ]

        seen_chunk_ids: set[str] = set()
        chunks: list[dict[str, Any]] = []
        for combination in combinations:
            result = self.collection.get(
                where=_chroma_where(combination),
                limit=result_limit,
                include=["documents", "metadatas"],
            )
            for chunk in _format_collection_records(result):
                if chunk["chunk_id"] not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk["chunk_id"])
                    chunks.append(chunk)
        return chunks

    def get_indexed_sources(self) -> list[str]:
        """Return the distinct source documents actually present in the index.

        This reads the live collection rather than any build-time config, so
        it always reflects what is really indexed even if the index was
        built or modified through a different path than the current config.
        """
        chunks = self.load_chunks()
        sources = {
            str(chunk.get("metadata", {}).get("source"))
            for chunk in chunks
            if chunk.get("metadata", {}).get("source")
        }
        return sorted(sources)

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


def _chroma_where(where: Mapping[str, Any]) -> dict[str, Any]:
    """Translate a flat equality/membership filter into Chroma's syntax.

    Chroma's `where` clause accepts exactly one top-level key; combining
    multiple conditions requires wrapping them in `$and`. A list or tuple
    value means "match any of these" and is translated to `$in`; any other
    value is matched by plain equality.
    """
    conditions = [
        {key: {"$in": list(value)} if isinstance(value, (list, tuple)) else value}
        for key, value in where.items()
    ]
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


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
