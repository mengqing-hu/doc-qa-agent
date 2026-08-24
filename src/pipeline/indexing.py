"""Build the persistent document index used by the online query runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core.config import PROJECT_ROOT, Config
from src.document.chunker import prepare_chunks
from src.document.pdf_parser import parse_pdf_document
from src.document.word_parser import parse_word_document
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore


def build_document_index(config: Config) -> int:
    """Parse configured documents and replace their Chroma collection."""
    _, chunks = prepare_chunks(parse_configured_documents(config), config)
    vector_store = VectorStore(config, embedder=Embedder(config))
    vector_store.replace_chunks(chunks)
    return len(chunks)


def parse_configured_documents(config: Config) -> list[dict[str, Any]]:
    """Parse every configured PDF and Word document into section records."""
    configured_paths = config.get("documents", "paths", default=[])
    if not configured_paths:
        raise ValueError("documents.paths must contain at least one document")

    sections: list[dict[str, Any]] = []
    for configured_path in configured_paths:
        document_path = Path(configured_path)
        if not document_path.is_absolute():
            document_path = PROJECT_ROOT / document_path
        suffix = document_path.suffix.lower()
        if suffix == ".pdf":
            sections.extend(parse_pdf_document(document_path, config))
        elif suffix == ".docx":
            sections.extend(parse_word_document(document_path))
        else:
            raise ValueError(f"Unsupported document type: {document_path}")
    return sections
