"""Split parsed document sections into indexable chunks."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.config import Config


logger = logging.getLogger(__name__)
DEFAULT_CHUNK_SIZE = 2000
DEFAULT_CHUNK_OVERLAP = 200


def chunk_sections(
    sections: Sequence[Mapping[str, Any]],
    config: Config | None = None,
) -> list[dict[str, Any]]:
    """Split parsed sections into chunks with retrieval metadata.

    Args:
        sections: Sections returned by a document parser.
        config: Optional project configuration. Defaults to ``Config()``.

    Returns:
        Chunks containing content, source metadata, and a stable chunk ID.
    """
    chunking_config = config if config is not None else Config()
    chunk_size = int(
        chunking_config.get(
            "chunking", "chunk_size", default=DEFAULT_CHUNK_SIZE
        )
    )
    chunk_overlap = int(
        chunking_config.get(
            "chunking", "chunk_overlap", default=DEFAULT_CHUNK_OVERLAP
        )
    )
    _validate_chunking_parameters(chunk_size, chunk_overlap)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=("\n\n", "\n", ". ", " ", ""),
    )
    chunks: list[dict[str, Any]] = []
    source_indices: dict[str, int] = {}

    for section in sections:
        section_type = str(section.get("type", ""))
        if section_type == "text":
            section_chunks = splitter.split_text(str(section.get("content", "")))
        elif section_type == "table":
            # Preserve table structure; splitting would break row and column relationships.
            section_chunks = [str(section.get("content", ""))]
        else:
            logger.error("Unsupported section type: %s", section_type or "<missing>")
            raise ValueError(f"Unsupported section type: {section_type!r}")

        source_name = str(section.get("source", "document"))
        source_id = Path(source_name).stem or "document"
        for content in section_chunks:
            source_indices[source_id] = source_indices.get(source_id, 0) + 1
            chunk_index = source_indices[source_id]
            chunks.append(
                {
                    "chunk_id": f"{source_id}_chunk_{chunk_index:03d}",
                    "text": content,
                    "metadata": {
                        "source": section.get("source"),
                        "page": section.get("page"),
                        "section_title": section.get("title", ""),
                        "chunk_type": section_type,
                        "chunk_index": chunk_index,
                    },
                }
            )

    logger.info("Created %d chunk(s) from %d section(s)", len(chunks), len(sections))
    return chunks


def _validate_chunking_parameters(chunk_size: int, chunk_overlap: int) -> None:
    """Validate splitter parameters before constructing the splitter."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be in [0, chunk_size)")
