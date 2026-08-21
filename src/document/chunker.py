"""Split parsed document sections into indexable chunks."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.config import Config


logger = logging.getLogger(__name__)
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_EXCLUDED_SECTION_KINDS = (
    "front_matter",
    "toc",
    "references",
    "appendix",
    "header_footer",
    "footnote",
)


def prepare_chunks(
    sections: Sequence[Mapping[str, Any]],
    config: Config | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Filter parsed sections and create retrieval chunks from the survivors."""
    chunking_config = config if config is not None else Config()
    filtered_sections = filter_sections(sections, chunking_config)
    chunks = _chunk_filtered_sections(filtered_sections, chunking_config)
    return filtered_sections, chunks


def filter_sections(
    sections: Sequence[Mapping[str, Any]],
    config: Config | None = None,
) -> list[dict[str, Any]]:
    """Apply configured retrieval filters without mutating the parsed sections."""
    chunking_config = config if config is not None else Config()
    configured_kinds = chunking_config.get(
        "chunking",
        "filters",
        "exclude_section_kinds",
        default=DEFAULT_EXCLUDED_SECTION_KINDS,
    )
    if not isinstance(configured_kinds, (list, tuple, set)):
        raise ValueError("chunking.filters.exclude_section_kinds must be a sequence")
    excluded_kinds = {str(kind).strip().casefold() for kind in configured_kinds}
    filtered_sections: list[dict[str, Any]] = []
    for section in sections:
        section_copy = dict(section)
        section_kind = str(section_copy.get("section_kind", "body")).casefold()
        if section_kind in excluded_kinds:
            continue
        filtered_sections.append(section_copy)
    logger.info(
        "Retained %d of %d parsed section(s) for retrieval",
        len(filtered_sections),
        len(sections),
    )
    return filtered_sections


def chunk_sections(
    sections: Sequence[Mapping[str, Any]],
    config: Config | None = None,
) -> list[dict[str, Any]]:
    """Filter parsed sections and split survivors into chunks with metadata.

    Args:
        sections: Sections returned by a document parser.
        config: Optional project configuration. Defaults to ``Config()``.

    Returns:
        Chunks containing content, source metadata, and a stable chunk ID.
    """
    chunking_config = config if config is not None else Config()
    filtered_sections = filter_sections(sections, chunking_config)
    return _chunk_filtered_sections(filtered_sections, chunking_config)


def _chunk_filtered_sections(
    sections: Sequence[Mapping[str, Any]],
    chunking_config: Config,
) -> list[dict[str, Any]]:
    """Split already-filtered sections into indexable chunks."""
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
