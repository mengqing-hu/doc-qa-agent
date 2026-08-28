"""Represent grounded answers and their document sources."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceReference:
    """Identify one chunk that was supplied as answer evidence."""

    chunk_id: str
    source: str
    page: int | str | None
    section_title: str | None


@dataclass(frozen=True)
class RAGResponse:
    """Return generated answer text with the retrieved evidence references."""

    answer: str
    sources: tuple[SourceReference, ...]
    truncated: bool = False
