"""State contracts shared by Agentic RAG graph nodes."""

from __future__ import annotations

from typing import NotRequired, TypedDict


class SourceState(TypedDict):
    """Store one source reference in a checkpoint-compatible format."""

    chunk_id: str
    source: str
    page: int | str | None
    section_title: str | None


class ResponseState(TypedDict):
    """Store a generated response in a checkpoint-compatible format."""

    answer: str
    sources: list[SourceState]


class AgentState(TypedDict):
    """Represent the minimum state required by the initial single-hop graph."""

    question: str
    response: NotRequired[ResponseState]
