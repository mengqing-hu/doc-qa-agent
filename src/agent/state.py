"""State contracts shared by Agentic RAG graph nodes."""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict


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


class ConversationMessage(TypedDict):
    """Store one checkpoint-compatible user or assistant message."""

    role: Literal["user", "assistant"]
    content: str


class AgentState(TypedDict):
    """Represent the minimum state required by the initial routed graph."""

    question: str
    retrieval_action: NotRequired[Literal["retrieve", "abstain"]]
    retrieval_confidence: NotRequired[float]
    retrieval_reason: NotRequired[str]
    conversation_history: NotRequired[list[ConversationMessage]]
    conversation_context: NotRequired[list[ConversationMessage]]
    original_query: NotRequired[str]
    rewritten_query: NotRequired[str]
    rewrite_used_conversation_context: NotRequired[bool]
    rewrite_reason: NotRequired[str]
    retrieved_chunks: NotRequired[list[dict[str, Any]]]
    retrieval_attempts: NotRequired[int]
    relevant_chunks: NotRequired[list[dict[str, Any]]]
    relevant_chunk_ids: NotRequired[list[str]]
    relevance_decisions: NotRequired[list[dict[str, str]]]
    relevance_status: NotRequired[Literal["relevant", "none"]]
    relevance_reason: NotRequired[str]
    support_status: NotRequired[
        Literal["pending", "supported", "partially_supported", "unsupported"]
    ]
    support_claims: NotRequired[list[dict[str, Any]]]
    support_reason: NotRequired[str]
    utility_status: NotRequired[Literal["pending", "useful", "not_useful"]]
    utility_missing_requirements: NotRequired[list[str]]
    utility_reason: NotRequired[str]
    response: NotRequired[ResponseState]
