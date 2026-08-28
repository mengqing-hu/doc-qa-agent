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


class Evidence(TypedDict):
    """Store one piece of evidence in a shape shared by RAG and web results."""

    chunk_id: str
    text: str
    origin: Literal["rag", "web"]
    metadata: dict[str, Any]
    score: float | None


class PlannedQueryState(TypedDict):
    """One retrieval the planner asked for, in a checkpoint-compatible format."""

    query: str
    tool: Literal["vector_retrieve", "web_search"]
    metadata_filter: NotRequired[dict[str, str | list[str]] | None]


class RetrievalPlanState(TypedDict):
    """One round's plan: the queries to run and whether retrieval is complete."""

    queries: list[PlannedQueryState]
    done: bool
    reason: str


class RetrievalRound(TypedDict):
    """A completed planning + gather round, kept for the trace and evaluation."""

    round: int
    queries: list[PlannedQueryState]
    gathered_count: int
    added_chunk_ids: list[str]
    gather_errors: list[str]


class AgentState(TypedDict):
    """Represent the state shared by every node of the flattened retrieval graph."""

    question: str
    retrieval_action: NotRequired[Literal["retrieve", "chitchat", "abstain"]]
    retrieval_confidence: NotRequired[float]
    retrieval_reason: NotRequired[str]
    # The bounded window assembled from the transcript store by the caller; the
    # graph never holds the full transcript.
    conversation_context: NotRequired[list[ConversationMessage]]
    conversation_summary: NotRequired[str | None]
    original_query: NotRequired[str]

    retrieval_rounds: NotRequired[int]
    max_retrieval_rounds: NotRequired[int]
    retrieval_plan: NotRequired[RetrievalPlanState]
    ungraded_evidence: NotRequired[list[Evidence]]
    gather_errors: NotRequired[list[str]]
    accumulated_evidence: NotRequired[list[Evidence]]
    retrieval_history: NotRequired[list[RetrievalRound]]
    last_round_added_relevant: NotRequired[bool]

    synthesis_attempts: NotRequired[int]
    synthesis_truncated: NotRequired[bool]
    support_status: NotRequired[
        Literal["supported", "partially_supported", "unsupported", "error"]
    ]
    support_reason: NotRequired[str]
    support_claims: NotRequired[list[dict[str, Any]]]

    response: NotRequired[ResponseState]
    error: NotRequired[str | None]
