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


class ScratchpadEntry(TypedDict):
    """Record one completed ReAct iteration: what was asked, found, and cited."""

    thought: str
    action: Literal["vector_retrieve", "web_search"]
    action_input: str
    fact: str
    sources: list[SourceState]


class AgentState(TypedDict):
    """Represent the state shared by every ReAct + Self-RAG graph node."""

    question: str
    retrieval_action: NotRequired[Literal["retrieve", "abstain"]]
    retrieval_confidence: NotRequired[float]
    retrieval_reason: NotRequired[str]
    conversation_history: NotRequired[list[ConversationMessage]]
    conversation_context: NotRequired[list[ConversationMessage]]
    original_query: NotRequired[str]
    iteration_count: NotRequired[int]
    max_iterations: NotRequired[int]
    scratchpad: NotRequired[list[ScratchpadEntry]]
    action: NotRequired[Literal["vector_retrieve", "web_search", "finish"]]
    action_input: NotRequired[str]
    action_thought: NotRequired[str]
    tool_attempts: NotRequired[int]
    current_action: NotRequired[Literal["vector_retrieve", "web_search"]]
    current_evidence: NotRequired[list[Evidence]]
    relevance_status: NotRequired[Literal["relevant", "none"]]
    relevance_reason: NotRequired[str]
    relevant_evidence: NotRequired[list[Evidence]]
    current_fact: NotRequired[str]
    current_sources: NotRequired[list[SourceState]]
    support_status: NotRequired[
        Literal["supported", "partially_supported", "unsupported"]
    ]
    support_reason: NotRequired[str]
    response: NotRequired[ResponseState]
