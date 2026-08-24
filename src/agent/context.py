"""Maintain bounded conversation context for query understanding."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.agent.state import AgentState, ConversationMessage
from src.core.config import Config


DEFAULT_MAX_HISTORY_MESSAGES = 8


class ContextManager:
    """Store recent conversation messages in checkpoint-compatible state."""

    def __init__(self, config: Config) -> None:
        """Read the maximum number of recent messages retained per thread."""
        self.max_history_messages = int(
            config.get(
                "agent",
                "context",
                "max_history_messages",
                default=DEFAULT_MAX_HISTORY_MESSAGES,
            )
        )
        if self.max_history_messages <= 0:
            raise ValueError("agent.context.max_history_messages must be greater than zero")

    def prepare_query(self, state: AgentState) -> dict[str, Any]:
        """Store the current user request and select prior messages for rewriting."""
        question = state["question"].strip()
        if not question:
            raise ValueError("question must not be empty")

        history = _history_from_state(state)
        return {
            "original_query": question,
            "rewritten_query": question,
            "rewrite_used_conversation_context": False,
            "rewrite_reason": "Query rewriting has not run for this turn.",
            "conversation_context": history[-self.max_history_messages :],
            "conversation_history": _trim_history(
                [*history, {"role": "user", "content": question}],
                self.max_history_messages,
            ),
            "retrieved_chunks": [],
            "retrieval_attempts": 0,
            "relevant_chunks": [],
            "relevant_chunk_ids": [],
            "relevance_decisions": [],
            "relevance_status": "none",
            "relevance_reason": "Relevance grading has not run for this turn.",
            "support_status": "pending",
            "support_claims": [],
            "support_reason": "Support verification has not run for this turn.",
            "utility_status": "pending",
            "utility_missing_requirements": [],
            "utility_reason": "Utility verification has not run for this turn.",
        }

    def persist_response(self, state: AgentState) -> dict[str, list[ConversationMessage]]:
        """Append the current user request and final response to conversation state."""
        history = _history_from_state(state)
        question = state["question"].strip()
        response = state.get("response")
        if not isinstance(response, Mapping) or not isinstance(response.get("answer"), str):
            raise RuntimeError("Agent graph cannot persist an invalid response")

        if not history or history[-1] != {"role": "user", "content": question}:
            history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": response["answer"]})
        return {
            "conversation_history": _trim_history(
                history,
                self.max_history_messages,
            )
        }


def _history_from_state(state: AgentState) -> list[ConversationMessage]:
    """Read validated conversation messages from graph state."""
    raw_history = state.get("conversation_history", [])
    if not isinstance(raw_history, list):
        raise RuntimeError("Conversation history must be a list")

    history: list[ConversationMessage] = []
    for message in raw_history:
        if not isinstance(message, Mapping):
            raise RuntimeError("Conversation history contains an invalid message")
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            raise RuntimeError("Conversation history contains an invalid message")
        history.append({"role": role, "content": content})
    return history


def _trim_history(
    history: list[ConversationMessage],
    max_history_messages: int,
) -> list[ConversationMessage]:
    """Keep only the most recent bounded set of conversation messages."""
    return history[-max_history_messages:]
