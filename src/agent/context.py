"""Reset per-turn retrieval state at the start of each graph invocation.

Conversation history no longer lives in the graph. The caller assembles a
bounded window from the transcript store (``src/store/messages.py`` via
``src/agent/context_window.py``) and passes it in as ``conversation_context`` /
``conversation_summary``; this node does not touch it.
"""

from __future__ import annotations

from typing import Any

from src.agent.state import AgentState
from src.core.config import Config


DEFAULT_MAX_RETRIEVAL_ROUNDS = 3


class ContextManager:
    """Prepare each turn's retrieval-loop state."""

    def __init__(self, config: Config) -> None:
        """Read the retrieval-round budget."""
        self.max_retrieval_rounds = int(
            config.get(
                "agent", "max_retrieval_rounds", default=DEFAULT_MAX_RETRIEVAL_ROUNDS
            )
        )
        if self.max_retrieval_rounds <= 0:
            raise ValueError("agent.max_retrieval_rounds must be greater than zero")

    def prepare_query(self, state: AgentState) -> dict[str, Any]:
        """Store the normalized question and reset this turn's retrieval state."""
        question = state["question"].strip()
        if not question:
            raise ValueError("question must not be empty")

        return {
            "original_query": question,
            "retrieval_rounds": 0,
            "max_retrieval_rounds": self.max_retrieval_rounds,
            "retrieval_plan": {"queries": [], "done": False, "reason": ""},
            "ungraded_evidence": [],
            "accumulated_evidence": [],
            "retrieval_history": [],
            "last_round_added_relevant": False,
            "synthesis_attempts": 0,
            "synthesis_truncated": False,
        }
