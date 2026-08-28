"""Assemble the bounded conversation window the graph's LLMs are allowed to see.

The store holds the full transcript. This module turns it into a token-bounded
view: a rolling summary of the older turns plus as many recent turns verbatim as
fit in the budget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.agent.state import ConversationMessage
from src.store.messages import StoredMessage


# Rough chars-per-token for mixed English / Chinese text. Good enough for a
# window budget; not used for billing.
CHARS_PER_TOKEN = 3.5
# Per-message framing overhead (role tag, separators) in estimated tokens.
MESSAGE_OVERHEAD_TOKENS = 4


def estimate_tokens(text: str) -> int:
    """Return a cheap deterministic token estimate for a string."""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


@dataclass(frozen=True)
class ContextWindow:
    """The conversation view passed into one graph invocation."""

    summary: str | None
    messages: list[ConversationMessage]


def build_context_window(
    history: list[StoredMessage],
    summary_text: str | None,
    summary_upto_seq: int,
    *,
    max_tokens: int,
) -> ContextWindow:
    """Return the rolling summary plus the most recent messages that fit the budget."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero")

    summary = (
        summary_text.strip() if summary_text and summary_text.strip() else None
    )
    budget = max_tokens - (estimate_tokens(summary) if summary else 0)

    # Only messages the summary does not already cover are eligible verbatim.
    uncovered = [message for message in history if message.seq > summary_upto_seq]

    selected: list[ConversationMessage] = []
    for message in reversed(uncovered):
        cost = estimate_tokens(message.content) + MESSAGE_OVERHEAD_TOKENS
        if cost > budget and selected:
            break
        budget -= cost
        selected.append({"role": message.role, "content": message.content})
    selected.reverse()
    return ContextWindow(summary=summary, messages=selected)
