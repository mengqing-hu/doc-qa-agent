"""Maintain a rolling summary of a conversation as it outgrows the context window."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from src.generation.llm import LLM
from src.store.messages import StoredMessage


logger = logging.getLogger(__name__)
DEFAULT_SUMMARY_MAX_TOKENS = 512
SUMMARIZER_PROMPT = """You maintain a running summary of a conversation between a user and a
document question-answering assistant. Update the summary below so it also
covers the new turns.

Preserve every entity, numeric value, document name,
and decision that a later turn might refer back to. Write plain prose with no preamble
and no bullet list.

Current summary:
{summary}

New turns:
{turns}

Updated summary:
"""


class ConversationSummarizer:
    """Fold older conversation turns into a compact running summary."""

    def __init__(self, llm: LLM, *, max_tokens: int = DEFAULT_SUMMARY_MAX_TOKENS) -> None:
        """Store the configured LLM used to write summaries."""
        self.llm = llm
        if max_tokens <= 0:
            raise ValueError("summary max_tokens must be greater than zero")
        self.max_tokens = max_tokens

    def summarize(
        self, prior_summary: str | None, messages: Sequence[StoredMessage]
    ) -> str:
        """Return an updated summary folding ``messages`` into ``prior_summary``."""
        if not messages:
            raise ValueError("messages must not be empty")
        turns = "\n".join(f"{message.role}: {message.content}" for message in messages)
        prompt = SUMMARIZER_PROMPT.format(
            summary=(prior_summary or "(none yet)"), turns=turns
        )
        summary = self.llm.generate(prompt, max_tokens=self.max_tokens).strip()
        if not summary:
            raise RuntimeError("Conversation summarizer returned an empty summary")
        return summary
