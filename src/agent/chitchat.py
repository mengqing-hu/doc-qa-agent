"""Answer conversational requests that do not need document evidence."""

from __future__ import annotations

import json
from collections.abc import Sequence

from src.agent.state import ConversationMessage
from src.generation.llm import LLM


CHITCHAT_PROMPT = """You are a helpful assistant chatting alongside a document
question-answering system. This request does not need the indexed documents.

If the request asks about this conversation itself, answer it using the summary
and history below. Otherwise, answer using your own general knowledge. Keep the
reply concise and conversational. Do not claim the answer comes from any
document, and do not invent citations.

Earlier conversation summary:
{summary}

Recent conversation history:
{history}

User request:
{question}
"""


class LLMChitchatResponder:
    """Generate a conversational reply without retrieving document evidence."""

    def __init__(self, llm: LLM) -> None:
        """Store the configured LLM used to answer chitchat requests."""
        self.llm = llm

    def respond(
        self,
        question: str,
        conversation_context: Sequence[ConversationMessage] = (),
        summary: str | None = None,
    ) -> str:
        """Return a conversational reply, using history only when relevant."""
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")
        prompt = CHITCHAT_PROMPT.format(
            summary=(summary.strip() if summary and summary.strip() else "(none)"),
            history=json.dumps(list(conversation_context), ensure_ascii=False),
            question=json.dumps(normalized_question, ensure_ascii=False),
        )
        reply = self.llm.generate(prompt).strip()
        if not reply:
            raise RuntimeError("Chitchat responder returned an empty reply")
        return reply
