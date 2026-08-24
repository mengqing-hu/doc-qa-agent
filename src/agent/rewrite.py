"""Rewrite conversation-dependent requests into standalone retrieval queries."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from src.agent.state import ConversationMessage
from src.generation.llm import LLM


CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
QUERY_REWRITE_PROMPT = """Rewrite the current user request into a standalone query for document retrieval.

Use the conversation history only to resolve references, omitted entities, or context.
Do not answer the question. Do not add facts that are absent from the user request and history.
If the request is already standalone, preserve its meaning with minimal changes.

Return only a JSON object with this exact schema:
{{
  "rewritten_query": "A standalone retrieval query.",
  "used_conversation_context": true,
  "reason": "A concise explanation."
}}

Conversation history:
{history}

Current user request:
{question}
"""


@dataclass(frozen=True)
class RewriteDecision:
    """Capture the standalone query produced by the rewriting LLM."""

    rewritten_query: str
    used_conversation_context: bool
    reason: str


class LLMQueryRewriter:
    """Rewrite document questions with bounded prior conversation context."""

    def __init__(self, llm: LLM) -> None:
        """Store the configured LLM used to rewrite retrieval queries."""
        self.llm = llm

    def rewrite(
        self,
        question: str,
        conversation_context: Sequence[ConversationMessage],
    ) -> RewriteDecision:
        """Return a validated standalone retrieval query."""
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")
        prompt = QUERY_REWRITE_PROMPT.format(
            history=json.dumps(list(conversation_context), ensure_ascii=False),
            question=json.dumps(normalized_question, ensure_ascii=False),
        )
        return _parse_rewrite_decision(self.llm.generate(prompt))


def _parse_rewrite_decision(response: str) -> RewriteDecision:
    """Validate the JSON-only response required from the rewriting LLM."""
    response_text = response.strip()
    code_fence_match = CODE_FENCE_PATTERN.fullmatch(response_text)
    if code_fence_match:
        response_text = code_fence_match.group(1).strip()

    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise RuntimeError("LLM query rewriter did not return valid JSON") from error
    if not isinstance(payload, Mapping):
        raise RuntimeError("LLM query rewriter response must be a JSON object")

    rewritten_query = payload.get("rewritten_query")
    used_conversation_context = payload.get("used_conversation_context")
    reason = payload.get("reason")
    if not isinstance(rewritten_query, str) or not rewritten_query.strip():
        raise RuntimeError("LLM query rewriter returned an invalid rewritten_query")
    if not isinstance(used_conversation_context, bool):
        raise RuntimeError(
            "LLM query rewriter returned an invalid used_conversation_context"
        )
    if not isinstance(reason, str) or not reason.strip():
        raise RuntimeError("LLM query rewriter returned an invalid reason")
    return RewriteDecision(
        rewritten_query=rewritten_query.strip(),
        used_conversation_context=used_conversation_context,
        reason=reason.strip(),
    )
