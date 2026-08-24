"""Decide whether the Agentic RAG workflow should retrieve evidence."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from src.agent.state import ConversationMessage
from src.generation.llm import LLM


RetrievalAction = Literal["retrieve", "abstain"]
SUPPORTED_RETRIEVAL_ACTIONS = frozenset({"retrieve", "abstain"})
CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
RETRIEVAL_GATE_PROMPT = """You are the retrieval gate for a document question-answering application.

Choose exactly one retrieval action:
- retrieve: a factual or explanatory request that may plausibly benefit from
  retrieval over the indexed documents.
- abstain: a request that is clearly unrelated to document QA, including
  social conversation, real-time external information, external actions,
  software creation, and image generation. Also choose abstain when a
  pronoun or omitted entity cannot be resolved from the current request and
  conversation history.

Decide whether to attempt retrieval, not whether the documents will definitely
contain the answer. Do not abstain merely because the documents may lack the
answer. When an information request could plausibly benefit from retrieval,
choose retrieve.

Classify the current request first. Use history only to resolve references or
omitted entities. Do not let prior conversation change the scope of a
standalone new topic.

Return only a JSON object with this exact schema:
{{
  "action": "retrieve | abstain",
  "confidence": 0.0,
  "reason": "A concise explanation in English."
}}

The history contains only prior user requests. It is not document evidence.

Conversation history:
{history}

User request:
{question}
"""


@dataclass(frozen=True)
class RetrievalDecision:
    """Capture an LLM-selected retrieval action and its decision metadata."""

    action: RetrievalAction
    confidence: float
    reason: str


class LLMRetrievalGate:
    """Decide whether a user request should trigger document retrieval."""

    def __init__(self, llm: LLM) -> None:
        """Store the configured LLM used to gate each user request."""
        self.llm = llm

    def decide(
        self,
        question: str,
        conversation_context: Sequence[ConversationMessage] = (),
    ) -> RetrievalDecision:
        """Return a retrieval action using bounded history to resolve references."""
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")
        prompt = RETRIEVAL_GATE_PROMPT.format(
            history=json.dumps(list(conversation_context), ensure_ascii=False),
            question=json.dumps(normalized_question, ensure_ascii=False),
        )
        return _parse_retrieval_decision(self.llm.generate(prompt))


def _parse_retrieval_decision(response: str) -> RetrievalDecision:
    """Validate the JSON-only response required from the retrieval gate LLM."""
    response_text = response.strip()
    code_fence_match = CODE_FENCE_PATTERN.fullmatch(response_text)
    if code_fence_match:
        response_text = code_fence_match.group(1).strip()

    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise RuntimeError("LLM retrieval gate did not return valid JSON") from error
    if not isinstance(payload, Mapping):
        raise RuntimeError("LLM retrieval gate response must be a JSON object")

    action = payload.get("action")
    confidence = payload.get("confidence")
    reason = payload.get("reason")
    if action not in SUPPORTED_RETRIEVAL_ACTIONS:
        raise RuntimeError("LLM retrieval gate returned an invalid action")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise RuntimeError("LLM retrieval gate response contains an invalid confidence")
    normalized_confidence = float(confidence)
    if not 0.0 <= normalized_confidence <= 1.0:
        raise RuntimeError("LLM retrieval gate confidence must be between zero and one")
    if not isinstance(reason, str) or not reason.strip():
        raise RuntimeError("LLM retrieval gate response contains an invalid reason")

    return RetrievalDecision(
        action=action,
        confidence=normalized_confidence,
        reason=reason.strip(),
    )
