"""Verify that supported answers are useful for the user's request."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from src.generation.llm import LLM


UtilityStatus = Literal["useful", "not_useful"]
CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
UTILITY_VERIFIER_PROMPT = """You are the utility verifier for a document question-answering system.

Evaluate whether the supported answer directly and completely addresses the
user's request. Check every explicit part of the question, preserve important
qualifications, and avoid treating unrelated detail as an answer. Do not add
facts and do not reject an answer merely because it is concise. The claims have
already passed document support verification.

Return only a JSON object with this exact schema:
{{
  "status": "useful | not_useful",
  "missing_requirements": ["A requested part that the answer does not cover."],
  "reason": "A concise explanation in English."
}}

Question:
{question}

Supported answer:
{answer}

Supported claims:
{claims}
"""


@dataclass(frozen=True)
class UtilityDecision:
    """Capture whether a supported answer is useful for the request."""

    status: UtilityStatus
    missing_requirements: tuple[str, ...]
    reason: str


class LLMUtilityVerifier:
    """Verify that a supported answer addresses the user's request."""

    def __init__(self, llm: LLM) -> None:
        """Store the configured LLM used to verify answer utility."""
        self.llm = llm

    def verify(
        self,
        question: str,
        answer: str,
        claims: Sequence[Mapping[str, Any]],
    ) -> UtilityDecision:
        """Return a validated utility decision for one supported answer."""
        normalized_question = question.strip()
        normalized_answer = answer.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")
        if not normalized_answer:
            raise ValueError("answer must not be empty")
        claim_payload = [dict(claim) for claim in claims]
        if not claim_payload:
            return UtilityDecision(
                status="not_useful",
                missing_requirements=("A supported answer claim is required.",),
                reason="No supported claims were available for utility verification.",
            )

        prompt = UTILITY_VERIFIER_PROMPT.format(
            question=json.dumps(normalized_question, ensure_ascii=False),
            answer=json.dumps(normalized_answer, ensure_ascii=False),
            claims=json.dumps(claim_payload, ensure_ascii=False),
        )
        return _parse_utility_decision(self.llm.generate(prompt))


def _parse_utility_decision(response: str) -> UtilityDecision:
    """Validate the JSON-only response returned by the utility verifier."""
    response_text = response.strip()
    code_fence_match = CODE_FENCE_PATTERN.fullmatch(response_text)
    if code_fence_match:
        response_text = code_fence_match.group(1).strip()

    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise RuntimeError("Utility verifier did not return valid JSON") from error
    if not isinstance(payload, Mapping):
        raise RuntimeError("Utility verifier response must be a JSON object")

    status = payload.get("status")
    missing_requirements = payload.get("missing_requirements")
    reason = payload.get("reason")
    if status not in {"useful", "not_useful"}:
        raise RuntimeError("Utility verifier returned an invalid status")
    if not isinstance(missing_requirements, list) or not all(
        isinstance(item, str) and item.strip() for item in missing_requirements
    ):
        raise RuntimeError("Utility verifier returned invalid missing_requirements")
    if not isinstance(reason, str) or not reason.strip():
        raise RuntimeError("Utility verifier returned an invalid reason")

    normalized_missing_requirements = tuple(
        dict.fromkeys(item.strip() for item in missing_requirements)
    )
    if status == "useful" and normalized_missing_requirements:
        raise RuntimeError("Useful answers cannot have missing requirements")
    return UtilityDecision(
        status=status,
        missing_requirements=normalized_missing_requirements,
        reason=reason.strip(),
    )
