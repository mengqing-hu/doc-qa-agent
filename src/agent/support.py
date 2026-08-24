"""Verify that generated answer claims are supported by relevant passages."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from src.generation.llm import LLM


ClaimSupport = Literal["supported", "unsupported"]
SupportStatus = Literal["supported", "partially_supported", "unsupported"]
CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
SUPPORT_VERIFIER_PROMPT = """You are the support verifier for a document question-answering system.

Evaluate every material factual claim in the answer against the supplied
relevant passages. Use only those passages as evidence; do not use general
knowledge. A claim is supported only when the passages entail it. Mark a claim
unsupported when it is absent, contradicted, or more specific than the
passages. Do not judge writing quality or whether the answer is useful.

Return only a JSON object with this exact schema:
{{
  "status": "supported | partially_supported | unsupported",
  "claims": [
    {{
      "claim": "One material factual claim from the answer.",
      "support": "supported | unsupported",
      "chunk_ids": ["chunk_id"],
      "reason": "A concise explanation in English."
    }}
  ],
  "reason": "A concise explanation of the overall support decision."
}}

Question:
{question}

Generated answer:
{answer}

Relevant passages:
{chunks}
"""


@dataclass(frozen=True)
class ClaimSupportDecision:
    """Capture support for one material answer claim."""

    claim: str
    support: ClaimSupport
    chunk_ids: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class SupportDecision:
    """Capture the overall support decision for one generated answer."""

    status: SupportStatus
    claims: tuple[ClaimSupportDecision, ...]
    reason: str


class LLMSupportVerifier:
    """Verify generated claims against the relevant retrieved passages."""

    def __init__(self, llm: LLM) -> None:
        """Store the configured LLM used to verify generated claims."""
        self.llm = llm

    def verify(
        self,
        question: str,
        answer: str,
        chunks: Sequence[Mapping[str, Any]],
    ) -> SupportDecision:
        """Return a validated support decision for a generated answer."""
        normalized_question = question.strip()
        normalized_answer = answer.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")
        if not normalized_answer:
            raise ValueError("answer must not be empty")
        chunk_payload = [
            {
                "chunk_id": str(chunk.get("chunk_id", "")),
                "text": str(chunk.get("text", "")),
            }
            for chunk in chunks
        ]
        if not chunk_payload:
            return SupportDecision(
                status="unsupported",
                claims=(),
                reason="No relevant passages were available to support the answer.",
            )

        prompt = SUPPORT_VERIFIER_PROMPT.format(
            question=json.dumps(normalized_question, ensure_ascii=False),
            answer=json.dumps(normalized_answer, ensure_ascii=False),
            chunks=json.dumps(chunk_payload, ensure_ascii=False),
        )
        available_chunk_ids = {
            item["chunk_id"] for item in chunk_payload if item["chunk_id"]
        }
        return _parse_support_decision(
            self.llm.generate(prompt),
            available_chunk_ids=available_chunk_ids,
        )


def _parse_support_decision(
    response: str,
    *,
    available_chunk_ids: set[str],
) -> SupportDecision:
    """Validate the JSON-only response returned by the support verifier."""
    response_text = response.strip()
    code_fence_match = CODE_FENCE_PATTERN.fullmatch(response_text)
    if code_fence_match:
        response_text = code_fence_match.group(1).strip()

    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise RuntimeError("Support verifier did not return valid JSON") from error
    if not isinstance(payload, Mapping):
        raise RuntimeError("Support verifier response must be a JSON object")

    raw_status = payload.get("status")
    raw_claims = payload.get("claims")
    reason = payload.get("reason")
    if raw_status not in {"supported", "partially_supported", "unsupported"}:
        raise RuntimeError("Support verifier returned an invalid status")
    if not isinstance(raw_claims, list) or not raw_claims:
        raise RuntimeError("Support verifier returned invalid claims")
    if not isinstance(reason, str) or not reason.strip():
        raise RuntimeError("Support verifier returned an invalid reason")

    claims: list[ClaimSupportDecision] = []
    for raw_claim in raw_claims:
        if not isinstance(raw_claim, Mapping):
            raise RuntimeError("Support verifier returned an invalid claim")
        claim = raw_claim.get("claim")
        support = raw_claim.get("support")
        raw_chunk_ids = raw_claim.get("chunk_ids")
        claim_reason = raw_claim.get("reason")
        if not isinstance(claim, str) or not claim.strip():
            raise RuntimeError("Support verifier returned an invalid claim text")
        if support not in {"supported", "unsupported"}:
            raise RuntimeError("Support verifier returned an invalid claim support label")
        if not isinstance(raw_chunk_ids, list) or not all(
            isinstance(chunk_id, str) and chunk_id.strip() for chunk_id in raw_chunk_ids
        ):
            raise RuntimeError("Support verifier returned invalid claim chunk IDs")
        chunk_ids = tuple(dict.fromkeys(chunk_id.strip() for chunk_id in raw_chunk_ids))
        if any(chunk_id not in available_chunk_ids for chunk_id in chunk_ids):
            raise RuntimeError("Support verifier returned a chunk ID that was not relevant")
        if support == "supported" and not chunk_ids:
            raise RuntimeError("Supported claims must include at least one chunk ID")
        if not isinstance(claim_reason, str) or not claim_reason.strip():
            raise RuntimeError("Support verifier returned an invalid claim reason")
        claims.append(
            ClaimSupportDecision(
                claim=claim.strip(),
                support=support,
                chunk_ids=chunk_ids,
                reason=claim_reason.strip(),
            )
        )

    supported_count = sum(claim.support == "supported" for claim in claims)
    if raw_status == "supported" and supported_count != len(claims):
        raise RuntimeError("Supported status requires every claim to be supported")
    if raw_status == "unsupported" and supported_count != 0:
        raise RuntimeError("Unsupported status cannot contain supported claims")
    if raw_status == "partially_supported" and not (
        0 < supported_count < len(claims)
    ):
        raise RuntimeError(
            "Partially supported status requires both supported and unsupported claims"
        )

    return SupportDecision(
        status=raw_status,
        claims=tuple(claims),
        reason=reason.strip(),
    )
