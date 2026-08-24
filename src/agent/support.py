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
DEFAULT_SUPPORT_MAX_TOKENS = 1024
CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
SUPPORT_VERIFIER_PROMPT = """You are the support verifier for a document question-answering system.

Evaluate every material factual claim in the answer against the supplied
relevant passages. Use only those passages as evidence; do not use general
knowledge. A claim is supported only when the passages entail it. Mark a claim
unsupported when it is absent, more specific than the passages, or presents
one value as definitive despite conflicting evidence. If passages disagree,
an answer that explicitly attributes each value to its source and reports the
discrepancy is supported; do not mark an attributed value unsupported merely
because another passage contains a different value. Do not judge writing
quality or whether the answer is useful. Treat a name variant (for example,
a model, method, or dataset name with an added qualifier such as a depth,
version, or size suffix) as referring to the same entity only when the
supplied passages establish that relationship — for example, one passage
describes the base architecture and another passage names the specific
configured variant. Keep a parameter value and its stated rationale as
separate claims so that an explicitly stated rationale can be supported
even when the entity name appears only in a neighboring passage.

Entity-resolution context below is conversation context, not document
evidence. Use it only to resolve a model alias or omitted reference; use the
relevant passages for all factual support decisions.

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

Keep claims and reasons concise. Do not repeat the passages or the full answer.
The status must match the claim labels: use "supported" when every claim is
supported, "unsupported" when none is supported, and "partially_supported"
only when both labels are present.

Question:
{question}

Generated answer:
{answer}

Relevant passages:
{chunks}

Entity-resolution context:
{conversation_context}
"""
SUPPORT_TRIM_PROMPT = """You are editing a document question-answering system's draft answer.

Some claims in the draft answer below were not supported by the retrieved
passages and must be removed. Rewrite the answer using only the supported
facts listed below. Do not add new facts, do not hedge about what was
removed, and do not mention the verification process. Keep the answer
concise and directly address the question using only the supported facts.

Question:
{question}

Draft answer:
{answer}

Supported facts:
{supported_facts}

Return only the rewritten answer text, with no preamble or JSON wrapper.
"""
SUPPORT_JSON_REPAIR_PROMPT = """Convert the following support-verifier response into a valid JSON object.

Return only compact JSON using exactly this schema:
{{
  "status": "supported | partially_supported | unsupported",
  "claims": [
    {{
      "claim": "One material factual claim.",
      "support": "supported | unsupported",
      "chunk_ids": ["chunk_id"],
      "reason": "A concise reason."
    }}
  ],
  "reason": "A concise overall reason."
}}

Do not add commentary. Preserve only claims and chunk IDs present in the response.

Response to repair:
{response}
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

    def __init__(self, llm: LLM, *, max_tokens: int = DEFAULT_SUPPORT_MAX_TOKENS) -> None:
        """Store the configured LLM used to verify generated claims."""
        self.llm = llm
        if max_tokens <= 0:
            raise ValueError("support max_tokens must be greater than zero")
        self.max_tokens = max_tokens

    def verify(
        self,
        question: str,
        answer: str,
        chunks: Sequence[Mapping[str, Any]],
        conversation_context: Sequence[Mapping[str, Any]] = (),
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
            conversation_context=json.dumps(
                list(conversation_context), ensure_ascii=False
            ),
        )
        available_chunk_ids = {
            item["chunk_id"] for item in chunk_payload if item["chunk_id"]
        }
        raw_response = self._generate(prompt)
        try:
            return _parse_support_decision(
                raw_response,
                available_chunk_ids=available_chunk_ids,
            )
        except RuntimeError as error:
            if str(error) != "Support verifier did not return valid JSON":
                raise
            repaired_response = self._generate(
                SUPPORT_JSON_REPAIR_PROMPT.format(response=raw_response)
            )
            return _parse_support_decision(
                repaired_response,
                available_chunk_ids=available_chunk_ids,
            )

    def trim_to_supported_claims(
        self,
        question: str,
        answer: str,
        claims: Sequence[Mapping[str, Any]],
    ) -> str:
        """Rewrite an answer to include only claims marked as supported."""
        normalized_question = question.strip()
        normalized_answer = answer.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")
        if not normalized_answer:
            raise ValueError("answer must not be empty")
        supported_facts = [
            str(claim.get("claim", ""))
            for claim in claims
            if claim.get("support") == "supported"
        ]
        if not supported_facts:
            raise ValueError("claims must include at least one supported claim")

        prompt = SUPPORT_TRIM_PROMPT.format(
            question=json.dumps(normalized_question, ensure_ascii=False),
            answer=json.dumps(normalized_answer, ensure_ascii=False),
            supported_facts=json.dumps(supported_facts, ensure_ascii=False),
        )
        trimmed = self._generate(prompt).strip()
        if not trimmed:
            raise RuntimeError("Support verifier returned an empty trimmed answer")
        return trimmed

    def _generate(self, prompt: str) -> str:
        """Generate verifier output with compatibility for simple test doubles."""
        try:
            return self.llm.generate(prompt, max_tokens=self.max_tokens)
        except TypeError as error:
            if "max_tokens" not in str(error):
                raise
            return self.llm.generate(prompt)


def _parse_support_decision(
    response: str,
    *,
    available_chunk_ids: set[str],
) -> SupportDecision:
    """Validate the JSON-only response returned by the support verifier."""
    response_text = _extract_json_text(response)
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
    if supported_count == 0:
        normalized_status: SupportStatus = "unsupported"
    elif supported_count == len(claims):
        normalized_status = "supported"
    else:
        normalized_status = "partially_supported"

    return SupportDecision(
        status=normalized_status,
        claims=tuple(claims),
        reason=reason.strip(),
    )


def _extract_json_text(response: str) -> str:
    """Extract a likely JSON object from common LLM wrapper text."""
    response_text = response.strip()
    response_text = THINK_BLOCK_PATTERN.sub("", response_text).strip()
    code_fence_match = CODE_FENCE_PATTERN.fullmatch(response_text)
    if code_fence_match:
        return code_fence_match.group(1).strip()

    try:
        json.loads(response_text)
    except json.JSONDecodeError:
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start >= 0 and end > start:
            return response_text[start : end + 1].strip()
    return response_text
