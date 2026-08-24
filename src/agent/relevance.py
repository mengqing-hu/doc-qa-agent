"""Grade retrieved passages for relevance to the current query."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from src.generation.llm import LLM


RelevanceLabel = Literal["relevant", "irrelevant"]
CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
RELEVANCE_GRADER_PROMPT = """You are the passage relevance grader for a document question-answering system.

For every retrieved passage, decide whether it contains information that is
directly useful for answering the question. Mark a passage irrelevant when it
only shares superficial vocabulary or discusses a different entity, task, or
experiment. Do not judge whether the passages are sufficient for a complete
answer; only judge passage-level relevance.

Return only a JSON object with this exact schema:
{{
  "passages": [
    {{
      "chunk_id": "chunk_id",
      "relevance": "relevant | irrelevant",
      "reason": "A concise explanation in English."
    }}
  ],
  "reason": "A concise explanation of the overall relevance decision."
}}

Question:
{question}

Retrieved passages:
{chunks}
"""


@dataclass(frozen=True)
class PassageRelevance:
    """Capture one passage-level relevance decision."""

    chunk_id: str
    relevance: RelevanceLabel
    reason: str


@dataclass(frozen=True)
class RelevanceDecision:
    """Capture the relevance decisions for one retrieved candidate set."""

    passages: tuple[PassageRelevance, ...]
    reason: str

    @property
    def relevant_chunk_ids(self) -> tuple[str, ...]:
        """Return chunk IDs selected for grounded answer generation."""
        return tuple(
            passage.chunk_id
            for passage in self.passages
            if passage.relevance == "relevant"
        )

    @property
    def status(self) -> Literal["relevant", "none"]:
        """Return whether at least one retrieved passage is relevant."""
        return "relevant" if self.relevant_chunk_ids else "none"


class LLMRelevanceGrader:
    """Select retrieved passages that are relevant to a user question."""

    def __init__(self, llm: LLM) -> None:
        """Store the configured LLM used to grade retrieved passages."""
        self.llm = llm

    def grade(
        self,
        question: str,
        chunks: Sequence[Mapping[str, Any]],
    ) -> RelevanceDecision:
        """Return one validated relevance decision for every retrieved chunk."""
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")
        chunk_payload = [
            {
                "chunk_id": str(chunk.get("chunk_id", "")),
                "text": str(chunk.get("text", "")),
            }
            for chunk in chunks
        ]
        if not chunk_payload:
            return RelevanceDecision(passages=(), reason="No retrieved passages were available.")

        prompt = RELEVANCE_GRADER_PROMPT.format(
            question=json.dumps(normalized_question, ensure_ascii=False),
            chunks=json.dumps(chunk_payload, ensure_ascii=False),
        )
        available_chunk_ids = {
            item["chunk_id"] for item in chunk_payload if item["chunk_id"]
        }
        return _parse_relevance_decision(
            self.llm.generate(prompt),
            available_chunk_ids=available_chunk_ids,
        )


def _parse_relevance_decision(
    response: str,
    *,
    available_chunk_ids: set[str],
) -> RelevanceDecision:
    """Validate the JSON-only response returned by the relevance grader."""
    response_text = response.strip()
    code_fence_match = CODE_FENCE_PATTERN.fullmatch(response_text)
    if code_fence_match:
        response_text = code_fence_match.group(1).strip()

    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise RuntimeError("Relevance grader did not return valid JSON") from error
    if not isinstance(payload, Mapping):
        raise RuntimeError("Relevance grader response must be a JSON object")

    raw_passages = payload.get("passages")
    reason = payload.get("reason")
    if not isinstance(raw_passages, list):
        raise RuntimeError("Relevance grader returned invalid passages")
    if not isinstance(reason, str) or not reason.strip():
        raise RuntimeError("Relevance grader returned an invalid reason")

    passages: list[PassageRelevance] = []
    seen_chunk_ids: set[str] = set()
    for raw_passage in raw_passages:
        if not isinstance(raw_passage, Mapping):
            raise RuntimeError("Relevance grader returned an invalid passage")
        chunk_id = raw_passage.get("chunk_id")
        relevance = raw_passage.get("relevance")
        passage_reason = raw_passage.get("reason")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise RuntimeError("Relevance grader returned an invalid chunk ID")
        normalized_chunk_id = chunk_id.strip()
        if normalized_chunk_id not in available_chunk_ids:
            raise RuntimeError("Relevance grader returned a chunk ID that was not retrieved")
        if normalized_chunk_id in seen_chunk_ids:
            raise RuntimeError("Relevance grader returned a duplicate chunk ID")
        if relevance not in {"relevant", "irrelevant"}:
            raise RuntimeError("Relevance grader returned an invalid relevance label")
        if not isinstance(passage_reason, str) or not passage_reason.strip():
            raise RuntimeError("Relevance grader returned an invalid passage reason")
        seen_chunk_ids.add(normalized_chunk_id)
        passages.append(
            PassageRelevance(
                chunk_id=normalized_chunk_id,
                relevance=relevance,
                reason=passage_reason.strip(),
            )
        )

    if seen_chunk_ids != available_chunk_ids:
        raise RuntimeError("Relevance grader did not grade every retrieved chunk")
    return RelevanceDecision(passages=tuple(passages), reason=reason.strip())
