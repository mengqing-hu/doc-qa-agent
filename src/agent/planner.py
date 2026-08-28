"""Plan one round of retrieval given the question and evidence gathered so far.

The planner replaces the per-iteration ReAct action selector. It never writes
the answer; it only decides which queries (and tools) to run next so that the
accumulated evidence can eventually answer the whole question. Generation and
support verification happen once, after retrieval is complete.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from src.agent.state import Evidence
from src.generation.llm import LLM


logger = logging.getLogger(__name__)
ToolName = Literal["vector_retrieve", "web_search"]
SUPPORTED_TOOLS = frozenset({"vector_retrieve", "web_search"})
CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
MAX_QUERIES_PER_ROUND = 3
EVIDENCE_SNIPPET_CHARACTERS = 220

PLANNER_PROMPT = """You are the retrieval planner for a document question-answering agent.

The private indexed documents are exactly: {indexed_documents}. Resolve any
reference such as "the pdf", "the report", or "this document" to one of those.

Each round you decide what to search for next so that the accumulated evidence
below can fully answer the original question. You do NOT write the answer.

Tools:
- vector_retrieve: search the private indexed documents. Use this by default.
- web_search: search the public web. Use it only for information that is
  clearly external to the private documents (publication years, who proposed a
  method, hardware specifications, results on public benchmarks, and similar).

Rules:
- Round 1: break the question into every INDEPENDENT sub-question you can and
  emit one query for each now. At least one query must use vector_retrieve.
- Round 2+: set "done" to true and return NO queries as soon as the accumulated
  evidence already touches every part of the question - even if some detail is
  thin. Only add a query for a part that has NO evidence at all yet, especially
  a fact that depends on something you just learned (for example, once you know
  the GPU model, search for its specifications). NEVER re-issue a query whose
  answer is already visible in the accumulated evidence, and never reword an
  earlier query.
- Every query must be self-contained: resolve all pronouns using the
  accumulated evidence. Write it as a focused natural-language question about
  ONE fact. Do not append a filename, do not name a specific document, and do
  not ask for a whole document or section.
- When "done" is false, "queries" must be non-empty. Emit at most {max_queries}
  queries per round. Prefer finishing over adding marginal queries.

Return ONLY a JSON object with this exact schema:
{{
  "queries": [
    {{"query": "...", "tool": "vector_retrieve | web_search"}}
  ],
  "done": false,
  "reason": "What is still missing, or why retrieval is complete."
}}

Original question:
{question}

Accumulated evidence so far:
{evidence_summary}
"""


@dataclass(frozen=True)
class PlannedQuery:
    """One retrieval the planner asked for."""

    query: str
    tool: ToolName
    metadata_filter: Mapping[str, str | tuple[str, ...]] | None = None


@dataclass(frozen=True)
class RetrievalPlan:
    """One round's plan: the queries to run and whether retrieval is complete."""

    queries: tuple[PlannedQuery, ...]
    done: bool
    reason: str


class LLMRetrievalPlanner:
    """Decide the next round of retrieval queries for the current question."""

    def __init__(self, llm: LLM, *, document_names: Sequence[str] = ()) -> None:
        """Store the configured LLM and the names of the indexed documents."""
        self.llm = llm
        self.document_names = tuple(document_names)

    def plan(
        self,
        question: str,
        accumulated_evidence: Sequence[Evidence],
        round_number: int,
    ) -> RetrievalPlan:
        """Return a validated retrieval plan for the current round."""
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")
        if round_number < 1:
            raise ValueError("round_number must be at least one")

        indexed_documents = (
            ", ".join(self.document_names) if self.document_names else "none indexed"
        )
        prompt = PLANNER_PROMPT.format(
            indexed_documents=indexed_documents,
            max_queries=MAX_QUERIES_PER_ROUND,
            question=json.dumps(normalized_question, ensure_ascii=False),
            evidence_summary=_summarize_evidence(accumulated_evidence),
        )
        plan = self._plan_with_retry(prompt)
        return _enforce_first_round_rules(plan, round_number, normalized_question)

    def _plan_with_retry(self, prompt: str) -> RetrievalPlan:
        """Parse the plan, retrying once on invalid model output."""
        raw_response = self.llm.generate(prompt)
        try:
            return _parse_plan(raw_response)
        except RuntimeError as error:
            logger.warning(
                "Retrieval planner returned invalid output (%s); retrying once", error
            )
            strict_prompt = f"{prompt}\n\nReturn ONLY the JSON object described above."
            return _parse_plan(self.llm.generate(strict_prompt))


def _summarize_evidence(evidence: Sequence[Evidence]) -> str:
    """Render a compact bullet list of the evidence gathered so far."""
    if not evidence:
        return "(nothing retrieved yet)"
    lines: list[str] = []
    for item in evidence:
        metadata = item.get("metadata") or {}
        source = str(metadata.get("source", item.get("origin", "unknown")))
        snippet = " ".join(str(item.get("text", "")).split())[:EVIDENCE_SNIPPET_CHARACTERS]
        lines.append(f"- [{item.get('chunk_id')}] ({source}) {snippet}")
    return "\n".join(lines)


def _enforce_first_round_rules(
    plan: RetrievalPlan, round_number: int, question: str
) -> RetrievalPlan:
    """Guarantee the first round performs at least one document search."""
    if round_number > 1:
        return plan
    if not plan.queries:
        return RetrievalPlan(
            queries=(PlannedQuery(query=question, tool="vector_retrieve"),),
            done=False,
            reason=(
                "Planner returned no queries on the first round; forcing an "
                "initial document search."
            ),
        )
    if not any(query.tool == "vector_retrieve" for query in plan.queries):
        logger.info("First-round plan had no vector_retrieve query; prepending one")
        return RetrievalPlan(
            queries=(
                PlannedQuery(query=question, tool="vector_retrieve"),
                *plan.queries,
            ),
            done=plan.done,
            reason=plan.reason,
        )
    return plan


def _parse_plan(response: str) -> RetrievalPlan:
    """Validate the JSON-only response returned by the retrieval planner."""
    response_text = response.strip()
    code_fence_match = CODE_FENCE_PATTERN.fullmatch(response_text)
    if code_fence_match:
        response_text = code_fence_match.group(1).strip()
    start = response_text.find("{")
    end = response_text.rfind("}")
    if start != -1 and end > start:
        response_text = response_text[start : end + 1]

    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise RuntimeError("Retrieval planner did not return valid JSON") from error
    if not isinstance(payload, Mapping):
        raise RuntimeError("Retrieval planner response must be a JSON object")

    raw_done = payload.get("done")
    raw_queries = payload.get("queries")
    reason = payload.get("reason")
    if not isinstance(raw_done, bool):
        raise RuntimeError("Retrieval planner returned an invalid done flag")
    if not isinstance(reason, str) or not reason.strip():
        raise RuntimeError("Retrieval planner returned an invalid reason")
    if not isinstance(raw_queries, list):
        raise RuntimeError("Retrieval planner returned invalid queries")

    queries: list[PlannedQuery] = []
    for raw_query in raw_queries[:MAX_QUERIES_PER_ROUND]:
        if not isinstance(raw_query, Mapping):
            raise RuntimeError("Retrieval planner returned an invalid query entry")
        query_text = raw_query.get("query")
        tool = raw_query.get("tool")
        if not isinstance(query_text, str) or not query_text.strip():
            raise RuntimeError("Retrieval planner returned an invalid query text")
        if tool not in SUPPORTED_TOOLS:
            raise RuntimeError("Retrieval planner returned an invalid tool")
        # metadata_filter is intentionally not exposed to the planner: every
        # query goes through ordinary reranked similarity search so one round
        # cannot pull an entire document into the context.
        queries.append(PlannedQuery(query=query_text.strip(), tool=tool))

    if not raw_done and not queries:
        raise RuntimeError(
            "Retrieval planner must return at least one query when not done"
        )

    return RetrievalPlan(queries=tuple(queries), done=raw_done, reason=reason.strip())
