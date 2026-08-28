"""Flattened Agentic RAG graph: plan -> gather -> grade (loop) -> synthesize -> verify.

The reasoning loop only decides *what to retrieve next*. Answer generation and
support verification each happen exactly once, after retrieval is complete, so a
single failed LLM check can no longer abstain a whole turn mid-loop.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

try:  # get_stream_writer is a no-op outside a streaming run; guard the import too
    from langgraph.config import get_stream_writer
except ImportError:  # pragma: no cover - older langgraph
    get_stream_writer = None  # type: ignore[assignment]

from src.agent.chitchat import LLMChitchatResponder
from src.agent.context import ContextManager
from src.agent.context_window import build_context_window
from src.agent.planner import LLMRetrievalPlanner, PlannedQuery, RetrievalPlan
from src.agent.relevance import LLMRelevanceGrader
from src.agent.routes import LLMRetrievalGate, RetrievalAction
from src.agent.state import (
    AgentState,
    ConversationMessage,
    Evidence,
    PlannedQueryState,
    ResponseState,
)
from src.agent.summarizer import ConversationSummarizer
from src.agent.support import LLMSupportVerifier
from src.agent.tools.base import Tool
from src.generation.rag_pipeline import RAGPipeline
from src.generation.response import RAGResponse, SourceReference
from src.store.messages import ConversationStore


logger = logging.getLogger(__name__)
# Starting point for the reranker's [0, 1]-scale relevance score. Evidence that
# already scored above this from the cross-encoder skips a redundant LLM grade.
# Not empirically calibrated - tune against real observed scores.
RELEVANCE_SCORE_THRESHOLD = 0.7
SYNTHESIS_MAX_CONTEXT_CHARACTERS = 24_000
# A compound multi-part question needs room for a full answer; 512 (the LLM
# default) and even 1200 truncate these mid-sentence.
SYNTHESIS_MAX_TOKENS = 2048
MAX_SYNTHESIS_ATTEMPTS = 2
# Cap on chunks carried into synthesis so a multi-round turn cannot overflow
# the generation context (roughly one 1000-char chunk each).
MAX_ACCUMULATED_EVIDENCE = 20
STRICT_GROUNDING_SUFFIX = (
    " Answer strictly and only from the provided context. Omit any fact that "
    "is not stated there; do not refuse solely because an exact name differs."
)
# Conversation-window defaults (overridable per invoke_agent_graph call).
DEFAULT_MAX_CONTEXT_TOKENS = 3000
DEFAULT_RESUMMARIZE_AFTER = 12
DEFAULT_KEEP_RECENT_VERBATIM = 6


def conversation_settings(config: Any) -> dict[str, int]:
    """Read the conversation-window knobs from config, with defaults."""
    return {
        "max_context_tokens": int(
            config.get(
                "agent", "context", "max_context_tokens",
                default=DEFAULT_MAX_CONTEXT_TOKENS,
            )
        ),
        "resummarize_after": int(
            config.get(
                "agent", "context", "resummarize_after",
                default=DEFAULT_RESUMMARIZE_AFTER,
            )
        ),
        "keep_recent_verbatim": int(
            config.get(
                "agent", "context", "keep_recent_verbatim",
                default=DEFAULT_KEEP_RECENT_VERBATIM,
            )
        ),
    }


def build_agent_graph(
    pipeline: RAGPipeline,
    *,
    retrieval_gate: LLMRetrievalGate,
    context_manager: ContextManager,
    chitchat_responder: LLMChitchatResponder,
    retrieval_planner: LLMRetrievalPlanner,
    relevance_grader: LLMRelevanceGrader,
    support_verifier: LLMSupportVerifier,
    vector_retrieve_tool: Tool,
    web_search_tool: Tool,
    checkpointer: BaseCheckpointSaver | None = None,
) -> Any:
    """Compile the flattened retrieval workflow."""
    builder = StateGraph(AgentState)
    builder.add_node("context_manager", _context_manager_node(context_manager))
    builder.add_node("retrieval_gate", _retrieval_gate_node(retrieval_gate))
    builder.add_node("chitchat", _chitchat_node(chitchat_responder))
    builder.add_node("plan_retrieval", _plan_retrieval_node(retrieval_planner))
    builder.add_node(
        "gather_evidence", _gather_evidence_node(vector_retrieve_tool, web_search_tool)
    )
    builder.add_node("grade_evidence", _grade_evidence_node(relevance_grader))
    builder.add_node("synthesize", _synthesize_node(pipeline))
    builder.add_node("verify_synthesis", _verify_synthesis_node(support_verifier))
    builder.add_node("trim_answer", _trim_answer_node(support_verifier))
    builder.add_node("abstain", _abstain_node)

    builder.add_edge(START, "context_manager")
    builder.add_edge("context_manager", "retrieval_gate")
    builder.add_conditional_edges(
        "retrieval_gate",
        _select_retrieval_action,
        {"retrieve": "plan_retrieval", "chitchat": "chitchat", "abstain": "abstain"},
    )
    builder.add_edge("chitchat", END)
    builder.add_edge("plan_retrieval", "gather_evidence")
    builder.add_edge("gather_evidence", "grade_evidence")
    builder.add_conditional_edges(
        "grade_evidence",
        _decide_continue,
        {
            "continue": "plan_retrieval",
            "synthesize": "synthesize",
            "abstain": "abstain",
        },
    )
    builder.add_conditional_edges(
        "synthesize",
        _route_after_synthesize,
        {"verify": "verify_synthesis", "abstain": "abstain"},
    )
    builder.add_conditional_edges(
        "verify_synthesis",
        _route_after_verify,
        {
            "done": END,
            "trim": "trim_answer",
            "retry": "synthesize",
            "abstain": "abstain",
        },
    )
    builder.add_edge("trim_answer", END)
    builder.add_edge("abstain", END)
    return builder.compile(checkpointer=checkpointer)


def _graph_input(
    normalized_question: str,
    store: ConversationStore,
    thread_id: str,
    max_context_tokens: int,
) -> dict[str, Any]:
    """Assemble one turn's graph input from the transcript store."""
    thread_summary = store.summary(thread_id)
    window = build_context_window(
        store.history(thread_id),
        thread_summary.text,
        thread_summary.upto_seq,
        max_tokens=max_context_tokens,
    )
    return {
        "question": normalized_question,
        "conversation_context": window.messages,
        "conversation_summary": window.summary,
    }


def _graph_config(thread_id: str) -> dict[str, Any]:
    """Return the LangGraph run config for one turn."""
    return {
        "run_name": "agentic_rag",
        "tags": ["agentic-rag", "flattened-retrieval"],
        "metadata": {"thread_id": thread_id},
        "configurable": {"thread_id": thread_id},
    }


def invoke_agent_graph(
    graph: Any,
    question: str,
    *,
    thread_id: str,
    store: ConversationStore,
    summarizer: ConversationSummarizer | None = None,
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    resummarize_after: int = DEFAULT_RESUMMARIZE_AFTER,
    keep_recent_verbatim: int = DEFAULT_KEEP_RECENT_VERBATIM,
) -> RAGResponse:
    """Run the graph for one question, persisting the turn to the transcript store."""
    normalized_question = _validate_turn_inputs(question, thread_id)

    result = graph.invoke(
        _graph_input(normalized_question, store, thread_id, max_context_tokens),
        config=_graph_config(thread_id),
    )
    response = _response_from_state(_require_response_state(result))
    store.append_turn(thread_id, normalized_question, response.answer)
    _maybe_resummarize(
        store, summarizer, thread_id, resummarize_after, keep_recent_verbatim
    )
    return response


def stream_agent_graph(
    graph: Any,
    question: str,
    *,
    thread_id: str,
    store: ConversationStore,
    summarizer: ConversationSummarizer | None = None,
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    resummarize_after: int = DEFAULT_RESUMMARIZE_AFTER,
    keep_recent_verbatim: int = DEFAULT_KEEP_RECENT_VERBATIM,
) -> Iterator[Any]:
    """Yield synthesis tokens as they are produced, then a final ``RAGResponse``.

    Each yielded ``str`` is an answer chunk; the final yielded value is the
    completed ``RAGResponse``. Non-synthesis paths (chitchat, abstain) yield no
    strings, only the final response.
    """
    normalized_question = _validate_turn_inputs(question, thread_id)

    final_state: dict[str, Any] = {}
    for mode, payload in graph.stream(
        _graph_input(normalized_question, store, thread_id, max_context_tokens),
        config=_graph_config(thread_id),
        stream_mode=["custom", "values"],
    ):
        if mode == "custom" and isinstance(payload, Mapping):
            token = payload.get("synthesis_token")
            if token:
                yield token
        elif mode == "values" and isinstance(payload, Mapping):
            final_state = dict(payload)

    response = _response_from_state(_require_response_state(final_state))
    store.append_turn(thread_id, normalized_question, response.answer)
    _maybe_resummarize(
        store, summarizer, thread_id, resummarize_after, keep_recent_verbatim
    )
    yield response


def _validate_turn_inputs(question: str, thread_id: str) -> str:
    """Validate and normalize the per-turn inputs shared by both entry points."""
    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("question must not be empty")
    if not thread_id.strip():
        raise ValueError("thread_id must not be empty")
    return normalized_question


def _require_response_state(result: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the graph's response state or raise if the graph produced none."""
    response_state = result.get("response")
    if not isinstance(response_state, Mapping):
        raise RuntimeError("Agent graph did not return a RAG response")
    return response_state


def _maybe_resummarize(
    store: ConversationStore,
    summarizer: ConversationSummarizer | None,
    thread_id: str,
    resummarize_after: int,
    keep_recent_verbatim: int,
) -> None:
    """Fold older uncovered turns into the rolling summary when there are enough."""
    if summarizer is None:
        return
    thread_summary = store.summary(thread_id)
    uncovered = [
        message
        for message in store.history(thread_id)
        if message.seq > thread_summary.upto_seq
    ]
    to_summarize = uncovered[: max(0, len(uncovered) - keep_recent_verbatim)]
    if len(to_summarize) < resummarize_after:
        return
    try:
        new_summary = summarizer.summarize(thread_summary.text, to_summarize)
    except Exception:
        logger.exception("conversation summarization failed; keeping the old summary")
        return
    store.update_summary(thread_id, new_summary, to_summarize[-1].seq)


# --------------------------------------------------------------------------- #
# Gate + chitchat (unchanged behaviour)
# --------------------------------------------------------------------------- #
def _context_manager_node(context_manager: ContextManager):
    """Create the node that prepares bounded history and resets retrieval state."""

    def run_context_manager(state: AgentState) -> dict[str, Any]:
        return context_manager.prepare_query(state)

    return run_context_manager


def _retrieval_gate_node(retrieval_gate: LLMRetrievalGate):
    """Create the node that decides whether to retrieve, chat, or abstain."""

    def run_retrieval_gate(state: AgentState) -> dict[str, Any]:
        try:
            decision = retrieval_gate.decide(
                state["original_query"],
                _user_messages(state.get("conversation_context", [])),
                summary=state.get("conversation_summary"),
            )
        except Exception:
            logger.exception("retrieval_gate failed; abstaining")
            return {
                "retrieval_action": "abstain",
                "retrieval_confidence": 0.0,
                "retrieval_reason": "retrieval_gate failed",
                "error": "retrieval_gate failed",
            }
        return {
            "retrieval_action": decision.action,
            "retrieval_confidence": decision.confidence,
            "retrieval_reason": decision.reason,
        }

    return run_retrieval_gate


def _user_messages(
    conversation_context: list[ConversationMessage],
) -> list[ConversationMessage]:
    """Keep assistant responses out of routing context."""
    return [message for message in conversation_context if message["role"] == "user"]


def _select_retrieval_action(state: AgentState) -> RetrievalAction:
    """Return the validated retrieval action used by conditional edges."""
    action = state.get("retrieval_action")
    if action in {"retrieve", "chitchat", "abstain"}:
        return action
    raise RuntimeError("Retrieval gate did not return a supported action")


def _chitchat_node(chitchat_responder: LLMChitchatResponder):
    """Create the node that answers requests that need no document evidence."""

    def run_chitchat(state: AgentState) -> dict[str, ResponseState]:
        try:
            reply = chitchat_responder.respond(
                state["original_query"],
                state.get("conversation_context", []),
                summary=state.get("conversation_summary"),
            )
        except Exception:
            logger.exception("chitchat failed; returning fallback reply")
            reply = "Sorry, I ran into a problem answering that - please try again."
        return {"response": {"answer": reply, "sources": []}}

    return run_chitchat


# --------------------------------------------------------------------------- #
# Retrieval loop: plan -> gather -> grade
# --------------------------------------------------------------------------- #
def _plan_retrieval_node(planner: LLMRetrievalPlanner):
    """Create the node that plans this round's retrieval queries."""

    def run_plan_retrieval(state: AgentState) -> dict[str, Any]:
        round_number = int(state.get("retrieval_rounds", 0)) + 1
        try:
            plan = planner.plan(
                state["original_query"],
                state.get("accumulated_evidence", []),
                round_number,
            )
        except Exception:
            logger.exception("plan_retrieval failed on round %d", round_number)
            if round_number == 1:
                plan = RetrievalPlan(
                    queries=(
                        PlannedQuery(state["original_query"], "vector_retrieve"),
                    ),
                    done=False,
                    reason="planner failed; running an initial document search",
                )
            else:
                plan = RetrievalPlan(
                    queries=(),
                    done=True,
                    reason="planner failed; synthesizing with evidence gathered so far",
                )
        else:
            plan = _drop_repeated_queries(plan, state.get("retrieval_history", []))
        return {
            "retrieval_plan": {
                "queries": [_planned_query_to_state(query) for query in plan.queries],
                "done": plan.done,
                "reason": plan.reason,
            },
            "retrieval_rounds": round_number,
        }

    return run_plan_retrieval


REPEATED_QUERY_SIMILARITY = 0.5


def _content_words(text: str) -> set[str]:
    """Return the lowercase word set of a query, ignoring common filler words."""
    stop = {
        "what", "which", "how", "why", "is", "are", "was", "were", "the", "a",
        "an", "of", "for", "in", "on", "to", "and", "or", "did", "does", "do",
        "this", "that", "these", "study", "report", "pdf", "document", "used",
        "use", "chosen", "selected",
    }
    return {word for word in re.findall(r"[a-z0-9]+", text.casefold())} - stop


def _drop_repeated_queries(
    plan: RetrievalPlan, history: Sequence[Mapping[str, Any]]
) -> RetrievalPlan:
    """Remove planned queries that just reword a query an earlier round already ran.

    Small models loop by re-asking the same sub-question every round. If every
    query in the plan is a near-duplicate of an earlier one, treat retrieval as
    complete rather than spending another round on it.
    """
    prior = [
        _content_words(str(query.get("query", "")))
        for entry in history
        for query in entry.get("queries", [])
    ]
    prior = [words for words in prior if words]
    if not prior:
        return plan

    def is_repeat(query_words: set[str]) -> bool:
        if not query_words:
            return False
        return any(
            len(query_words & earlier) / len(query_words | earlier)
            >= REPEATED_QUERY_SIMILARITY
            for earlier in prior
        )

    fresh = [
        query for query in plan.queries if not is_repeat(_content_words(query.query))
    ]
    if len(fresh) == len(plan.queries):
        return plan
    if not fresh:
        return RetrievalPlan(
            queries=(),
            done=True,
            reason="planner only repeated earlier queries; retrieval treated as complete",
        )
    return RetrievalPlan(queries=tuple(fresh), done=plan.done, reason=plan.reason)


def _gather_evidence_node(vector_retrieve_tool: Tool, web_search_tool: Tool):
    """Create the node that executes every query in the current round's plan."""
    tools: dict[str, Tool] = {
        "vector_retrieve": vector_retrieve_tool,
        "web_search": web_search_tool,
    }

    def run_gather_evidence(state: AgentState) -> dict[str, Any]:
        plan = state.get("retrieval_plan", {"queries": []})
        seen_chunk_ids = {
            item["chunk_id"] for item in state.get("accumulated_evidence", [])
        }
        gathered: list[Evidence] = []
        gather_errors: list[str] = []
        for query in plan["queries"]:
            tool = tools.get(query["tool"])
            if tool is None:
                continue
            try:
                results = tool.run(
                    query["query"],
                    metadata_filter=_state_filter_to_tool(query.get("metadata_filter")),
                )
            except Exception as tool_error:  # noqa: BLE001 - one query must not kill the round
                logger.exception(
                    "tool %s failed for query %r", query["tool"], query["query"]
                )
                gather_errors.append(f"{query['tool']}: {tool_error}")
                results = []
            for item in results:
                if item["chunk_id"] not in seen_chunk_ids:
                    seen_chunk_ids.add(item["chunk_id"])
                    gathered.append(item)
        if gather_errors and not gathered:
            logger.warning(
                "gather_evidence round produced no evidence; all %d tool call(s) failed",
                len(gather_errors),
            )
        return {
            "ungraded_evidence": gathered,
            "gather_errors": gather_errors,
        }

    return run_gather_evidence


def _grade_evidence_node(relevance_grader: LLMRelevanceGrader):
    """Create the node that keeps only this round's relevant new evidence."""

    def run_grade_evidence(state: AgentState) -> dict[str, Any]:
        ungraded = state.get("ungraded_evidence", [])
        plan = state.get("retrieval_plan", {"queries": []})
        round_number = int(state.get("retrieval_rounds", 0))

        if not ungraded:
            relevant: list[Evidence] = []
        elif _all_scored_above_threshold(ungraded):
            relevant = list(ungraded)
        else:
            try:
                decision = relevance_grader.grade(state["original_query"], ungraded)
            except Exception:
                logger.exception(
                    "grade_evidence raised; treating this round's evidence as relevant"
                )
                relevant = list(ungraded)
            else:
                if not decision.passages:
                    # The grader gave up (invalid output after a repair attempt);
                    # an empty passage list is not a real "all irrelevant" verdict.
                    logger.warning(
                        "grade_evidence returned no graded passages; keeping all "
                        "%d chunk(s) this round",
                        len(ungraded),
                    )
                    relevant = list(ungraded)
                else:
                    relevant_ids = set(decision.relevant_chunk_ids)
                    relevant = [
                        item for item in ungraded if item["chunk_id"] in relevant_ids
                    ]

        accumulated = _cap_evidence(
            [*state.get("accumulated_evidence", []), *relevant]
        )
        kept_ids = {item["chunk_id"] for item in accumulated}
        history_entry = {
            "round": round_number,
            "queries": plan["queries"],
            "gathered_count": len(ungraded),
            "added_chunk_ids": [
                item["chunk_id"] for item in relevant if item["chunk_id"] in kept_ids
            ],
            "gather_errors": list(state.get("gather_errors", [])),
        }
        return {
            "accumulated_evidence": accumulated,
            "retrieval_history": [*state.get("retrieval_history", []), history_entry],
            "last_round_added_relevant": bool(relevant),
            "ungraded_evidence": [],
            "gather_errors": [],
        }

    return run_grade_evidence


def _cap_evidence(evidence: Sequence[Evidence]) -> list[Evidence]:
    """Keep the highest-scored evidence so synthesis context cannot balloon.

    Web results carry no comparable score; they are always kept. RAG chunks are
    kept in descending reranker score up to the cap.
    """
    web = [item for item in evidence if item.get("score") is None]
    rag = [item for item in evidence if item.get("score") is not None]
    rag.sort(key=lambda item: item["score"], reverse=True)
    keep_rag = max(0, MAX_ACCUMULATED_EVIDENCE - len(web))
    kept = web + rag[:keep_rag]
    # Preserve the original arrival order for a stable, readable trace.
    kept_ids = {item["chunk_id"] for item in kept}
    return [item for item in evidence if item["chunk_id"] in kept_ids]


def _all_scored_above_threshold(ungraded: Sequence[Evidence]) -> bool:
    """Skip a redundant LLM relevance grade when every chunk already scored high."""
    return all(
        item.get("score") is not None and item["score"] >= RELEVANCE_SCORE_THRESHOLD
        for item in ungraded
    )


def _decide_continue(state: AgentState) -> str:
    """Route to another retrieval round, to synthesis, or to abstention."""
    plan = state.get("retrieval_plan", {"done": True})
    rounds = int(state.get("retrieval_rounds", 0))
    max_rounds = int(state.get("max_retrieval_rounds", 0))
    has_evidence = bool(state.get("accumulated_evidence"))
    made_progress = bool(state.get("last_round_added_relevant"))

    stop = plan.get("done") or rounds >= max_rounds or not made_progress
    if stop:
        return "synthesize" if has_evidence else "abstain"
    return "continue"


# --------------------------------------------------------------------------- #
# Synthesis + verification (each runs once)
# --------------------------------------------------------------------------- #
def _synthesize_node(pipeline: RAGPipeline):
    """Create the node that generates one grounded answer from all evidence."""

    def run_synthesize(state: AgentState) -> dict[str, Any]:
        evidence = state.get("accumulated_evidence", [])
        attempts = int(state.get("synthesis_attempts", 0))
        if not evidence:
            return {"error": "synthesize reached with no evidence"}

        query = state["original_query"]
        if attempts >= 1:
            query = f"{query}{STRICT_GROUNDING_SUFFIX}"
        chunks = [_evidence_to_chunk(item) for item in evidence]
        writer = _stream_writer()
        try:
            if writer is not None:
                answer, sources, truncated = _stream_synthesis(
                    pipeline, query, chunks, writer
                )
            else:
                response = pipeline.generate(
                    query,
                    chunks,
                    max_context_characters=SYNTHESIS_MAX_CONTEXT_CHARACTERS,
                    max_tokens=SYNTHESIS_MAX_TOKENS,
                    group_fairly_by_source=True,
                    max_continuations=1,
                )
                answer = response.answer
                sources = [_source_to_state(s) for s in response.sources]
                truncated = response.truncated
        except Exception:
            logger.exception("synthesize failed")
            return {"error": "synthesize failed", "synthesis_attempts": attempts + 1}
        return {
            "response": {"answer": answer, "sources": sources},
            "synthesis_attempts": attempts + 1,
            "synthesis_truncated": truncated,
            "error": None,
        }

    return run_synthesize


def _stream_writer() -> Any:
    """Return the LangGraph custom-stream writer, or None outside a streaming run."""
    if get_stream_writer is None:
        return None
    try:
        return get_stream_writer()
    except Exception:  # noqa: BLE001 - not in a streaming context
        return None


def _stream_synthesis(
    pipeline: RAGPipeline,
    query: str,
    chunks: list[dict[str, Any]],
    writer: Any,
) -> tuple[str, list[dict[str, Any]], bool]:
    """Stream the drafted answer token by token through the writer."""
    parts: list[str] = []
    for token in pipeline.generate_stream(
        query,
        chunks,
        max_context_characters=SYNTHESIS_MAX_CONTEXT_CHARACTERS,
        max_tokens=SYNTHESIS_MAX_TOKENS,
        group_fairly_by_source=True,
    ):
        parts.append(token)
        try:
            writer({"synthesis_token": token})
        except Exception:  # noqa: BLE001 - never let a UI writer failure abort synthesis
            pass
    answer = "".join(parts).strip()
    sources = [
        _source_to_state(_source_from_chunk(chunk)) for chunk in chunks
    ]
    return answer, sources, False


def _source_from_chunk(chunk: Mapping[str, Any]) -> SourceReference:
    """Build a source reference from a synthesis evidence chunk."""
    metadata = chunk.get("metadata") or {}
    return SourceReference(
        chunk_id=str(chunk.get("chunk_id", "unknown")),
        source=str(metadata.get("source", "unknown source")),
        page=metadata.get("page"),
        section_title=metadata.get("section_title"),
    )


def _route_after_synthesize(state: AgentState) -> str:
    """Verify a drafted answer, or abstain when synthesis produced nothing."""
    if state.get("error") or not isinstance(state.get("response"), Mapping):
        return "abstain"
    return "verify"


def _verify_synthesis_node(support_verifier: LLMSupportVerifier):
    """Create the node that checks the drafted answer against all the evidence."""

    def run_verify_synthesis(state: AgentState) -> dict[str, Any]:
        draft = state["response"]["answer"]
        evidence = state.get("accumulated_evidence", [])
        try:
            decision = support_verifier.verify(
                state["original_query"],
                draft,
                [
                    {"chunk_id": item["chunk_id"], "text": item["text"]}
                    for item in evidence
                ],
                state.get("conversation_context", []),
            )
        except Exception:
            logger.exception("verify_synthesis failed; keeping the drafted answer")
            return {
                "support_status": "error",
                "support_reason": "support verification failed",
            }
        return {
            "support_status": decision.status,
            "support_reason": decision.reason,
            "support_claims": [
                {
                    "claim": claim.claim,
                    "support": claim.support,
                    "chunk_ids": list(claim.chunk_ids),
                    "reason": claim.reason,
                }
                for claim in decision.claims
            ],
        }

    return run_verify_synthesis


def _route_after_verify(state: AgentState) -> str:
    """Finalize, trim to supported claims, retry synthesis, or abstain."""
    status = state.get("support_status")
    if status in {"supported", "error"}:
        return "done"
    if status == "partially_supported":
        return "trim"
    # unsupported
    if int(state.get("synthesis_attempts", 0)) < MAX_SYNTHESIS_ATTEMPTS:
        return "retry"
    return "abstain"


def _trim_answer_node(support_verifier: LLMSupportVerifier):
    """Create the node that rewrites a partly supported answer to supported facts."""

    def run_trim_answer(state: AgentState) -> dict[str, ResponseState]:
        draft = state["response"]["answer"]
        claims = state.get("support_claims", [])
        try:
            trimmed = support_verifier.trim_to_supported_claims(
                state["original_query"], draft, claims
            )
        except Exception:
            logger.exception("trim_answer failed; keeping the drafted answer")
            trimmed = draft
        return {
            "response": {"answer": trimmed, "sources": state["response"]["sources"]}
        }

    return run_trim_answer


def _abstain_node(state: AgentState) -> dict[str, ResponseState]:
    """Return a grounded refusal, distinguishing 'nothing found' from 'unverifiable'."""
    if state.get("error"):
        answer = (
            "Sorry, I ran into a problem while answering this question. "
            "Please try again in a moment."
        )
    elif state.get("retrieval_action") == "abstain":
        answer = "I can only answer questions about the indexed documents."
    elif state.get("accumulated_evidence"):
        answer = (
            "I found related information in the documents but could not produce "
            "an answer I can fully support from it."
        )
    else:
        answer = (
            "The retrieved documents and web search did not contain evidence "
            "that supports an answer to this question."
        )
    return {"response": {"answer": answer, "sources": []}}


# --------------------------------------------------------------------------- #
# Conversions
# --------------------------------------------------------------------------- #
def _planned_query_to_state(query: PlannedQuery) -> PlannedQueryState:
    """Convert a planner query into a checkpoint-compatible dict."""
    metadata_filter = query.metadata_filter
    serialized: dict[str, str | list[str]] | None = None
    if metadata_filter:
        serialized = {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in metadata_filter.items()
        }
    return {
        "query": query.query,
        "tool": query.tool,
        "metadata_filter": serialized,
    }


def _state_filter_to_tool(
    metadata_filter: Mapping[str, Any] | None,
) -> dict[str, str | tuple[str, ...]] | None:
    """Convert a checkpointed metadata_filter back into the tool's expected shape."""
    if not metadata_filter:
        return None
    return {
        key: tuple(value) if isinstance(value, list) else value
        for key, value in metadata_filter.items()
    }


def _evidence_to_chunk(evidence: Evidence) -> dict[str, Any]:
    """Adapt one Evidence record into the chunk dict the RAG pipeline expects."""
    return {
        "chunk_id": evidence["chunk_id"],
        "text": evidence["text"],
        "metadata": dict(evidence.get("metadata") or {}),
        "rerank_score": evidence.get("score"),
    }


def _source_to_state(source: SourceReference) -> dict[str, Any]:
    """Convert a public source reference into checkpoint-compatible state."""
    return {
        "chunk_id": source.chunk_id,
        "source": source.source,
        "page": source.page,
        "section_title": source.section_title,
    }


def _response_from_state(response_state: Mapping[str, Any]) -> RAGResponse:
    """Convert checkpoint-compatible graph state into a public response."""
    answer = response_state.get("answer")
    sources = response_state.get("sources")
    if not isinstance(answer, str) or not isinstance(sources, list):
        raise RuntimeError("Agent graph returned an invalid response state")

    try:
        source_references = tuple(
            SourceReference(
                chunk_id=str(source["chunk_id"]),
                source=str(source["source"]),
                page=source.get("page"),
                section_title=source.get("section_title"),
            )
            for source in sources
            if isinstance(source, Mapping)
        )
    except KeyError as error:
        raise RuntimeError("Agent graph returned an invalid source state") from error
    if len(source_references) != len(sources):
        raise RuntimeError("Agent graph returned an invalid source state")

    return RAGResponse(answer=answer, sources=source_references)
