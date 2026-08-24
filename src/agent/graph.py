"""Build the initial LangGraph wrapper around the existing RAG pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from src.agent.context import ContextManager
from src.agent.relevance import LLMRelevanceGrader
from src.agent.rewrite import LLMQueryRewriter
from src.agent.routes import LLMRetrievalGate, RetrievalAction
from src.agent.state import AgentState, ConversationMessage, ResponseState
from src.agent.support import LLMSupportVerifier
from src.agent.utility import LLMUtilityVerifier
from src.generation.rag_pipeline import RAGPipeline
from src.generation.response import RAGResponse, SourceReference


def build_agent_graph(
    pipeline: RAGPipeline,
    *,
    retrieval_gate: LLMRetrievalGate,
    context_manager: ContextManager,
    query_rewriter: LLMQueryRewriter,
    relevance_grader: LLMRelevanceGrader,
    support_verifier: LLMSupportVerifier,
    utility_verifier: LLMUtilityVerifier,
    checkpointer: BaseCheckpointSaver | None = None,
) -> Any:
    """Compile the complete Self-RAG reflection workflow."""
    builder = StateGraph(AgentState)
    builder.add_node("retrieval_gate", _retrieval_gate_node(retrieval_gate))
    builder.add_node("context_manager", _context_manager_node(context_manager))
    builder.add_node("query_rewriter", _query_rewriter_node(query_rewriter))
    builder.add_node("retrieve", _retrieve_node(pipeline))
    builder.add_node("grade_relevance", _grade_relevance_node(relevance_grader))
    builder.add_node("generate", _generate_node(pipeline))
    builder.add_node("verify_support", _verify_support_node(support_verifier))
    builder.add_node("verify_utility", _verify_utility_node(utility_verifier))
    builder.add_node("abstain", _abstain_node)
    builder.add_node("persist_turn", _persist_turn_node(context_manager))
    builder.add_edge(START, "context_manager")
    builder.add_edge("context_manager", "retrieval_gate")
    builder.add_conditional_edges(
        "retrieval_gate",
        _select_retrieval_action,
        {
            "retrieve": "query_rewriter",
            "abstain": "abstain",
        },
    )
    builder.add_edge("query_rewriter", "retrieve")
    builder.add_edge("retrieve", "grade_relevance")
    builder.add_conditional_edges(
        "grade_relevance",
        _select_relevance_action,
        {
            "generate": "generate",
            "abstain": "abstain",
        },
    )
    builder.add_edge("generate", "verify_support")
    builder.add_conditional_edges(
        "verify_support",
        _select_support_action,
        {
            "verify": "verify_utility",
            "abstain": "abstain",
        },
    )
    builder.add_conditional_edges(
        "verify_utility",
        _select_utility_action,
        {
            "persist": "persist_turn",
            "abstain": "abstain",
        },
    )
    builder.add_edge("abstain", "persist_turn")
    builder.add_edge("persist_turn", END)
    return builder.compile(checkpointer=checkpointer)


def invoke_agent_graph(
    graph: Any,
    question: str,
    *,
    thread_id: str,
) -> RAGResponse:
    """Run the graph for one question and return its grounded response."""
    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("question must not be empty")
    if not thread_id.strip():
        raise ValueError("thread_id must not be empty")

    result = graph.invoke(
        {"question": normalized_question},
        config={
            "run_name": "agentic_rag",
            "tags": ["agentic-rag"],
            "metadata": {"thread_id": thread_id},
            "configurable": {"thread_id": thread_id},
        },
    )
    response_state = result.get("response")
    if not isinstance(response_state, Mapping):
        raise RuntimeError("Agent graph did not return a RAG response")
    return _response_from_state(response_state)


def _retrieval_gate_node(retrieval_gate: LLMRetrievalGate):
    """Create the node that records the selected retrieval action."""
    def run_retrieval_gate(state: AgentState) -> dict[str, str | float]:
        decision = retrieval_gate.decide(
            state["original_query"],
            _user_messages(state.get("conversation_context", [])),
        )
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
    if action in {"retrieve", "abstain"}:
        return action
    raise RuntimeError("Retrieval gate did not return a supported action")


def _context_manager_node(context_manager: ContextManager):
    """Create the node that prepares bounded history for routing and rewriting."""
    def run_context_manager(state: AgentState) -> dict[str, Any]:
        return context_manager.prepare_query(state)

    return run_context_manager


def _query_rewriter_node(query_rewriter: LLMQueryRewriter):
    """Create the node that resolves references in document retrieval queries."""
    def run_query_rewriter(state: AgentState) -> dict[str, str | bool]:
        decision = query_rewriter.rewrite(
            state["original_query"],
            state.get("conversation_context", []),
        )
        return {
            "rewritten_query": decision.rewritten_query,
            "rewrite_used_conversation_context": decision.used_conversation_context,
            "rewrite_reason": decision.reason,
        }

    return run_query_rewriter


def _retrieve_node(pipeline: RAGPipeline):
    """Create the node that retrieves and reranks one evidence candidate set."""
    def run_retrieve(state: AgentState) -> dict[str, Any]:
        retrieval_attempts = int(state.get("retrieval_attempts", 0)) + 1
        return {
            "retrieved_chunks": pipeline.retrieve(state["rewritten_query"]),
            "retrieval_attempts": retrieval_attempts,
        }

    return run_retrieve


def _grade_relevance_node(relevance_grader: LLMRelevanceGrader):
    """Create the node that filters retrieved passages using the Rel reflection."""
    def run_grade_relevance(state: AgentState) -> dict[str, Any]:
        retrieved_chunks = state.get("retrieved_chunks", [])
        decision = relevance_grader.grade(state["rewritten_query"], retrieved_chunks)
        relevant_chunk_ids = set(decision.relevant_chunk_ids)
        relevance_decisions = [
            {
                "chunk_id": passage.chunk_id,
                "relevance": passage.relevance,
                "reason": passage.reason,
            }
            for passage in decision.passages
        ]
        return {
            "relevant_chunks": [
                chunk
                for chunk in retrieved_chunks
                if str(chunk.get("chunk_id")) in relevant_chunk_ids
            ],
            "relevant_chunk_ids": list(decision.relevant_chunk_ids),
            "relevance_decisions": relevance_decisions,
            "relevance_status": decision.status,
            "relevance_reason": decision.reason,
        }

    return run_grade_relevance


def _select_relevance_action(state: AgentState) -> str:
    """Select generation only when at least one passage is relevant."""
    relevance_status = state.get("relevance_status")
    if relevance_status == "relevant":
        return "generate"
    if relevance_status == "none":
        return "abstain"
    raise RuntimeError("Relevance grader did not return a supported status")


def _verify_support_node(support_verifier: LLMSupportVerifier):
    """Create the node that verifies generated claims against relevant chunks."""
    def run_verify_support(state: AgentState) -> dict[str, Any]:
        response = state.get("response")
        if not isinstance(response, Mapping) or not isinstance(response.get("answer"), str):
            raise RuntimeError("Support verification requires a generated answer")
        decision = support_verifier.verify(
            state["rewritten_query"],
            response["answer"],
            state.get("relevant_chunks", []),
        )
        return {
            "support_status": decision.status,
            "support_claims": [
                {
                    "claim": claim.claim,
                    "support": claim.support,
                    "chunk_ids": list(claim.chunk_ids),
                    "reason": claim.reason,
                }
                for claim in decision.claims
            ],
            "support_reason": decision.reason,
        }

    return run_verify_support


def _select_support_action(state: AgentState) -> str:
    """Verify utility only after all material claims are supported."""
    support_status = state.get("support_status")
    if support_status == "supported":
        return "verify"
    if support_status in {"partially_supported", "unsupported"}:
        return "abstain"
    raise RuntimeError("Support verifier did not return a supported status")


def _verify_utility_node(utility_verifier: LLMUtilityVerifier):
    """Create the node that verifies whether a supported answer is useful."""
    def run_verify_utility(state: AgentState) -> dict[str, Any]:
        response = state.get("response")
        if not isinstance(response, Mapping) or not isinstance(response.get("answer"), str):
            raise RuntimeError("Utility verification requires a generated answer")
        decision = utility_verifier.verify(
            state["original_query"],
            response["answer"],
            state.get("support_claims", []),
        )
        return {
            "utility_status": decision.status,
            "utility_missing_requirements": list(decision.missing_requirements),
            "utility_reason": decision.reason,
        }

    return run_verify_utility


def _select_utility_action(state: AgentState) -> str:
    """Persist only supported answers that directly address the request."""
    utility_status = state.get("utility_status")
    if utility_status == "useful":
        return "persist"
    if utility_status == "not_useful":
        return "abstain"
    raise RuntimeError("Utility verifier did not return a supported status")


def _persist_turn_node(context_manager: ContextManager):
    """Create the node that persists the final user and assistant messages."""
    def run_persist_turn(state: AgentState) -> dict[str, Any]:
        return context_manager.persist_response(state)

    return run_persist_turn


def _abstain_node(state: AgentState) -> dict[str, ResponseState]:
    """Return a clear response when a reflection rejects the current answer."""
    if state.get("utility_status") == "not_useful":
        answer = "The generated answer does not fully address the question."
    elif state.get("support_status") in {"partially_supported", "unsupported"}:
        answer = "The generated answer could not be fully supported by the retrieved documents."
    elif state.get("retrieval_action") == "retrieve":
        answer = "The retrieved documents do not contain a passage relevant to this question."
    else:
        answer = "I can only answer questions about the indexed documents."
    return {
        "response": {
            "answer": answer,
            "sources": [],
        }
    }


def _generate_node(pipeline: RAGPipeline):
    """Create the node that generates an answer from relevant chunks only."""
    def run_generate(state: AgentState) -> dict[str, ResponseState]:
        return {
            "response": _response_to_state(
                pipeline.generate(
                    state["rewritten_query"],
                    state.get("relevant_chunks", []),
                )
            )
        }

    return run_generate


def _response_to_state(response: RAGResponse) -> ResponseState:
    """Convert a public response into checkpoint-compatible graph state."""
    return {
        "answer": response.answer,
        "sources": [
            {
                "chunk_id": source.chunk_id,
                "source": source.source,
                "page": source.page,
                "section_title": source.section_title,
            }
            for source in response.sources
        ],
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
