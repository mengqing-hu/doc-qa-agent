"""Build the ReAct + Self-RAG graph around the existing RAG pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from src.agent.actions import LLMActionSelector
from src.agent.chitchat import LLMChitchatResponder
from src.agent.context import ContextManager
from src.agent.relevance import LLMRelevanceGrader
from src.agent.routes import LLMRetrievalGate, RetrievalAction
from src.agent.state import AgentState, ConversationMessage, ScratchpadEntry, ResponseState
from src.agent.support import LLMSupportVerifier
from src.agent.tools.base import Tool
from src.generation.rag_pipeline import RAGPipeline
from src.generation.response import RAGResponse, SourceReference


OTHER_ACTION: dict[str, str] = {
    "vector_retrieve": "web_search",
    "web_search": "vector_retrieve",
}
MAX_TOOL_ATTEMPTS_PER_ITERATION = 2


def build_agent_graph(
    pipeline: RAGPipeline,
    *,
    retrieval_gate: LLMRetrievalGate,
    context_manager: ContextManager,
    chitchat_responder: LLMChitchatResponder,
    action_selector: LLMActionSelector,
    relevance_grader: LLMRelevanceGrader,
    support_verifier: LLMSupportVerifier,
    vector_retrieve_tool: Tool,
    web_search_tool: Tool,
    checkpointer: BaseCheckpointSaver | None = None,
) -> Any:
    """Compile the ReAct + Self-RAG reasoning workflow."""
    builder = StateGraph(AgentState)
    builder.add_node("context_manager", _context_manager_node(context_manager))
    builder.add_node("retrieval_gate", _retrieval_gate_node(retrieval_gate))
    builder.add_node("chitchat", _chitchat_node(chitchat_responder))
    builder.add_node("select_action", _select_action_node(action_selector))
    builder.add_node(
        "run_action", _run_action_node(vector_retrieve_tool, web_search_tool)
    )
    builder.add_node("grade_relevance", _grade_relevance_node(relevance_grader))
    builder.add_node("generate_answer", _generate_answer_node(pipeline))
    builder.add_node("verify_answer", _verify_answer_node(support_verifier))
    builder.add_node("record_evidence", _record_evidence_node)
    builder.add_node("finish_answer", _finish_answer_node)
    builder.add_node("abstain", _abstain_node)
    builder.add_node("persist_turn", _persist_turn_node(context_manager))

    builder.add_edge(START, "context_manager")
    builder.add_edge("context_manager", "retrieval_gate")
    builder.add_conditional_edges(
        "retrieval_gate",
        _select_retrieval_action,
        {"retrieve": "select_action", "chitchat": "chitchat", "abstain": "abstain"},
    )
    builder.add_edge("chitchat", "persist_turn")
    builder.add_conditional_edges(
        "select_action",
        _select_after_action,
        {"finish": "finish_answer", "retrieve": "run_action"},
    )
    builder.add_edge("run_action", "grade_relevance")
    builder.add_conditional_edges(
        "grade_relevance",
        _select_after_relevance,
        {"generate": "generate_answer", "retry": "run_action", "abstain": "abstain"},
    )
    builder.add_edge("generate_answer", "verify_answer")
    builder.add_conditional_edges(
        "verify_answer",
        _select_after_support,
        {"record": "record_evidence", "retry": "run_action", "abstain": "abstain"},
    )
    builder.add_conditional_edges(
        "record_evidence",
        _select_after_record,
        {"continue": "select_action", "abstain": "abstain"},
    )
    builder.add_edge("finish_answer", "persist_turn")
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
            "tags": ["agentic-rag", "react"],
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
    if action in {"retrieve", "chitchat", "abstain"}:
        return action
    raise RuntimeError("Retrieval gate did not return a supported action")


def _chitchat_node(chitchat_responder: LLMChitchatResponder):
    """Create the node that answers requests that need no document evidence."""

    def run_chitchat(state: AgentState) -> dict[str, ResponseState]:
        reply = chitchat_responder.respond(
            state["original_query"], state.get("conversation_context", [])
        )
        return {"response": {"answer": reply, "sources": []}}

    return run_chitchat


def _context_manager_node(context_manager: ContextManager):
    """Create the node that prepares bounded history and resets loop state."""

    def run_context_manager(state: AgentState) -> dict[str, Any]:
        return context_manager.prepare_query(state)

    return run_context_manager


def _select_action_node(action_selector: LLMActionSelector):
    """Create the node that picks the next Thought+Action pair."""

    def run_select_action(state: AgentState) -> dict[str, Any]:
        decision = action_selector.select(
            state["original_query"], state.get("scratchpad", [])
        )
        return {
            "action": decision.action,
            "action_input": decision.action_input,
            "action_thought": decision.thought,
            "tool_attempts": 0,
        }

    return run_select_action


def _select_after_action(state: AgentState) -> str:
    """Route to finishing the trajectory or running the chosen retrieval tool."""
    action = state.get("action")
    if action == "finish":
        return "finish"
    if action in OTHER_ACTION:
        return "retrieve"
    raise RuntimeError("Action selector did not return a supported action")


def _run_action_node(vector_retrieve_tool: Tool, web_search_tool: Tool):
    """Create the node that executes the selected (or retried) retrieval tool."""
    tools: dict[str, Tool] = {
        "vector_retrieve": vector_retrieve_tool,
        "web_search": web_search_tool,
    }

    def run_action(state: AgentState) -> dict[str, Any]:
        attempts = int(state.get("tool_attempts", 0))
        chosen_action = (
            state["action"] if attempts == 0 else OTHER_ACTION[state["action"]]
        )
        try:
            evidence = tools[chosen_action].run(state["action_input"])
        except Exception:
            evidence = []
        return {
            "current_action": chosen_action,
            "current_evidence": evidence,
            "tool_attempts": attempts + 1,
        }

    return run_action


def _grade_relevance_node(relevance_grader: LLMRelevanceGrader):
    """Create the node that filters this iteration's evidence for relevance."""

    def run_grade_relevance(state: AgentState) -> dict[str, Any]:
        evidence = state.get("current_evidence", [])
        decision = relevance_grader.grade(state["action_input"], evidence)
        relevant_ids = set(decision.relevant_chunk_ids)
        relevant_evidence = [
            item for item in evidence if item["chunk_id"] in relevant_ids
        ]
        return {
            "relevant_evidence": relevant_evidence,
            "relevance_status": decision.status,
            "relevance_reason": decision.reason,
        }

    return run_grade_relevance


def _select_after_relevance(state: AgentState) -> str:
    """Route to generation, a same-iteration retry, or abstention."""
    if state.get("relevance_status") == "relevant":
        return "generate"
    if int(state.get("tool_attempts", 0)) >= MAX_TOOL_ATTEMPTS_PER_ITERATION:
        return "abstain"
    return "retry"


def _generate_answer_node(pipeline: RAGPipeline):
    """Create the node that extracts one fact from this iteration's evidence."""

    def run_generate_answer(state: AgentState) -> dict[str, Any]:
        response = pipeline.generate(
            state["action_input"], state.get("relevant_evidence", [])
        )
        return {
            "current_fact": response.answer,
            "current_sources": [
                {
                    "chunk_id": source.chunk_id,
                    "source": source.source,
                    "page": source.page,
                    "section_title": source.section_title,
                }
                for source in response.sources
            ],
        }

    return run_generate_answer


def _verify_answer_node(support_verifier: LLMSupportVerifier):
    """Create the node that checks this iteration's fact against its evidence."""

    def run_verify_answer(state: AgentState) -> dict[str, Any]:
        decision = support_verifier.verify(
            state["action_input"],
            state["current_fact"],
            state.get("relevant_evidence", []),
        )
        return {
            "support_status": decision.status,
            "support_reason": decision.reason,
        }

    return run_verify_answer


def _select_after_support(state: AgentState) -> str:
    """Route to recording the evidence, a same-iteration retry, or abstention."""
    if state.get("support_status") == "supported":
        return "record"
    if int(state.get("tool_attempts", 0)) >= MAX_TOOL_ATTEMPTS_PER_ITERATION:
        return "abstain"
    return "retry"


def _record_evidence_node(state: AgentState) -> dict[str, Any]:
    """Append this iteration's verified fact to the scratchpad."""
    entry: ScratchpadEntry = {
        "thought": state.get("action_thought", ""),
        "action": state["current_action"],
        "action_input": state["action_input"],
        "fact": state["current_fact"],
        "sources": state.get("current_sources", []),
    }
    return {
        "scratchpad": [*state.get("scratchpad", []), entry],
        "iteration_count": int(state.get("iteration_count", 0)) + 1,
    }


def _select_after_record(state: AgentState) -> str:
    """Enforce the hard iteration cap before returning to action selection."""
    if int(state.get("iteration_count", 0)) >= int(state.get("max_iterations", 0)):
        return "abstain"
    return "continue"


def _finish_answer_node(state: AgentState) -> dict[str, ResponseState]:
    """Build the final response from the model's chosen finish action."""
    seen_chunk_ids: set[str] = set()
    sources = []
    for entry in state.get("scratchpad", []):
        for source in entry.get("sources", []):
            if source["chunk_id"] not in seen_chunk_ids:
                seen_chunk_ids.add(source["chunk_id"])
                sources.append(source)
    return {"response": {"answer": state["action_input"], "sources": sources}}


def _abstain_node(state: AgentState) -> dict[str, ResponseState]:
    """Return a grounded refusal when the gate, tools, or budget close off an answer."""
    if state.get("retrieval_action") == "abstain":
        answer = "I can only answer questions about the indexed documents."
    elif int(state.get("iteration_count", 0)) >= int(state.get("max_iterations", 0)):
        answer = "This question could not be answered within the allowed number of reasoning steps."
    else:
        answer = (
            "The retrieved documents and web search did not contain evidence "
            "that supports an answer to this question."
        )
    return {"response": {"answer": answer, "sources": []}}


def _persist_turn_node(context_manager: ContextManager):
    """Create the node that persists the final user and assistant messages."""

    def run_persist_turn(state: AgentState) -> dict[str, Any]:
        return context_manager.persist_response(state)

    return run_persist_turn


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
