"""Build the initial LangGraph wrapper around the existing RAG pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from src.agent.state import AgentState, ResponseState, SourceState
from src.generation.rag_pipeline import RAGPipeline
from src.generation.response import RAGResponse, SourceReference


def build_single_hop_rag_graph(
    pipeline: RAGPipeline,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
) -> Any:
    """Compile a graph that preserves the existing single-hop RAG behavior."""
    builder = StateGraph(AgentState)
    builder.add_node("current_rag", _current_rag_node(pipeline))
    builder.add_edge(START, "current_rag")
    builder.add_edge("current_rag", END)
    return builder.compile(checkpointer=checkpointer)


def invoke_single_hop_rag(
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
            "run_name": "single_hop_rag",
            "tags": ["agentic-rag", "single-hop"],
            "metadata": {"thread_id": thread_id},
            "configurable": {"thread_id": thread_id},
        },
    )
    response_state = result.get("response")
    if not isinstance(response_state, Mapping):
        raise RuntimeError("Agent graph did not return a RAG response")
    return _response_from_state(response_state)


def _current_rag_node(pipeline: RAGPipeline):
    """Create the graph node that delegates to the existing RAG pipeline."""
    def run_current_rag(state: AgentState) -> dict[str, ResponseState]:
        return {"response": _response_to_state(pipeline.answer(state["question"]))}

    return run_current_rag


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
