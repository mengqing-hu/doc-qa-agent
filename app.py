"""Streamlit chat UI for the Self-RAG document QA agent."""

from __future__ import annotations

import os
import sqlite3
import uuid

import streamlit as st
from langgraph.checkpoint.sqlite import SqliteSaver

from src.agent.chitchat import LLMChitchatResponder
from src.agent.context import ContextManager
from src.agent.graph import (
    build_agent_graph,
    conversation_settings,
    stream_agent_graph,
)
from src.agent.planner import LLMRetrievalPlanner
from src.agent.relevance import LLMRelevanceGrader
from src.agent.routes import LLMRetrievalGate
from src.agent.summarizer import ConversationSummarizer
from src.agent.support import LLMSupportVerifier
from src.agent.tools.vector_retrieve import VectorRetrieveTool
from src.agent.tools.web_search import WebSearchTool
from src.core.config import PROJECT_ROOT, Config
from src.core.logger import setup_logging
from src.generation.response import RAGResponse
from src.pipeline.query_runtime import build_query_pipeline
from src.store.conversations import create_conversation, list_conversations
from src.store.messages import SqliteConversationStore


DB_PATH = PROJECT_ROOT / "data" / "checkpoints.sqlite"


def _load_secrets_into_environment() -> None:
    """Make Streamlit Secrets visible to the existing os.getenv-based Config."""
    try:
        secrets_items = list(st.secrets.items())
    except Exception:
        return
    for key, value in secrets_items:
        os.environ.setdefault(key, str(value))


def _get_or_create_visitor_id() -> str:
    """Return a stable anonymous id for this visitor, carried in the URL."""
    existing_uid = st.query_params.get("uid")
    if existing_uid:
        return existing_uid
    new_uid = uuid.uuid4().hex
    st.query_params["uid"] = new_uid
    st.rerun()


@st.cache_resource(show_spinner="Loading models and index...")
def _load_runtime():
    """Build the graph, transcript store, and summarizer once per server process."""
    config = Config()
    setup_logging(config)
    pipeline = build_query_pipeline(config)

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    checkpointer = SqliteSaver(connection)

    graph = build_agent_graph(
        pipeline,
        retrieval_gate=LLMRetrievalGate(pipeline.llm),
        context_manager=ContextManager(config),
        chitchat_responder=LLMChitchatResponder(pipeline.llm),
        retrieval_planner=LLMRetrievalPlanner(
            pipeline.llm,
            document_names=pipeline.hybrid_retriever.vector_store.get_indexed_sources(),
        ),
        relevance_grader=LLMRelevanceGrader(pipeline.llm),
        support_verifier=LLMSupportVerifier(pipeline.llm),
        vector_retrieve_tool=VectorRetrieveTool(pipeline),
        web_search_tool=WebSearchTool(config),
        checkpointer=checkpointer,
    )
    store = SqliteConversationStore(DB_PATH)
    summarizer = ConversationSummarizer(pipeline.llm)
    return graph, store, summarizer, conversation_settings(config)


def _render_sidebar(owner_uid: str) -> None:
    """Render the new-conversation button and this visitor's conversation list."""
    with st.sidebar:
        st.header("Conversation History")
        if st.button("+ New conversation", use_container_width=True):
            st.session_state.thread_id = None
            st.rerun()

        for conversation in list_conversations(DB_PATH, owner_uid):
            label = conversation.title or "(untitled)"
            if st.button(label, key=conversation.thread_id, use_container_width=True):
                st.session_state.thread_id = conversation.thread_id
                st.rerun()


def _render_reasoning_trace(state: dict) -> None:
    """Show this turn's retrieval decision and ReAct trajectory in a collapsed panel."""
    with st.expander("Reasoning trace"):
        st.markdown(
            f"**Retrieval decision**: `{state.get('retrieval_action')}` "
            f"(confidence {state.get('retrieval_confidence')})  \n{state.get('retrieval_reason')}"
        )
        st.markdown(
            f"**Retrieval rounds**: {state.get('retrieval_rounds')} / "
            f"{state.get('max_retrieval_rounds')}"
        )
        for entry in state.get("retrieval_history", []):
            queries = "  \n".join(
                f"- `{query.get('tool')}`: {query.get('query')}"
                for query in entry.get("queries", [])
            )
            added = ", ".join(entry.get("added_chunk_ids", [])) or "none"
            st.markdown(
                f"**Round {entry.get('round')}**  \n{queries}  \n"
                f"Relevant chunks added: {added}"
            )
        if state.get("support_status"):
            st.markdown(
                f"**Support check**: `{state.get('support_status')}` - "
                f"{state.get('support_reason')}"
            )


def _format_sources(sources) -> str:
    """Render a compact citation line for a response's source references."""
    parts = []
    for source in sources:
        if source.page is not None:
            parts.append(f"{source.source} p.{source.page}")
        else:
            parts.append(source.source)
    return ", ".join(parts)


def main() -> None:
    """Render the chat UI: sidebar history plus the active conversation."""
    st.set_page_config(page_title="Self-RAG Document QA", page_icon="📄")
    _load_secrets_into_environment()

    owner_uid = _get_or_create_visitor_id()
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = None

    graph, store, summarizer, conv_settings = _load_runtime()
    _render_sidebar(owner_uid)

    st.title("Self-RAG Document QA")

    thread_id = st.session_state.thread_id
    if thread_id:
        for message in store.history(thread_id):
            with st.chat_message(message.role):
                st.markdown(message.content)

    question = st.chat_input("Ask a question...")
    if not question:
        return

    if st.session_state.thread_id is None:
        st.session_state.thread_id = create_conversation(DB_PATH, owner_uid, question)
    thread_id = st.session_state.thread_id

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        response = _run_turn(
            graph, store, summarizer, question, thread_id, conv_settings
        )
        if response is not None:
            if response.sources:
                st.caption("Sources: " + _format_sources(response.sources))
            config = {"configurable": {"thread_id": thread_id}}
            _render_reasoning_trace(graph.get_state(config).values)

    if response is not None:
        st.rerun()


def _run_turn(graph, store, summarizer, question: str, thread_id: str, conv_settings):
    """Stream the answer into the chat message, returning the final RAGResponse."""
    response_box: dict[str, RAGResponse] = {}

    def token_stream():
        for item in stream_agent_graph(
            graph,
            question,
            thread_id=thread_id,
            store=store,
            summarizer=summarizer,
            **conv_settings,
        ):
            if isinstance(item, RAGResponse):
                response_box["response"] = item
            else:
                yield item

    try:
        streamed = st.write_stream(token_stream())
    except Exception as error:  # noqa: BLE001 - surface any turn failure in the UI
        st.error(f"Error: {error}")
        return None

    response = response_box.get("response")
    if response is None:
        return None
    # chitchat / abstain paths stream no tokens; render their answer now.
    if not streamed:
        st.markdown(response.answer)
    return response


if __name__ == "__main__":
    main()
