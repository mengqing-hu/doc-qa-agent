"""Streamlit chat UI for the Self-RAG document QA agent."""

from __future__ import annotations

import os
import sqlite3
import uuid

import streamlit as st
from langgraph.checkpoint.sqlite import SqliteSaver

from src.agent.context import ContextManager
from src.agent.graph import build_agent_graph, invoke_agent_graph
from src.agent.relevance import LLMRelevanceGrader
from src.agent.rewrite import LLMQueryRewriter
from src.agent.routes import LLMRetrievalGate
from src.agent.support import LLMSupportVerifier
from src.agent.utility import LLMUtilityVerifier
from src.core.config import PROJECT_ROOT, Config
from src.core.logger import setup_logging
from src.pipeline.query_runtime import build_query_pipeline
from src.store.conversations import create_conversation, list_conversations


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
def _load_graph():
    """Build the pipeline and compiled agent graph once per server process."""
    config = Config()
    setup_logging(config)
    pipeline = build_query_pipeline(config)

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    checkpointer = SqliteSaver(connection)

    return build_agent_graph(
        pipeline,
        retrieval_gate=LLMRetrievalGate(pipeline.llm),
        context_manager=ContextManager(config),
        query_rewriter=LLMQueryRewriter(pipeline.llm),
        relevance_grader=LLMRelevanceGrader(pipeline.llm),
        support_verifier=LLMSupportVerifier(pipeline.llm),
        utility_verifier=LLMUtilityVerifier(pipeline.llm),
        checkpointer=checkpointer,
    )


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
    """Show this turn's Ret/Rel/Sup/Use decisions in a collapsed panel."""
    with st.expander("Reasoning trace"):
        st.markdown(
            f"**[Ret] Retrieval decision**: `{state.get('retrieval_action')}` "
            f"(confidence {state.get('retrieval_confidence')})  \n{state.get('retrieval_reason')}"
        )
        st.markdown(f"**Rewritten query**: {state.get('rewritten_query')}")
        st.markdown(
            f"**[Rel] Relevance verdict**: `{state.get('relevance_status')}`  \n"
            f"{state.get('relevance_reason')}"
        )
        st.markdown(
            f"**[Sup] Support verdict**: `{state.get('support_status')}`  \n"
            f"{state.get('support_reason')}"
        )
        st.markdown(
            f"**[Use] Utility verdict**: `{state.get('utility_status')}`  \n"
            f"{state.get('utility_reason')}"
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

    graph = _load_graph()
    _render_sidebar(owner_uid)

    st.title("Self-RAG Document QA")

    thread_id = st.session_state.thread_id
    history = []
    if thread_id:
        config = {"configurable": {"thread_id": thread_id}}
        history = graph.get_state(config).values.get("conversation_history", [])

    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question...")
    if not question:
        return

    if st.session_state.thread_id is None:
        st.session_state.thread_id = create_conversation(DB_PATH, owner_uid, question)

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        response = None
        with st.spinner("Thinking..."):
            try:
                response = invoke_agent_graph(
                    graph, question, thread_id=st.session_state.thread_id
                )
            except Exception as error:
                st.error(f"Error: {error}")

        if response is not None:
            st.markdown(response.answer)
            if response.sources:
                st.caption("Sources: " + _format_sources(response.sources))
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            _render_reasoning_trace(graph.get_state(config).values)

    if response is not None:
        st.rerun()


if __name__ == "__main__":
    main()
