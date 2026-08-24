"""Run the document question-answering pipeline from the command line."""

from __future__ import annotations

import argparse

from langgraph.checkpoint.memory import InMemorySaver

from src.agent.context import ContextManager
from src.agent.graph import build_agent_graph, invoke_agent_graph
from src.agent.relevance import LLMRelevanceGrader
from src.agent.rewrite import LLMQueryRewriter
from src.agent.routes import LLMRetrievalGate
from src.agent.support import LLMSupportVerifier
from src.agent.utility import LLMUtilityVerifier
from src.core.config import Config
from src.core.logger import setup_logging
from src.pipeline.query_runtime import build_query_pipeline


def main() -> None:
    """Parse a question, build the local pipeline, and print a grounded answer."""
    arguments = _parse_arguments()
    config = Config()
    setup_logging(config)
    pipeline = build_query_pipeline(config)
    graph = build_agent_graph(
        pipeline,
        retrieval_gate=LLMRetrievalGate(pipeline.llm),
        context_manager=ContextManager(config),
        query_rewriter=LLMQueryRewriter(pipeline.llm),
        relevance_grader=LLMRelevanceGrader(pipeline.llm),
        support_verifier=LLMSupportVerifier(pipeline.llm),
        utility_verifier=LLMUtilityVerifier(pipeline.llm),
        checkpointer=InMemorySaver(),
    )
    response = invoke_agent_graph(
        graph,
        arguments.question,
        thread_id=arguments.thread_id,
    )

    print(response.answer)
    print("\nSources:")
    for source in response.sources:
        location = f"page {source.page}" if source.page is not None else "page unknown"
        section = source.section_title or "section unknown"
        print(f"- {source.chunk_id} | {source.source} | {location} | {section}")


def _parse_arguments() -> argparse.Namespace:
    """Read the required question from command-line arguments."""
    parser = argparse.ArgumentParser(description="Answer questions from local documents.")
    parser.add_argument("question", help="Question to answer from the indexed documents.")
    parser.add_argument(
        "--thread-id",
        default="cli-session",
        help="Conversation identifier used by the LangGraph checkpointer.",
    )
    return parser.parse_args()
if __name__ == "__main__":
    main()
