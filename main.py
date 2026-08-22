"""Run the document question-answering pipeline from the command line."""

from __future__ import annotations

import argparse

from langgraph.checkpoint.memory import InMemorySaver

from src.agent.graph import build_single_hop_rag_graph, invoke_single_hop_rag
from src.core.config import Config
from src.core.logger import setup_logging
from src.pipeline.query_runtime import build_query_pipeline


def main() -> None:
    """Parse a question, build the local pipeline, and print a grounded answer."""
    arguments = _parse_arguments()
    config = Config()
    setup_logging(config)
    graph = build_single_hop_rag_graph(
        build_query_pipeline(config),
        checkpointer=InMemorySaver(),
    )
    response = invoke_single_hop_rag(
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
