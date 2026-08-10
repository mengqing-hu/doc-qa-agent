"""Run the document question-answering pipeline from the command line."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.core.config import PROJECT_ROOT, Config
from src.core.logger import setup_logging
from src.document.chunker import chunk_sections
from src.document.pdf_parser import parse_pdf_document
from src.document.word_parser import parse_word_document
from src.generation.llm import LLM
from src.generation.prompts import PromptBuilder
from src.generation.rag_pipeline import RAGPipeline
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.embedder import Embedder
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import VectorStore


def main() -> None:
    """Parse a question, build the local pipeline, and print a grounded answer."""
    arguments = _parse_arguments()
    config = Config()
    setup_logging(config)
    pipeline = _build_pipeline(config)
    response = pipeline.answer(arguments.question)

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
    return parser.parse_args()


def _build_pipeline(config: Config) -> RAGPipeline:
    """Parse configured documents and assemble the online RAG components."""
    chunks = chunk_sections(_parse_documents(config), config)
    embedder = Embedder(config)
    vector_store = VectorStore(config, embedder=embedder)
    vector_store.add_chunks(chunks)
    bm25_retriever = BM25Retriever(chunks, config)
    hybrid_retriever = HybridRetriever(vector_store, bm25_retriever, config)
    return RAGPipeline(
        hybrid_retriever,
        Reranker(config),
        PromptBuilder(config),
        LLM(config),
    )


def _parse_documents(config: Config) -> list[dict[str, object]]:
    """Parse configured PDF and Word documents into one section list."""
    configured_paths = config.get("documents", "paths", default=[])
    if not configured_paths:
        raise ValueError("documents.paths must contain at least one document")

    sections: list[dict[str, object]] = []
    for configured_path in configured_paths:
        document_path = Path(configured_path)
        if not document_path.is_absolute():
            document_path = PROJECT_ROOT / document_path
        suffix = document_path.suffix.lower()
        if suffix == ".pdf":
            sections.extend(parse_pdf_document(document_path, config))
        elif suffix == ".docx":
            sections.extend(parse_word_document(document_path))
        else:
            raise ValueError(f"Unsupported document type: {document_path}")
    return sections


if __name__ == "__main__":
    main()
