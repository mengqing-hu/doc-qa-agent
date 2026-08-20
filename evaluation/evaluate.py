"""Evaluate retrieval and no-answer behavior for the RAG pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.core.config import PROJECT_ROOT, Config
from src.core.logger import setup_logging
from src.document.chunker import chunk_sections
from src.document.pdf_parser import parse_pdf_document
from src.document.word_parser import parse_word_document
from src.generation.llm import LLM
from src.generation.prompts import PromptBuilder
from src.generation.rag_pipeline import RAGPipeline
from src.generation.response import SourceReference
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.embedder import Embedder
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import VectorStore


DEFAULT_INPUT_PATH = PROJECT_ROOT / "evaluation" / "test_queries.json"
DEFAULT_RESULTS_DIRECTORY = PROJECT_ROOT / "evaluation" / "results"
DEFAULT_TOP_KS = (1, 3, 5, 10)
UNANSWERABLE_QUERY_TYPE = "unanswerable"
REFUSAL_PATTERNS = (
    "cannot answer",
    "cannot be answered",
    "cannot determine",
    "could not determine",
    "does not provide",
    "does not report",
    "does not mention",
    "do not report",
    "do not mention",
    "insufficient information",
    "not enough information",
    "not available",
    "not discussed",
    "not provided",
    "not reported",
    "not mentioned",
    "unable to answer",
    "unable to determine",
    "文档中未提及",
    "文档没有提到",
    "没有提到",
    "未提及",
    "无法回答",
    "无法确定",
)


def main() -> None:
    """Run retrieval comparison and no-answer rejection evaluation."""
    arguments = _parse_arguments()
    config = Config()
    setup_logging(config)

    top_ks = _normalize_top_ks(arguments.top_ks)
    queries = _load_queries(arguments.input)
    if arguments.limit is not None:
        queries = queries[: arguments.limit]
    _validate_queries(queries)

    answerable_queries = _answerable_queries(queries)
    unanswerable_queries = _unanswerable_queries(queries)

    runtime = _build_runtime(config)
    retrieval_results = _evaluate_retrieval_variants(
        answerable_queries,
        runtime["vector_store"],
        runtime["hybrid_retriever"],
        runtime["reranker"],
        top_ks=top_ks,
    )
    rejection_result = (
        _evaluate_rejection(unanswerable_queries, runtime["rag_pipeline"])
        if not arguments.skip_rejection
        else _skipped_rejection_result(unanswerable_queries)
    )

    arguments.results_dir.mkdir(parents=True, exist_ok=True)
    for result in retrieval_results.values():
        _write_json(arguments.results_dir / f"{result['variant']}.json", result)
    _write_json(arguments.results_dir / "unanswerable_rejection.json", rejection_result)
    _write_json(
        arguments.results_dir / "comparison.json",
        _build_comparison_result(retrieval_results, rejection_result),
    )

    _print_retrieval_summary(retrieval_results, top_ks)
    _print_rejection_summary(rejection_result)


def _parse_arguments() -> argparse.Namespace:
    """Read evaluation settings from the command line."""
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval and no-answer rejection quality."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Annotated query file with relevant_chunk_ids.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIRECTORY,
        help="Directory where evaluation result JSON files are written.",
    )
    parser.add_argument(
        "--top-ks",
        type=int,
        nargs="+",
        default=list(DEFAULT_TOP_KS),
        help="K values used for Recall@K and Hit Rate@K.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of queries to evaluate.",
    )
    parser.add_argument(
        "--skip-rejection",
        action="store_true",
        help="Skip LLM-based evaluation for unanswerable queries.",
    )
    return parser.parse_args()


def _normalize_top_ks(top_ks: list[int]) -> list[int]:
    """Return sorted unique positive K values."""
    normalized_top_ks = sorted(set(top_ks))
    if not normalized_top_ks or any(top_k <= 0 for top_k in normalized_top_ks):
        raise ValueError("top_ks must contain at least one positive integer")
    return normalized_top_ks


def _load_queries(input_path: Path) -> list[dict[str, Any]]:
    """Load annotated evaluation queries."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Evaluation file was not found: {input_path}")

    with input_path.open(encoding="utf-8") as file:
        queries = json.load(file)
    if not isinstance(queries, list):
        raise ValueError("Evaluation file must contain a JSON list")
    return [dict(query) for query in queries]


def _validate_queries(queries: list[dict[str, Any]]) -> None:
    """Validate answerable and unanswerable query labels."""
    if not queries:
        raise ValueError("At least one evaluation query is required")

    missing_labels = [
        str(query.get("query_id", "<unknown>"))
        for query in queries
        if query.get("query_type") != UNANSWERABLE_QUERY_TYPE
        and not query.get("relevant_chunk_ids")
    ]
    if missing_labels:
        raise ValueError(
            "Missing relevant_chunk_ids for query_id(s): "
            + ", ".join(missing_labels)
        )

    invalid_unanswerable = [
        str(query.get("query_id", "<unknown>"))
        for query in queries
        if query.get("query_type") == UNANSWERABLE_QUERY_TYPE
        and query.get("relevant_chunk_ids")
    ]
    if invalid_unanswerable:
        raise ValueError(
            "Unanswerable query_type must not define relevant_chunk_ids for "
            "query_id(s): "
            + ", ".join(invalid_unanswerable)
        )


def _answerable_queries(queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return queries with ground-truth relevant chunks."""
    return [
        query
        for query in queries
        if query.get("query_type") != UNANSWERABLE_QUERY_TYPE
    ]


def _unanswerable_queries(queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return queries that should be rejected by the generation layer."""
    return [
        query
        for query in queries
        if query.get("query_type") == UNANSWERABLE_QUERY_TYPE
    ]


def _build_runtime(config: Config) -> dict[str, Any]:
    """Build shared chunks, indexes, retrievers, and the final RAG pipeline."""
    chunks = chunk_sections(_parse_documents(config), config)

    embedder = Embedder(config)
    for chunk, embedding in zip(chunks, embedder.embed_chunks(chunks), strict=True):
        chunk["embedding"] = embedding

    vector_store = VectorStore(config, embedder=embedder)
    vector_store.add_chunks(chunks)
    bm25_retriever = BM25Retriever(chunks, config)
    hybrid_retriever = HybridRetriever(vector_store, bm25_retriever, config)
    reranker = Reranker(config)
    rag_pipeline = RAGPipeline(
        hybrid_retriever,
        reranker,
        PromptBuilder(config),
        LLM(config),
    )
    return {
        "chunks": chunks,
        "vector_store": vector_store,
        "bm25_retriever": bm25_retriever,
        "hybrid_retriever": hybrid_retriever,
        "reranker": reranker,
        "rag_pipeline": rag_pipeline,
    }


def _parse_documents(config: Config) -> list[dict[str, Any]]:
    """Parse configured PDF and Word documents into one section list."""
    configured_paths = config.get("documents", "paths", default=[])
    if not configured_paths:
        raise ValueError("documents.paths must contain at least one document")

    sections: list[dict[str, Any]] = []
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


def _evaluate_retrieval_variants(
    answerable_queries: list[dict[str, Any]],
    vector_store: VectorStore,
    hybrid_retriever: HybridRetriever,
    reranker: Reranker,
    *,
    top_ks: list[int],
) -> dict[str, dict[str, Any]]:
    """Evaluate V1, V2a, and V2 with the same chunks and labels."""
    rerank_candidate_top_k = int(
        hybrid_retriever.config.get("retrieval", "hybrid_top_k", default=30)
    )
    if rerank_candidate_top_k < max(top_ks):
        raise ValueError(
            "retrieval.hybrid_top_k must be at least the largest requested top_k "
            "when evaluating reranked retrieval"
        )

    return {
        "v1_dense": _evaluate_retrieval_variant(
            "v1_dense",
            "Dense vector retrieval with ChromaDB only.",
            answerable_queries,
            top_ks,
            lambda query: vector_store.search(query, top_k=max(top_ks)),
        ),
        "v2a_hybrid": _evaluate_retrieval_variant(
            "v2a_hybrid",
            "Dense plus BM25 retrieval with RRF fusion, without Cross-Encoder.",
            answerable_queries,
            top_ks,
            lambda query: hybrid_retriever.search(query, top_k=max(top_ks)),
        ),
        "v2_hybrid_rerank": _evaluate_retrieval_variant(
            "v2_hybrid_rerank",
            "Dense plus BM25 retrieval with RRF fusion and Cross-Encoder reranking.",
            answerable_queries,
            top_ks,
            lambda query: reranker.rerank(
                query,
                hybrid_retriever.search(query, top_k=rerank_candidate_top_k),
                top_k=max(top_ks),
            ),
        ),
    }


def _evaluate_retrieval_variant(
    variant: str,
    description: str,
    queries: list[dict[str, Any]],
    top_ks: list[int],
    retrieve: Any,
) -> dict[str, Any]:
    """Evaluate one retrieval strategy over all answerable queries."""
    query_results = [
        _score_retrieval_query(query, retrieve(str(query["query"])), top_ks)
        for query in queries
    ]
    return {
        "variant": variant,
        "description": description,
        "query_count": len(query_results),
        "top_ks": top_ks,
        "metrics": _aggregate_retrieval_metrics(query_results, top_ks),
        "queries": query_results,
    }


def _score_retrieval_query(
    query: dict[str, Any],
    retrieved_chunks: list[dict[str, Any]],
    top_ks: list[int],
) -> dict[str, Any]:
    """Compute Recall@K and reciprocal rank for one answerable query."""
    relevant_ids = {str(chunk_id) for chunk_id in query["relevant_chunk_ids"]}
    retrieved_ids = [str(chunk["chunk_id"]) for chunk in retrieved_chunks[: max(top_ks)]]
    matched_ids = [chunk_id for chunk_id in retrieved_ids if chunk_id in relevant_ids]
    first_relevant_rank = _first_relevant_rank(retrieved_ids, relevant_ids)
    reciprocal_rank = 0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank

    metrics = {
        f"recall_at_{top_k}": _recall_at_k(retrieved_ids, relevant_ids, top_k)
        for top_k in top_ks
    }
    metrics.update(
        {
            f"hit_at_{top_k}": _hit_at_k(retrieved_ids, relevant_ids, top_k)
            for top_k in top_ks
        }
    )
    metrics["reciprocal_rank"] = reciprocal_rank

    return {
        "query_id": query.get("query_id"),
        "query": query.get("query"),
        "query_type": query.get("query_type"),
        "relevant_chunk_ids": sorted(relevant_ids),
        "retrieved_chunk_ids": retrieved_ids,
        "matched_chunk_ids": matched_ids,
        "first_relevant_rank": first_relevant_rank,
        "metrics": metrics,
    }


def _recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    top_k: int,
) -> float:
    """Return the fraction of relevant chunks found in Top-K."""
    retrieved_top_k = set(retrieved_ids[:top_k])
    return len(retrieved_top_k.intersection(relevant_ids)) / len(relevant_ids)


def _hit_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    top_k: int,
) -> bool:
    """Return whether at least one relevant chunk appears in Top-K."""
    return bool(set(retrieved_ids[:top_k]).intersection(relevant_ids))


def _first_relevant_rank(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> int | None:
    """Return the one-based rank of the first relevant result."""
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return rank
    return None


def _aggregate_retrieval_metrics(
    query_results: list[dict[str, Any]],
    top_ks: list[int],
) -> dict[str, float]:
    """Average per-query metrics for one retrieval variant."""
    if not query_results:
        return {
            **{f"recall_at_{top_k}": 0.0 for top_k in top_ks},
            **{f"hit_rate_at_{top_k}": 0.0 for top_k in top_ks},
            "mrr": 0.0,
        }

    query_count = len(query_results)
    metrics: dict[str, float] = {}
    for top_k in top_ks:
        metrics[f"recall_at_{top_k}"] = (
            sum(result["metrics"][f"recall_at_{top_k}"] for result in query_results)
            / query_count
        )
        metrics[f"hit_rate_at_{top_k}"] = (
            sum(1 for result in query_results if result["metrics"][f"hit_at_{top_k}"])
            / query_count
        )
    metrics["mrr"] = (
        sum(result["metrics"]["reciprocal_rank"] for result in query_results)
        / query_count
    )
    return metrics


def _evaluate_rejection(
    unanswerable_queries: list[dict[str, Any]],
    rag_pipeline: RAGPipeline,
) -> dict[str, Any]:
    """Evaluate whether the full RAG pipeline refuses unanswerable queries."""
    query_results = [
        _score_rejection_query(query, rag_pipeline)
        for query in unanswerable_queries
    ]
    query_count = len(query_results)
    refused_count = sum(1 for result in query_results if result["refusal_detected"])
    failed_count = sum(1 for result in query_results if result["status"] == "failed")
    rejection_accuracy = refused_count / query_count if query_count else 0.0
    return {
        "variant": "unanswerable_rejection",
        "description": (
            "Full RAG pipeline refusal behavior on queries with no supporting chunks."
        ),
        "query_count": query_count,
        "metrics": {
            "rejection_accuracy": rejection_accuracy,
            "refused_count": refused_count,
            "failed_count": failed_count,
        },
        "queries": query_results,
    }


def _score_rejection_query(
    query: dict[str, Any],
    rag_pipeline: RAGPipeline,
) -> dict[str, Any]:
    """Generate one answer and detect whether it contains a refusal phrase."""
    try:
        response = rag_pipeline.answer(str(query["query"]))
    except Exception as error:
        return {
            "query_id": query.get("query_id"),
            "query": query.get("query"),
            "expected_answer": query.get("answer"),
            "status": "failed",
            "model_answer": None,
            "refusal_detected": False,
            "matched_refusal_pattern": None,
            "sources": [],
            "error": str(error),
        }

    refusal_detected, matched_pattern = _detect_refusal(response.answer)
    return {
        "query_id": query.get("query_id"),
        "query": query.get("query"),
        "expected_answer": query.get("answer"),
        "status": "passed",
        "model_answer": response.answer,
        "refusal_detected": refusal_detected,
        "matched_refusal_pattern": matched_pattern,
        "sources": [_source_to_dict(source) for source in response.sources],
        "error": None,
    }


def _detect_refusal(answer: str) -> tuple[bool, str | None]:
    """Detect simple refusal wording in a generated answer."""
    normalized_answer = answer.lower()
    for pattern in REFUSAL_PATTERNS:
        if pattern.lower() in normalized_answer:
            return True, pattern
    return False, None


def _source_to_dict(source: SourceReference) -> dict[str, Any]:
    """Serialize a source reference for JSON output."""
    return {
        "chunk_id": source.chunk_id,
        "source": source.source,
        "page": source.page,
        "section_title": source.section_title,
    }


def _skipped_rejection_result(unanswerable_queries: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a placeholder when LLM-based rejection evaluation is skipped."""
    return {
        "variant": "unanswerable_rejection",
        "description": "Skipped by --skip-rejection.",
        "query_count": len(unanswerable_queries),
        "metrics": {
            "rejection_accuracy": None,
            "refused_count": None,
            "failed_count": None,
        },
        "queries": [
            {
                "query_id": query.get("query_id"),
                "query": query.get("query"),
                "expected_answer": query.get("answer"),
                "status": "skipped",
                "model_answer": None,
                "refusal_detected": None,
                "matched_refusal_pattern": None,
                "sources": [],
                "error": None,
            }
            for query in unanswerable_queries
        ],
    }


def _build_comparison_result(
    retrieval_results: dict[str, dict[str, Any]],
    rejection_result: dict[str, Any],
) -> dict[str, Any]:
    """Create a compact file for README tables and interview discussion."""
    return {
        "retrieval": {
            variant: result["metrics"]
            for variant, result in retrieval_results.items()
        },
        "unanswerable_rejection": rejection_result["metrics"],
    }


def _write_json(output_path: Path, payload: dict[str, Any]) -> None:
    """Write one evaluation result file."""
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _print_retrieval_summary(
    retrieval_results: dict[str, dict[str, Any]],
    top_ks: list[int],
) -> None:
    """Print retrieval metrics for each variant."""
    print()
    print("Retrieval metrics")
    for result in retrieval_results.values():
        metrics = result["metrics"]
        print(f"{result['variant']} ({result['query_count']} answerable queries)")
        for top_k in top_ks:
            print(f"  Recall@{top_k}: {metrics[f'recall_at_{top_k}']:.3f}")
        print(f"  MRR: {metrics['mrr']:.3f}")


def _print_rejection_summary(rejection_result: dict[str, Any]) -> None:
    """Print the no-answer rejection metric."""
    metrics = rejection_result["metrics"]
    print()
    print(f"{rejection_result['variant']} ({rejection_result['query_count']} queries)")
    if metrics["rejection_accuracy"] is None:
        print("  Rejection accuracy: skipped")
        return
    print(f"  Rejection accuracy: {metrics['rejection_accuracy']:.3f}")
    print(f"  Refused: {metrics['refused_count']}")
    print(f"  Failed: {metrics['failed_count']}")


if __name__ == "__main__":
    main()
