"""Evaluate grounded answer quality with RAGAS."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    ContextPrecisionWithReference,
    ContextRecall,
    Faithfulness,
    FactualCorrectness,
)

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
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.embedder import Embedder
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import VectorStore


logger = logging.getLogger(__name__)
DEFAULT_INPUT_PATH = PROJECT_ROOT / "evaluation" / "test_queries.json"
DEFAULT_RESULTS_DIRECTORY = PROJECT_ROOT / "evaluation" / "results"
DEFAULT_OUTPUT_FILE_NAME = "ragas_hybrid_rerank.json"
UNANSWERABLE_QUERY_TYPE = "unanswerable"
RAGAS_METRIC_NAMES = (
    "context_precision_with_reference",
    "context_recall",
    "faithfulness",
    "factual_correctness",
)


def main() -> None:
    """Run RAGAS metrics for answerable questions using final RAG evidence."""
    arguments = _parse_arguments()
    config = Config()
    setup_logging(config)

    queries = _load_queries(arguments.input)
    answerable_queries = _answerable_queries(queries)
    if arguments.limit is not None:
        answerable_queries = answerable_queries[: arguments.limit]
    _validate_answerable_queries(answerable_queries)

    runtime = _build_runtime(config)
    query_records = _generate_query_records(answerable_queries, runtime)
    metric_scores = asyncio.run(
        _evaluate_query_records(
            query_records,
            runtime["chunks"],
            _build_judge_llm(config, arguments),
        )
    )

    evaluation_result = _build_evaluation_result(query_records, metric_scores)
    arguments.results_dir.mkdir(parents=True, exist_ok=True)
    output_path = arguments.results_dir / arguments.output_file
    _write_json(output_path, evaluation_result)
    _print_summary(evaluation_result, output_path)


def _parse_arguments() -> argparse.Namespace:
    """Read RAGAS evaluation settings from the command line."""
    parser = argparse.ArgumentParser(
        description="Evaluate grounded RAG answers with RAGAS."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Annotated query file containing reference answers and chunk IDs.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIRECTORY,
        help="Directory where the RAGAS result JSON file is written.",
    )
    parser.add_argument(
        "--output-file",
        default=DEFAULT_OUTPUT_FILE_NAME,
        help="Name of the RAGAS result file inside results-dir.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of answerable queries to evaluate.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="OpenAI-compatible model used by RAGAS; defaults to llm.model.",
    )
    parser.add_argument(
        "--judge-base-url",
        default=None,
        help="OpenAI-compatible endpoint used by RAGAS; defaults to llm.base_url.",
    )
    arguments = parser.parse_args()
    if arguments.limit is not None and arguments.limit <= 0:
        parser.error("--limit must be greater than zero")
    return arguments


def _load_queries(input_path: Path) -> list[dict[str, Any]]:
    """Load the annotated evaluation queries from JSON."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Evaluation file was not found: {input_path}")

    with input_path.open(encoding="utf-8") as file:
        queries = json.load(file)
    if not isinstance(queries, list):
        raise ValueError("Evaluation file must contain a JSON list")
    return [dict(query) for query in queries]


def _answerable_queries(queries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only queries with an expected document-grounded answer."""
    return [
        query
        for query in queries
        if query.get("query_type") != UNANSWERABLE_QUERY_TYPE
    ]


def _validate_answerable_queries(queries: Sequence[dict[str, Any]]) -> None:
    """Validate fields needed to build RAGAS evaluation inputs."""
    if not queries:
        raise ValueError("At least one answerable query is required")

    invalid_query_ids = [
        str(query.get("query_id", "<unknown>"))
        for query in queries
        if not str(query.get("query", "")).strip()
        or not str(query.get("answer", "")).strip()
        or not query.get("relevant_chunk_ids")
    ]
    if invalid_query_ids:
        raise ValueError(
            "Answerable queries require query, answer, and relevant_chunk_ids: "
            + ", ".join(invalid_query_ids)
        )


def _build_runtime(config: Config) -> dict[str, Any]:
    """Build the same parsing, indexing, retrieval, and generation stack as the CLI."""
    chunks = chunk_sections(_parse_documents(config), config)
    embedder = Embedder(config)
    vector_store = VectorStore(config, embedder=embedder)
    vector_store.add_chunks(chunks)
    bm25_retriever = BM25Retriever(chunks, config)
    hybrid_retriever = HybridRetriever(vector_store, bm25_retriever, config)
    pipeline = RAGPipeline(
        hybrid_retriever,
        Reranker(config),
        PromptBuilder(config),
        LLM(config),
    )
    return {"chunks": chunks, "pipeline": pipeline}


def _parse_documents(config: Config) -> list[dict[str, Any]]:
    """Parse all configured PDF and Word documents into sections."""
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


def _generate_query_records(
    queries: Sequence[dict[str, Any]],
    runtime: dict[str, Any],
) -> list[dict[str, Any]]:
    """Generate answers while retaining the exact chunks supplied to the LLM."""
    pipeline = runtime["pipeline"]
    if not isinstance(pipeline, RAGPipeline):
        raise TypeError("runtime.pipeline must be a RAGPipeline")

    records: list[dict[str, Any]] = []
    for query in queries:
        question = str(query["query"]).strip()
        hybrid_chunks = pipeline.hybrid_retriever.search(question)
        final_chunks = pipeline.reranker.rerank(question, hybrid_chunks)
        prompt = pipeline.prompt_builder.build(question, final_chunks)
        answer = pipeline.llm.generate(prompt)
        records.append(
            {
                "query_id": query.get("query_id"),
                "query": question,
                "reference_answer": str(query["answer"]),
                "reference_chunk_ids": [
                    str(chunk_id) for chunk_id in query["relevant_chunk_ids"]
                ],
                "model_answer": answer,
                "retrieved_chunks": final_chunks,
            }
        )
    return records


async def _evaluate_query_records(
    query_records: Sequence[dict[str, Any]],
    chunks: Sequence[dict[str, Any]],
    judge_llm: Any,
) -> list[dict[str, Any]]:
    """Score generated RAG outputs with modern RAGAS metric components."""
    chunk_text_by_id = {
        str(chunk["chunk_id"]): str(chunk.get("text", "")) for chunk in chunks
    }
    metrics = _build_metrics(judge_llm)
    scores: list[dict[str, Any]] = []

    for record in query_records:
        reference_chunk_ids = record["reference_chunk_ids"]
        missing_chunk_ids = [
            chunk_id for chunk_id in reference_chunk_ids if chunk_id not in chunk_text_by_id
        ]
        if missing_chunk_ids:
            raise ValueError(
                f"Unknown reference chunk ID(s) for {record['query_id']}: "
                + ", ".join(missing_chunk_ids)
            )

        retrieved_contexts = [
            str(chunk.get("text", "")) for chunk in record["retrieved_chunks"]
        ]
        scores.append(
            await _score_query(
                metrics,
                query=str(record["query"]),
                response=str(record["model_answer"]),
                reference=str(record["reference_answer"]),
                retrieved_contexts=retrieved_contexts,
            )
        )
    return scores


def _build_metrics(judge_llm: Any) -> dict[str, Any]:
    """Create modern RAGAS metric components using one evaluator LLM."""
    return {
        "context_precision_with_reference": ContextPrecisionWithReference(
            llm=judge_llm
        ),
        "context_recall": ContextRecall(llm=judge_llm),
        "faithfulness": Faithfulness(llm=judge_llm),
        "factual_correctness": FactualCorrectness(llm=judge_llm),
    }


async def _score_query(
    metrics: dict[str, Any],
    *,
    query: str,
    response: str,
    reference: str,
    retrieved_contexts: list[str],
) -> dict[str, Any]:
    """Score one generated answer and retain individual metric errors."""
    metric_calls = {
        "context_precision_with_reference": lambda: metrics[
            "context_precision_with_reference"
        ].ascore(
            user_input=query,
            reference=reference,
            retrieved_contexts=retrieved_contexts,
        ),
        "context_recall": lambda: metrics["context_recall"].ascore(
            user_input=query,
            reference=reference,
            retrieved_contexts=retrieved_contexts,
        ),
        "faithfulness": lambda: metrics["faithfulness"].ascore(
            user_input=query,
            response=response,
            retrieved_contexts=retrieved_contexts,
        ),
        "factual_correctness": lambda: metrics["factual_correctness"].ascore(
            response=response,
            reference=reference,
        ),
    }
    scores: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for metric_name, metric_call in metric_calls.items():
        try:
            metric_result = await metric_call()
        except Exception as error:
            logger.exception("RAGAS metric failed: %s", metric_name)
            scores[metric_name] = None
            errors[metric_name] = str(error)
            continue
        scores[metric_name] = _json_safe_value(metric_result.value)
    return {"metrics": scores, "errors": errors}


def _build_judge_llm(config: Config, arguments: argparse.Namespace) -> Any:
    """Build the OpenAI-compatible structured-output LLM used by RAGAS."""
    judge_model = arguments.judge_model or str(config.get("llm", "model"))
    judge_base_url = arguments.judge_base_url or str(config.get("llm", "base_url"))
    client = AsyncOpenAI(api_key=config.scadsai_api_key, base_url=judge_base_url)
    return llm_factory(judge_model, provider="openai", client=client, temperature=0.0)


def _build_evaluation_result(
    query_records: Sequence[dict[str, Any]],
    score_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Build a JSON-safe result with aggregate and per-query RAGAS scores."""
    if len(query_records) != len(score_records):
        raise ValueError("RAGAS returned an unexpected number of score records")

    query_results = [
        {
            "query_id": record["query_id"],
            "query": record["query"],
            "reference_answer": record["reference_answer"],
            "model_answer": record["model_answer"],
            "reference_chunk_ids": record["reference_chunk_ids"],
            "retrieved_chunk_ids": [
                str(chunk["chunk_id"]) for chunk in record["retrieved_chunks"]
            ],
            "metrics": score_record["metrics"],
            "metric_errors": score_record["errors"],
        }
        for record, score_record in zip(query_records, score_records, strict=True)
    ]
    return {
        "variant": "ragas_hybrid_rerank",
        "description": (
            "RAGAS evaluation of hybrid retrieval, Cross-Encoder reranking, and "
            "LLM-generated answers for answerable queries."
        ),
        "query_count": len(query_results),
        "metrics": _aggregate_metrics(query_results),
        "queries": query_results,
    }


def _aggregate_metrics(query_results: Sequence[dict[str, Any]]) -> dict[str, float | None]:
    """Average finite per-query values for each configured RAGAS metric."""
    return {
        metric_name: _mean_finite(
            [result["metrics"].get(metric_name) for result in query_results]
        )
        for metric_name in RAGAS_METRIC_NAMES
    }


def _mean_finite(values: Sequence[Any]) -> float | None:
    """Return the arithmetic mean of finite numeric values, if available."""
    finite_values = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    if not finite_values:
        return None
    return sum(finite_values) / len(finite_values)


def _json_safe_value(value: Any) -> Any:
    """Return a JSON-compatible scalar without NaN or infinity values."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(output_path: Path, payload: dict[str, Any]) -> None:
    """Write one evaluation result file with stable Unicode output."""
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)


def _print_summary(result: dict[str, Any], output_path: Path) -> None:
    """Print aggregate RAGAS metrics and the output location."""
    print(f"RAGAS metrics ({result['query_count']} answerable queries)")
    for metric_name, value in result["metrics"].items():
        formatted_value = "unavailable" if value is None else f"{value:.3f}"
        print(f"  {metric_name}: {formatted_value}")
    print(f"Saved RAGAS results to {output_path}")


if __name__ == "__main__":
    main()
