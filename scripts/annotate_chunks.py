"""Annotate evaluation queries with relevant chunk IDs."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.core.config import PROJECT_ROOT, Config
from src.core.logger import setup_logging
from src.document.chunker import chunk_sections
from src.document.pdf_parser import parse_pdf_document
from src.document.word_parser import parse_word_document
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.embedder import Embedder
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import VectorStore


DEFAULT_INPUT_PATH = PROJECT_ROOT / "evaluation" / "test_queries_draft.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "test_queries.json"
DEFAULT_CANDIDATE_OUTPUT_PATH = (
    PROJECT_ROOT / "evaluation" / "test_queries_candidates.json"
)
DEFAULT_CANDIDATE_TOP_K = 10


def main() -> None:
    """Build retrieval candidates for interactive labeling or JSON review."""
    arguments = _parse_arguments()
    config = Config()
    setup_logging(config)

    if arguments.candidate_top_k <= 0:
        raise ValueError("candidate_top_k must be greater than zero")

    queries = _load_queries(arguments.input)
    if arguments.limit is not None:
        queries = queries[: arguments.limit]

    sections = _parse_documents(config)
    chunks = chunk_sections(sections, config)

    embedder = Embedder(config)
    for chunk, embedding in zip(chunks, embedder.embed_chunks(chunks), strict=True):
        chunk["embedding"] = embedding

    vector_store = VectorStore(config, embedder=embedder)
    vector_store.add_chunks(chunks)
    bm25_retriever = BM25Retriever(chunks, config)
    hybrid_retriever = HybridRetriever(vector_store, bm25_retriever, config)
    reranker = Reranker(config) if arguments.rerank_candidates else None

    output_path = _resolve_output_path(arguments)
    annotated_queries = _annotate_queries(
        queries,
        hybrid_retriever,
        reranker,
        output_path=output_path,
        candidate_top_k=arguments.candidate_top_k,
        interactive=not (arguments.non_interactive or arguments.export_candidates),
        include_candidates=arguments.export_candidates,
    )
    _write_queries(output_path, annotated_queries)
    saved_kind = (
        "candidate export" if arguments.export_candidates else "annotated queries"
    )
    print(f"Saved {saved_kind} to {output_path}")


def _parse_arguments() -> argparse.Namespace:
    """Read annotation settings from the command line."""
    parser = argparse.ArgumentParser(
        description="Create relevant-chunk annotations for evaluation queries."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Draft query file to annotate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file; uses a mode-specific default when omitted.",
    )
    parser.add_argument(
        "--candidate-top-k",
        type=int,
        default=DEFAULT_CANDIDATE_TOP_K,
        help="Number of retrieval candidates to display for each query.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of queries to process.",
    )
    parser.add_argument(
        "--rerank-candidates",
        action="store_true",
        help="Rerank hybrid candidates with the Cross-Encoder before annotation.",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Skip prompts and keep any existing relevant_chunk_ids values.",
    )
    parser.add_argument(
        "--export-candidates",
        action="store_true",
        help=(
            "Write full retrieval candidates to JSON without prompting; defaults to "
            "evaluation/test_queries_candidates.json."
        ),
    )
    return parser.parse_args()


def _resolve_output_path(arguments: argparse.Namespace) -> Path:
    """Return an explicit output path or the safe default for the selected mode."""
    if arguments.output is not None:
        return arguments.output
    if arguments.export_candidates:
        return DEFAULT_CANDIDATE_OUTPUT_PATH
    return DEFAULT_OUTPUT_PATH


def _load_queries(input_path: Path) -> list[dict[str, Any]]:
    """Load the draft evaluation queries from disk."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file was not found: {input_path}")

    with input_path.open(encoding="utf-8") as file:
        queries = json.load(file)
    if not isinstance(queries, list):
        raise ValueError("Input file must contain a JSON list of queries")
    return [dict(query) for query in queries]


def _load_existing_annotations(output_path: Path) -> list[dict[str, Any]]:
    """Load previous annotations so interrupted runs can resume cleanly."""
    if not output_path.is_file():
        return []

    with output_path.open(encoding="utf-8") as file:
        existing_queries = json.load(file)
    if not isinstance(existing_queries, list):
        raise ValueError("Output file must contain a JSON list of queries")
    return [dict(query) for query in existing_queries if isinstance(query, dict)]


def _annotate_queries(
    queries: list[dict[str, Any]],
    hybrid_retriever: HybridRetriever,
    reranker: Reranker | None,
    *,
    output_path: Path,
    candidate_top_k: int,
    interactive: bool,
    include_candidates: bool,
) -> list[dict[str, Any]]:
    """Collect relevant chunk IDs or export the candidate records for each query."""
    annotated_queries: list[dict[str, Any]] = []
    for position, query in enumerate(queries, start=1):
        query_id = str(query.get("query_id", f"query_{position:03d}"))
        query_text = str(query.get("query", "")).strip()
        if not query_text:
            raise ValueError(f"Query text is empty for query_id={query_id}")

        print()
        print(f"[{position}/{len(queries)}] {query_id}")
        print(f"Query: {query_text}")
        print(f"Answer: {query.get('answer', '')}")
        source_info = query.get("source_info")
        if source_info:
            print(f"Source: {source_info}")

        candidates = hybrid_retriever.search(query_text, top_k=candidate_top_k)
        if reranker is not None:
            candidates = reranker.rerank(query_text, candidates, top_k=candidate_top_k)

        if interactive:
            _print_candidates(candidates)
        relevant_chunk_ids = _resolve_relevant_chunk_ids(
            query,
            candidates,
            interactive=interactive,
        )

        annotated_query = dict(query)
        annotated_query["relevant_chunk_ids"] = relevant_chunk_ids
        if include_candidates:
            annotated_query["retrieval_candidates"] = _serialize_candidates(candidates)
        annotated_queries.append(annotated_query)
        _write_queries(output_path, annotated_queries)

    return annotated_queries


def _serialize_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the chunk ID and complete text needed for manual relevance review."""
    return [
        {
            "chunk_id": str(candidate.get("chunk_id", "unknown")),
            "text": str(candidate.get("text", "")),
        }
        for candidate in candidates
    ]


def _resolve_relevant_chunk_ids(
    query: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    interactive: bool,
) -> list[str]:
    """Return the manually selected chunk IDs for one query."""
    existing_ids = [
        str(chunk_id)
        for chunk_id in query.get("relevant_chunk_ids", [])
        if str(chunk_id).strip()
    ]
    if not interactive:
        return existing_ids

    prompt = (
        "Relevant chunk ids or ranks (comma-separated, "
        "blank keeps existing, 's' skips, 'q' quits): "
    )
    while True:
        response = input(f"{prompt}").strip()
        if response == "":
            return existing_ids
        if response.lower() == "s":
            return existing_ids
        if response.lower() == "q":
            raise KeyboardInterrupt

        try:
            selected_ids = _resolve_selection(response, candidates)
        except ValueError as error:
            print(str(error))
            continue
        candidate_ids = {str(candidate["chunk_id"]) for candidate in candidates}
        invalid_ids = [chunk_id for chunk_id in selected_ids if chunk_id not in candidate_ids]
        if invalid_ids:
            print(
                "Unknown chunk id(s): "
                + ", ".join(invalid_ids)
                + f". Available ids: {', '.join(sorted(candidate_ids))}"
            )
            continue

        return selected_ids


def _resolve_selection(raw_value: str, candidates: list[dict[str, Any]]) -> list[str]:
    """Resolve comma-separated chunk IDs or row numbers to chunk IDs."""
    tokens = [token.strip() for token in raw_value.split(",")]
    selected_ids: list[str] = []
    for token in tokens:
        if not token:
            continue
        if token.isdigit():
            candidate_index = int(token) - 1
            if candidate_index < 0 or candidate_index >= len(candidates):
                raise ValueError(
                    f"Candidate rank {token} is out of range 1-{len(candidates)}"
                )
            selected_ids.append(str(candidates[candidate_index]["chunk_id"]))
            continue
        selected_ids.append(token)
    return selected_ids


def _print_candidates(candidates: list[dict[str, Any]]) -> None:
    """Display retrieval candidates in a compact review-friendly form."""
    print("Candidates:")
    for index, candidate in enumerate(candidates, start=1):
        metadata = candidate.get("metadata") or {}
        source = metadata.get("source", "unknown source")
        page = metadata.get("page", "unknown page")
        section_title = metadata.get("section_title", "unknown section")
        text = textwrap.shorten(
            str(candidate.get("text", "")),
            width=220,
            placeholder=" ...",
        )
        score_bits = []
        if candidate.get("rrf_score") is not None:
            score_bits.append(f"rrf={candidate['rrf_score']:.6f}")
        if candidate.get("rerank_score") is not None:
            score_bits.append(f"rerank={candidate['rerank_score']:.4f}")
        if candidate.get("bm25_score") is not None:
            score_bits.append(f"bm25={candidate['bm25_score']:.4f}")
        if candidate.get("distance") is not None:
            score_bits.append(f"distance={candidate['distance']:.4f}")
        score_text = ", ".join(score_bits) if score_bits else "no scores"
        print(
            f"  {index:02d}. rank={index} | chunk_id={candidate['chunk_id']} | "
            f"{source} | page {page} | {section_title} | {score_text}"
        )
        print(f"      {text}")


def _write_queries(output_path: Path, queries: list[dict[str, Any]]) -> None:
    """Write annotated queries to disk, preserving previous edits when possible."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_queries = _load_existing_annotations(output_path)
    existing_by_id: dict[str, dict[str, Any]] = {}
    existing_order: list[str] = []
    for query in existing_queries:
        query_id = str(query.get("query_id", "")).strip()
        if not query_id:
            continue
        existing_by_id[query_id] = dict(query)
        existing_order.append(query_id)

    merged_by_id = dict(existing_by_id)
    for query in queries:
        query_id = str(query.get("query_id", "")).strip()
        if not query_id:
            continue
        merged_by_id[query_id] = dict(query)

    ordered_queries: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for query_id in existing_order:
        if query_id in merged_by_id and query_id not in seen_ids:
            ordered_queries.append(merged_by_id[query_id])
            seen_ids.add(query_id)
    for query in queries:
        query_id = str(query.get("query_id", "")).strip()
        if not query_id or query_id in seen_ids:
            continue
        ordered_queries.append(merged_by_id[query_id])
        seen_ids.add(query_id)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(ordered_queries, file, ensure_ascii=False, indent=2)


def _parse_documents(config: Config) -> list[dict[str, Any]]:
    """Parse the configured PDF and Word documents into one section list."""
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


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAnnotation interrupted. Partial progress was preserved.")
