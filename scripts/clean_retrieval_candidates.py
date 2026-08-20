"""Remove exported retrieval candidates from annotated evaluation queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RETRIEVAL_CANDIDATES_FIELD = "retrieval_candidates"


def main() -> None:
    """Remove retrieval candidate payloads and write the cleaned query file."""
    arguments = _parse_arguments()
    queries = _load_queries(arguments.input)
    cleaned_queries, removed_count = _remove_retrieval_candidates(queries)
    output_path = arguments.input if arguments.in_place else arguments.output
    if output_path is None:
        raise ValueError("An output path is required when --in-place is not set")

    _write_queries(output_path, cleaned_queries)
    print(
        f"Removed {RETRIEVAL_CANDIDATES_FIELD} from {removed_count} query record(s)."
    )
    print(f"Saved cleaned queries to {output_path}")


def _parse_arguments() -> argparse.Namespace:
    """Read input and output settings for candidate cleanup."""
    parser = argparse.ArgumentParser(
        description="Remove retrieval_candidates fields from annotated query JSON."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Annotated query JSON containing optional retrieval_candidates fields.",
    )
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument(
        "--output",
        type=Path,
        help="New JSON path for the cleaned query records.",
    )
    output_group.add_argument(
        "--in-place",
        action="store_true",
        help="Replace the input file with its cleaned content.",
    )
    return parser.parse_args()


def _load_queries(input_path: Path) -> list[dict[str, Any]]:
    """Load and validate a JSON list of query records."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file was not found: {input_path}")

    with input_path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError("Input file must contain a JSON list of query objects")
    return [dict(query) for query in payload]


def _remove_retrieval_candidates(
    queries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Return copied query records without exported candidate payloads."""
    cleaned_queries: list[dict[str, Any]] = []
    removed_count = 0
    for query in queries:
        cleaned_query = dict(query)
        if RETRIEVAL_CANDIDATES_FIELD in cleaned_query:
            del cleaned_query[RETRIEVAL_CANDIDATES_FIELD]
            removed_count += 1
        cleaned_queries.append(cleaned_query)
    return cleaned_queries, removed_count


def _write_queries(output_path: Path, queries: list[dict[str, Any]]) -> None:
    """Write cleaned query records as formatted UTF-8 JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(queries, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
