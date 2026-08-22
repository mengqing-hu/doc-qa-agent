"""Build or refresh the persistent Chroma document index."""

from __future__ import annotations

from src.core.config import Config
from src.core.logger import setup_logging
from src.pipeline.indexing import build_document_index


def main() -> None:
    """Build the configured document index and report its chunk count."""
    config = Config()
    setup_logging(config)
    chunk_count = build_document_index(config)
    print(f"Indexed {chunk_count} chunk(s) into the configured Chroma collection.")


if __name__ == "__main__":
    main()
