"""Generate one document-level summary chunk per indexed source document."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.core.config import Config
from src.generation.llm import LLM


DEFAULT_SUMMARY_MAX_TOKENS = 1024
SUMMARY_PROMPT = """You are creating a document-level summary for a retrieval index.

Write a concise, comprehensive summary of the document below that captures
its purpose, scope, and main points, so a reader can tell what the document
covers without reading it in full. Base the summary only on the text
provided; do not add information that is not present in it.

Document:
{document_text}
"""


def generate_document_summaries(
    chunks: Sequence[dict[str, Any]],
    config: Config | None = None,
    *,
    llm: LLM | None = None,
) -> list[dict[str, Any]]:
    """Return one summary chunk per distinct source document in `chunks`."""
    summarizer_config = config if config is not None else Config()
    summarizer_llm = llm if llm is not None else LLM(summarizer_config)

    chunks_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        source = str(chunk.get("metadata", {}).get("source") or "document")
        chunks_by_source[source].append(chunk)

    summaries: list[dict[str, Any]] = []
    for source, source_chunks in chunks_by_source.items():
        document_text = "\n\n".join(chunk.get("text", "") for chunk in source_chunks)
        prompt = SUMMARY_PROMPT.format(document_text=document_text)
        summary_text = summarizer_llm.generate(
            prompt, max_tokens=DEFAULT_SUMMARY_MAX_TOKENS
        ).strip()
        source_id = Path(source).stem or "document"
        summaries.append(
            {
                "chunk_id": f"{source_id}_summary",
                "text": summary_text,
                "metadata": {
                    "source": source,
                    "page": None,
                    "section_title": "Document Summary",
                    "chunk_type": "document_summary",
                    "chunk_index": 0,
                },
            }
        )
    return summaries
