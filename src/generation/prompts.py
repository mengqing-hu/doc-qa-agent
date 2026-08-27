"""Build grounded question-answering prompts from retrieved chunks."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.core.config import PROJECT_ROOT, Config


DEFAULT_PROMPT_PATH = "prompts/qa_prompt.md"
DEFAULT_MAX_CONTEXT_CHARACTERS = 10_000


class PromptTemplateError(ValueError):
    """Raised when a prompt template cannot be used safely."""


class PromptBuilder:
    """Render a question-answering template with ranked document evidence."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        template: str | None = None,
    ) -> None:
        """Load the configured prompt template, or use an injected test template."""
        self.config = config if config is not None else Config()
        self.max_context_characters = int(
            self.config.get(
                "generation",
                "max_context_characters",
                default=DEFAULT_MAX_CONTEXT_CHARACTERS,
            )
        )
        if self.max_context_characters <= 0:
            raise ValueError("generation.max_context_characters must be greater than zero")

        self.template = template if template is not None else self._load_template()
        _validate_template(self.template)

    def build(
        self,
        query: str,
        chunks: Sequence[dict[str, Any]],
        *,
        max_context_characters: int | None = None,
        group_fairly_by_source: bool = False,
    ) -> str:
        """Return a prompt containing the highest-ranked chunks within its budget.

        `group_fairly_by_source` interleaves chunks by their `metadata.source`
        before applying the budget, so one large document cannot exhaust the
        budget before a smaller one gets any chunks in at all. Use it only
        when the chunk order carries no relevance ranking to preserve (e.g.
        a metadata-filtered structural match); leave it off for the default
        similarity-ranked path, where the existing order is the priority.
        """
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")

        budget = (
            max_context_characters
            if max_context_characters is not None
            else self.max_context_characters
        )
        if budget <= 0:
            raise ValueError("max_context_characters must be greater than zero")
        ordered_chunks = (
            _interleave_by_source(chunks) if group_fairly_by_source else chunks
        )
        context = _build_context(ordered_chunks, budget)
        return self.template.format(context=context, query=normalized_query)

    def _load_template(self) -> str:
        """Read the prompt file from the configured project-relative path."""
        configured_path = Path(
            self.config.get("generation", "prompt_path", default=DEFAULT_PROMPT_PATH)
        )
        prompt_path = (
            configured_path
            if configured_path.is_absolute()
            else PROJECT_ROOT / configured_path
        )
        try:
            return prompt_path.read_text(encoding="utf-8")
        except OSError as error:
            raise PromptTemplateError(
                f"Prompt template could not be read: {prompt_path}"
            ) from error


def _validate_template(template: str) -> None:
    """Ensure the template includes both required RAG variables."""
    missing_variables = [
        variable
        for variable in ("{context}", "{query}")
        if variable not in template
    ]
    if missing_variables:
        raise PromptTemplateError(
            "Prompt template is missing required variable(s): "
            + ", ".join(missing_variables)
        )


def _interleave_by_source(
    chunks: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Round-robin chunks across their source documents.

    Preserves each source's internal ordering, but takes one chunk per
    source in turn — so the budget-truncation loop that follows sees every
    source early, instead of exhausting the budget on whichever source
    happens to come first in the input order.
    """
    chunks_by_source: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        source = str((chunk.get("metadata") or {}).get("source", "unknown source"))
        chunks_by_source.setdefault(source, []).append(chunk)

    interleaved: list[dict[str, Any]] = []
    for group in itertools.zip_longest(*chunks_by_source.values()):
        interleaved.extend(chunk for chunk in group if chunk is not None)
    return interleaved


def _build_context(
    chunks: Sequence[dict[str, Any]],
    max_characters: int,
) -> str:
    """Format ranked chunks until the configured context budget is reached."""
    context_parts: list[str] = []
    remaining_characters = max_characters
    for chunk in chunks:
        formatted_chunk = _format_chunk(chunk)
        if len(formatted_chunk) > remaining_characters:
            break
        context_parts.append(formatted_chunk)
        remaining_characters -= len(formatted_chunk)

    if not context_parts:
        return "No retrieved context is available."
    return "\n\n".join(context_parts)


def _format_chunk(chunk: dict[str, Any]) -> str:
    """Attach source metadata so the model can cite an evidence identifier."""
    metadata = chunk.get("metadata") or {}
    source = metadata.get("source", "unknown source")
    page = metadata.get("page", "unknown page")
    section_title = metadata.get("section_title", "unknown section")
    return (
        f"[Chunk ID: {chunk.get('chunk_id', 'unknown')}]\n"
        f"Source: {source} | Page: {page} | Section: {section_title}\n"
        f"Content:\n{chunk.get('text', '')}"
    )
