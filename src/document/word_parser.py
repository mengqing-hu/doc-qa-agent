"""Parse Word documents into text and table sections."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

from docx import Document
from docx.document import Document as WordDocument
from docx.oxml.ns import qn
from docx.opc.exceptions import PackageNotFoundError
from docx.table import Table
from docx.text.paragraph import Paragraph


logger = logging.getLogger(__name__)
INTRODUCTION_TITLE = "Introduction"
HEADING_STYLE_PATTERN = re.compile(
    r"^(?:heading|überschrift)(?:\s+(\d+))?\b", re.IGNORECASE
)
TABLE_CAPTION_PATTERN = re.compile(r"^(?:table|tab\.)\b", re.IGNORECASE)
TOC_STYLE_PATTERN = re.compile(r"^(?:toc\s*\d+|table of figures|table of tables)$", re.IGNORECASE)
NON_RETRIEVABLE_HEADING_PATTERN = re.compile(
    r"^(?:table of contents|contents|list of figures|list of tables|list of table)$",
    re.IGNORECASE,
)
NUMBER_PLACEHOLDER_PATTERN = re.compile(r"%(\d+)")
ABSTRACT_HEADING_PATTERN = re.compile(r"^abstract$", re.IGNORECASE)
REFERENCES_HEADING_PATTERN = re.compile(
    r"^(?:\d+(?:\.\d+)*\s+)?(?:references|bibliography|literaturverzeichnis)$",
    re.IGNORECASE,
)


def parse_word_document(document_path: str | Path) -> list[dict[str, Any]]:
    """Parse a Word document into text and Markdown table sections.

    Args:
        document_path: Path to the Word document.

    Returns:
        Sections with title, content, type, and source metadata.
    """
    document_file = Path(document_path)
    if not document_file.is_file():
        raise FileNotFoundError(f"Word document was not found: {document_file}")

    # Open Word document
    try:
        document = Document(document_file)
    except (BadZipFile, OSError, PackageNotFoundError) as error:
        logger.error("Failed to open Word document: %s", document_file)
        raise ValueError(f"Unable to parse Word document: {document_file}") from error

    source_name = document_file.name
    heading_numbers = _HeadingNumberResolver(document)

    # Parse document body in physical order.
    sections = _parse_document_body(document, source_name, heading_numbers)

    logger.info("Parsed %s into %d section(s)", source_name, len(sections))
    return sections


def _parse_document_body(
    document: WordDocument,
    source_name: str,
    heading_numbers: "_HeadingNumberResolver",
) -> list[dict[str, Any]]:
    """Build text and data-table sections in Word body order."""
    sections: list[dict[str, Any]] = []
    current_title = INTRODUCTION_TITLE
    current_heading_level = 0
    heading_path: list[str] = []
    current_paragraphs: list[str] = []
    pending_table_caption: str | None = None
    content_started = False
    content_ended = False

    for body_element in document.element.body.iterchildren():
        if body_element.tag.endswith("}p"):
            paragraph = Paragraph(body_element, document)
            paragraph_text = paragraph.text.strip()
            if not paragraph_text:
                continue

            if _is_non_retrievable_paragraph(paragraph):
                continue

            is_abstract_heading = _is_abstract_heading(paragraph_text)
            if not content_started:
                if not is_abstract_heading:
                    continue
                content_started = True

            if _is_references_heading(paragraph_text):
                _append_text_section(
                    sections,
                    current_title,
                    current_paragraphs,
                    source_name,
                    heading_level=current_heading_level,
                    heading_path=heading_path,
                )
                content_ended = True
                break

            heading_level = _heading_level(paragraph)
            if heading_level is not None or is_abstract_heading:
                _append_text_section(
                    sections,
                    current_title,
                    current_paragraphs,
                    source_name,
                    heading_level=current_heading_level,
                    heading_path=heading_path,
                )
                current_paragraphs = []
                pending_table_caption = None
                if NON_RETRIEVABLE_HEADING_PATTERN.fullmatch(paragraph_text):
                    continue

                resolved_heading_level = heading_level or 1
                heading_number = (
                    heading_numbers.next_number(paragraph, resolved_heading_level)
                    if heading_level is not None
                    else None
                )
                displayed_heading = _display_heading(paragraph_text, heading_number)
                heading_path = _update_heading_path(
                    heading_path, resolved_heading_level, displayed_heading
                )
                current_title = displayed_heading
                current_heading_level = resolved_heading_level
                continue

            # Remember a caption only for the following data table
            if (
                paragraph.style.name == "Caption"
                and TABLE_CAPTION_PATTERN.match(paragraph_text)
            ):
                pending_table_caption = paragraph_text
                continue

            current_paragraphs.append(paragraph_text)
            pending_table_caption = None
            continue

        if not body_element.tag.endswith("}tbl"):
            continue

        if not content_started:
            continue

        table = Table(body_element, document)

        # Skip layout tables used for formulas and visual alignment
        if len(table.rows) <= 2 or len(table.columns) <= 1:
            logger.debug(
                "Skipped layout table with %d row(s) and %d column(s)",
                len(table.rows),
                len(table.columns),
            )
            continue

        _append_text_section(
            sections,
            current_title,
            current_paragraphs,
            source_name,
            heading_level=current_heading_level,
            heading_path=heading_path,
        )
        current_paragraphs = []
        table_number = sum(section["type"] == "table" for section in sections) + 1
        table_title = pending_table_caption or current_title or f"Table {table_number}"
        sections.append(
            {
                "title": table_title,
                "content": _with_heading_context(
                    _table_to_markdown(table), heading_path
                ),
                "type": "table",
                "source": source_name,
                "heading_level": current_heading_level,
                "heading_path": " > ".join(heading_path),
                "heading_number": _heading_number_from_title(current_title),
            }
        )
        pending_table_caption = None

    _append_text_section(
        sections,
        current_title,
        current_paragraphs if not content_ended else [],
        source_name,
        heading_level=current_heading_level,
        heading_path=heading_path,
    )
    if not content_started:
        logger.warning("No Abstract heading was found; no Word sections were retained")
    return sections


def _append_text_section(
    sections: list[dict[str, Any]],
    title: str,
    paragraphs: list[str],
    source_name: str,
    *,
    heading_level: int,
    heading_path: list[str],
) -> None:
    """Append a text section only when it contains paragraph content."""
    if not paragraphs:
        return

    sections.append(
        {
            "title": title,
            "content": _with_heading_context("\n\n".join(paragraphs), heading_path),
            "type": "text",
            "source": source_name,
            "heading_level": heading_level,
            "heading_path": " > ".join(heading_path),
            "heading_number": _heading_number_from_title(title),
        }
    )


def _is_non_retrievable_paragraph(paragraph: Paragraph) -> bool:
    """Return whether a paragraph belongs to a generated Word contents list."""
    return bool(TOC_STYLE_PATTERN.fullmatch(paragraph.style.name.strip()))


def _is_abstract_heading(text: str) -> bool:
    """Return whether a paragraph marks the first retrievable Word section."""
    return bool(ABSTRACT_HEADING_PATTERN.fullmatch(text.strip()))


def _is_references_heading(text: str) -> bool:
    """Return whether a paragraph marks the exclusive end of indexable content."""
    return bool(REFERENCES_HEADING_PATTERN.fullmatch(text.strip()))


def _heading_level(paragraph: Paragraph) -> int | None:
    """Return a paragraph's outline level or a supported heading-style level."""
    outline_level = _outline_level(paragraph)
    if outline_level is not None:
        return outline_level

    style_name = paragraph.style.name.strip()
    matched_style = HEADING_STYLE_PATTERN.match(style_name)
    if not matched_style:
        return None
    level = matched_style.group(1)
    return int(level) if level is not None else 1


def _outline_level(paragraph: Paragraph) -> int | None:
    """Read a direct or style-defined Word outline level when available."""
    paragraph_properties = paragraph._p.pPr
    if paragraph_properties is None:
        paragraph_properties = paragraph.style._element.pPr
    if paragraph_properties is None:
        return None

    outline_level = paragraph_properties.find(qn("w:outlineLvl"))
    if outline_level is None:
        return None
    value = outline_level.get(qn("w:val"))
    try:
        return int(value) + 1
    except (TypeError, ValueError):
        return None


def _update_heading_path(
    heading_path: list[str], heading_level: int, heading_text: str
) -> list[str]:
    """Replace the current heading at its level and discard deeper headings."""
    prefix = heading_path[: max(heading_level - 1, 0)]
    return [*prefix, heading_text]


def _with_heading_context(content: str, heading_path: list[str]) -> str:
    """Prefix indexable content with its complete document heading path."""
    if not heading_path:
        return content
    return f"Section: {' > '.join(heading_path)}\n\n{content}"


def _display_heading(heading_text: str, heading_number: str | None) -> str:
    """Add a Word-rendered multilevel number to a heading when available."""
    return f"{heading_number} {heading_text}" if heading_number else heading_text


def _heading_number_from_title(title: str) -> str | None:
    """Return a leading hierarchical heading number, if the title has one."""
    matched_number = re.match(r"^(\d+(?:\.\d+)*)\s+", title)
    return matched_number.group(1) if matched_number else None


class _HeadingNumberResolver:
    """Render Word multilevel numbering for heading paragraphs in document order."""

    def __init__(self, document: WordDocument) -> None:
        self._definitions = _numbering_definitions(document)
        self._style_numbering_ids = _style_numbering_ids(document)
        self._counters: dict[str, list[int]] = {}

    def next_number(self, paragraph: Paragraph, heading_level: int) -> str | None:
        """Advance the linked numbering list and return its displayed label."""
        numbering_id = self._numbering_id(paragraph)
        if numbering_id is None:
            return None
        levels = self._definitions.get(numbering_id)
        level_index = heading_level - 1
        if levels is None or level_index not in levels:
            return None

        counters = self._counters.setdefault(numbering_id, [0] * 9)
        start, level_text = levels[level_index]
        counters[level_index] = counters[level_index] + 1 if counters[level_index] else start
        for index in range(level_index + 1, len(counters)):
            counters[index] = 0

        return NUMBER_PLACEHOLDER_PATTERN.sub(
            lambda match: str(counters[int(match.group(1)) - 1]), level_text
        )

    def _numbering_id(self, paragraph: Paragraph) -> str | None:
        """Find a paragraph's numbering ID through direct or style properties."""
        direct_numbering_id = _numbering_id_from_properties(paragraph._p.pPr)
        if direct_numbering_id is not None:
            return direct_numbering_id

        style = paragraph.style
        while style is not None:
            style_numbering_id = _numbering_id_from_properties(style._element.pPr)
            if style_numbering_id is not None:
                return style_numbering_id
            style = style.base_style

        style = paragraph.style
        while style is not None:
            mapped_numbering_id = self._style_numbering_ids.get(style.style_id)
            if mapped_numbering_id is not None:
                return mapped_numbering_id
            style = style.base_style
        return None


def _numbering_definitions(
    document: WordDocument,
) -> dict[str, dict[int, tuple[int, str]]]:
    """Read numbering starts and display templates from Word numbering XML."""
    numbering = document.part.numbering_part.element
    abstract_numbers = {
        abstract.get(qn("w:abstractNumId")): abstract
        for abstract in numbering.findall(qn("w:abstractNum"))
    }
    definitions: dict[str, dict[int, tuple[int, str]]] = {}
    for number in numbering.findall(qn("w:num")):
        numbering_id = number.get(qn("w:numId"))
        abstract_number = number.find(qn("w:abstractNumId"))
        if numbering_id is None or abstract_number is None:
            continue
        abstract = abstract_numbers.get(abstract_number.get(qn("w:val")))
        if abstract is None:
            continue
        levels: dict[int, tuple[int, str]] = {}
        for level in abstract.findall(qn("w:lvl")):
            level_index = level.get(qn("w:ilvl"))
            start = level.find(qn("w:start"))
            level_text = level.find(qn("w:lvlText"))
            if level_index is None or start is None or level_text is None:
                continue
            try:
                levels[int(level_index)] = (
                    int(start.get(qn("w:val"))),
                    str(level_text.get(qn("w:val"))),
                )
            except (TypeError, ValueError):
                continue
        if levels:
            definitions[numbering_id] = levels
    return definitions


def _style_numbering_ids(document: WordDocument) -> dict[str, str]:
    """Map heading style IDs to the numbering definitions that reference them."""
    numbering = document.part.numbering_part.element
    abstract_numbers = {
        abstract.get(qn("w:abstractNumId")): abstract
        for abstract in numbering.findall(qn("w:abstractNum"))
    }
    style_numbers: dict[str, str] = {}
    for number in numbering.findall(qn("w:num")):
        numbering_id = number.get(qn("w:numId"))
        abstract_number = number.find(qn("w:abstractNumId"))
        if numbering_id is None or abstract_number is None:
            continue
        abstract = abstract_numbers.get(abstract_number.get(qn("w:val")))
        if abstract is None:
            continue
        for level in abstract.findall(qn("w:lvl")):
            style = level.find(qn("w:pStyle"))
            if style is not None and style.get(qn("w:val")):
                style_numbers[str(style.get(qn("w:val")))] = numbering_id
    return style_numbers


def _numbering_id_from_properties(properties: Any) -> str | None:
    """Return a numbering ID from a paragraph or style property element."""
    if properties is None or properties.numPr is None or properties.numPr.numId is None:
        return None
    value = properties.numPr.numId.val
    return str(value) if value is not None else None


def _table_to_markdown(table: Table) -> str:
    """Convert a Word table into Markdown while preserving row boundaries."""
    rows = [
        [_normalize_cell_text(cell.text) for cell in row.cells]
        for row in table.rows
    ]
    if not rows or not rows[0]:
        return ""

    # Create Markdown header
    header = rows[0]
    markdown_lines = [
        f"| {' | '.join(header)} |",
        f"| {' | '.join('---' for _ in header)} |",
    ]

    # Create Markdown body
    for row in rows[1:]:
        markdown_lines.append(f"| {' | '.join(row)} |")

    return "\n".join(markdown_lines)


def _normalize_cell_text(cell_text: str) -> str:
    """Normalize cell content so it remains valid inside a Markdown table."""
    return cell_text.strip().replace("|", "\\|").replace("\n", "<br>")
