"""Parse Word documents into text and table sections."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from zipfile import BadZipFile

from docx import Document
from docx.document import Document as WordDocument
from docx.opc.exceptions import PackageNotFoundError
from docx.table import Table
from docx.text.paragraph import Paragraph


logger = logging.getLogger(__name__)
HEADING_STYLES = {"Heading 1", "Heading 2", "Heading 3"}
INTRODUCTION_TITLE = "Introduction"
TABLE_CAPTION_PATTERN = re.compile(r"^table\b", re.IGNORECASE)


def parse_word_document(document_path: str | Path) -> list[dict[str, str]]:
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

    # Parse document body in physical order
    text_sections, table_sections = _parse_document_body(document, source_name)
    sections = text_sections + table_sections

    logger.info("Parsed %s into %d section(s)", source_name, len(sections))
    return sections


def _parse_document_body(
    document: WordDocument,
    source_name: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Build text and data-table sections in Word body order."""
    text_sections: list[dict[str, str]] = []
    table_sections: list[dict[str, str]] = []
    current_title = INTRODUCTION_TITLE
    current_paragraphs: list[str] = []
    pending_table_caption: str | None = None

    for body_element in document.element.body.iterchildren():
        if body_element.tag.endswith("}p"):
            paragraph = Paragraph(body_element, document)
            paragraph_text = paragraph.text.strip()
            if not paragraph_text:
                continue

            # Remember a caption only for the following data table
            if (
                paragraph.style.name == "Caption"
                and TABLE_CAPTION_PATTERN.match(paragraph_text)
            ):
                pending_table_caption = paragraph_text
                continue

            # Start a section at supported heading levels
            if paragraph.style.name in HEADING_STYLES:
                _append_text_section(
                    text_sections,
                    current_title,
                    current_paragraphs,
                    source_name,
                )
                current_title = paragraph_text
                current_paragraphs = []
                pending_table_caption = None
                continue

            current_paragraphs.append(paragraph_text)
            pending_table_caption = None
            continue

        if not body_element.tag.endswith("}tbl"):
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

        table_number = len(table_sections) + 1
        table_title = pending_table_caption or current_title or f"Table {table_number}"
        table_sections.append(
            {
                "title": table_title,
                "content": _table_to_markdown(table),
                "type": "table",
                "source": source_name,
            }
        )
        pending_table_caption = None

    _append_text_section(text_sections, current_title, current_paragraphs, source_name)
    return text_sections, table_sections


def _append_text_section(
    sections: list[dict[str, str]],
    title: str,
    paragraphs: list[str],
    source_name: str,
) -> None:
    """Append a text section only when it contains paragraph content."""
    if not paragraphs:
        return

    sections.append(
        {
            "title": title,
            "content": "\n\n".join(paragraphs),
            "type": "text",
            "source": source_name,
        }
    )


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
