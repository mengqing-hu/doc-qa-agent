"""Parse PDF documents into text and table sections."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pdfplumber
from pdfminer.pdfparser import PDFSyntaxError
from pdfplumber.page import Page

from src.core.config import Config


logger = logging.getLogger(__name__)
INTRODUCTION_TITLE = "Introduction"
LINE_POSITION_TOLERANCE = 2.0


def parse_pdf_document(
    document_path: str | Path,
    config: Config,
) -> list[dict[str, str | int]]:
    """Parse a PDF document into text and Markdown table sections.

    Args:
        document_path: Path to the PDF document.
        config: Loaded project configuration.

    Returns:
        Sections with title, content, type, page, and source metadata.
    """
    document_file = Path(document_path)
    if not document_file.is_file():
        raise FileNotFoundError(f"PDF document was not found: {document_file}")

    # Read parsing settings
    title_font_size_threshold = float(
        config.get("document_parsing", "pdf_title_font_size_threshold", default=12)
    )
    page_number_offset = int(
        config.get("document_parsing", "pdf_page_number_offset", default=0)
    )

    # Open PDF document
    try:
        with pdfplumber.open(document_file) as pdf:
            sections = _parse_pdf_pages(
                pdf.pages,
                document_file.name,
                title_font_size_threshold,
                page_number_offset,
            )
    except (OSError, PDFSyntaxError) as error:
        logger.error("Failed to open PDF document: %s", document_file)
        raise ValueError(f"Unable to parse PDF document: {document_file}") from error

    logger.info("Parsed %s into %d section(s)", document_file.name, len(sections))
    return sections


def _parse_pdf_pages(
    pages: list[Page],
    source_name: str,
    title_font_size_threshold: float,
    page_number_offset: int,
) -> list[dict[str, str | int]]:
    """Extract text sections and collect tables across all PDF pages."""
    text_sections: list[dict[str, str | int]] = []
    table_sections: list[dict[str, Any]] = []
    current_title = INTRODUCTION_TITLE
    current_page = 1 - page_number_offset
    current_lines: list[str] = []
    previous_page_last_table: dict[str, Any] | None = None

    for physical_page_number, page in enumerate(pages, start=1):
        printed_page_number = physical_page_number - page_number_offset

        # Parse text lines
        for line in _extract_lines(page, title_font_size_threshold):
            if line["is_title"]:
                _append_text_section(
                    text_sections,
                    current_title,
                    current_lines,
                    current_page,
                    source_name,
                )
                current_title = str(line["text"])
                current_page = printed_page_number
                current_lines = []
            else:
                current_lines.append(str(line["text"]))

        # Parse page tables
        page_tables = page.extract_tables() or []
        previous_page_last_table = _append_page_tables(
            table_sections,
            page_tables,
            printed_page_number,
            source_name,
            previous_page_last_table,
        )

    _append_text_section(
        text_sections,
        current_title,
        current_lines,
        current_page,
        source_name,
    )
    return text_sections + _render_table_sections(table_sections)


def _extract_lines(page: Page, title_threshold: float) -> list[dict[str, str | bool]]:
    """Group PDF characters into lines and identify bold large-font titles."""
    line_groups: list[list[dict[str, Any]]] = []

    # Group characters by vertical position
    for character in sorted(page.chars, key=lambda item: (item["top"], item["x0"])):
        if not line_groups or abs(character["top"] - line_groups[-1][0]["top"]) > LINE_POSITION_TOLERANCE:
            line_groups.append([character])
        else:
            line_groups[-1].append(character)

    lines: list[dict[str, str | bool]] = []
    for characters in line_groups:
        text = "".join(character["text"] for character in characters).strip()
        if not text:
            continue

        largest_font_size = max(float(character["size"]) for character in characters)
        is_bold = any("bold" in character["fontname"].lower() for character in characters)

        # Restore the separator lost during coordinate-based character joining
        if is_bold and largest_font_size > title_threshold:
            text = re.sub(r"(\d)([A-Z])", r"\1 \2", text)

        lines.append(
            {
                "text": text,
                "is_title": largest_font_size > title_threshold and is_bold,
            }
        )

    return lines


def _append_text_section(
    sections: list[dict[str, str | int]],
    title: str,
    lines: list[str],
    page_number: int,
    source_name: str,
) -> None:
    """Append a text section only when it contains extracted content."""
    if not lines:
        return

    sections.append(
        {
            "title": title,
            "content": "\n".join(lines),
            "type": "text",
            "page": page_number,
            "source": source_name,
        }
    )


def _append_page_tables(
    table_sections: list[dict[str, Any]],
    page_tables: list[list[list[str | None]]],
    page_number: int,
    source_name: str,
    previous_page_last_table: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Append page tables and merge a compatible continuation table."""
    if not page_tables:
        return None

    for table_index, table_rows in enumerate(page_tables):
        normalized_rows = _normalize_table_rows(table_rows)

        # Merge a continuation table from the previous page
        if (
            table_index == 0
            and previous_page_last_table is not None
            and _has_matching_column_count(previous_page_last_table["rows"], normalized_rows)
        ):
            _merge_table_rows(previous_page_last_table["rows"], normalized_rows)
            continue

        table_sections.append(
            {
                "title": f"Table {len(table_sections) + 1}",
                "rows": normalized_rows,
                "page": page_number,
                "source": source_name,
            }
        )

    return table_sections[-1]


def _normalize_table_rows(table_rows: list[list[str | None]]) -> list[list[str]]:
    """Normalize PDF table cells for Markdown rendering."""
    return [
        [((cell or "").strip().replace("|", "\\|").replace("\n", "<br>")) for cell in row]
        for row in table_rows
    ]


def _has_matching_column_count(
    previous_rows: list[list[str]],
    current_rows: list[list[str]],
) -> bool:
    """Check whether two tables can represent one cross-page table."""
    return bool(previous_rows and current_rows and len(previous_rows[0]) == len(current_rows[0]))


def _merge_table_rows(previous_rows: list[list[str]], current_rows: list[list[str]]) -> None:
    """Merge continuation rows while retaining the first table header."""
    if previous_rows[0] == current_rows[0]:
        previous_rows.extend(current_rows[1:])
    else:
        previous_rows.extend(current_rows)


def _render_table_sections(
    table_sections: list[dict[str, Any]],
) -> list[dict[str, str | int]]:
    """Convert collected table rows into final Markdown sections."""
    return [
        {
            "title": table["title"],
            "content": _table_to_markdown(table["rows"]),
            "type": "table",
            "page": table["page"],
            "source": table["source"],
        }
        for table in table_sections
    ]


def _table_to_markdown(rows: list[list[str]]) -> str:
    """Render normalized table rows as a Markdown table."""
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
