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
LAYOUT_MARGIN_RATIO = 0.12
MIN_REPEATED_LAYOUT_PAGES = 3
TABLE_CONTINUATION_MARGIN_RATIO = 0.15
TOC_HEADING_PATTERN = re.compile(
    r"^(?:table\s+of\s+contents|contents|list\s+of\s+figures|list\s+of\s+tables)$",
    re.IGNORECASE,
)
TOC_ENTRY_PATTERN = re.compile(r"(?:\.{3,}|…{2,})\s*\d+\s*$")
PAGE_NUMBER_PATTERN = re.compile(
    r"^(?:page\s+)?[-–—]?\s*\d+\s*[-–—]?$", re.IGNORECASE
)
SIGNATURE_LINE_PATTERN = re.compile(r"^[\s._\-–—]{8,}$")
DOT_LEADER_PATTERN = re.compile(r"^(?:\s*[.\u2026·]{3,}\s*)+$")
ABSTRACT_HEADING_PATTERN = re.compile(r"^abstract$", re.IGNORECASE)
REFERENCES_HEADING_PATTERN = re.compile(
    r"^(?:\d+(?:\.\d+)*\s+)?(?:references|bibliography|literaturverzeichnis)$",
    re.IGNORECASE,
)
TABLE_CAPTION_PATTERN = re.compile(r"^table\s+\d+(?:\.\d+)*\b", re.IGNORECASE)


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
    """Extract text and table sections in their original page order."""
    sections: list[dict[str, Any]] = []
    current_title = INTRODUCTION_TITLE
    current_page = 1 - page_number_offset
    current_lines: list[str] = []
    previous_page_last_table: dict[str, Any] | None = None
    table_count = 0
    extracted_pages: list[tuple[int, float, list[dict[str, Any]]]] = []
    content_started = False
    content_ended = False

    for physical_page_number, page in enumerate(pages, start=1):
        printed_page_number = physical_page_number - page_number_offset
        extracted_pages.append(
            (
                printed_page_number,
                float(page.height),
                _extract_lines(page, title_font_size_threshold),
            )
        )

    repeated_layout_lines = _detect_repeated_layout_lines(extracted_pages)

    for (physical_page_number, page), (
        printed_page_number,
        page_height,
        page_lines,
    ) in zip(enumerate(pages, start=1), extracted_pages, strict=True):
        filtered_lines = _filter_page_lines(
            page_lines,
            page_height=page_height,
            repeated_layout_lines=repeated_layout_lines,
        )
        if _is_table_of_contents_page(filtered_lines):
            logger.debug("Skipped PDF contents page %d", physical_page_number)
            continue

        table_events = _extract_table_events(page)
        content_events = [
            (float(line["top"]), "line", line)
            for line in filtered_lines
            if not _line_belongs_to_table(line, table_events)
        ]
        content_events.extend(
            (float(table_event["top"]), "table", table_event)
            for table_event in table_events
        )

        for _, event_type, event in sorted(content_events, key=lambda item: item[0]):
            if event_type == "table":
                if not content_started:
                    continue
                table_caption = _extract_table_caption(current_lines)
                _append_text_section(
                    sections,
                    current_title,
                    current_lines,
                    current_page,
                    source_name,
                )
                current_lines = []
                previous_page_last_table, table_count = _append_table_section(
                    sections,
                    event,
                    printed_page_number,
                    page_height,
                    source_name,
                    previous_page_last_table,
                    table_count,
                    table_caption=table_caption,
                )
                continue

            text = str(event["text"])
            is_abstract_heading = _is_abstract_heading(text)
            if not content_started:
                if not is_abstract_heading:
                    continue
                content_started = True

            if _is_references_heading(text):
                _append_text_section(
                    sections,
                    current_title,
                    current_lines,
                    current_page,
                    source_name,
                )
                current_lines = []
                content_ended = True
                break

            if event["is_title"] or is_abstract_heading:
                _append_text_section(
                    sections,
                    current_title,
                    current_lines,
                    current_page,
                    source_name,
                )
                current_title = text
                current_page = printed_page_number
                current_lines = []
            else:
                current_lines.append(text)

        if content_ended:
            break

    if not content_ended:
        _append_text_section(
            sections,
            current_title,
            current_lines,
            current_page,
            source_name,
        )
    if not content_started:
        logger.warning("No Abstract heading was found; no PDF sections were retained")
    return sections


def _extract_lines(page: Page, title_threshold: float) -> list[dict[str, Any]]:
    """Group PDF characters into lines and identify bold large-font titles."""
    line_groups: list[list[dict[str, Any]]] = []

    # Group characters by vertical position
    for character in sorted(page.chars, key=lambda item: (item["top"], item["x0"])):
        if not line_groups or abs(character["top"] - line_groups[-1][0]["top"]) > LINE_POSITION_TOLERANCE:
            line_groups.append([character])
        else:
            line_groups[-1].append(character)

    lines: list[dict[str, Any]] = []
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
                "top": min(float(character["top"]) for character in characters),
                "bottom": max(float(character["bottom"]) for character in characters),
                "is_title": largest_font_size > title_threshold and is_bold,
            }
        )

    return lines


def _detect_repeated_layout_lines(
    extracted_pages: list[tuple[int, float, list[dict[str, Any]]]],
) -> set[str]:
    """Find identical lines repeated in page header or footer regions."""
    occurrences: dict[str, set[int]] = {}
    for page_number, page_height, lines in extracted_pages:
        for line in lines:
            if not _is_layout_line(line, page_height):
                continue
            key = _normalize_layout_line(str(line["text"]))
            if key:
                occurrences.setdefault(key, set()).add(page_number)

    return {
        key
        for key, page_numbers in occurrences.items()
        if len(page_numbers) >= MIN_REPEATED_LAYOUT_PAGES
    }


def _filter_page_lines(
    lines: list[dict[str, Any]],
    *,
    page_height: float,
    repeated_layout_lines: set[str],
) -> list[dict[str, Any]]:
    """Remove repeated layout text and line-level PDF presentation noise."""
    filtered_lines: list[dict[str, Any]] = []
    for line in lines:
        text = str(line["text"]).strip()
        normalized_text = _normalize_layout_line(text)
        if normalized_text in repeated_layout_lines:
            continue
        if _is_noise_line(text):
            continue
        filtered_lines.append(line)
    return filtered_lines


def _is_layout_line(line: dict[str, Any], page_height: float) -> bool:
    """Return whether a line lies in a conventional header or footer region."""
    return (
        float(line["top"]) <= page_height * LAYOUT_MARGIN_RATIO
        or float(line["bottom"]) >= page_height * (1 - LAYOUT_MARGIN_RATIO)
    )


def _normalize_layout_line(text: str) -> str:
    """Normalize whitespace and case for repeated-layout comparison."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _is_noise_line(text: str) -> bool:
    """Identify standalone page numbers, signature lines, and leader lines."""
    if not text:
        return True
    return bool(
        PAGE_NUMBER_PATTERN.fullmatch(text)
        or SIGNATURE_LINE_PATTERN.fullmatch(text)
        or DOT_LEADER_PATTERN.fullmatch(text)
    )


def _is_table_of_contents_page(lines: list[dict[str, Any]]) -> bool:
    """Detect contents pages without removing ordinary dotted body text."""
    texts = [str(line["text"]).strip() for line in lines]
    has_contents_heading = any(TOC_HEADING_PATTERN.fullmatch(text) for text in texts)
    toc_entry_count = sum(bool(TOC_ENTRY_PATTERN.search(text)) for text in texts)
    return has_contents_heading or toc_entry_count >= 2


def _is_abstract_heading(text: str) -> bool:
    """Return whether a line marks the first retrievable PDF section."""
    return bool(ABSTRACT_HEADING_PATTERN.fullmatch(text.strip()))


def _is_references_heading(text: str) -> bool:
    """Return whether a line marks the exclusive end of retrievable content."""
    return bool(REFERENCES_HEADING_PATTERN.fullmatch(text.strip()))


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


def _extract_table_events(page: Page) -> list[dict[str, Any]]:
    """Extract table rows with their page positions for ordered insertion."""
    return [
        {
            "top": float(table.bbox[1]),
            "bottom": float(table.bbox[3]),
            "rows": _normalize_table_rows(table.extract() or []),
        }
        for table in page.find_tables()
    ]


def _line_belongs_to_table(
    line: dict[str, Any], table_events: list[dict[str, Any]]
) -> bool:
    """Return whether a line is vertically contained by an extracted table."""
    line_midpoint = (float(line["top"]) + float(line["bottom"])) / 2
    return any(
        float(table_event["top"]) <= line_midpoint <= float(table_event["bottom"])
        for table_event in table_events
    )


def _append_table_section(
    sections: list[dict[str, Any]],
    table_event: dict[str, Any],
    page_number: int,
    page_height: float,
    source_name: str,
    previous_page_last_table: dict[str, Any] | None,
    table_count: int,
    *,
    table_caption: str | None,
) -> tuple[dict[str, Any] | None, int]:
    """Insert one table or merge it into its immediately preceding continuation."""
    rows = list(table_event["rows"])
    if not rows:
        return previous_page_last_table, table_count

    if (
        previous_page_last_table is not None
        and int(previous_page_last_table["end_page"]) == page_number - 1
        and float(previous_page_last_table["bottom"]) >= page_height * (1 - TABLE_CONTINUATION_MARGIN_RATIO)
        and float(table_event["top"]) <= page_height * TABLE_CONTINUATION_MARGIN_RATIO
        and _has_matching_column_count(previous_page_last_table["rows"], rows)
    ):
        _merge_table_rows(previous_page_last_table["rows"], rows)
        previous_page_last_table["end_page"] = page_number
        previous_page_last_table["bottom"] = float(table_event["bottom"])
        previous_page_last_table["section"]["content"] = _table_content(
            previous_page_last_table["rows"],
            previous_page_last_table["caption"],
        )
        return previous_page_last_table, table_count

    table_count += 1
    section = {
        "title": table_caption or f"Table {table_count}",
        "content": _table_content(rows, table_caption),
        "type": "table",
        "page": page_number,
        "source": source_name,
    }
    sections.append(section)
    return {
        "rows": rows,
        "end_page": page_number,
        "bottom": float(table_event["bottom"]),
        "caption": table_caption,
        "section": section,
    }, table_count


def _extract_table_caption(lines: list[str]) -> str | None:
    """Remove and return the table caption immediately before a table."""
    if not lines:
        return None
    candidate = lines[-1].strip()
    if not TABLE_CAPTION_PATTERN.match(candidate):
        return None
    lines.pop()
    return candidate


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


def _table_content(rows: list[list[str]], table_caption: str | None) -> str:
    """Include a table caption in the indexable Markdown representation."""
    markdown = _table_to_markdown(rows)
    return f"Table: {table_caption}\n\n{markdown}" if table_caption else markdown
