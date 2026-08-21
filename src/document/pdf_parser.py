"""Parse PDF documents through the MinerU API."""

from __future__ import annotations

from io import BytesIO
import logging
import re
import time
from pathlib import Path
from typing import Any
from uuid import uuid4
from zipfile import BadZipFile, ZipFile, is_zipfile

import requests
import pypdfium2 as pdfium

from src.core.config import Config, get_required_environment_variable


logger = logging.getLogger(__name__)
INTRODUCTION_TITLE = "Introduction"
ABSTRACT_HEADING_PATTERN = re.compile(r"^(?:abstract|zusammenfassung)$", re.IGNORECASE)
REFERENCES_HEADING_PATTERN = re.compile(
    r"^(?:\d+(?:\.\d+)*\s+)?(?:references|bibliography|literaturverzeichnis)$",
    re.IGNORECASE,
)
TOC_HEADING_PATTERN = re.compile(
    r"^(?:\d+(?:\.\d+)*\s+)?(?:table\s+of\s+contents|contents|目录|list\s+of\s+figures|图目录|list\s+of\s+tables|表目录)$",
    re.IGNORECASE,
)
TOC_ENTRY_PATTERN = re.compile(
    r".+(?:\.{3,}|…{2,}|·{3,})\s*\d+\s*$",
    re.IGNORECASE,
)
TABLE_CAPTION_PATTERN = re.compile(r"^table\s+\d+(?:\.\d+)*\b", re.IGNORECASE)
MARKDOWN_HEADING_PATTERN = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")
MARKDOWN_TABLE_DIVIDER_PATTERN = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
)
HTML_TABLE_START_PATTERN = re.compile(r"^\s*<table\b", re.IGNORECASE)
HTML_TABLE_OPEN_PATTERN = re.compile(r"<table\b", re.IGNORECASE)
HTML_TABLE_CLOSE_PATTERN = re.compile(r"</table>", re.IGNORECASE)
IMAGE_LINE_PATTERN = re.compile(r"^!\[(?P<caption>.*)\]\([^)]*\)$")
PAGE_MARKER_PATTERN = re.compile(
    r"^\s*<!--\s*(?P<label>page(?:[\s_-]*(?:number|idx))?)\s*[:=_-]?\s*(?P<number>\d+)\s*-->\s*$",
    re.IGNORECASE,
)
PAGE_BREAK_PATTERN = re.compile(
    r"^\s*<!--\s*page[\s_-]*break\s*-->\s*$",
    re.IGNORECASE,
)
PAGE_NUMBER_PATTERN = re.compile(r"^(?:page\s+)?[-–—]?\s*\d+\s*[-–—]?$", re.IGNORECASE)
SIGNATURE_LINE_PATTERN = re.compile(r"^[\s._\-–—]{8,}$")
DOT_LEADER_PATTERN = re.compile(r"^(?:\s*[.\u2026·]{3,}\s*)+$")
DOT_LEADER_TAIL_PATTERN = re.compile(r"[.\u2026·\s]{3,}\d*\s*$")


class MinerURequestError(RuntimeError):
    """Raised when MinerU cannot return a usable Markdown result."""


def parse_pdf_document(
    document_path: str | Path,
    config: Config,
) -> list[dict[str, Any]]:
    """Parse a PDF into text and table sections using MinerU's cloud API."""
    document_file = Path(document_path)
    if not document_file.is_file():
        raise FileNotFoundError(f"PDF document was not found: {document_file}")

    markdown = _request_mineru_markdown(document_file, config)
    markdown = _add_physical_page_markers(markdown, document_file)
    sections = _parse_mineru_markdown(markdown, document_file.name)
    logger.info("Parsed %s into %d section(s) through MinerU", document_file.name, len(sections))
    return sections


def _request_mineru_markdown(document_file: Path, config: Config) -> str:
    """Upload one PDF to MinerU, wait for parsing, and download its result."""
    settings = config.get("document_parsing", "mineru", default={})
    if not isinstance(settings, dict):
        raise ValueError("document_parsing.mineru must be a mapping")

    base_url = str(settings.get("api_base_url", "https://mineru.net")).rstrip("/")
    token_variable = str(settings.get("token_env_var", "MINERU_API_TOKEN"))
    token = get_required_environment_variable(token_variable)
    request_timeout = float(settings.get("request_timeout_seconds", 30))
    upload_timeout = float(settings.get("upload_timeout_seconds", 300))
    task_timeout = float(settings.get("task_timeout_seconds", 600))
    poll_interval = float(settings.get("poll_interval_seconds", 5))
    if request_timeout <= 0 or upload_timeout <= 0 or task_timeout <= 0 or poll_interval <= 0:
        raise ValueError("MinerU timeout and polling settings must be greater than zero")

    data_id = uuid4().hex
    headers = {"Authorization": f"Bearer {token}"}
    upload_url, batch_id = _request_mineru_upload_url(
        base_url,
        headers,
        document_file,
        data_id,
        request_timeout,
        bool(settings.get("is_ocr", False)),
        str(settings.get("model_version", "vlm")),
    )
    _upload_mineru_file(upload_url, document_file, upload_timeout)
    if batch_id is None:
        raise MinerURequestError("MinerU did not return a parsing batch ID")
    result_url = _wait_for_mineru_result(
        base_url,
        headers,
        batch_id,
        data_id,
        request_timeout,
        task_timeout,
        poll_interval,
    )
    return _download_mineru_markdown(result_url, request_timeout)


def _request_mineru_upload_url(
    base_url: str,
    headers: dict[str, str],
    document_file: Path,
    data_id: str,
    timeout_seconds: float,
    is_ocr: bool,
    model_version: str,
) -> tuple[str, str | None]:
    """Request a pre-signed upload URL from the MinerU cloud API."""
    payload = {
        "files": [{"name": document_file.name, "data_id": data_id, "is_ocr": is_ocr}],
        "model_version": model_version,
    }
    response = _mineru_request(
        "post",
        f"{base_url}/api/v4/file-urls/batch",
        headers=headers,
        json=payload,
        timeout=timeout_seconds,
    )
    data = _mineru_response_data(response, "request an upload URL")
    file_urls = data.get("file_urls") if isinstance(data, dict) else None
    upload_url = _first_url(file_urls)
    if upload_url is None:
        raise MinerURequestError("MinerU did not return an upload URL")
    batch_id = data.get("batch_id") if isinstance(data, dict) else None
    return upload_url, str(batch_id) if batch_id else None


def _upload_mineru_file(upload_url: str, document_file: Path, timeout_seconds: float) -> None:
    """Upload a local PDF to the pre-signed storage URL supplied by MinerU."""
    try:
        with document_file.open("rb") as document_stream:
            response = requests.put(
                upload_url,
                data=document_stream,
                timeout=timeout_seconds,
            )
        response.raise_for_status()
    except (OSError, requests.RequestException) as error:
        raise MinerURequestError(
            f"MinerU file upload failed for {document_file.name}: {error}"
        ) from error


def _wait_for_mineru_result(
    base_url: str,
    headers: dict[str, str],
    batch_id: str,
    data_id: str,
    request_timeout: float,
    task_timeout: float,
    poll_interval: float,
) -> str:
    """Poll a MinerU batch until the submitted document finishes or fails."""
    deadline = time.monotonic() + task_timeout
    endpoints = (
        f"{base_url}/api/v4/extract-results/batch/{batch_id}",
        f"{base_url}/api/v4/extract/task/batch/{batch_id}",
    )
    while time.monotonic() < deadline:
        response = _mineru_task_status_response(endpoints, headers, request_timeout)
        data = _mineru_response_data(response, "read the extraction task status")
        result = _mineru_file_result(data, data_id)
        state = str(result.get("state", "")).lower()
        if state in {"done", "success", "completed"}:
            result_url = _result_download_url(result)
            if result_url is None:
                raise MinerURequestError("MinerU completed parsing without a result download URL")
            return result_url
        if state in {"failed", "error"}:
            detail = str(result.get("err_msg") or result.get("error") or "unknown error")
            raise MinerURequestError(f"MinerU parsing failed: {detail}")
        time.sleep(poll_interval)
    raise MinerURequestError(f"MinerU parsing timed out after {task_timeout:g} seconds")


def _mineru_task_status_response(
    endpoints: tuple[str, ...],
    headers: dict[str, str],
    timeout_seconds: float,
) -> requests.Response:
    """Request task status across MinerU cloud API path versions."""
    for index, endpoint in enumerate(endpoints):
        is_last_endpoint = index == len(endpoints) - 1
        try:
            response = requests.get(endpoint, headers=headers, timeout=timeout_seconds)
            if response.status_code == requests.codes.not_found and not is_last_endpoint:
                continue
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            if is_last_endpoint:
                raise MinerURequestError(f"MinerU task status request failed: {error}") from error
    raise MinerURequestError(
        "MinerU did not recognize the extraction batch. "
        "The upload and task-submission batch IDs may not match."
    )


def _download_mineru_markdown(result_url: str, timeout_seconds: float) -> str:
    """Download and read the Markdown archive generated by MinerU."""
    try:
        response = requests.get(result_url, timeout=timeout_seconds)
        response.raise_for_status()
    except requests.RequestException as error:
        raise MinerURequestError(f"MinerU result download failed: {error}") from error
    return _markdown_from_mineru_response(response)


def _mineru_request(method: str, url: str, **kwargs: Any) -> requests.Response:
    """Send one cloud API request and convert HTTP failures to parser errors."""
    try:
        response = requests.request(method, url, **kwargs)
        response.raise_for_status()
        return response
    except requests.RequestException as error:
        raise MinerURequestError(f"MinerU API request failed: {error}") from error


def _mineru_response_data(response: requests.Response, action: str) -> Any:
    """Validate MinerU's JSON envelope and return its data field."""
    try:
        payload = response.json()
    except ValueError as error:
        raise MinerURequestError(f"MinerU returned invalid JSON while attempting to {action}") from error
    if not isinstance(payload, dict):
        raise MinerURequestError(f"MinerU returned an invalid response while attempting to {action}")
    if payload.get("code", 0) not in (0, "0", None):
        detail = str(payload.get("msg") or payload.get("message") or "unknown error")
        raise MinerURequestError(f"MinerU could not {action}: {detail}")
    if "data" not in payload:
        raise MinerURequestError(f"MinerU returned no data while attempting to {action}")
    return payload["data"]


def _first_url(value: Any) -> str | None:
    """Return the first URL from MinerU's upload URL response."""
    if isinstance(value, str) and value:
        return value
    if isinstance(value, list) and value:
        candidate = value[0]
        if isinstance(candidate, str) and candidate:
            return candidate
        if isinstance(candidate, dict):
            for key in ("url", "file_url", "upload_url"):
                url = candidate.get(key)
                if isinstance(url, str) and url:
                    return url
    return None


def _mineru_file_result(data: Any, data_id: str) -> dict[str, Any]:
    """Find the status record for the PDF submitted in this parser call."""
    if not isinstance(data, dict):
        raise MinerURequestError("MinerU returned an invalid task status payload")
    results = data.get("extract_result") or data.get("results")
    if not isinstance(results, list):
        raise MinerURequestError("MinerU task status did not contain extraction results")
    for result in results:
        if isinstance(result, dict) and str(result.get("data_id")) == data_id:
            return result
    if len(results) == 1 and isinstance(results[0], dict):
        return results[0]
    raise MinerURequestError("MinerU task status did not contain the submitted PDF")


def _result_download_url(result: dict[str, Any]) -> str | None:
    """Return a compatible MinerU ZIP or Markdown result URL."""
    for key in ("full_zip_url", "zip_url", "markdown_url", "md_url"):
        value = result.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _markdown_from_mineru_response(response: requests.Response) -> str:
    """Read one Markdown file from a MinerU ZIP or JSON response."""
    if is_zipfile(BytesIO(response.content)):
        return _markdown_from_zip(response.content)

    try:
        payload = response.json()
    except ValueError as error:
        raise MinerURequestError("MinerU returned neither a ZIP archive nor JSON") from error

    markdown = _find_markdown_value(payload)
    if markdown is None:
        raise MinerURequestError("MinerU response did not contain Markdown content")
    return markdown


def _markdown_from_zip(payload: bytes) -> str:
    """Read the primary Markdown document from MinerU's in-memory ZIP result."""
    try:
        with ZipFile(BytesIO(payload)) as archive:
            candidates = [
                entry
                for entry in archive.infolist()
                if entry.filename.lower().endswith(".md")
                and Path(entry.filename).name.lower() != "readme.md"
            ]
            if not candidates:
                raise MinerURequestError("MinerU ZIP result does not contain Markdown")
            document = max(candidates, key=lambda entry: entry.file_size)
            return archive.read(document).decode("utf-8")
    except (BadZipFile, OSError, UnicodeDecodeError) as error:
        raise MinerURequestError("MinerU ZIP result could not be read") from error


def _find_content_position(markdown: str, text: str, start: int) -> int | None:
    """Find content text despite minor whitespace differences in Markdown."""
    exact_position = markdown.find(text, start)
    if exact_position >= 0:
        return exact_position
    compact_text = re.sub(r"\s+", " ", text).strip()
    if len(compact_text) < 20:
        return None
    whitespace_pattern = r"\s+".join(re.escape(token) for token in compact_text.split())
    match = re.search(whitespace_pattern, markdown[start:])
    if match is None:
        return None
    return start + match.start()


def _add_physical_page_markers(markdown: str, document_file: Path) -> str:
    """Map MinerU text back to one-based physical PDF pages."""
    try:
        document = pdfium.PdfDocument(str(document_file))
    except (OSError, RuntimeError) as error:
        logger.warning("Could not open PDF for physical page mapping: %s", error)
        return markdown

    markdown = PAGE_MARKER_PATTERN.sub("", markdown)
    markdown = PAGE_BREAK_PATTERN.sub("", markdown)
    cursor = 0
    insertions: list[tuple[int, int]] = []
    unmatched_pages: list[int] = []
    for page_index, page in enumerate(document):
        page_text = page.get_textpage().get_text_range()
        snippets = _page_text_snippets(page_text)
        match = _find_page_position(markdown, snippets, cursor)
        if match is None:
            if snippets:
                unmatched_pages.append(page_index + 1)
            continue
        position, matched_length = match
        line_position = markdown.rfind("\n", 0, position) + 1
        insertions.append((line_position, page_index + 1))
        cursor = position + matched_length

    for position, page_number in reversed(insertions):
        markdown = f"{markdown[:position]}<!-- page: {page_number} -->\n{markdown[position:]}"
    if not insertions:
        logger.warning("Could not map MinerU Markdown text to physical PDF pages")
    elif unmatched_pages:
        logger.debug(
            "Could not anchor %d physical page(s) to MinerU Markdown: %s",
            len(unmatched_pages),
            unmatched_pages,
        )
    return markdown


def _page_text_snippets(page_text: str) -> list[str]:
    """Select candidate stable text anchors for page-level matching.

    Table-of-contents style pages are dominated by dot-leader lines
    (``"1 Introduction .......... 5"``) that MinerU's Markdown conversion
    typically reformats or strips, so both the raw and leader-stripped forms
    of each line are offered as candidates.
    """
    candidates: list[str] = []
    seen: set[str] = set()
    for line in page_text.replace("\r", "\n").splitlines():
        candidate = re.sub(r"\s+", " ", line).strip()
        if not candidate or PAGE_NUMBER_PATTERN.fullmatch(candidate):
            continue
        cleaned = DOT_LEADER_TAIL_PATTERN.sub("", candidate).strip()
        for text in (cleaned, candidate):
            if len(text) >= 15 and text not in seen:
                seen.add(text)
                candidates.append(text)
    return candidates


def _find_page_position(
    markdown: str, snippets: list[str], cursor: int
) -> tuple[int, int] | None:
    """Return the earliest markdown match among candidate page snippets."""
    best: tuple[int, int] | None = None
    for snippet in snippets:
        position = _find_content_position(markdown, snippet, cursor)
        if position is None:
            continue
        if best is None or position < best[0]:
            best = (position, len(snippet))
    return best


def _find_markdown_value(value: Any) -> str | None:
    """Find an inline Markdown string in compatible MinerU JSON responses."""
    if isinstance(value, dict):
        for key in ("markdown", "md", "md_content"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate
        for nested_value in value.values():
            markdown = _find_markdown_value(nested_value)
            if markdown is not None:
                return markdown
    if isinstance(value, list):
        for item in value:
            markdown = _find_markdown_value(item)
            if markdown is not None:
                return markdown
    return None


def _parse_mineru_markdown(markdown: str, source_name: str) -> list[dict[str, Any]]:
    """Convert all ordered MinerU Markdown into classified document sections."""
    sections: list[dict[str, Any]] = []
    current_title = INTRODUCTION_TITLE
    current_page: int | str = "?"
    section_page: int | str = current_page
    current_lines: list[str] = []
    current_kind = "front_matter"
    content_started = False
    table_count = 0
    lines = markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    line_index = 0

    while line_index < len(lines):
        line = lines[line_index].strip()
        page_marker = PAGE_MARKER_PATTERN.fullmatch(line)
        if page_marker:
            current_page = _physical_page_number(page_marker)
            if not current_lines:
                section_page = current_page
            line_index += 1
            continue
        if PAGE_BREAK_PATTERN.fullmatch(line):
            if isinstance(current_page, int):
                current_page += 1
                if not current_lines:
                    section_page = current_page
            line_index += 1
            continue
        if not line or _is_noise_line(line):
            if line:
                current_lines.append(line)
            line_index += 1
            continue

        image_caption = _image_line_caption(line)
        if image_caption is not None:
            if image_caption:
                current_lines.append(image_caption)
            line_index += 1
            continue

        heading = _markdown_heading(line)
        if heading is None and _is_toc_heading(line):
            heading = _normalize_markdown_text(line)
        if heading is not None:
            _append_text_section(
                sections,
                current_title,
                current_lines,
                section_page,
                source_name,
                current_kind,
            )
            current_title = heading
            heading_kind = _classify_section_kind(heading)
            if heading_kind == "body" and current_kind in {"references", "appendix"}:
                heading_kind = current_kind
            current_kind = heading_kind
            if _is_abstract_heading(heading):
                content_started = True
            elif not content_started and current_kind == "body":
                current_kind = "front_matter"
            current_lines = []
            section_page = current_page
            line_index += 1
            continue

        if _is_toc_entry(line) and current_kind in {"front_matter", "abstract"}:
            _append_text_section(
                sections,
                current_title,
                current_lines,
                section_page,
                source_name,
                current_kind,
            )
            current_title = "Contents"
            current_kind = "toc"
            current_lines = []
            section_page = current_page

        table_end = _table_end_index(lines, line_index)
        if table_end is not None:
            table_content = "\n".join(lines[line_index : table_end + 1]).strip()
            caption = _extract_table_caption(current_lines)
            _append_text_section(
                sections,
                current_title,
                current_lines,
                section_page,
                source_name,
                current_kind,
            )
            table_count += 1
            table_title = caption or f"Table {table_count}"
            sections.append(
                {
                    "title": table_title,
                    "content": _table_content(table_content, caption),
                    "type": "table",
                    "page": current_page,
                    "source": source_name,
                    "section_kind": current_kind,
                }
            )
            current_lines = []
            section_page = current_page
            line_index = table_end + 1
            continue

        current_lines.append(line)
        line_index += 1

    _append_text_section(
        sections,
        current_title,
        current_lines,
        section_page,
        source_name,
        current_kind,
    )
    if not any(section.get("page") != "?" for section in sections):
        logger.warning(
            "MinerU Markdown contains no recognized page markers; PDF section pages are unknown"
        )
    return sections


def _table_end_index(lines: list[str], start_index: int) -> int | None:
    """Return the final index of a Markdown or HTML table starting at this line."""
    line = lines[start_index].strip()
    if HTML_TABLE_START_PATTERN.match(line):
        depth = len(HTML_TABLE_OPEN_PATTERN.findall(line)) - len(HTML_TABLE_CLOSE_PATTERN.findall(line))
        if depth <= 0:
            return start_index
        for index in range(start_index + 1, len(lines)):
            depth += len(HTML_TABLE_OPEN_PATTERN.findall(lines[index]))
            depth -= len(HTML_TABLE_CLOSE_PATTERN.findall(lines[index]))
            if depth <= 0:
                return index
        return None
    if (
        "|" not in line
        or start_index + 1 >= len(lines)
        or not MARKDOWN_TABLE_DIVIDER_PATTERN.fullmatch(lines[start_index + 1].strip())
    ):
        return None
    index = start_index + 2
    while index < len(lines) and "|" in lines[index]:
        index += 1
    return index - 1


def _append_text_section(
    sections: list[dict[str, Any]],
    title: str,
    lines: list[str],
    page_number: int | str,
    source_name: str,
    section_kind: str,
) -> None:
    """Append one non-empty text section."""
    content = "\n".join(lines).strip()
    if not content:
        return
    sections.append(
        {
            "title": title,
            "content": content,
            "type": "text",
            "page": page_number,
            "source": source_name,
            "section_kind": section_kind,
        }
    )


def _physical_page_number(page_marker: re.Match[str]) -> int:
    """Convert a MinerU page marker to a one-based physical page number."""
    page_number = int(page_marker.group("number"))
    label = page_marker.group("label").casefold().replace("_", "-").replace(" ", "-")
    return page_number + 1 if label.endswith("-idx") or page_number == 0 else page_number


def _classify_section_kind(title: str) -> str:
    """Classify a heading without dropping its content from parsed sections."""
    normalized_title = title.strip()
    if _is_toc_heading(normalized_title):
        return "toc"
    if _is_references_heading(normalized_title):
        return "references"
    if re.match(r"^(?:\d+(?:\.\d+)*\s+)?(?:appendix|annex|anhang)\b", normalized_title, re.IGNORECASE):
        return "appendix"
    if _is_abstract_heading(normalized_title):
        return "abstract"
    return "body"


def _markdown_heading(line: str) -> str | None:
    """Return a normalized ATX Markdown heading when present."""
    matched_heading = MARKDOWN_HEADING_PATTERN.fullmatch(line)
    if matched_heading is None:
        return None
    return _normalize_markdown_text(matched_heading.group(1))


def _extract_table_caption(lines: list[str]) -> str | None:
    """Remove a preceding Table-number caption and return its plain text."""
    if not lines:
        return None
    candidate = _normalize_markdown_text(lines[-1])
    if not TABLE_CAPTION_PATTERN.match(candidate):
        return None
    lines.pop()
    return candidate


def _table_content(table: str, caption: str | None) -> str:
    """Prefix table Markdown or HTML with its searchable caption."""
    return f"Table: {caption}\n\n{table}" if caption else table


def _normalize_markdown_text(text: str) -> str:
    """Strip lightweight Markdown emphasis from headings and captions."""
    return re.sub(r"^(?:\*\*|__)(.*?)(?:\*\*|__)$", r"\1", text.strip()).strip()


def _is_abstract_heading(text: str) -> bool:
    """Return whether this heading starts the retrievable document content."""
    return bool(ABSTRACT_HEADING_PATTERN.fullmatch(text.strip()))


def _is_toc_heading(text: str) -> bool:
    """Return whether a heading starts a non-retrievable contents section."""
    return bool(TOC_HEADING_PATTERN.fullmatch(text.strip()))


def _is_toc_entry(text: str) -> bool:
    """Return whether a line looks like a contents entry with a page number."""
    return bool(TOC_ENTRY_PATTERN.fullmatch(text.strip()))


def _is_references_heading(text: str) -> bool:
    """Return whether this heading ends the retrievable document content."""
    return bool(REFERENCES_HEADING_PATTERN.fullmatch(text.strip()))


def _is_noise_line(text: str) -> bool:
    """Filter residual page numbers and leader lines not removed by MinerU."""
    return bool(
        PAGE_NUMBER_PATTERN.fullmatch(text)
        or SIGNATURE_LINE_PATTERN.fullmatch(text)
        or DOT_LEADER_PATTERN.fullmatch(text)
    )


def _image_line_caption(text: str) -> str | None:
    """Return a standalone image line's caption, or None if not an image line."""
    match = IMAGE_LINE_PATTERN.fullmatch(text)
    if match is None:
        return None
    return _normalize_markdown_text(match.group("caption"))
