"""Wrap the Tavily web search API as a ReAct tool."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from langchain_tavily import TavilySearch

from src.agent.state import Evidence
from src.core.config import Config, get_required_environment_variable


DEFAULT_MAX_RESULTS = 5


class WebSearchTool:
    """Query the public web through Tavily when the private index has nothing."""

    def __init__(self, config: Config, *, client: Any | None = None) -> None:
        """Build a Tavily client, or use an injected client for testing."""
        max_results = int(
            config.get("web_search", "max_results", default=DEFAULT_MAX_RESULTS)
        )
        if max_results <= 0:
            raise ValueError("web_search.max_results must be greater than zero")
        self.client = client if client is not None else TavilySearch(
            max_results=max_results,
            tavily_api_key=get_required_environment_variable("TAVILY_API_KEY"),
        )

    def run(
        self,
        query: str,
        *,
        metadata_filter: Mapping[str, str] | None = None,
    ) -> list[Evidence]:
        """Return web search results adapted into the shared Evidence shape.

        `metadata_filter` is accepted for interface conformance with `Tool`
        but has no meaning for live web results, and is ignored.
        """
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")

        raw_results = self.client.invoke({"query": normalized_query})
        if not isinstance(raw_results, Mapping) or "error" in raw_results:
            return []
        results = raw_results.get("results")
        if not isinstance(results, list):
            return []
        return [
            web_result_to_evidence(result)
            for result in results
            if isinstance(result, Mapping)
        ]


def web_result_to_evidence(result: Mapping[str, Any]) -> Evidence:
    """Adapt one Tavily search result into the shared Evidence shape.

    `score` is always None: Tavily results are not scored on the same scale
    as the reranker, so they never qualify for the high-confidence bypass
    and are always sent through the normal relevance grading.
    """
    url = str(result.get("url", ""))
    return {
        "chunk_id": f"web_{hashlib.sha1(url.encode('utf-8')).hexdigest()[:12]}",
        "text": str(result.get("content", "")),
        "origin": "web",
        "metadata": {
            "source": url,
            "page": None,
            "section_title": result.get("title"),
        },
        "score": None,
    }
