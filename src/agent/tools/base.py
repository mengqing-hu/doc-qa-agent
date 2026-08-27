"""Define the common interface shared by every ReAct retrieval tool."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from src.agent.state import Evidence


class Tool(Protocol):
    """Run one retrieval action and return adapted evidence candidates."""

    def run(
        self,
        query: str,
        *,
        metadata_filter: Mapping[str, str | tuple[str, ...]] | None = None,
    ) -> list[Evidence]:
        """Return evidence candidates for the given self-contained query.

        When `metadata_filter` is set, return every chunk whose metadata
        matches it (a structural request) instead of the chunks most
        similar to `query`. A value may be a single string, or a tuple of
        strings meaning "match any of these".
        """
        ...
