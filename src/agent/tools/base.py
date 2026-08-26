"""Define the common interface shared by every ReAct retrieval tool."""

from __future__ import annotations

from typing import Protocol

from src.agent.state import Evidence


class Tool(Protocol):
    """Run one retrieval action and return adapted evidence candidates."""

    def run(self, query: str) -> list[Evidence]:
        """Return evidence candidates for the given self-contained query."""
        ...
