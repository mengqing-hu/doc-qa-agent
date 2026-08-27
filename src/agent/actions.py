"""Select the next ReAct action given the accumulated reasoning trajectory."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from src.agent.state import ScratchpadEntry
from src.generation.llm import LLM


logger = logging.getLogger(__name__)
ActionName = Literal["vector_retrieve", "web_search", "finish"]
SUPPORTED_ACTIONS = frozenset({"vector_retrieve", "web_search", "finish"})
ALLOWED_METADATA_FILTER_KEYS = frozenset({"source", "chunk_type"})
CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
ACTION_SELECTOR_PROMPT = """You are the reasoning controller for a document question-answering agent that follows the Thought-Action-Observation loop.

The private indexed documents are exactly: {indexed_documents}. When the
original question or an observation refers to "the pdf file", "the word
document", "this article", "this report", or similar, resolve that reference
to the matching document above and treat the request as being about that
document's content — never reinterpret it as a general question about a
file format or file type in the abstract.

At each step you choose exactly one action:
- vector_retrieve: search the private indexed documents. Prefer this action
  whenever the missing information could plausibly be recorded in those
  documents.
- web_search: search the public web. Choose this only when the missing
  information is clearly external to the private documents,
  or after the private documents have already failed to answer a closely
  related sub-question.
- finish: end the reasoning process and answer the original question. You may
  only choose finish after at least one prior observation is listed below;
  never choose finish as the very first action.

Every action_input for vector_retrieve or web_search must be a self-contained
question: resolve every pronoun and reference using the prior observations so
the query makes sense on its own, without needing this conversation's context.

For vector_retrieve only, you may also set metadata_filter when the request
is scoped to a structural property of the indexed documents rather than to
semantic similarity — for example, every chunk of one particular kind, or
every chunk from one or more particular source documents. metadata_filter is
a JSON object mapping one or both of these exact keys to the value to match:
- source: the exact source document filename, or a JSON array of filenames
  to match any one of them (for example, when a request spans more than one
  document).
- chunk_type: the exact kind of chunk (for example "table", "text", or
  "document_summary"), or a JSON array to match any one of them.
Never combine multiple filenames or chunk kinds into one string — always use
a JSON array for "any of these", never a single string like
"[a.pdf, b.pdf]". When present, every matching chunk is returned instead of
only the chunks most similar to action_input. Omit metadata_filter (or set
it to null) for an ordinary similarity-based request. Never invent a key
other than these two.

Return only a JSON object with this exact schema:
{{
  "thought": "Your reasoning about what is known and what is still needed.",
  "action": "vector_retrieve | web_search | finish",
  "action_input": "A self-contained retrieval query, or the final answer when action is finish.",
  "metadata_filter": {{"source": "..." or ["...", "..."], "chunk_type": "..." or ["...", "..."]}} or null
}}

Original question:
{question}

Prior observations (oldest first):
{scratchpad}
"""


@dataclass(frozen=True)
class ActionDecision:
    """Capture one selected ReAct action and the reasoning behind it."""

    thought: str
    action: ActionName
    action_input: str
    metadata_filter: Mapping[str, str | tuple[str, ...]] | None = None


class LLMActionSelector:
    """Select the next Thought+Action pair in the ReAct loop."""

    def __init__(self, llm: LLM, *, document_names: Sequence[str] = ()) -> None:
        """Store the configured LLM and the names of the indexed documents."""
        self.llm = llm
        self.document_names = tuple(document_names)

    def select(
        self,
        question: str,
        scratchpad: Sequence[ScratchpadEntry],
    ) -> ActionDecision:
        """Return a validated action decision for the current trajectory."""
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")

        indexed_documents = (
            ", ".join(self.document_names) if self.document_names else "none indexed"
        )
        scratchpad_payload = [
            {
                "thought": entry["thought"],
                "action": entry["action"],
                "action_input": entry["action_input"],
                "observation": entry["fact"],
            }
            for entry in scratchpad
        ]
        prompt = ACTION_SELECTOR_PROMPT.format(
            indexed_documents=indexed_documents,
            question=json.dumps(normalized_question, ensure_ascii=False),
            scratchpad=json.dumps(scratchpad_payload, ensure_ascii=False),
        )
        decision = _parse_action_decision(self.llm.generate(prompt))
        if decision.action == "finish" and not scratchpad:
            raise RuntimeError(
                "Action selector chose finish before any evidence was gathered"
            )
        return decision


def _parse_action_decision(response: str) -> ActionDecision:
    """Validate the JSON-only response required from the action selector LLM."""
    response_text = response.strip()
    code_fence_match = CODE_FENCE_PATTERN.fullmatch(response_text)
    if code_fence_match:
        response_text = code_fence_match.group(1).strip()

    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise RuntimeError("Action selector did not return valid JSON") from error
    if not isinstance(payload, Mapping):
        raise RuntimeError("Action selector response must be a JSON object")

    thought = payload.get("thought")
    action = payload.get("action")
    action_input = payload.get("action_input")
    raw_metadata_filter = payload.get("metadata_filter")
    if not isinstance(thought, str) or not thought.strip():
        raise RuntimeError("Action selector response contains an invalid thought")
    if action not in SUPPORTED_ACTIONS:
        raise RuntimeError("Action selector returned an invalid action")
    if not isinstance(action_input, str) or not action_input.strip():
        raise RuntimeError("Action selector response contains an invalid action_input")
    metadata_filter = _parse_metadata_filter(raw_metadata_filter)

    return ActionDecision(
        thought=thought.strip(),
        action=action,
        action_input=action_input.strip(),
        metadata_filter=metadata_filter,
    )


def _parse_metadata_filter(
    raw_metadata_filter: object,
) -> Mapping[str, str | tuple[str, ...]] | None:
    """Best-effort parse of the optional metadata_filter object.

    metadata_filter is an optional enhancement, not a required field: an
    unsupported key or an unusable value degrades to omitting that key
    rather than failing the whole action decision, so a malformed field
    never crashes a turn that would otherwise have a valid action and
    action_input. Each value may be a single string, or a JSON array of
    strings meaning "match any of these" (e.g. one filter spanning several
    source documents).
    """
    if raw_metadata_filter is None:
        return None
    if not isinstance(raw_metadata_filter, Mapping):
        logger.warning(
            "Action selector returned a non-object metadata_filter; ignoring it"
        )
        return None

    metadata_filter: dict[str, str | tuple[str, ...]] = {}
    for key, value in raw_metadata_filter.items():
        if key not in ALLOWED_METADATA_FILTER_KEYS:
            logger.warning(
                "Action selector returned an unsupported metadata_filter key %r; "
                "dropping it",
                key,
            )
            continue
        parsed_value = _parse_metadata_filter_value(key, value)
        if parsed_value is None:
            continue
        metadata_filter[key] = parsed_value
    return metadata_filter or None


def _parse_metadata_filter_value(
    key: str, value: object
) -> str | tuple[str, ...] | None:
    """Parse one metadata_filter value as a single string or list of strings."""
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            logger.warning(
                "Action selector returned an invalid metadata_filter value for %r; "
                "dropping it",
                key,
            )
            return None
        return normalized

    if isinstance(value, list):
        values = tuple(
            item.strip()
            for item in value
            if isinstance(item, str) and item.strip()
        )
        if not values:
            logger.warning(
                "Action selector returned an invalid metadata_filter list for %r; "
                "dropping it",
                key,
            )
            return None
        return values

    logger.warning(
        "Action selector returned an invalid metadata_filter value for %r; "
        "dropping it",
        key,
    )
    return None
