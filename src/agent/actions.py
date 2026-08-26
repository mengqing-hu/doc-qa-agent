"""Select the next ReAct action given the accumulated reasoning trajectory."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from src.agent.state import ScratchpadEntry
from src.generation.llm import LLM


ActionName = Literal["vector_retrieve", "web_search", "finish"]
SUPPORTED_ACTIONS = frozenset({"vector_retrieve", "web_search", "finish"})
CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
ACTION_SELECTOR_PROMPT = """You are the reasoning controller for a document question-answering agent that follows the Thought-Action-Observation loop.

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

Return only a JSON object with this exact schema:
{{
  "thought": "Your reasoning about what is known and what is still needed.",
  "action": "vector_retrieve | web_search | finish",
  "action_input": "A self-contained retrieval query, or the final answer when action is finish."
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


class LLMActionSelector:
    """Select the next Thought+Action pair in the ReAct loop."""

    def __init__(self, llm: LLM) -> None:
        """Store the configured LLM used to select each iteration's action."""
        self.llm = llm

    def select(
        self,
        question: str,
        scratchpad: Sequence[ScratchpadEntry],
    ) -> ActionDecision:
        """Return a validated action decision for the current trajectory."""
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")

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
    if not isinstance(thought, str) or not thought.strip():
        raise RuntimeError("Action selector response contains an invalid thought")
    if action not in SUPPORTED_ACTIONS:
        raise RuntimeError("Action selector returned an invalid action")
    if not isinstance(action_input, str) or not action_input.strip():
        raise RuntimeError("Action selector response contains an invalid action_input")

    return ActionDecision(
        thought=thought.strip(),
        action=action,
        action_input=action_input.strip(),
    )
