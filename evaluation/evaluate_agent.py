"""Evaluate the flattened retrieval agent's tool use and reasoning without golden answers.

This script runs the real agent graph over ``evaluation/agent_test.json`` and scores
each turn along six agent-level dimensions:

- Action Validity        - every chosen action is protocol-legal and executed cleanly
- Step Efficiency        - iteration count, redundant retrieval, over-budget rate
- Tool Usage Correctness - vector_retrieve vs web_search chosen (and used) appropriately
- Trajectory Coherence   - thoughts follow from observations, no contradictions (LLM judge)
- Goal Completion        - the final answer covers every part of the question (LLM judge)
- Task Success Rate      - a per-turn composite of the checks above

Everything is reference-free: the test set carries only questions plus lightweight
expectation labels (expected_route / expected_tools / expected_hops / must_cover).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from openai import OpenAI, OpenAIError

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.agent.chitchat import LLMChitchatResponder
from src.agent.context import ContextManager
from src.agent.graph import build_agent_graph, invoke_agent_graph
from src.store.messages import InMemoryConversationStore
from src.agent.planner import LLMRetrievalPlanner
from src.agent.relevance import LLMRelevanceGrader
from src.agent.routes import LLMRetrievalGate
from src.agent.support import LLMSupportVerifier
from src.agent.tools.vector_retrieve import VectorRetrieveTool
from src.agent.tools.web_search import WebSearchTool
from src.core.config import PROJECT_ROOT, Config
from src.core.logger import setup_logging
from src.pipeline.query_runtime import build_query_pipeline


DEFAULT_INPUT_PATH = PROJECT_ROOT / "evaluation" / "agent_test.json"
DEFAULT_RESULTS_DIRECTORY = PROJECT_ROOT / "evaluation" / "results" / "agent"
DEFAULT_EXPERIMENT_NAME = "baseline"
DEFAULT_JUDGE_MODEL = "openai/gpt-oss-120b"
# Conversations run concurrently; the reranker serialises its own rate-limited
# endpoint call across threads, so the LLM calls (gate/plan/grade/synth/verify)
# parallelise while rerank stays at ~1 req/s. Keep this modest - the ScaDS LLM
# endpoint is shared too.
DEFAULT_WORKER_COUNT = 3


def _resolve_judge_model(config: Config, override: str | None) -> str:
    """Resolve the judge model: CLI override > config > built-in default."""
    if override:
        return override
    return str(
        config.get("evaluation", "judge", "model", default=DEFAULT_JUDGE_MODEL)
    )

VALID_ROUTES = frozenset({"retrieve", "chitchat", "abstain"})
VALID_TOOLS = frozenset({"vector_retrieve", "web_search"})
VALID_EXPECTED_TOOLS = frozenset({"docs_only", "needs_web", "either", "none"})
REDUNDANT_QUERY_SIMILARITY = 0.90
CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
REFUSAL_PATTERNS = (
    "i can only answer questions about the indexed documents",
    "could not be answered within the allowed number of reasoning steps",
    "did not contain evidence that supports an answer",
    "ran into a problem",
    "cannot answer",
    "cannot be answered",
    "cannot determine",
    "does not provide",
    "does not mention",
    "insufficient information",
    "not enough information",
    "not available",
    "not discussed",
    "not provided",
    "not reported",
    "not mentioned",
    "unable to answer",
    "unable to determine",
    "无法回答",
    "无法确定",
    "未提及",
    "没有提到",
)

JUDGE_SYSTEM_POLICY = """The agent plans retrieval in rounds, then generates one answer:
- Each round a planner emits one or more search queries over two tools:
  - vector_retrieve: searches the PRIVATE indexed documents ({indexed_documents}).
    Preferred whenever the missing information could plausibly be in those documents.
  - web_search: searches the PUBLIC web. Appropriate only when the missing
    information is clearly external to the private documents.
- Round 1 should decompose the question into its independent sub-questions.
  Later rounds should only add queries for what is still missing, especially a
  fact that depends on something just learned.
- After retrieval stops, the agent generates one grounded answer from all the
  accumulated evidence and a verifier checks it.
The agent may abstain (refuse) when the evidence does not support an answer."""

JUDGE_PROMPT = """You are a strict evaluator of a document question-answering agent. Judge ONLY what
the retrieval rounds and answer show. Do not reward confident-sounding but unsupported answers.

{policy}

## Original user question
{question}

## The agent's retrieval rounds and the evidence they gathered (oldest first)
{trajectory}

## Retrieval-gate decision for this turn
{retrieval_action} - {retrieval_reason}

## The agent's final answer
{answer}

## Information points a COMPLETE answer should address (for goal_completion only)
{must_cover}

Return ONLY a JSON object with this exact schema (each *_reason is one short sentence):
{{
  "goal_completion": "complete | partial | none",
  "goal_completion_reason": "...",
  "groundedness": "grounded | partially_grounded | unsupported | not_applicable",
  "groundedness_reason": "...",
  "tool_choice_appropriateness": "appropriate | questionable | inappropriate",
  "tool_choice_reason": "...",
  "trajectory_coherence": "coherent | minor_issues | incoherent",
  "trajectory_coherence_reason": "...",
  "query_formulation": "good | acceptable | poor",
  "query_formulation_reason": "..."
}}

Guidance:
- goal_completion: does the final answer address every part of the question? Use the
  information points above as a checklist; "complete" only if all are covered.
- groundedness: are the answer's claims supported by the gathered evidence?
  Use "not_applicable" only when the agent abstained / made small talk.
- tool_choice_appropriateness: for each query, was vector_retrieve vs web_search the
  right tool given the policy, and was the decomposition into rounds sensible?
- trajectory_coherence: do later rounds build on earlier evidence, with no
  contradictions, no repeated identical searches, and no ignored findings?
- query_formulation: were the search queries self-contained (pronouns resolved) and
  well targeted?"""


def main() -> None:
    """Run the agent over the test set and write an agent-evaluation report."""
    arguments = _parse_arguments()
    testset = _load_testset(arguments.input)
    if arguments.limit is not None:
        testset = testset[: arguments.limit]
    _validate_testset(testset)

    config = Config()
    setup_logging(config)
    judge_model = _resolve_judge_model(config, arguments.judge_model)
    graph = _build_graph(config)
    judge = JudgeClient(config, judge_model, run_judge=not arguments.no_judge)
    judge.set_indexed_documents(
        build_query_pipeline(config).hybrid_retriever.vector_store.get_indexed_sources()
    )

    conversation_results = _evaluate_conversations(
        graph, judge, testset, arguments.experiment_name, arguments.workers
    )
    report = {
        "experiment_name": arguments.experiment_name,
        "input_path": str(arguments.input),
        "system_model": str(config.get("llm", "model")),
        "judge_model": None if arguments.no_judge else judge_model,
        "conversation_count": len(conversation_results),
        "turn_count": sum(len(result["turns"]) for result in conversation_results),
        "metrics": _aggregate_metrics(conversation_results),
        "conversations": conversation_results,
    }

    arguments.results_dir.mkdir(parents=True, exist_ok=True)
    output_path = arguments.results_dir / f"{arguments.experiment_name}.json"
    _write_json(output_path, report)
    _print_summary(report, output_path)


def _parse_arguments() -> argparse.Namespace:
    """Read evaluation settings from the command line."""
    parser = argparse.ArgumentParser(
        description="Evaluate agent tool use and reasoning (reference-free)."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIRECTORY)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Judge model id. Overrides evaluation.judge.model in config.yaml; "
        f"falls back to that, then to {DEFAULT_JUDGE_MODEL}.",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip all LLM-judge scoring; report only the automated checks.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKER_COUNT)
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Test set loading and validation
# --------------------------------------------------------------------------- #
def _load_testset(input_path: Path) -> list[dict[str, Any]]:
    """Load the agent test set from a JSON list."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Agent test set was not found: {input_path}")
    with input_path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError("Agent test set must contain a JSON list")
    return [dict(conversation) for conversation in payload]


def _validate_testset(testset: Sequence[Mapping[str, Any]]) -> None:
    """Validate the labels the automated checks depend on."""
    if not testset:
        raise ValueError("At least one conversation is required")
    seen_ids: set[str] = set()
    for conversation in testset:
        conversation_id = _required_string(conversation, "conversation_id", "conversation")
        if conversation_id in seen_ids:
            raise ValueError(f"Duplicate conversation_id: {conversation_id}")
        seen_ids.add(conversation_id)
        turns = conversation.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"{conversation_id}.turns must be a non-empty list")
        for expected_turn_id, turn in enumerate(turns, start=1):
            label = f"{conversation_id}.turns[{expected_turn_id}]"
            if not isinstance(turn, Mapping):
                raise ValueError(f"{label} must be an object")
            if turn.get("turn_id") != expected_turn_id:
                raise ValueError(f"{label}.turn_id must be {expected_turn_id}")
            _required_string(turn, "question", label)
            route = _required_string(turn, "expected_route", label)
            if route not in VALID_ROUTES:
                raise ValueError(f"{label}.expected_route is invalid: {route}")
            tools = _required_string(turn, "expected_tools", label)
            if tools not in VALID_EXPECTED_TOOLS:
                raise ValueError(f"{label}.expected_tools is invalid: {tools}")
            if not isinstance(turn.get("expected_hops", 0), int):
                raise ValueError(f"{label}.expected_hops must be an integer")
            if not isinstance(turn.get("expected_abstention", False), bool):
                raise ValueError(f"{label}.expected_abstention must be a boolean")
            if not isinstance(turn.get("must_cover", []), list):
                raise ValueError(f"{label}.must_cover must be a list")


def _required_string(record: Mapping[str, Any], key: str, label: str) -> str:
    """Read one required non-empty string field."""
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}.{key} must be a non-empty string")
    return value.strip()


# --------------------------------------------------------------------------- #
# Graph construction (mirrors main.py)
# --------------------------------------------------------------------------- #
def _build_graph(config: Config) -> Any:
    """Build the real agent graph with one shared in-memory checkpointer."""
    pipeline = build_query_pipeline(config)
    return build_agent_graph(
        pipeline,
        retrieval_gate=LLMRetrievalGate(pipeline.llm),
        context_manager=ContextManager(config),
        chitchat_responder=LLMChitchatResponder(pipeline.llm),
        retrieval_planner=LLMRetrievalPlanner(
            pipeline.llm,
            document_names=pipeline.hybrid_retriever.vector_store.get_indexed_sources(),
        ),
        relevance_grader=LLMRelevanceGrader(pipeline.llm),
        support_verifier=LLMSupportVerifier(pipeline.llm),
        vector_retrieve_tool=VectorRetrieveTool(pipeline),
        web_search_tool=WebSearchTool(config),
        checkpointer=InMemorySaver(),
    )


# --------------------------------------------------------------------------- #
# LLM judge
# --------------------------------------------------------------------------- #
class JudgeClient:
    """Score one turn's trajectory along the reference-free judge dimensions."""

    def __init__(self, config: Config, model: str, *, run_judge: bool = True) -> None:
        """Build an OpenAI-compatible client for the judge model."""
        self.model = model
        self.run_judge = run_judge
        self._indexed_documents = "the indexed documents"
        base_url = str(
            config.get(
                "evaluation",
                "judge",
                "base_url",
                default=config.get("llm", "base_url"),
            )
        )
        self.client = (
            OpenAI(api_key=config.scadsai_api_key, base_url=base_url)
            if run_judge
            else None
        )

    def set_indexed_documents(self, document_names: Sequence[str]) -> None:
        """Record the indexed document names for the judge policy text."""
        if document_names:
            self._indexed_documents = ", ".join(document_names)

    def score(self, turn: Mapping[str, Any], observed: Mapping[str, Any]) -> dict[str, Any]:
        """Return the judge's structured verdict, or a skipped placeholder."""
        if not self.run_judge or self.client is None:
            return {"skipped": True}

        prompt = JUDGE_PROMPT.format(
            policy=JUDGE_SYSTEM_POLICY.format(indexed_documents=self._indexed_documents),
            question=turn["question"],
            trajectory=_format_trajectory(observed),
            retrieval_action=observed.get("retrieval_action"),
            retrieval_reason=observed.get("retrieval_reason") or "(none)",
            answer=observed.get("answer") or "(no answer)",
            must_cover=_format_must_cover(turn.get("must_cover", [])),
        )
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1600,
            )
        except OpenAIError as error:
            return {"skipped": False, "error": f"judge request failed: {error}"}

        content = completion.choices[0].message.content if completion.choices else None
        verdict = _parse_judge_json(content or "")
        if verdict is None:
            return {"skipped": False, "error": "judge did not return valid JSON",
                    "raw": (content or "")[:2000]}
        return verdict


def _format_trajectory(observed: Mapping[str, Any]) -> str:
    """Render the retrieval rounds and the evidence they gathered for the judge."""
    rounds = observed.get("steps", [])
    if not rounds:
        return "(no retrieval rounds - the agent answered directly or abstained)"
    lines: list[str] = []
    for entry in rounds:
        queries = "\n".join(
            f"    - [{query.get('tool')}] {query.get('query')}"
            for query in entry.get("queries", [])
        )
        added = ", ".join(entry.get("added_chunk_ids", [])) or "none"
        errors = entry.get("gather_errors", [])
        error_line = f"\n    tool errors: {'; '.join(errors)}" if errors else ""
        lines.append(
            f"Round {entry.get('round')}:\n{queries}\n"
            f"    chunks gathered: {entry.get('gathered_count', 0)}, "
            f"relevant kept: {added}{error_line}"
        )
    evidence = observed.get("evidence_snippets", [])
    if evidence:
        lines.append("\nAccumulated evidence passed to synthesis:")
        lines.extend(f"  - {snippet}" for snippet in evidence)
    support = observed.get("support_status")
    if support:
        lines.append(f"\nSupport verifier verdict on the drafted answer: {support}")
    return "\n".join(lines)


def _format_must_cover(must_cover: Sequence[str]) -> str:
    """Render the must_cover checklist for the judge prompt."""
    if not must_cover:
        return "(none specified - judge whether the answer fully addresses the question)"
    return "\n".join(f"- {point}" for point in must_cover)


def _parse_judge_json(text: str) -> dict[str, Any] | None:
    """Best-effort parse of the judge's JSON object."""
    stripped = text.strip()
    fence = CODE_FENCE_PATTERN.fullmatch(stripped)
    if fence:
        stripped = fence.group(1).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        payload = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


# --------------------------------------------------------------------------- #
# Running and scoring
# --------------------------------------------------------------------------- #
def _evaluate_conversations(
    graph: Any,
    judge: JudgeClient,
    testset: Sequence[Mapping[str, Any]],
    experiment_name: str,
    worker_count: int,
) -> list[dict[str, Any]]:
    """Run independent conversations concurrently, each on its own graph thread."""
    if worker_count <= 0:
        raise ValueError("workers must be greater than zero")
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(
            executor.map(
                lambda conversation: _evaluate_conversation(
                    graph, judge, conversation, experiment_name
                ),
                testset,
            )
        )


def _evaluate_conversation(
    graph: Any,
    judge: JudgeClient,
    conversation: Mapping[str, Any],
    experiment_name: str,
) -> dict[str, Any]:
    """Run all turns of one conversation in order on a single stateful thread."""
    conversation_id = str(conversation["conversation_id"])
    thread_id = f"agent-eval:{experiment_name}:{conversation_id}"
    store = InMemoryConversationStore()
    turn_results = [
        _evaluate_turn(graph, judge, turn, thread_id, store)
        for turn in conversation["turns"]
    ]
    return {
        "conversation_id": conversation_id,
        "thread_id": thread_id,
        "scenario_types": conversation.get("scenario_types", []),
        "description": conversation.get("description"),
        "task_success": all(turn["task_success"] for turn in turn_results),
        "turns": turn_results,
    }


def _evaluate_turn(
    graph: Any,
    judge: JudgeClient,
    turn: Mapping[str, Any],
    thread_id: str,
    store: InMemoryConversationStore,
) -> dict[str, Any]:
    """Run one user message, read the checkpointed state, and score it."""
    question = str(turn["question"])
    start_time = time.perf_counter()
    try:
        response = invoke_agent_graph(
            graph, question, thread_id=thread_id, store=store
        )
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        state = _state_values(graph, thread_id)
        observed = _extract_observed(state, response)
        error = None
    except Exception as invocation_error:  # noqa: BLE001 - record and continue
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        observed = _empty_observed()
        error = str(invocation_error)

    checks = _automated_checks(turn, observed, error)
    verdict = (
        {"skipped": True}
        if error is not None
        else judge.score(turn, observed)
    )
    task_success = _task_success(turn, checks, verdict)

    return {
        "turn_id": turn["turn_id"],
        "question": question,
        "expected": {
            "route": turn["expected_route"],
            "tools": turn["expected_tools"],
            "tool_sequence": turn.get("expected_tool_sequence", []),
            "hops": turn.get("expected_hops", 0),
            "abstention": bool(turn.get("expected_abstention", False)),
            "must_cover": turn.get("must_cover", []),
        },
        "observed": {
            "retrieval_action": observed["retrieval_action"],
            "tool_path": observed["tool_path"],
            "iterations": observed["iterations"],
            "support_status": observed["support_status"],
            "relevance_status": observed["relevance_status"],
            "answer": observed["answer"],
            "source_ids": observed["source_ids"],
            "steps": observed["steps"],
            "abstained": observed["abstained"],
        },
        "checks": checks,
        "judge": verdict,
        "task_success": task_success,
        "latency_ms": elapsed_ms,
        "error": error,
    }


def _state_values(graph: Any, thread_id: str) -> Mapping[str, Any]:
    """Read the checkpointed graph state for the current turn."""
    snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
    values = snapshot.values
    if not isinstance(values, Mapping):
        raise RuntimeError("Agent graph checkpoint did not contain a state mapping")
    return values


def _empty_observed() -> dict[str, Any]:
    """Return the observed shape used when a turn raised before producing state."""
    return {
        "retrieval_action": None,
        "retrieval_reason": None,
        "tool_path": [],
        "iterations": 0,
        "retrieval_rounds": 0,
        "max_retrieval_rounds": 0,
        "support_status": None,
        "relevance_status": None,
        "synthesis_truncated": False,
        "answer": None,
        "source_ids": [],
        "steps": [],
        "evidence_snippets": [],
        "action_inputs": [],
        "abstained": False,
        "state_error": "invocation raised",
    }


def _extract_observed(
    state: Mapping[str, Any], response: Any
) -> dict[str, Any]:
    """Pull the fields the checks and the judge need out of graph state."""
    history = state.get("retrieval_history", [])
    steps = [
        {
            "round": entry.get("round"),
            "queries": entry.get("queries", []),
            "gathered_count": entry.get("gathered_count", 0),
            "added_chunk_ids": entry.get("added_chunk_ids", []),
            "gather_errors": entry.get("gather_errors", []),
        }
        for entry in history
    ]
    planned_queries = [
        query for entry in history for query in entry.get("queries", [])
    ]
    tool_path = [query.get("tool") for query in planned_queries]
    action_inputs = [str(query.get("query", "")) for query in planned_queries]
    accumulated = state.get("accumulated_evidence", [])
    evidence_snippets = [
        f"[{item.get('chunk_id')}] "
        + " ".join(str(item.get("text", "")).split())[:200]
        for item in accumulated
    ]
    answer = getattr(response, "answer", None)
    retrieval_action = state.get("retrieval_action")
    return {
        "retrieval_action": retrieval_action,
        "retrieval_reason": state.get("retrieval_reason"),
        "tool_path": tool_path,
        "iterations": len(history),
        "retrieval_rounds": int(state.get("retrieval_rounds", 0)),
        "max_retrieval_rounds": int(state.get("max_retrieval_rounds", 0)),
        "support_status": state.get("support_status"),
        "relevance_status": "relevant" if accumulated else "none",
        "synthesis_truncated": bool(state.get("synthesis_truncated", False)),
        "answer": answer,
        "source_ids": [str(source.chunk_id) for source in getattr(response, "sources", [])],
        "steps": steps,
        "evidence_snippets": evidence_snippets,
        "action_inputs": action_inputs,
        "abstained": _detect_abstention(answer or "", retrieval_action),
        "state_error": state.get("error"),
    }


def _detect_abstention(answer: str, retrieval_action: str | None) -> bool:
    """Recognise a scope refusal or a grounded no-evidence answer."""
    if retrieval_action == "abstain":
        return True
    normalized = answer.casefold()
    return any(pattern in normalized for pattern in REFUSAL_PATTERNS)


# --------------------------------------------------------------------------- #
# Automated checks
# --------------------------------------------------------------------------- #
def _automated_checks(
    turn: Mapping[str, Any],
    observed: Mapping[str, Any],
    error: str | None,
) -> dict[str, Any]:
    """Compute the label-free checks for Action Validity / Tool Use / Efficiency."""
    expected_route = turn["expected_route"]
    expected_tools = turn["expected_tools"]
    expected_hops = int(turn.get("expected_hops", 0))
    expected_abstention = bool(turn.get("expected_abstention", False))
    tool_path = list(observed["tool_path"])

    route_match = observed["retrieval_action"] == expected_route
    action_validity = (
        error is None
        and observed.get("state_error") in (None, "")
        and all(action in VALID_TOOLS for action in tool_path)
    )

    used_web = "web_search" in tool_path
    used_vector = "vector_retrieve" in tool_path
    if expected_tools == "none":
        tool_usage_pass = len(tool_path) == 0
    elif expected_tools == "docs_only":
        tool_usage_pass = used_vector and not used_web
    elif expected_tools == "needs_web":
        tool_usage_pass = used_web
    else:  # "either"
        tool_usage_pass = True

    unnecessary_web_call = expected_tools == "docs_only" and used_web
    missing_web_call = expected_tools == "needs_web" and not used_web
    premature_web_call = bool(tool_path) and tool_path[0] == "web_search" and (
        expected_tools == "docs_only"
    )

    abstention_match = observed["abstained"] == expected_abstention

    over_budget = (
        observed["max_retrieval_rounds"] > 0
        and observed["retrieval_rounds"] >= observed["max_retrieval_rounds"]
        and observed["abstained"]
    )
    redundant_retrieval = _has_redundant_queries(observed["action_inputs"])
    hop_delta = (
        observed["iterations"] - expected_hops if expected_hops > 0 else None
    )

    joined_queries = " ".join(observed["action_inputs"]).casefold()
    required_terms = [str(term) for term in turn.get("required_query_terms", [])]
    forbidden_terms = [str(term) for term in turn.get("forbidden_query_terms", [])]
    required_terms_pass = all(
        term.casefold() in joined_queries for term in required_terms
    ) if required_terms else None
    forbidden_terms_pass = (
        not any(term.casefold() in joined_queries for term in forbidden_terms)
        if forbidden_terms
        else None
    )

    tool_sequence_exact_match = (
        tool_path == list(turn.get("expected_tool_sequence", []))
        if turn.get("expected_tool_sequence")
        else None
    )

    return {
        "route_match": route_match,
        "action_validity": action_validity,
        "tool_usage_pass": tool_usage_pass,
        "unnecessary_web_call": unnecessary_web_call,
        "missing_web_call": missing_web_call,
        "premature_web_call": premature_web_call,
        "abstention_match": abstention_match,
        "over_budget": over_budget,
        "redundant_retrieval": redundant_retrieval,
        "hop_delta": hop_delta,
        "tool_sequence_exact_match": tool_sequence_exact_match,
        "required_terms_pass": required_terms_pass,
        "forbidden_terms_pass": forbidden_terms_pass,
    }


def _has_redundant_queries(action_inputs: Sequence[str]) -> bool:
    """Return True when two search queries in one turn are near-duplicates."""
    normalized = [query.strip().casefold() for query in action_inputs if query.strip()]
    for first_index in range(len(normalized)):
        for second_index in range(first_index + 1, len(normalized)):
            ratio = SequenceMatcher(
                None, normalized[first_index], normalized[second_index]
            ).ratio()
            if ratio >= REDUNDANT_QUERY_SIMILARITY:
                return True
    return False


def _task_success(
    turn: Mapping[str, Any],
    checks: Mapping[str, Any],
    verdict: Mapping[str, Any],
) -> bool:
    """Compose a per-turn success verdict from the checks and the judge."""
    expected_route = turn["expected_route"]
    expected_abstention = bool(turn.get("expected_abstention", False))

    if expected_abstention:
        return bool(checks["abstention_match"])
    if expected_route in {"chitchat", "abstain"}:
        return bool(checks["route_match"] and checks["action_validity"])

    # An answerable retrieval turn. expected_abstention is False here, so
    # abstention_match is True only when the agent did NOT abstain.
    base = bool(
        checks["route_match"]
        and checks["action_validity"]
        and checks["tool_usage_pass"]
        and not checks["unnecessary_web_call"]
        and checks["abstention_match"]
    )

    if verdict.get("skipped") or "error" in verdict:
        return bool(base)  # no judge signal available; fall back to the automated checks
    return bool(
        base
        and verdict.get("goal_completion") == "complete"
        and verdict.get("groundedness") in {"grounded", "partially_grounded"}
    )


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _aggregate_metrics(conversations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate turn-level checks and judge verdicts into headline metrics."""
    turns = [turn for conversation in conversations for turn in conversation["turns"]]
    docs_only = [t for t in turns if t["expected"]["tools"] == "docs_only"]
    needs_web = [t for t in turns if t["expected"]["tools"] == "needs_web"]
    scored_tools = [t for t in turns if t["expected"]["tools"] != "either"]
    abstain_turns = [t for t in turns if t["expected"]["abstention"]]
    hop_turns = [t for t in turns if t["checks"]["hop_delta"] is not None]
    judged = [
        t for t in turns
        if isinstance(t["judge"], Mapping)
        and not t["judge"].get("skipped")
        and "error" not in t["judge"]
    ]
    required_term_turns = [
        t for t in turns if t["checks"]["required_terms_pass"] is not None
    ]
    forbidden_term_turns = [
        t for t in turns if t["checks"]["forbidden_terms_pass"] is not None
    ]

    return {
        "retrieval_action_accuracy": _rate(turns, lambda t: t["checks"]["route_match"]),
        "action_validity_rate": _rate(turns, lambda t: t["checks"]["action_validity"]),
        "tool_usage_correctness_rate": _rate(
            scored_tools, lambda t: t["checks"]["tool_usage_pass"]
        ),
        "unnecessary_web_call_rate": _rate(
            docs_only, lambda t: t["checks"]["unnecessary_web_call"]
        ),
        "missing_web_call_rate": _rate(
            needs_web, lambda t: t["checks"]["missing_web_call"]
        ),
        "premature_web_call_rate": _rate(
            turns, lambda t: t["checks"]["premature_web_call"]
        ),
        "grounded_abstention_accuracy": _rate(
            abstain_turns, lambda t: t["checks"]["abstention_match"]
        ),
        "required_query_term_pass_rate": _rate(
            required_term_turns, lambda t: t["checks"]["required_terms_pass"]
        ),
        "forbidden_query_term_pass_rate": _rate(
            forbidden_term_turns, lambda t: t["checks"]["forbidden_terms_pass"]
        ),
        "step_efficiency": {
            "mean_rounds": _round(mean([t["observed"]["iterations"] for t in turns]))
            if turns
            else None,
            "max_rounds": max((t["observed"]["iterations"] for t in turns), default=0),
            "mean_hop_delta": _round(
                mean([t["checks"]["hop_delta"] for t in hop_turns])
            )
            if hop_turns
            else None,
            "over_budget_rate": _rate(turns, lambda t: t["checks"]["over_budget"]),
            "redundant_retrieval_rate": _rate(
                turns, lambda t: t["checks"]["redundant_retrieval"]
            ),
        },
        "judge": {
            "judged_turns": len(judged),
            "goal_completion_complete_rate": _rate(
                judged, lambda t: t["judge"].get("goal_completion") == "complete"
            ),
            "goal_completion_partial_or_better_rate": _rate(
                judged,
                lambda t: t["judge"].get("goal_completion") in {"complete", "partial"},
            ),
            "groundedness_grounded_rate": _rate(
                judged,
                lambda t: t["judge"].get("groundedness")
                in {"grounded", "partially_grounded"},
            ),
            "tool_choice_appropriate_rate": _rate(
                judged,
                lambda t: t["judge"].get("tool_choice_appropriateness") == "appropriate",
            ),
            "trajectory_coherence_rate": _rate(
                judged,
                lambda t: t["judge"].get("trajectory_coherence") == "coherent",
            ),
            "query_formulation_good_rate": _rate(
                judged, lambda t: t["judge"].get("query_formulation") == "good"
            ),
        },
        "task_success_rate": _rate(turns, lambda t: t["task_success"]),
        "conversation_success_rate": _rate(
            conversations, lambda c: c["task_success"]
        ),
        "synthesis_truncation_rate": _rate(
            turns, lambda t: t["observed"].get("synthesis_truncated")
        ),
        "mean_latency_ms": _round(mean([t["latency_ms"] for t in turns])) if turns else None,
        "failed_turn_count": sum(1 for t in turns if t["error"] is not None),
    }


def _rate(items: Sequence[Any], predicate: Any) -> float | None:
    """Return the share of items satisfying a predicate, or None when empty."""
    if not items:
        return None
    return round(sum(1 for item in items if predicate(item)) / len(items), 3)


def _round(value: float | None) -> float | None:
    """Round a float to three places, passing None through."""
    return None if value is None else round(value, 3)


def _write_json(output_path: Path, payload: Mapping[str, Any]) -> None:
    """Write one readable evaluation report."""
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)


def _print_summary(report: Mapping[str, Any], output_path: Path) -> None:
    """Print the headline agent metrics."""
    metrics = report["metrics"]
    print()
    print(f"System model: {report['system_model']}")
    print(f"Judge model:  {report['judge_model'] or '(judge disabled)'}")
    print(f"Conversations: {report['conversation_count']}  Turns: {report['turn_count']}")
    print()
    _print_metric("Retrieval-action accuracy", metrics["retrieval_action_accuracy"])
    _print_metric("Action validity rate", metrics["action_validity_rate"])
    _print_metric("Tool-usage correctness rate", metrics["tool_usage_correctness_rate"])
    _print_metric("Unnecessary web-call rate (docs_only)", metrics["unnecessary_web_call_rate"])
    _print_metric("Missing web-call rate (needs_web)", metrics["missing_web_call_rate"])
    _print_metric("Grounded-abstention accuracy", metrics["grounded_abstention_accuracy"])
    efficiency = metrics["step_efficiency"]
    print(
        f"  Step efficiency: mean_rounds={efficiency['mean_rounds']} "
        f"max_rounds={efficiency['max_rounds']} "
        f"mean_hop_delta={efficiency['mean_hop_delta']} "
        f"over_budget={efficiency['over_budget_rate']} "
        f"redundant={efficiency['redundant_retrieval_rate']}"
    )
    judge = metrics["judge"]
    print(f"  Judge ({judge['judged_turns']} turns):")
    _print_metric("    Goal completion (complete)", judge["goal_completion_complete_rate"])
    _print_metric("    Groundedness", judge["groundedness_grounded_rate"])
    _print_metric("    Tool choice appropriate", judge["tool_choice_appropriate_rate"])
    _print_metric("    Trajectory coherence", judge["trajectory_coherence_rate"])
    _print_metric("    Query formulation good", judge["query_formulation_good_rate"])
    print()
    _print_metric("Task success rate (turns)", metrics["task_success_rate"])
    _print_metric("Conversation success rate", metrics["conversation_success_rate"])
    print(f"  Mean latency: {metrics['mean_latency_ms']} ms")
    print(f"  Failed turns: {metrics['failed_turn_count']}")
    print()
    print(f"Saved agent evaluation to {output_path}")


def _print_metric(label: str, value: float | None) -> None:
    """Print one metric line, tolerating a None value."""
    print(f"  {label}: {'n/a' if value is None else f'{value:.3f}'}")


if __name__ == "__main__":
    main()
