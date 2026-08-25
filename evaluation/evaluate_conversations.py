"""Evaluate stateful LangGraph document QA conversations."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import chromadb
from langgraph.checkpoint.memory import InMemorySaver

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.agent.context import ContextManager
from src.agent.graph import build_agent_graph, invoke_agent_graph
from src.agent.relevance import LLMRelevanceGrader
from src.agent.rewrite import LLMQueryRewriter
from src.agent.routes import LLMRetrievalGate
from src.agent.support import LLMSupportVerifier
from src.core.config import PROJECT_ROOT, Config
from src.core.logger import setup_logging
from src.pipeline.query_runtime import build_query_pipeline


DEFAULT_INPUT_PATH = (
    PROJECT_ROOT / "evaluation" / "conversations" / "development" / "v1" / "trajectories.json"
)
DEFAULT_RESULTS_DIRECTORY = (
    PROJECT_ROOT / "evaluation" / "results" / "conversations" / "development" / "v1"
)
DEFAULT_EXPERIMENT_NAME = "baseline-context-rewrite"
DEFAULT_WORKER_COUNT = 4
EXPECTED_ACTIONS = frozenset({"retrieve", "abstain"})
EXPECTED_ANSWERABILITY = frozenset({"answerable", "unanswerable"})
REFUSAL_PATTERNS = (
    "cannot answer",
    "cannot be answered",
    "cannot determine",
    "could not determine",
    "does not provide",
    "does not report",
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
    "i can only answer questions about the indexed documents",
)


def main() -> None:
    """Validate or execute labeled conversation trajectories."""
    arguments = _parse_arguments()
    trajectories = _load_trajectories(arguments.input)
    if arguments.limit is not None:
        trajectories = trajectories[: arguments.limit]
    _validate_trajectories(trajectories)

    config = Config()
    indexed_chunk_count = _validate_indexed_chunk_ids(trajectories, config)
    print(
        f"Validated {len(trajectories)} trajectory(s) against "
        f"{indexed_chunk_count} indexed chunk(s)."
    )
    if arguments.validate_only:
        return

    setup_logging(config)
    graph = _build_graph(config)
    result = _evaluate_trajectories(
        graph, trajectories, arguments.experiment_name, arguments.workers
    )
    result["input_path"] = str(arguments.input)
    result["indexed_chunk_count"] = indexed_chunk_count
    result["experiment_name"] = arguments.experiment_name
    result["limits"] = {
        "trajectory_count": arguments.limit,
    }

    arguments.results_dir.mkdir(parents=True, exist_ok=True)
    output_path = arguments.results_dir / f"{arguments.experiment_name}.json"
    _write_json(output_path, result)
    _print_summary(result, output_path)


def _parse_arguments() -> argparse.Namespace:
    """Read command-line arguments for conversation evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate stateful LangGraph document QA conversations."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Trajectory JSON file to validate or execute.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIRECTORY,
        help="Directory for the JSON experiment result.",
    )
    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_EXPERIMENT_NAME,
        help="Stable name used for the output result file and thread IDs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of trajectories to execute.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate labels and indexed chunk IDs without calling the graph.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKER_COUNT,
        help="Number of trajectories to execute concurrently (each trajectory's "
        "own turns still run in order).",
    )
    return parser.parse_args()


def _load_trajectories(input_path: Path) -> list[dict[str, Any]]:
    """Load a trajectory JSON list from disk."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Trajectory file was not found: {input_path}")
    with input_path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError("Trajectory file must contain a JSON list")
    return [dict(trajectory) for trajectory in payload]


def _validate_trajectories(trajectories: Sequence[Mapping[str, Any]]) -> None:
    """Validate labels required for deterministic trajectory scoring."""
    if not trajectories:
        raise ValueError("At least one conversation trajectory is required")

    conversation_ids: set[str] = set()
    for trajectory in trajectories:
        conversation_id = _required_string(trajectory, "conversation_id", "trajectory")
        if conversation_id in conversation_ids:
            raise ValueError(f"Duplicate conversation_id: {conversation_id}")
        conversation_ids.add(conversation_id)
        _required_string(trajectory, "split", conversation_id)
        _validate_string_list(trajectory, "scenario_types", conversation_id, non_empty=True)
        _validate_string_list(trajectory, "source_documents", conversation_id, non_empty=True)

        turns = trajectory.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"{conversation_id}.turns must be a non-empty list")
        for expected_turn_id, turn in enumerate(turns, start=1):
            if not isinstance(turn, Mapping):
                raise ValueError(f"{conversation_id}.turns[{expected_turn_id}] must be an object")
            _validate_turn(turn, conversation_id, expected_turn_id)


def _validate_turn(
    turn: Mapping[str, Any],
    conversation_id: str,
    expected_turn_id: int,
) -> None:
    """Validate one turn and its evidence labels."""
    label = f"{conversation_id}.turns[{expected_turn_id}]"
    if turn.get("turn_id") != expected_turn_id:
        raise ValueError(f"{label}.turn_id must be {expected_turn_id}")
    _required_string(turn, "question", label)
    action = _required_string(turn, "expected_retrieval_action", label)
    if action not in EXPECTED_ACTIONS:
        raise ValueError(f"{label}.expected_retrieval_action is invalid: {action}")
    answerability = _required_string(turn, "answerability", label)
    if answerability not in EXPECTED_ANSWERABILITY:
        raise ValueError(f"{label}.answerability is invalid: {answerability}")
    _validate_string_list(turn, "required_query_terms", label)
    _validate_string_list(turn, "forbidden_query_terms", label)
    relevant_chunk_ids = _validate_string_list(turn, "relevant_chunk_ids", label)
    _required_string(turn, "reference_answer", label)

    expected_abstention = turn.get("expected_abstention", False)
    if not isinstance(expected_abstention, bool):
        raise ValueError(f"{label}.expected_abstention must be a boolean")
    if answerability == "answerable" and not relevant_chunk_ids:
        raise ValueError(f"{label} must define relevant_chunk_ids for an answerable turn")
    if answerability == "unanswerable":
        if relevant_chunk_ids:
            raise ValueError(f"{label} must not define relevant_chunk_ids for an unanswerable turn")
        if not expected_abstention:
            raise ValueError(f"{label} must set expected_abstention to true")


def _required_string(record: Mapping[str, Any], key: str, label: str) -> str:
    """Read one required non-empty string label."""
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}.{key} must be a non-empty string")
    return value.strip()


def _validate_string_list(
    record: Mapping[str, Any],
    key: str,
    label: str,
    *,
    non_empty: bool = False,
) -> list[str]:
    """Validate a JSON list of non-empty strings."""
    value = record.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{label}.{key} must be a list")
    if non_empty and not value:
        raise ValueError(f"{label}.{key} must not be empty")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{label}.{key} must contain non-empty strings")
    return [item.strip() for item in value]


def _validate_indexed_chunk_ids(
    trajectories: Sequence[Mapping[str, Any]],
    config: Config,
) -> int:
    """Ensure every annotated evidence ID exists in the configured Chroma collection."""
    expected_ids = {
        chunk_id
        for trajectory in trajectories
        for turn in trajectory["turns"]
        for chunk_id in turn["relevant_chunk_ids"]
    }
    persist_directory = Path(config.get("vector_store", "persist_dir", default="data/chroma_db"))
    if not persist_directory.is_absolute():
        persist_directory = PROJECT_ROOT / persist_directory
    collection_name = str(config.get("vector_store", "collection_name", default="doc_chunks"))
    client = chromadb.PersistentClient(path=str(persist_directory))
    try:
        collection = client.get_collection(collection_name)
    except ValueError as error:
        raise RuntimeError(
            f"Configured Chroma collection does not exist: {collection_name}"
        ) from error

    indexed_ids = set(collection.get(ids=sorted(expected_ids), include=[])["ids"])
    missing_ids = sorted(expected_ids.difference(indexed_ids))
    if missing_ids:
        raise ValueError(
            "Trajectory labels reference chunk IDs missing from the configured "
            f"Chroma collection: {', '.join(missing_ids)}"
        )
    return collection.count()


def _build_graph(config: Config) -> Any:
    """Build one real graph and shared in-memory checkpointer for an experiment."""
    pipeline = build_query_pipeline(config)
    return build_agent_graph(
        pipeline,
        retrieval_gate=LLMRetrievalGate(pipeline.llm),
        context_manager=ContextManager(config),
        query_rewriter=LLMQueryRewriter(pipeline.llm),
        relevance_grader=LLMRelevanceGrader(pipeline.llm),
        support_verifier=LLMSupportVerifier(pipeline.llm),
        checkpointer=InMemorySaver(),
    )


def _evaluate_trajectories(
    graph: Any,
    trajectories: Sequence[Mapping[str, Any]],
    experiment_name: str,
    worker_count: int,
) -> dict[str, Any]:
    """Execute independent trajectories concurrently, each on its own LangGraph thread."""
    if worker_count <= 0:
        raise ValueError("workers must be greater than zero")
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        trajectory_results = list(
            executor.map(
                lambda trajectory: _evaluate_trajectory(graph, trajectory, experiment_name),
                trajectories,
            )
        )
    return {
        "trajectory_count": len(trajectory_results),
        "turn_count": sum(len(result["turns"]) for result in trajectory_results),
        "metrics": _aggregate_metrics(trajectory_results),
        "trajectories": trajectory_results,
    }


def _evaluate_trajectory(
    graph: Any,
    trajectory: Mapping[str, Any],
    experiment_name: str,
) -> dict[str, Any]:
    """Run all turns from one trajectory in its own stateful graph thread."""
    conversation_id = str(trajectory["conversation_id"])
    thread_id = f"evaluation:{experiment_name}:{conversation_id}"
    turn_results = [
        _evaluate_turn(graph, turn, thread_id)
        for turn in trajectory["turns"]
    ]
    return {
        "conversation_id": conversation_id,
        "thread_id": thread_id,
        "split": trajectory["split"],
        "scenario_types": trajectory["scenario_types"],
        "source_documents": trajectory["source_documents"],
        "description": trajectory.get("description"),
        "automated_success": all(turn["automated_success"] for turn in turn_results),
        "turns": turn_results,
    }


def _evaluate_turn(graph: Any, turn: Mapping[str, Any], thread_id: str) -> dict[str, Any]:
    """Run and score one user message without exposing labels to the graph."""
    question = str(turn["question"])
    start_time = time.perf_counter()
    try:
        response = invoke_agent_graph(graph, question, thread_id=thread_id)
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        state = _state_values(graph, thread_id)
        return _score_turn(turn, response.answer, response.sources, state, elapsed_ms)
    except Exception as error:
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return _failed_turn_result(turn, elapsed_ms, str(error))


def _state_values(graph: Any, thread_id: str) -> Mapping[str, Any]:
    """Read the checkpointed graph state produced by the current turn."""
    snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
    values = snapshot.values
    if not isinstance(values, Mapping):
        raise RuntimeError("Agent graph checkpoint did not contain a state mapping")
    return values


def _score_turn(
    turn: Mapping[str, Any],
    answer: str,
    sources: Sequence[Any],
    state: Mapping[str, Any],
    elapsed_ms: float,
) -> dict[str, Any]:
    """Score route, rewriting, retrieval evidence, and abstention checks."""
    expected_action = str(turn["expected_retrieval_action"])
    observed_action = state.get("retrieval_action")
    rewritten_query = _optional_string(state.get("rewritten_query"))
    required_terms = [str(term) for term in turn["required_query_terms"]]
    forbidden_terms = [str(term) for term in turn["forbidden_query_terms"]]
    source_records = [_source_to_dict(source) for source in sources]
    source_ids = [source["chunk_id"] for source in source_records]
    relevant_ids = [str(chunk_id) for chunk_id in turn["relevant_chunk_ids"]]
    observed_relevant_ids = [
        str(chunk_id) for chunk_id in state.get("relevant_chunk_ids", [])
    ]
    expected_abstention = bool(turn.get("expected_abstention", False))
    abstention_detected = _detect_abstention(answer, observed_action)

    required_terms_pass = _contains_all_terms(rewritten_query, required_terms)
    forbidden_terms_pass = not _contains_any_term(rewritten_query, forbidden_terms)
    evidence_hit = (
        bool(set(source_ids).intersection(relevant_ids)) if relevant_ids else None
    )
    relevance_hit = (
        bool(set(observed_relevant_ids).intersection(relevant_ids))
        if relevant_ids
        else None
    )
    expected_relevance_status = "relevant" if relevant_ids else "none"
    relevance_status_matches = (
        state.get("relevance_status") == expected_relevance_status
    )
    support_status_matches = (
        state.get("support_status") == "supported" if relevant_ids else None
    )
    action_matches = observed_action == expected_action
    abstention_matches = abstention_detected == expected_abstention
    automated_success = (
        action_matches
        and required_terms_pass
        and forbidden_terms_pass
        and relevance_status_matches
        and (relevance_hit is not False)
        and (support_status_matches is not False)
        and (evidence_hit is not False)
        and abstention_matches
    )
    return {
        "turn_id": turn["turn_id"],
        "question": turn["question"],
        "expected": {
            "retrieval_action": expected_action,
            "answerability": turn["answerability"],
            "required_query_terms": required_terms,
            "forbidden_query_terms": forbidden_terms,
            "relevant_chunk_ids": relevant_ids,
            "expected_abstention": expected_abstention,
            "reference_answer": turn["reference_answer"],
        },
        "observed": {
            "retrieval_action": observed_action,
            "original_query": _optional_string(state.get("original_query")),
            "rewritten_query": rewritten_query,
            "rewrite_used_conversation_context": state.get(
                "rewrite_used_conversation_context"
            ),
            "rewrite_reason": _optional_string(state.get("rewrite_reason")),
            "conversation_history_length": _history_length(state),
            "retrieval_attempts": state.get("retrieval_attempts"),
            "relevant_chunk_ids": observed_relevant_ids,
            "relevance_decisions": state.get("relevance_decisions", []),
            "relevance_status": state.get("relevance_status"),
            "relevance_reason": _optional_string(state.get("relevance_reason")),
            "support_status": state.get("support_status"),
            "support_claims": state.get("support_claims", []),
            "support_reason": _optional_string(state.get("support_reason")),
            "answer": answer,
            "sources": source_records,
            "refusal_detected": abstention_detected,
        },
        "checks": {
            "action_matches": action_matches,
            "required_terms_pass": required_terms_pass,
            "forbidden_terms_pass": forbidden_terms_pass,
            "evidence_hit": evidence_hit,
            "relevance_hit": relevance_hit,
            "relevance_status_matches": relevance_status_matches,
            "support_status_matches": support_status_matches,
            "abstention_matches": abstention_matches,
        },
        "automated_success": automated_success,
        "latency_ms": elapsed_ms,
        "error": None,
    }


def _failed_turn_result(
    turn: Mapping[str, Any], elapsed_ms: float, error: str
) -> dict[str, Any]:
    """Return a stable failed result when a graph invocation raises."""
    return {
        "turn_id": turn["turn_id"],
        "question": turn["question"],
        "expected": dict(turn),
        "observed": None,
        "checks": {
            "action_matches": False,
            "required_terms_pass": False,
            "forbidden_terms_pass": False,
            "relevance_hit": False,
            "relevance_status_matches": False,
            "support_status_matches": False,
            "evidence_hit": False,
            "abstention_matches": False,
        },
        "automated_success": False,
        "latency_ms": elapsed_ms,
        "error": error,
    }


def _optional_string(value: Any) -> str | None:
    """Normalize optional strings returned from graph state."""
    return value.strip() if isinstance(value, str) else None


def _history_length(state: Mapping[str, Any]) -> int | None:
    """Return the saved message count when history is checkpoint-compatible."""
    history = state.get("conversation_history")
    return len(history) if isinstance(history, list) else None


def _source_to_dict(source: Any) -> dict[str, Any]:
    """Serialize a public source reference without relying on implementation type."""
    return {
        "chunk_id": str(source.chunk_id),
        "source": str(source.source),
        "page": source.page,
        "section_title": source.section_title,
    }


def _contains_all_terms(text: str | None, terms: Sequence[str]) -> bool:
    """Check all required terms when a trajectory declares them."""
    if not terms:
        return True
    if text is None:
        return False
    normalized_text = text.casefold()
    return all(term.casefold() in normalized_text for term in terms)


def _contains_any_term(text: str | None, terms: Sequence[str]) -> bool:
    """Check whether a rewritten query carries a forbidden prior-topic term."""
    if text is None:
        return False
    normalized_text = text.casefold()
    return any(term.casefold() in normalized_text for term in terms)


def _detect_abstention(answer: str, observed_action: str | None) -> bool:
    """Recognize scope rejection or a documented-evidence refusal answer."""
    if observed_action == "abstain":
        return True
    normalized_answer = answer.casefold()
    return any(pattern in normalized_answer for pattern in REFUSAL_PATTERNS)


def _aggregate_metrics(trajectory_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate transparent automated checks over all completed turns."""
    turns = [
        turn
        for trajectory in trajectory_results
        for turn in trajectory["turns"]
    ]
    answerable_turns = [
        turn for turn in turns if turn["expected"]["answerability"] == "answerable"
    ]
    unanswerable_turns = [
        turn for turn in turns if turn["expected"]["answerability"] == "unanswerable"
    ]
    required_term_turns = [
        turn for turn in turns if turn["expected"]["required_query_terms"]
    ]
    forbidden_term_turns = [
        turn for turn in turns if turn["expected"]["forbidden_query_terms"]
    ]
    return {
        "retrieval_action_accuracy": _check_rate(turns, "action_matches"),
        "relevance_status_accuracy": _check_rate(turns, "relevance_status_matches"),
        "required_query_term_pass_rate": _check_rate(
            required_term_turns, "required_terms_pass"
        ),
        "forbidden_query_term_pass_rate": _check_rate(
            forbidden_term_turns, "forbidden_terms_pass"
        ),
        "answerable_evidence_hit_rate": _check_rate(answerable_turns, "evidence_hit"),
        "answerable_relevance_hit_rate": _check_rate(
            answerable_turns, "relevance_hit"
        ),
        "answerable_support_rate": _check_rate(
            answerable_turns, "support_status_matches"
        ),
        "grounded_abstention_accuracy": _check_rate(
            unanswerable_turns, "abstention_matches"
        ),
        "trajectory_success_rate": _trajectory_success_rate(trajectory_results),
        "mean_latency_ms": _mean([float(turn["latency_ms"]) for turn in turns]),
        "failed_turn_count": sum(1 for turn in turns if turn["error"] is not None),
    }


def _check_rate(turns: Sequence[Mapping[str, Any]], check_name: str) -> float | None:
    """Compute a rate only when the corresponding label applies."""
    if not turns:
        return None
    passed_count = sum(1 for turn in turns if turn["checks"][check_name] is True)
    return passed_count / len(turns)


def _trajectory_success_rate(trajectory_results: Sequence[Mapping[str, Any]]) -> float:
    """Return the share of trajectories passing every automated check."""
    if not trajectory_results:
        return 0.0
    return sum(
        1 for trajectory in trajectory_results if trajectory["automated_success"]
    ) / len(trajectory_results)


def _mean(values: Sequence[float]) -> float:
    """Return a zero-safe arithmetic mean."""
    return sum(values) / len(values) if values else 0.0


def _write_json(output_path: Path, payload: Mapping[str, Any]) -> None:
    """Write a readable, reproducible evaluation artifact."""
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)


def _print_summary(result: Mapping[str, Any], output_path: Path) -> None:
    """Print the compact metrics required for a baseline comparison."""
    metrics = result["metrics"]
    print(f"Saved conversation evaluation to {output_path}")
    print(f"Trajectories: {result['trajectory_count']}")
    print(f"Turns: {result['turn_count']}")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.3f}")
        else:
            print(f"{name}: {value}")


if __name__ == "__main__":
    main()
