"""Rerank retrieval candidates with a local or ScaDS.AI model."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urljoin

import requests
from sentence_transformers import CrossEncoder

from src.core.config import Config


logger = logging.getLogger(__name__)
DEFAULT_PROVIDER = "local"
DEFAULT_LOCAL_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_SCADSAI_MODEL_NAME = "Qwen/Qwen3-Reranker-4B"
DEFAULT_BATCH_SIZE = 16
DEFAULT_TOP_K = 5
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MIN_REQUEST_INTERVAL_SECONDS = 1.0
DEFAULT_MAX_RETRIES = 5


class Reranker:
    """Rank query-chunk pairs through the configured reranking provider."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        model: Any | None = None,
        session: requests.Session | None = None,
    ) -> None:
        """Load a local model or configure the ScaDS.AI reranking endpoint."""
        self.config = config if config is not None else Config()
        self.provider = str(
            self.config.get("reranking", "provider", default=DEFAULT_PROVIDER)
        ).lower()
        default_model_name = (
            DEFAULT_SCADSAI_MODEL_NAME
            if self.provider == "scadsai"
            else DEFAULT_LOCAL_MODEL_NAME
        )
        self.model_name = str(
            self.config.get(
                "reranking",
                "model_name",
                default=self.config.get(
                    "reranking", "model", default=default_model_name
                ),
            )
        )
        self.batch_size = int(
            self.config.get("reranking", "batch_size", default=DEFAULT_BATCH_SIZE)
        )
        if self.batch_size <= 0:
            raise ValueError("reranking.batch_size must be greater than zero")

        self.model: Any | None = None
        self.session: requests.Session | None = None
        self.endpoint: str | None = None
        self.timeout_seconds: float | None = None
        self.min_request_interval_seconds: float | None = None
        self.max_retries: int | None = None
        self._last_request_monotonic: float | None = None

        if self.provider == "local":
            hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
            self.model = (
                model
                if model is not None
                else CrossEncoder(self.model_name, token=hf_token)
            )
        elif self.provider == "scadsai":
            self.session = session if session is not None else requests.Session()
            self.endpoint = _build_scadsai_endpoint(self.config)
            self.timeout_seconds = float(
                self.config.get(
                    "reranking",
                    "timeout_seconds",
                    default=DEFAULT_TIMEOUT_SECONDS,
                )
            )
            if self.timeout_seconds <= 0:
                raise ValueError("reranking.timeout_seconds must be greater than zero")
            self.min_request_interval_seconds = float(
                self.config.get(
                    "reranking",
                    "min_request_interval_seconds",
                    default=DEFAULT_MIN_REQUEST_INTERVAL_SECONDS,
                )
            )
            if self.min_request_interval_seconds < 0:
                raise ValueError(
                    "reranking.min_request_interval_seconds must be greater than or equal to zero"
                )
            self.max_retries = int(
                self.config.get(
                    "reranking", "max_retries", default=DEFAULT_MAX_RETRIES
                )
            )
            if self.max_retries < 0:
                raise ValueError("reranking.max_retries must be greater than or equal to zero")
        else:
            raise ValueError(
                "reranking.provider must be either 'local' or 'scadsai'"
            )

        logger.info("Configured %s reranker: %s", self.provider, self.model_name)

    def rerank(
        self,
        query: str,
        chunks: Sequence[dict[str, Any]],
        *,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return the highest-scoring chunks for a non-empty query."""
        result_count = (
            top_k
            if top_k is not None
            else int(
                self.config.get("retrieval", "final_top_k", default=DEFAULT_TOP_K)
            )
        )
        if result_count <= 0:
            raise ValueError("top_k must be greater than zero")
        if not query.strip():
            raise ValueError("query must not be empty")
        if not chunks:
            return []

        ranked_chunks = (
            self._rerank_locally(query, chunks)
            if self.provider == "local"
            else self._rerank_with_scadsai(query, chunks)
        )
        return ranked_chunks[:result_count]

    def _rerank_locally(
        self, query: str, chunks: Sequence[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Score query-chunk pairs with the configured local Cross-Encoder."""
        if self.model is None:
            raise RuntimeError("Local reranker model is not configured")

        scores = self.model.predict(
            [(query, str(chunk.get("text", ""))) for chunk in chunks],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        if len(scores) != len(chunks):
            raise ValueError("Cross-Encoder returned an unexpected number of scores")
        return _rank_chunks(chunks, scores)

    def _rerank_with_scadsai(
        self, query: str, chunks: Sequence[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Score all candidates through ScaDS.AI's reranking endpoint."""
        if (
            self.session is None
            or self.endpoint is None
            or self.timeout_seconds is None
            or self.min_request_interval_seconds is None
            or self.max_retries is None
        ):
            raise RuntimeError("ScaDS.AI reranker client is not configured")

        payload: dict[str, Any] = {
            "model": self.model_name,
            "query": query,
            "documents": [str(chunk.get("text", "")) for chunk in chunks],
            "top_n": len(chunks),
            "return_documents": False,
        }
        instruction = self.config.get("reranking", "instruction")
        if instruction:
            payload["instruction"] = str(instruction)

        try:
            response = self._post_to_scadsai(payload)
        except requests.RequestException as error:
            logger.exception("ScaDS.AI reranking request failed for %s", self.model_name)
            raise RuntimeError("ScaDS.AI reranking request failed") from error

        try:
            response_payload = response.json()
        except ValueError as error:
            raise RuntimeError("ScaDS.AI reranking response was not valid JSON") from error

        logger.info("Reranked %d candidate(s) through ScaDS.AI", len(chunks))
        return _rank_chunks(
            chunks, _scadsai_scores(response_payload, candidate_count=len(chunks))
        )

    def _post_to_scadsai(self, payload: Mapping[str, Any]) -> requests.Response:
        """Post one reranking request while respecting service rate limits."""
        if (
            self.session is None
            or self.endpoint is None
            or self.timeout_seconds is None
            or self.min_request_interval_seconds is None
            or self.max_retries is None
        ):
            raise RuntimeError("ScaDS.AI reranker client is not configured")

        for retry_count in range(self.max_retries + 1):
            self._wait_for_request_slot()
            response = self.session.post(
                self.endpoint,
                json=dict(payload),
                headers={"Authorization": f"Bearer {self.config.scadsai_api_key}"},
                timeout=self.timeout_seconds,
            )
            self._last_request_monotonic = time.monotonic()
            if response.status_code != requests.codes.too_many_requests:
                response.raise_for_status()
                return response

            if retry_count == self.max_retries:
                response.raise_for_status()
            delay_seconds = _retry_delay_seconds(response, retry_count)
            logger.warning(
                "ScaDS.AI rate limited reranking request; retrying in %.1f second(s) "
                "(%d/%d)",
                delay_seconds,
                retry_count + 1,
                self.max_retries,
            )
            time.sleep(delay_seconds)

        raise RuntimeError("ScaDS.AI reranking request exhausted retries")

    def _wait_for_request_slot(self) -> None:
        """Wait until the configured minimum interval since the previous call."""
        if (
            self._last_request_monotonic is None
            or self.min_request_interval_seconds is None
        ):
            return

        elapsed_seconds = time.monotonic() - self._last_request_monotonic
        delay_seconds = self.min_request_interval_seconds - elapsed_seconds
        if delay_seconds > 0:
            time.sleep(delay_seconds)


def _build_scadsai_endpoint(config: Config) -> str:
    """Resolve the configured rerank path against the ScaDS.AI API base URL."""
    base_url = str(config.get("reranking", "base_url", default="")).strip()
    endpoint = str(config.get("reranking", "endpoint", default="rerank")).strip()
    if not base_url:
        raise ValueError("reranking.base_url must not be empty for provider 'scadsai'")
    if not endpoint:
        raise ValueError("reranking.endpoint must not be empty for provider 'scadsai'")
    return urljoin(f"{base_url.rstrip('/')}/", endpoint.lstrip("/"))


def _retry_delay_seconds(response: requests.Response, retry_count: int) -> float:
    """Use a server-provided retry delay, or exponential backoff as a fallback."""
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            delay_seconds = float(retry_after)
        except ValueError:
            delay_seconds = 0.0
        if delay_seconds > 0:
            return delay_seconds
    return float(2**retry_count)


def _scadsai_scores(response_payload: Any, *, candidate_count: int) -> list[float]:
    """Validate API results and restore scores to their input-document order."""
    if not isinstance(response_payload, Mapping):
        raise RuntimeError("ScaDS.AI reranking response must be a JSON object")
    results = response_payload.get("results")
    if not isinstance(results, list) or len(results) != candidate_count:
        raise RuntimeError(
            "ScaDS.AI reranking response must contain one result per candidate"
        )

    scores: list[float | None] = [None] * candidate_count
    for result in results:
        if not isinstance(result, Mapping):
            raise RuntimeError("ScaDS.AI reranking result must be a JSON object")
        index = result.get("index")
        score = result.get("relevance_score")
        if not isinstance(index, int) or not 0 <= index < candidate_count:
            raise RuntimeError("ScaDS.AI reranking result contains an invalid index")
        if scores[index] is not None:
            raise RuntimeError("ScaDS.AI reranking response contains duplicate indices")
        try:
            scores[index] = float(score)
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                "ScaDS.AI reranking result contains an invalid relevance_score"
            ) from error

    if any(score is None for score in scores):
        raise RuntimeError("ScaDS.AI reranking response omitted a candidate index")
    return [float(score) for score in scores]


def _rank_chunks(
    chunks: Sequence[dict[str, Any]], scores: Sequence[float]
) -> list[dict[str, Any]]:
    """Attach scores and apply deterministic descending ranking."""
    ranked_chunks = [
        {**chunk, "rerank_score": float(score)}
        for chunk, score in zip(chunks, scores, strict=True)
    ]
    ranked_chunks.sort(
        key=lambda chunk: (-chunk["rerank_score"], str(chunk["chunk_id"]))
    )
    return ranked_chunks
