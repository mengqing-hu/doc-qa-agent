"""Invoke the configured OpenAI-compatible chat model."""

from __future__ import annotations

import logging
import time
from typing import Any

from openai import OpenAI, OpenAIError

from src.core.config import Config


logger = logging.getLogger(__name__)
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 512
DEFAULT_MAX_RETRIES = 2


class LLM:
    """Generate an answer from a rendered RAG prompt."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        client: Any | None = None,
    ) -> None:
        """Create an API client, or use an injected client for testing."""
        self.config = config if config is not None else Config()
        self.model = str(self.config.get("llm", "model", default=DEFAULT_MODEL))
        self.temperature = float(
            self.config.get("llm", "temperature", default=DEFAULT_TEMPERATURE)
        )
        self.max_tokens = int(
            self.config.get("llm", "max_tokens", default=DEFAULT_MAX_TOKENS)
        )
        if self.max_tokens <= 0:
            raise ValueError("llm.max_tokens must be greater than zero")
        if not 0 <= self.temperature <= 2:
            raise ValueError("llm.temperature must be between zero and two")
        self.max_retries = int(
            self.config.get("llm", "max_retries", default=DEFAULT_MAX_RETRIES)
        )
        if self.max_retries < 0:
            raise ValueError("llm.max_retries must be greater than or equal to zero")

        self.client = client if client is not None else OpenAI(
            api_key=self.config.scadsai_api_key,
            base_url=str(self.config.get("llm", "base_url")),
        )

    def generate(self, prompt: str, *, max_tokens: int | None = None) -> str:
        """Return one non-empty chat completion for a rendered prompt."""
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        token_limit = self.max_tokens if max_tokens is None else int(max_tokens)
        if token_limit <= 0:
            raise ValueError("max_tokens must be greater than zero")

        completion = self._create_completion_with_retry(prompt, token_limit)

        content = completion.choices[0].message.content if completion.choices else None
        if content is None or not content.strip():
            raise RuntimeError("LLM response did not contain answer text")

        finish_reason = completion.choices[0].finish_reason if completion.choices else None
        if finish_reason == "length":
            logger.warning(
                "LLM response for model %s was truncated by max_tokens=%d; "
                "the answer may be incomplete",
                self.model,
                token_limit,
            )

        answer = content.strip()
        logger.info("Generated answer with %d character(s)", len(answer))
        return answer

    def _create_completion_with_retry(self, prompt: str, token_limit: int) -> Any:
        """Retry a transient LLM request failure with exponential backoff.

        `OpenAIError` covers timeouts, rate limits, and connection errors —
        the same class of transient failure the ScaDS.AI reranker already
        retries. A non-transient failure (e.g. a malformed response) is not
        an `OpenAIError` and is not retried here.
        """
        for retry_count in range(self.max_retries + 1):
            try:
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=token_limit,
                )
            except OpenAIError as error:
                if retry_count == self.max_retries:
                    logger.exception(
                        "LLM request failed for model %s after %d retry(ies)",
                        self.model,
                        retry_count,
                    )
                    raise RuntimeError("LLM request failed") from error
                delay_seconds = float(2**retry_count)
                logger.warning(
                    "LLM request failed for model %s; retrying in %.1f second(s) "
                    "(%d/%d): %s",
                    self.model,
                    delay_seconds,
                    retry_count + 1,
                    self.max_retries,
                    error,
                )
                time.sleep(delay_seconds)
        raise AssertionError("unreachable")
