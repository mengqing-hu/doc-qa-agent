"""Invoke the configured OpenAI-compatible chat model."""

from __future__ import annotations

import logging
from typing import Any

from openai import OpenAI, OpenAIError

from src.core.config import Config


logger = logging.getLogger(__name__)
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 512


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

        self.client = client if client is not None else OpenAI(
            api_key=self.config.scadsai_api_key,
            base_url=str(self.config.get("llm", "base_url")),
        )

    def generate(self, prompt: str) -> str:
        """Return one non-empty chat completion for a rendered prompt."""
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except OpenAIError as error:
            logger.exception("LLM request failed for model %s", self.model)
            raise RuntimeError("LLM request failed") from error

        content = completion.choices[0].message.content if completion.choices else None
        if content is None or not content.strip():
            raise RuntimeError("LLM response did not contain answer text")

        answer = content.strip()
        logger.info("Generated answer with %d character(s)", len(answer))
        return answer
