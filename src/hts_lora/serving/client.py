"""Thin OpenAI client wrapper for talking to the upstream MLX server."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


@dataclass
class UpstreamResponse:
    """Result of a single upstream completion call."""

    text: str
    latency_ms: int


class MLXClient:
    """Tiny wrapper around the OpenAI client pointed at the MLX server.

    The MLX `mlx_lm.server` speaks the OpenAI Chat Completions protocol but
    does not require an API key — we still pass a placeholder so the SDK is
    happy.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self._client = OpenAI(base_url=base_url, api_key="not-needed", timeout=timeout)

    def complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> UpstreamResponse:
        """Call the upstream chat completions endpoint and return raw text + latency.

        NOTE: this path uses /v1/chat/completions which applies the model's
        chat template but does NOT add the Nemotron `<think>\\n</think>\\n\\n`
        prefix the model expects. For HTS classification, prefer
        `complete_prompt` with a manually rendered prompt.
        """
        t0 = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=temperature,
        )
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        text = resp.choices[0].message.content or ""
        return UpstreamResponse(text=text, latency_ms=elapsed_ms)

    def complete_prompt(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> UpstreamResponse:
        """Call the upstream /v1/completions endpoint with a raw prompt string.

        Used for the HTS classifier so we can pre-render the chat template and
        append the Nemotron `<think>\\n</think>\\n\\n` prefix that the model
        was trained to start every response with.
        """
        t0 = time.perf_counter()
        resp = self._client.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        text = resp.choices[0].text or ""
        return UpstreamResponse(text=text, latency_ms=elapsed_ms)

    def is_reachable(self) -> bool:
        """Check whether the upstream server responds to /v1/models."""
        try:
            self._client.models.list()
            return True
        except Exception:
            return False
