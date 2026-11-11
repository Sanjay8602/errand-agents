"""Live model via the official Anthropic SDK (manual tool loop is owned by the agent).

Only used interactively. Automated evals never touch the network.
"""
from __future__ import annotations

import os
from typing import Any

from .base import ModelResponse

DEFAULT_MODEL = os.environ.get("ERRAND_AGENT_MODEL", "claude-opus-5-5")


class AnthropicModel:
    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 4096, effort: str = "medium"):
        import anthropic  # imported lazily so offline tests never need network or a key

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.name = model
        self.max_tokens = max_tokens
        self.effort = effort

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> ModelResponse:
        kwargs: dict[str, Any] = dict(
            model=self.name,
            max_tokens=self.max_tokens,
            system=[{"type": "text", "text": system}],
            messages=messages,
            tools=tools,
            output_config={"effort": self.effort},
        )
        try:
            resp = self.client.messages.create(**kwargs)
        except self._anthropic.RateLimitError as e:
            raise ModelError("rate_limited", str(e), transient=True) from e
        except self._anthropic.APIStatusError as e:
            raise ModelError(f"api_{e.status_code}", str(e), transient=e.status_code >= 500) from e
        except self._anthropic.APIConnectionError as e:
            raise ModelError("connection", str(e), transient=True) from e

        content = [b.model_dump(exclude_none=True) for b in resp.content]
        if resp.stop_reason == "refusal":
            content.append({"type": "text", "text": "I can't help with that request."})
        return ModelResponse(
            content=content,
            stop_reason=resp.stop_reason or "end_turn",
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            model=resp.model,
            raw=resp,
        )


class ModelError(Exception):
    def __init__(self, code: str, message: str, transient: bool):
        super().__init__(message)
        self.code = code
        self.transient = transient
