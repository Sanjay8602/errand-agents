"""Live model via any OpenAI-compatible chat-completions API.

Works with Groq, OpenAI, OpenRouter, Gemini, Ollama and similar. Pick a provider
preset (see PROVIDERS) and set its key, e.g. for Groq:

    GROQ_API_KEY=gsk_...          (in the environment or in a .env file at the repo root)
    ERRAND_AGENT_MODEL=...        (optional; overrides the preset's default model id)

The agent stores conversation history in the Anthropic content-block shape
(text / tool_use / tool_result). This module converts to and from the OpenAI
messages shape so the orchestrator, gate and tests are provider-agnostic.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from .base import ModelResponse

DEFAULT_MODEL = os.environ.get("ERRAND_AGENT_MODEL", "gpt-4o-mini")

# Provider presets: which env var holds the key, the base URL, and a sensible default model.
PROVIDERS: dict[str, dict[str, str | None]] = {
    "groq": {"key_env": "GROQ_API_KEY", "base_url": "https://api.groq.com/openai/v1",
             "default_model": "openai/gpt-oss-120b"},
    "openai": {"key_env": "OPENAI_API_KEY", "base_url": None, "default_model": "gpt-4o-mini"},
    "openrouter": {"key_env": "OPENROUTER_API_KEY", "base_url": "https://openrouter.ai/api/v1",
                   "default_model": "meta-llama/llama-3.3-70b-instruct"},
    "gemini": {"key_env": "GEMINI_API_KEY", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
               "default_model": "gemini-2.0-flash"},
    "ollama": {"key_env": "OLLAMA_API_KEY", "base_url": "http://localhost:11434/v1", "default_model": "llama3.1"},
}


def load_dotenv(path: str | os.PathLike) -> int:
    """Minimal .env loader (KEY=VALUE lines, # comments). Existing env vars win. Returns count loaded."""
    n = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
                    n += 1
    except FileNotFoundError:
        pass
    return n


class ModelError(Exception):
    def __init__(self, code: str, message: str, transient: bool, retry_after: float | None = None):
        super().__init__(message)
        self.code = code
        self.transient = transient
        self.retry_after = retry_after  # seconds the provider asked us to wait, if it said


_RETRY_RE = re.compile(r"try again in\s*(?:(\d+)m)?(\d+(?:\.\d+)?)(ms|s)?", re.I)


def parse_retry_after(message: str, headers: Any = None) -> float | None:
    """Seconds to wait, from a Retry-After header or Groq/OpenAI's 'Please try again in 11m7.8s' text."""
    try:
        if headers is not None and headers.get("retry-after"):
            return float(headers.get("retry-after"))
    except (TypeError, ValueError):
        pass
    m = _RETRY_RE.search(message or "")
    if not m:
        return None
    minutes = float(m.group(1) or 0)
    value = float(m.group(2))
    if (m.group(3) or "s").lower() == "ms":
        value /= 1000.0
    return minutes * 60 + value


# ----------------------------------------------------------------- conversion
def to_openai_tools(tools: list[dict]) -> list[dict]:
    return [{"type": "function",
             "function": {"name": t["name"], "description": t.get("description", ""),
                          "parameters": t["input_schema"]}} for t in tools]


def to_openai_messages(system: str, messages: list[dict]) -> list[dict]:
    out: list[dict] = [{"role": "system", "content": system}]
    for m in messages:
        role, content = m.get("role"), m.get("content")
        if role == "user":
            if isinstance(content, str):
                out.append({"role": "user", "content": content})
                continue
            # A list: tool results (each becomes a `tool` message) and/or text.
            texts = []
            for b in content:
                if b.get("type") == "tool_result":
                    c = b.get("content")
                    if isinstance(c, list):
                        c = "".join(x.get("text", "") for x in c if isinstance(x, dict))
                    out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": str(c)})
                elif b.get("type") == "text":
                    texts.append(b["text"])
            if texts:
                out.append({"role": "user", "content": "\n".join(texts)})
        elif role == "assistant":
            blocks = content if isinstance(content, list) else [{"type": "text", "text": str(content)}]
            text = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
            calls = [{"id": b["id"], "type": "function",
                      "function": {"name": b["name"], "arguments": json.dumps(b.get("input") or {})}}
                     for b in blocks if b.get("type") == "tool_use"]
            msg: dict[str, Any] = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            if text or calls:
                out.append(msg)
    return out


def from_openai_response(resp: Any) -> ModelResponse:
    choice = resp.choices[0]
    msg = choice.message
    content: list[dict] = []
    if msg.content:
        content.append({"type": "text", "text": msg.content})
    for tc in msg.tool_calls or []:
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {"_raw": tc.function.arguments}
        content.append({"type": "tool_use", "id": tc.id, "name": tc.function.name, "input": args})
    finish = choice.finish_reason or "stop"
    stop = {"tool_calls": "tool_use", "stop": "end_turn", "length": "max_tokens",
            "content_filter": "refusal"}.get(finish, "end_turn")
    if any(b["type"] == "tool_use" for b in content):
        stop = "tool_use"
    usage = getattr(resp, "usage", None)
    return ModelResponse(content=content or [{"type": "text", "text": ""}], stop_reason=stop,
                         input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                         output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                         model=getattr(resp, "model", None) or "openai", raw=resp)


# ---------------------------------------------------------------------- client
class OpenAICompatibleModel:
    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None,
                 temperature: float = 0.2, provider: str = "openai"):
        import openai  # lazy: offline tests never import the SDK

        preset = PROVIDERS.get(provider, PROVIDERS["openai"])
        key = api_key or os.environ.get(preset["key_env"] or "") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ModelError("missing_api_key",
                             f"Set {preset['key_env']} (in the environment or a .env file) to use provider '{provider}'.",
                             transient=False)
        url = base_url or (os.environ.get("OPENAI_BASE_URL") if provider == "openai" else None) or preset["base_url"]
        self._openai = openai
        self.client = openai.OpenAI(api_key=key, base_url=url)
        self.provider = provider
        self.name = model or os.environ.get("ERRAND_AGENT_MODEL") or preset["default_model"]
        self.temperature = temperature

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> ModelResponse:
        try:
            resp = self.client.chat.completions.create(
                model=self.name,
                messages=to_openai_messages(system, messages),
                tools=to_openai_tools(tools),
                tool_choice="auto",
                temperature=self.temperature,
            )
        except self._openai.RateLimitError as e:
            headers = getattr(getattr(e, "response", None), "headers", None)
            raise ModelError("rate_limited", str(e), transient=True,
                             retry_after=parse_retry_after(str(e), headers)) from e
        except self._openai.APIStatusError as e:
            raise ModelError(f"api_{e.status_code}", str(e), transient=e.status_code >= 500) from e
        except self._openai.APIConnectionError as e:
            raise ModelError("connection", str(e), transient=True) from e
        return from_openai_response(resp)
