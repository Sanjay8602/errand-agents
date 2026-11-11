"""Run tracing: one JSONL file per run plus an in-memory list for tests and the CLI."""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

# USD per million tokens (input, output); scripted model is free.
PRICING = {
    "openai/gpt-oss-120b": (0.15, 0.75),   # Groq list price
    "openai/gpt-oss-20b": (0.075, 0.30),
    "qwen/qwen3.8-27b": (0.20, 0.60),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.0),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.40, 1.60),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-fable-5-1": (10.0, 50.0),
    "scripted": (0.0, 0.0),
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    inp, out = PRICING.get(model, (0.0, 0.0))
    return round((input_tokens * inp + output_tokens * out) / 1_000_000, 6)


class Tracer:
    def __init__(self, trace_dir: Optional[Path], run_id: Optional[str] = None,
                 now: Optional[Callable[[], datetime]] = None, listener: Optional[Callable[[dict], None]] = None):
        self.run_id = run_id or "run-" + uuid.uuid4().hex[:8]
        self.events: list[dict] = []
        self._now = now or datetime.now
        self._seq = 0
        self.listener = listener
        self.path: Optional[Path] = None
        if trace_dir is not None:
            Path(trace_dir).mkdir(parents=True, exist_ok=True)
            self.path = Path(trace_dir) / f"{self.run_id}.jsonl"

    def event(self, type_: str, **data: Any) -> dict:
        self._seq += 1
        ev = {"seq": self._seq, "run_id": self.run_id, "ts": self._now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
              "type": type_, **data}
        self.events.append(ev)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(ev, default=str, ensure_ascii=False) + "\n")
        if self.listener:
            self.listener(ev)
        return ev

    def of_type(self, *types: str) -> list[dict]:
        return [e for e in self.events if e["type"] in types]

    def totals(self) -> dict:
        calls = self.of_type("model_call")
        return {
            "model_calls": len(calls),
            "tool_calls": len(self.of_type("tool_call")),
            "input_tokens": sum(c.get("input_tokens", 0) for c in calls),
            "output_tokens": sum(c.get("output_tokens", 0) for c in calls),
            "estimated_cost_usd": round(sum(c.get("estimated_cost_usd", 0.0) for c in calls), 6),
            "model_latency_ms": sum(c.get("latency_ms", 0) for c in calls),
        }
