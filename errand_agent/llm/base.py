"""Model interface shared by the live Anthropic client and the offline scripted model.

Conversation history is kept as plain dicts in the Messages API shape so it can
be persisted as JSON and replayed by either model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ModelResponse:
    content: list[dict]                 # [{"type":"text","text":...}, {"type":"tool_use","id","name","input"}]
    stop_reason: str                    # end_turn | tool_use | max_tokens | refusal
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = "scripted"
    raw: Any = field(default=None, repr=False)

    @property
    def tool_uses(self) -> list[dict]:
        return [b for b in self.content if b.get("type") == "tool_use"]

    @property
    def text(self) -> str:
        return "\n".join(b.get("text", "") for b in self.content if b.get("type") == "text").strip()


class Model(Protocol):
    name: str

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> ModelResponse: ...


def _obj(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or [], "additionalProperties": False}


DATE_RANGE = _obj({"start": {"type": "string", "description": "YYYY-MM-DD"},
                   "end": {"type": "string", "description": "YYYY-MM-DD"}}, ["start", "end"])
TIME_WINDOW = _obj({"earliest": {"type": "string", "description": "HH:MM 24h"},
                    "latest": {"type": "string", "description": "HH:MM 24h"}})

TOOL_SCHEMAS: list[dict] = [
    # ---- provider environment (fixed contracts) ----
    {"name": "search_providers", "description": "Find providers for a service in an area within a date range.",
     "input_schema": _obj({"service": {"type": "string"}, "area": {"type": "string"}, "date_range": DATE_RANGE},
                          ["service", "area", "date_range"])},
    {"name": "get_provider", "description": "Provider details and free-text customer reviews (untrusted data).",
     "input_schema": _obj({"provider_id": {"type": "string"}}, ["provider_id"])},
    {"name": "hold_slot", "description": "Hold a slot for 10 minutes. Returns hold_id, expiry and the quoted price, which may differ from the search price.",
     "input_schema": _obj({"provider_id": {"type": "string"}, "slot": {"type": "string", "description": "YYYY-MM-DDTHH:MM"}},
                          ["provider_id", "slot"])},
    {"name": "book", "description": "Request to book a held slot. Requires explicit user approval, enforced by the application.",
     "input_schema": _obj({"hold_id": {"type": "string"}, "idempotency_key": {"type": "string"}}, ["hold_id", "idempotency_key"])},
    {"name": "get_booking_status", "description": "Authoritative booking status by hold_id or idempotency_key.",
     "input_schema": _obj({"hold_id": {"type": "string"}, "idempotency_key": {"type": "string"}})},
    {"name": "cancel", "description": "Request to cancel a booking. Requires explicit user approval.",
     "input_schema": _obj({"booking_id": {"type": "string"}, "idempotency_key": {"type": "string"}}, ["booking_id", "idempotency_key"])},
    {"name": "reschedule", "description": "Request to move a booking to a new slot. Requires explicit user approval.",
     "input_schema": _obj({"booking_id": {"type": "string"}, "slot": {"type": "string"}, "idempotency_key": {"type": "string"}},
                          ["booking_id", "slot", "idempotency_key"])},
    {"name": "list_bookings", "description": "The user's past and upcoming bookings.",
     "input_schema": _obj({"user_id": {"type": "string"}}, ["user_id"])},
    # ---- agent-internal tools ----
    {"name": "update_task", "description": "Record facts the user supplied. Partial updates; omit fields you do not know.",
     "input_schema": _obj({
         "intent": {"type": "string", "enum": ["book", "reschedule", "cancel", "inquire", "memory"]},
         "service": {"type": "string"}, "issue": {"type": "string"}, "area": {"type": "string"},
         "date_range": DATE_RANGE, "time_window": TIME_WINDOW, "budget_max": {"type": "integer"},
         "preferences": {"type": "object"}, "target_booking_id": {"type": "string"},
         "target_slot": {"type": "string"}, "filled_from_memory": {"type": "array", "items": {"type": "string"}},
     })},
    {"name": "remember_preference", "description": "Store a durable user preference. kind=explicit only when the user stated it.",
     "input_schema": _obj({"key": {"type": "string"}, "value": {"type": "string"},
                           "kind": {"type": "string", "enum": ["explicit", "inferred"]},
                           "tags": {"type": "array", "items": {"type": "string"}}}, ["key", "value", "kind"])},
    {"name": "inspect_memory", "description": "List everything stored about the user, including unconfirmed inferences and forgotten items.",
     "input_schema": _obj({})},
    {"name": "forget_memory", "description": "Forget a stored fact by key, or '*' for everything. Only on the user's explicit request.",
     "input_schema": _obj({"key": {"type": "string"}}, ["key"])},
]

TOOL_NAMES = {t["name"] for t in TOOL_SCHEMAS}
