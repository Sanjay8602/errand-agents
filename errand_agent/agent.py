"""The errand agent orchestrator.

Responsibilities that are deliberately *not* delegated to the model:
  * approval enforcement (ApprovalGate) for book / cancel / reschedule
  * idempotency keys, retries, timeout resolution via get_booking_status
  * restart recovery of in-flight side effects
  * hard limits on model turns, tool calls and wall-clock time
  * memory mutation policy (tool output can never trigger a forget)
  * proactive handling of provider cancellations
"""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from .approval import CONSEQUENTIAL_TOOLS, ApprovalGate, ApprovalRequest
from .clock import Clock
from .context import ContextBuilder
from .llm.base import TOOL_NAMES, TOOL_SCHEMAS, Model, ModelResponse
from .memory import MemoryStore
from .mocks.tools import MockToolEnvironment, ToolError, ToolTimeout
from .sanitize import truncate_for_trace, wrap_untrusted
from .state import TRANSITIONS, IllegalTransition, PendingOperation, Task, WorkflowState
from .storage import JsonStore
from .trace import Tracer, estimate_cost_usd

FORGET_INTENT = re.compile(r"\b(forget|delete|remove|erase|don'?t (remember|store|keep)|clear (my )?memory)\b", re.I)
REMEMBER_INTENT = re.compile(r"\b(prefer|always|never|from now on|remember|usually|only|my (area|budget|locality) is)\b", re.I)


class SimulatedCrash(RuntimeError):
    """Raised by tests to simulate the process dying at a chosen point."""


@dataclass
class Limits:
    max_model_turns: int = 8          # model calls per user message
    max_tool_calls: int = 12          # tool executions per user message
    wall_clock_seconds: float = 30.0  # per user message
    tool_retries: int = 2             # retries for transient tool errors (same idempotency key)
    model_retries: int = 1            # retries for transient model errors (429 / 5xx / network)
    model_retry_backoff_s: float = 0.0  # base delay before a model retry (doubles each attempt); 0 in tests
    max_retry_wait_s: float = 60.0    # if the provider asks us to wait longer than this, give up instead


@dataclass
class AgentReply:
    text: str
    state: str
    approval_id: Optional[str] = None
    outcome: Optional[str] = None     # approval_required | booked | cancelled | rescheduled | failed | info | limit_exhausted
    data: dict = field(default_factory=dict)


class ErrandAgent:
    def __init__(self, env: MockToolEnvironment, model: Model, data_dir: Path, clock: Clock,
                 tracer: Optional[Tracer] = None, limits: Optional[Limits] = None,
                 context: Optional[ContextBuilder] = None, user_id: str = "user-1",
                 crash_hook: Optional[Callable[[str], None]] = None):
        self.env = env
        self.model = model
        self.clock = clock
        self.limits = limits or Limits()
        self.context = context or ContextBuilder()
        self.user_id = user_id
        self.data_dir = Path(data_dir)
        self.tracer = tracer or Tracer(None)
        self.crash_hook = crash_hook or (lambda _p: None)
        self.memory = MemoryStore(self.data_dir / "memory.json", now=self.clock.now)
        self.store = JsonStore(self.data_dir / "agent_state.json")
        raw = self.store.load({})
        self.task = Task.from_dict(raw.get("task", {}))
        self.state = WorkflowState(raw.get("state", "IDLE"))
        self.history: list[dict] = raw.get("history", [])
        self.gate = ApprovalGate(ApprovalRequest.from_dict(raw["pending_approval"]) if raw.get("pending_approval") else None)
        self.pending_ops: list[PendingOperation] = [PendingOperation(**p) for p in raw.get("pending_ops", [])]
        self.holds: dict[str, dict] = raw.get("holds", {})
        self.bookings: dict[str, dict] = raw.get("bookings", {})
        self.turn = raw.get("turn", 0)
        self.last_user_text: str = raw.get("last_user_text", "")
        self.recovery_notices: list[str] = []
        self._turn_cache: dict[tuple, Any] = {}
        self.tracer.event("session_start", state=self.state.value, pending_ops=len(self.pending_ops),
                          model=getattr(model, "name", "?"))
        self._recover_in_flight()

    # =================================================================== public
    def handle(self, user_text: str) -> AgentReply:
        """Process one user message. Always returns a user-facing outcome."""
        self.turn += 1
        self.last_user_text = user_text
        self.tracer.event("user_message", turn=self.turn, text=user_text)
        notices = self._drain_notices()
        try:
            reply = self._handle(user_text)
        except SimulatedCrash:
            raise
        except Exception as e:  # last-resort: never hang or fail silently
            self.tracer.event("unexpected_error", error=repr(e))
            self._transition(WorkflowState.FAILED, reason=f"unexpected error: {e!r}")
            reply = self._reply("Something went wrong on my side and I stopped before taking any action. "
                                "Nothing was booked, cancelled or moved. Please try again.", outcome="failed")
        if notices:
            reply.text = "\n".join(notices) + "\n\n" + reply.text
        self._persist()
        return reply

    def check_events(self) -> list[AgentReply]:
        """Proactive follow-up: detect provider-side events without a user prompt."""
        replies = []
        for ev in self.env.poll_events():
            self.tracer.event("event_detected", event=ev)
            if ev["type"] == "provider_cancelled":
                replies.append(self._handle_provider_cancellation(ev))
        for n in self._drain_notices():
            replies.append(self._reply(n, outcome="info"))
        self._persist()
        return replies

    # ============================================================ main routing
    def _handle(self, user_text: str) -> AgentReply:
        if self.state == WorkflowState.AWAITING_APPROVAL and self.gate.pending:
            decision = ApprovalGate.classify_reply(user_text)
            self.tracer.event("approval_reply", decision=decision, approval_id=self.gate.pending.approval_id)
            if decision == "approve":
                req = self.gate.approve()
                self.tracer.event("approval_granted", approval_id=req.approval_id, action=req.action,
                                  fingerprint=req.fingerprint)
                self._append_user(user_text)
                return self._execute_approved(req)
            if decision == "reject":
                req = self.gate.reject()
                self.tracer.event("approval_rejected", approval_id=req.approval_id)
                self._append_user(user_text)
                self._transition(WorkflowState.IDLE, reason="user declined")
                return self._reply_and_record(f"Okay, I won't {req.action} anything. Tell me if you want a different option.",
                                              outcome="rejected")
            req = self.gate.invalidate("user sent a new request instead of confirming")
            self.tracer.event("approval_invalidated", approval_id=req.approval_id, reason="new user request")
            self._transition(WorkflowState.SEARCHING, reason="pending approval superseded")
            self._append_user(user_text)
            self._append_assistant_text(
                f"(The pending {req.action} request was withdrawn because the user did not confirm it. "
                f"Their next message is a new instruction - handle it on its own; do not re-file the {req.action} "
                f"unless they ask for it again.)")
            # fall through to the model with the fresh request
            return self._model_loop()

        if self.state == WorkflowState.COMPLETED and ApprovalGate.classify_reply(user_text) == "approve":
            # Repeated confirmation after success must be a no-op.
            last = self._last_completed()
            self._append_user(user_text)
            return self._reply_and_record(
                f"Already done - {last} Nothing further was booked, cancelled or moved.", outcome="noop")

        if self.state in (WorkflowState.COMPLETED, WorkflowState.FAILED, WorkflowState.IDLE):
            self._transition(WorkflowState.COLLECTING, reason="new user message")
        self._append_user(user_text)
        return self._model_loop()

    # ============================================================== model loop
    def _model_loop(self) -> AgentReply:
        deadline = self.clock.monotonic() + self.limits.wall_clock_seconds
        model_turns = 0
        tool_calls = 0
        self._turn_cache: dict[tuple, Any] = {}
        while True:
            if model_turns >= self.limits.max_model_turns:
                return self._exhausted("model_turns", model_turns, tool_calls)
            if self.clock.monotonic() > deadline:
                return self._exhausted("wall_clock", model_turns, tool_calls)

            system, msgs, report = self.context.build(
                self.turn, self.task, self.state, self.memory, self.history,
                self.clock.now().strftime("%Y-%m-%dT%H:%M"),
                extra_state={"Last completed action": self._last_completed() or "none",
                             "User id for list_bookings": self.user_id})
            self.tracer.event("context_sent", **report.to_dict(), system_prompt=report.rendered_system)

            resp = self._call_model(system, msgs)
            model_turns += 1
            if resp is None:
                return self._exhausted("model_error", model_turns, tool_calls)
            self.history.append({"role": "assistant", "content": resp.content})

            if not resp.tool_uses:
                text = resp.text or "(no response)"
                if self.state in (WorkflowState.COLLECTING, WorkflowState.SEARCHING):
                    self._transition(WorkflowState.COLLECTING if self.task.missing() else WorkflowState.SEARCHING,
                                     reason="model replied with text")
                return self._reply(text, outcome="info")

            results = []
            filed: Optional[ApprovalRequest] = None
            exhausted = False
            for tu in resp.tool_uses:
                if tool_calls >= self.limits.max_tool_calls:
                    results.append(self._tool_result(tu, {"error": "tool_budget_exhausted",
                                                          "message": "no more tool calls allowed this turn"}, is_error=True))
                    exhausted = True
                    continue
                tool_calls += 1
                if tu["name"] in CONSEQUENTIAL_TOOLS:
                    if filed is None:
                        filed, payload, is_err = self._file_approval(tu)
                    else:
                        payload, is_err = {"error": "one_action_at_a_time",
                                           "message": "another action is already awaiting approval"}, True
                    results.append(self._tool_result(tu, payload, is_error=is_err))
                else:
                    payload, is_err = self._execute_tool(tu)
                    results.append(self._tool_result(tu, payload, is_error=is_err))
            self.history.append({"role": "user", "content": results})

            if filed is not None:
                self._transition(WorkflowState.AWAITING_APPROVAL, reason=f"{filed.action} requires approval")
                self._append_assistant_text(filed.summary)
                return self._reply(filed.summary, approval_id=filed.approval_id, outcome="approval_required")
            if exhausted:
                return self._exhausted("tool_calls", model_turns, tool_calls)
            if self.state == WorkflowState.COLLECTING and not self.task.missing():
                self._transition(WorkflowState.SEARCHING, reason="all required details present")

    def _call_model(self, system: str, msgs: list[dict]) -> Optional[ModelResponse]:
        attempts = 0
        while True:
            t0 = time.perf_counter()  # real latency even under a fake clock
            try:
                resp = self.model.complete(system, msgs, TOOL_SCHEMAS)
            except Exception as e:  # ModelError or anything from the SDK
                attempts += 1
                transient = getattr(e, "transient", False)
                retry_after = getattr(e, "retry_after", None)
                self.tracer.event("model_error", error=repr(e), transient=transient, attempt=attempts,
                                  retry_after=retry_after)
                self._last_model_error = e
                if transient and attempts <= self.limits.model_retries:
                    # Rate limits (429) and 5xx deserve a pause, not an instant hammer. If the provider
                    # named a wait, honour it; if that wait is longer than we are willing to block, stop now.
                    delay = min(self.limits.model_retry_backoff_s * (2 ** (attempts - 1)), 30.0)
                    if retry_after is not None:
                        if retry_after > self.limits.max_retry_wait_s:
                            self.tracer.event("model_retry_abandoned", retry_after=retry_after)
                            return None
                        delay = max(delay, retry_after + 1.0)
                    self.tracer.event("model_retry_wait", seconds=round(delay, 1), attempt=attempts)
                    time.sleep(delay)
                    continue
                return None
            latency = int((time.perf_counter() - t0) * 1000)
            self.tracer.event("model_call", model=resp.model, latency_ms=latency, stop_reason=resp.stop_reason,
                              input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
                              estimated_cost_usd=estimate_cost_usd(resp.model, resp.input_tokens, resp.output_tokens),
                              tools_requested=[t["name"] for t in resp.tool_uses],
                              tool_inputs=[t["input"] for t in resp.tool_uses],
                              text_preview=resp.text[:300])
            if resp.stop_reason == "max_tokens":
                self.tracer.event("model_error", error="max_tokens", transient=False)
            return resp

    # ============================================================ tool routing
    READ_ONLY_TOOLS = ("search_providers", "get_provider", "list_bookings", "inspect_memory", "get_booking_status")

    def _execute_tool(self, tu: dict) -> tuple[Any, bool]:
        name, args = tu["name"], dict(tu.get("input") or {})
        if name not in TOOL_NAMES:
            return {"error": "unknown_tool", "message": f"{name} is not available"}, True
        # A model that repeats an identical read call in the same turn is looping; hand back the
        # cached result with a nudge instead of spending tool budget (seen live with a small model).
        key = (name, json.dumps(args, sort_keys=True, default=str))
        if name in self.READ_ONLY_TOOLS and key in self._turn_cache:
            self.tracer.event("duplicate_call", tool=name, args=args)
            cached = self._turn_cache[key]
            if isinstance(cached, dict):
                cached = {**cached, "application_note": "You already made this exact call this turn; this is the "
                                                        "same result. Use it and move to the next step."}
            return cached, False
        payload, is_err = self._execute_tool_inner(name, args)
        if name in self.READ_ONLY_TOOLS and not is_err:
            self._turn_cache[key] = payload
        return payload, is_err

    def _execute_tool_inner(self, name: str, args: dict) -> tuple[Any, bool]:
        if name == "update_task":
            return self._tool_update_task(args), False
        if name == "remember_preference":
            return self._tool_remember(args)
        if name == "inspect_memory":
            # Compact view for the model: key, value and provenance only (ids/timestamps/tags are noise here).
            data = {"memories": [{"key": m["key"], "value": m["value"], "kind": m["kind"], "confirmed": m["confirmed"],
                                  "status": m["status"]} for m in self.memory.inspect()],
                    "note": "status=forgotten items are tombstones and are never used; kind=inferred with confirmed=false "
                            "is a guess that is not used until the user confirms it."}
            self.tracer.event("tool_call", tool=name, args=args, result=truncate_for_trace(data), latency_ms=0)
            return data, False
        if name == "forget_memory":
            return self._tool_forget(args)
        if name == "list_bookings":
            # The session's user is fixed; a model-invented id (seen live: "current_user") would
            # silently return an empty list. The application owns identity, not the model (BUGS.md #4).
            if args.get("user_id") != self.user_id:
                self.tracer.event("arg_override", tool=name, field="user_id", from_value=args.get("user_id"),
                                  to_value=self.user_id)
            args["user_id"] = self.user_id
        payload, is_err = self._call_env(name, args)
        if name == "search_providers" and not is_err:
            payload = _compact_search(payload)
        return payload, is_err

    def _call_env(self, name: str, args: dict, retries: Optional[int] = None) -> tuple[Any, bool]:
        retries = self.limits.tool_retries if retries is None else retries
        attempt = 0
        while True:
            attempt += 1
            t0 = time.perf_counter()
            try:
                result = self.env.call(name, args)
            except ToolError as e:
                latency = int((time.perf_counter() - t0) * 1000)
                self.tracer.event("tool_call", tool=name, args=args, error=e.to_dict(), attempt=attempt, latency_ms=latency)
                if e.transient and attempt <= retries and not isinstance(e, ToolTimeout):
                    continue
                return e.to_dict(), True
            latency = int((time.perf_counter() - t0) * 1000)
            flagged: list[str] = []
            from .sanitize import sanitize  # local import to keep module import light
            sanitized = sanitize(result, flagged)
            self.tracer.event("tool_call", tool=name, args=args, result=truncate_for_trace(sanitized),
                              injection_flags=flagged, attempt=attempt, latency_ms=latency)
            if flagged:
                self.tracer.event("injection_detected", tool=name, paths=flagged)
            if name == "hold_slot" and isinstance(result, dict) and result.get("hold_id"):
                self.holds[result["hold_id"]] = result
            return result, False

    def _tool_result(self, tu: dict, payload: Any, is_error: bool = False) -> dict:
        envelope, _ = wrap_untrusted(tu["name"], payload)
        if tu["name"] == "hold_slot" and not is_error and isinstance(payload, dict) and payload.get("hold_id"):
            envelope += (f"\nApplication note: the slot is held. Call book(hold_id=\"{payload['hold_id']}\", "
                         f"idempotency_key=\"book:{payload['hold_id']}\") now; the application will ask the user to confirm. "
                         "Do not ask the user yourself.")
        block = {"type": "tool_result", "tool_use_id": tu["id"], "content": envelope}
        if is_error:
            block["is_error"] = True
        return block

    def _tool_update_task(self, args: dict) -> dict:
        before = self.task.to_dict()
        changed = self.task.update(**{k: v for k, v in args.items() if k != "filled_from_memory"})
        if args.get("filled_from_memory"):
            for f in args["filled_from_memory"]:
                if f not in self.task.filled_from_memory:
                    self.task.filled_from_memory.append(f)
        self.tracer.event("task_updated", changed=changed, task=self.task.to_dict(),
                          preserved=[k for k in before if k not in changed and before[k] not in (None, {}, [])])
        return {"ok": True, "task": self.task.to_dict(), "missing": self.task.missing()}

    def _tool_remember(self, args: dict) -> tuple[Any, bool]:
        kind = args.get("kind", "inferred")
        if kind == "explicit" and not REMEMBER_INTENT.search(self.last_user_text or ""):
            # The user did not state a preference in this message: store it as an unconfirmed inference.
            self.tracer.event("policy_downgrade", tool="remember_preference", from_kind="explicit", to_kind="inferred",
                              reason="no explicit preference statement in the user's message")
            kind = "inferred"
        m = self.memory.remember(args["key"], args.get("value"), kind=kind, tags=args.get("tags") or ["general"],
                                 source=f"turn {self.turn}: {self.last_user_text[:80]}")
        self.tracer.event("tool_call", tool="remember_preference", args=args, result={"id": m.id, "kind": m.kind,
                                                                                      "confirmed": m.confirmed}, latency_ms=0)
        return {"ok": True, "memory": asdict(m)}, False

    def _tool_forget(self, args: dict) -> tuple[Any, bool]:
        if not FORGET_INTENT.search(self.last_user_text or ""):
            self.tracer.event("policy_denied", tool="forget_memory", args=args,
                              reason="forgetting requires an explicit request in the user's own message")
            return {"error": "policy_denied",
                    "message": "Memory can only be forgotten when the user explicitly asks in their own message."}, True
        key = args.get("key", "")
        if key == "*":
            n = self.memory.forget_all()
            res = {"ok": True, "forgotten": n}
        else:
            m = self.memory.forget(key)
            res = {"ok": bool(m), "forgotten": m.key if m else None}
            if not m:
                res["message"] = f"nothing stored under '{key}'"
        self.tracer.event("tool_call", tool="forget_memory", args=args, result=res, latency_ms=0)
        return res, False

    # ============================================================== approvals
    def _file_approval(self, tu: dict) -> tuple[Optional[ApprovalRequest], Any, bool]:
        """Turn a model's book/cancel/reschedule call into a pending approval. Never executes."""
        name, args = tu["name"], dict(tu.get("input") or {})
        if name == "book":
            hold = self.holds.get(args.get("hold_id", ""))
            if not hold:
                return None, {"error": "unknown_hold", "message": "call hold_slot first; booking needs a valid hold"}, True
            if hold.get("booking_id") or any(b.get("hold_id") == hold["hold_id"] for b in self.bookings.values()):
                return None, {"error": "already_booked", "message": "this hold was already booked"}, True
            details = {"provider_id": hold["provider_id"], "provider_name": hold.get("provider_name"),
                       "service": hold.get("service"), "slot": hold["slot"], "price": hold["quoted_price"],
                       "search_price": hold.get("search_price"), "hold_id": hold["hold_id"],
                       "idempotency_key": args.get("idempotency_key") or f"book:{hold['hold_id']}"}
            summary = self._book_summary(details)
        elif name == "cancel":
            b = self._booking_from_env(args.get("booking_id"))
            if not b:
                return None, {"error": "unknown_booking", "message": "no such booking"}, True
            details = {"booking_id": b["booking_id"], "provider_name": b.get("provider_name"), "service": b["service"],
                       "slot": b["slot"], "idempotency_key": args.get("idempotency_key") or f"cancel:{b['booking_id']}"}
            summary = (f"Please confirm: cancel your {b['service']} booking with {b.get('provider_name') or b['provider_id']} "
                       f"on {_fmt_slot(b['slot'])} (booking {b['booking_id']})? Reply 'yes' to confirm or 'no' to keep it.")
        else:  # reschedule
            b = self._booking_from_env(args.get("booking_id"))
            if not b:
                return None, {"error": "unknown_booking", "message": "no such booking"}, True
            if not args.get("slot"):
                return None, {"error": "missing_slot", "message": "reschedule needs a slot"}, True
            details = {"booking_id": b["booking_id"], "slot": args["slot"], "previous_slot": b["slot"],
                       "provider_name": b.get("provider_name"), "service": b["service"],
                       "idempotency_key": args.get("idempotency_key") or f"reschedule:{b['booking_id']}:{args['slot']}"}
            summary = (f"Please confirm: move your {b['service']} booking with {b.get('provider_name') or b['provider_id']} "
                       f"from {_fmt_slot(b['slot'])} to {_fmt_slot(args['slot'])}? Reply 'yes' to confirm or 'no' to leave it.")
        req = self.gate.request(name, details, summary)
        self.tracer.event("approval_requested", approval_id=req.approval_id, action=name, details=details,
                          fingerprint=req.fingerprint)
        return req, {"status": "approval_required", "approval_id": req.approval_id,
                     "note": "The user must confirm. Do not claim the action happened."}, False

    def _book_summary(self, d: dict) -> str:
        notes = []
        if d.get("search_price") is not None and d["search_price"] != d["price"]:
            notes.append(f"Note: the quoted price ₹{d['price']} differs from the listed ₹{d['search_price']}.")
        if self.task.budget_max and d["price"] > self.task.budget_max:
            notes.append(f"This is ABOVE your ₹{self.task.budget_max} budget.")
        return (f"Please confirm: book {d.get('provider_name') or d['provider_id']} for {d.get('service')} on "
                f"{_fmt_slot(d['slot'])} at ₹{d['price']}? " + " ".join(notes) +
                (" " if notes else "") + "Reply 'yes' to confirm or 'no' to cancel.")

    def _execute_approved(self, req: ApprovalRequest) -> AgentReply:
        self._transition(WorkflowState.EXECUTING, reason=f"approved {req.action}")
        if req.action == "book":
            return self._execute_book(req)
        if req.action == "cancel":
            return self._execute_cancel(req)
        return self._execute_reschedule(req)

    # ----------------------------------------------------------------- book
    def _execute_book(self, req: ApprovalRequest) -> AgentReply:
        d = req.details
        hold = self.holds.get(d["hold_id"])
        # Re-validate the hold against the live clock; re-hold if expired.
        if hold is None or self._hold_expired(hold):
            self.tracer.event("hold_expired", hold_id=d["hold_id"])
            new_hold, err = self._call_env("hold_slot", {"provider_id": d["provider_id"], "slot": d["slot"]})
            if err:
                self.gate.invalidate("hold expired and could not be renewed")
                self._transition(WorkflowState.FAILED, reason="re-hold failed")
                return self._reply_and_record(
                    f"The 10-minute hold expired and I couldn't re-hold {d.get('provider_name')} at {_fmt_slot(d['slot'])} "
                    f"({new_hold.get('message', new_hold.get('error'))}). Nothing was booked. Want me to look for another slot?",
                    outcome="failed")
            hold = new_hold
            d = {**d, "hold_id": hold["hold_id"], "price": hold["quoted_price"],
                 "idempotency_key": f"book:{hold['hold_id']}"}
        live = {"provider_id": hold["provider_id"], "slot": hold["slot"], "price": hold["quoted_price"]}
        ok, why = self.gate.authorize("book", live)
        self.tracer.event("gate_check", action="book", ok=ok, reason=why, live=live)
        if not ok:
            old = self.gate.invalidate(why)
            new_req = self.gate.request("book", d, self._book_summary(d))
            self.tracer.event("approval_invalidated", approval_id=old.approval_id, reason=why)
            self.tracer.event("approval_requested", approval_id=new_req.approval_id, action="book", details=d,
                              fingerprint=new_req.fingerprint, superseded=old.approval_id)
            self._transition(WorkflowState.AWAITING_APPROVAL, reason="material details changed")
            msg = (f"Before I book: {why}. The hold had to be renewed and the terms changed "
                   f"(was ₹{old.details['price']}, now ₹{d['price']}). " + new_req.summary)
            return self._reply_and_record(msg, approval_id=new_req.approval_id, outcome="approval_required")

        op = PendingOperation(op="book", idempotency_key=d["idempotency_key"],
                              args={"hold_id": hold["hold_id"]}, approval_id=req.approval_id)
        self.pending_ops.append(op)
        self._persist()
        self.crash_hook("before_book")
        result = self._book_with_recovery(op, hold)
        return result

    def _book_with_recovery(self, op: PendingOperation, hold: dict, attempts_left: int = 1) -> AgentReply:
        try:
            res = self.env.book(hold_id=hold["hold_id"], idempotency_key=op.idempotency_key)
            self.tracer.event("tool_call", tool="book", args={"hold_id": hold["hold_id"], "idempotency_key": op.idempotency_key},
                              result=res, latency_ms=0)
        except ToolTimeout as e:
            self.tracer.event("tool_call", tool="book", args={"hold_id": hold["hold_id"], "idempotency_key": op.idempotency_key},
                              error=e.to_dict(), latency_ms=0)
            self.crash_hook("after_book_timeout")
            return self._resolve_ambiguous_book(op, hold, attempts_left)
        except ToolError as e:
            self.tracer.event("tool_call", tool="book", args={"hold_id": hold["hold_id"]}, error=e.to_dict(), latency_ms=0)
            if e.transient and attempts_left > 0:
                return self._book_with_recovery(op, hold, attempts_left - 1)
            return self._finish_op(op, "failed", f"The booking service returned an error ({e.code}). Nothing was booked.",
                                   WorkflowState.FAILED, outcome="failed")
        self.crash_hook("after_book_success")
        status = res.get("status")
        if status == "success":
            return self._confirm_booking(op, res["booking"])
        if status == "already_booked":
            b = self._booking_from_env(res.get("booking_id")) or {}
            return self._confirm_booking(op, b, note="(it was already booked earlier)")
        if status == "slot_taken":
            return self._finish_op(op, "failed",
                                   f"Sorry - the {_fmt_slot(hold['slot'])} slot with {hold.get('provider_name')} was taken "
                                   f"before I could book it. Nothing was booked. Shall I look for the next best option?",
                                   WorkflowState.FAILED, outcome="slot_taken")
        if status == "hold_expired":
            self.pending_ops.remove(op)
            self.holds.pop(hold["hold_id"], None)
            # Re-run the approved execution path, which will re-hold and re-check the fingerprint.
            return self._execute_book(self.gate.pending) if self.gate.pending else self._finish_op(
                op, "failed", "The hold expired. Nothing was booked.", WorkflowState.FAILED, outcome="failed")
        return self._finish_op(op, "failed", f"Unexpected booking result ({status}). Nothing was booked.",
                               WorkflowState.FAILED, outcome="failed")

    def _resolve_ambiguous_book(self, op: PendingOperation, hold: dict, attempts_left: int) -> AgentReply:
        """Timeout: ask the source of truth before deciding to retry or report."""
        self._transition(WorkflowState.VERIFYING, reason="book timed out; checking booking status")
        status, err = self._call_env("get_booking_status", {"idempotency_key": op.idempotency_key, "hold_id": hold["hold_id"]})
        if err:
            return self._finish_op(op, "failed",
                                   "The booking request timed out and I couldn't reach the booking status service to verify it. "
                                   "I have NOT confirmed anything. Please check your bookings in a minute or ask me to try again.",
                                   WorkflowState.FAILED, outcome="unverified")
        self.tracer.event("verification", source="get_booking_status", result=status)
        if status.get("status") in ("confirmed", "rescheduled") and status.get("booking"):
            return self._confirm_booking(op, status["booking"], verified_status=status,
                                         note="(the request timed out, but the booking service confirms it went through)")
        if attempts_left > 0 and not self._hold_expired(hold):
            self.tracer.event("retry", op="book", idempotency_key=op.idempotency_key, reason="status says not booked")
            self._transition(WorkflowState.EXECUTING, reason="retrying book with same idempotency key")
            return self._book_with_recovery(op, hold, attempts_left - 1)
        return self._finish_op(op, "failed",
                               "The booking request timed out and the booking service confirms nothing was booked. "
                               "I did not retry further. Want me to try once more or pick another slot?",
                               WorkflowState.FAILED, outcome="failed")

    def _confirm_booking(self, op: PendingOperation, booking: dict, note: str = "",
                         verified_status: Optional[dict] = None) -> AgentReply:
        # Verify against the source of truth even on a direct success (unless the caller just did).
        if verified_status is not None:
            status, err = verified_status, False
        else:
            status, err = self._call_env("get_booking_status", {"idempotency_key": op.idempotency_key})
        verified = (not err) and status.get("booking") and status["booking"]["booking_id"] == booking.get("booking_id")
        self.tracer.event("booking_verified" if verified else "verification_failed", booking_id=booking.get("booking_id"),
                          status=status)
        if not verified:
            return self._finish_op(op, "failed", "I got a success response but could not verify the booking with the "
                                   "booking service, so I am not treating it as confirmed. Please check your bookings.",
                                   WorkflowState.FAILED, outcome="unverified")
        rec = {**booking, "area": self.task.area, "time_window": self.task.time_window, "budget_max": self.task.budget_max}
        self.bookings[booking["booking_id"]] = rec
        if booking.get("hold_id") in self.holds:
            self.holds[booking["hold_id"]]["booking_id"] = booking["booking_id"]
        text = (f"Booked ✔ {booking.get('provider_name') or booking['provider_id']} for {booking['service']} on "
                f"{_fmt_slot(booking['slot'])} at ₹{booking['price']}. Booking ID {booking['booking_id']}. {note}").strip()
        return self._finish_op(op, "done", text, WorkflowState.COMPLETED, outcome="booked", result=booking)

    # --------------------------------------------------------------- cancel
    def _execute_cancel(self, req: ApprovalRequest) -> AgentReply:
        d = req.details
        ok, why = self.gate.authorize("cancel", {"booking_id": d["booking_id"]})
        self.tracer.event("gate_check", action="cancel", ok=ok, reason=why)
        if not ok:
            self._transition(WorkflowState.FAILED, reason=why)
            return self._reply_and_record(f"I can't cancel: {why}.", outcome="failed")
        op = PendingOperation(op="cancel", idempotency_key=d["idempotency_key"], args={"booking_id": d["booking_id"]},
                              approval_id=req.approval_id)
        self.pending_ops.append(op)
        self._persist()
        res, err = self._call_env("cancel", {"booking_id": d["booking_id"], "idempotency_key": d["idempotency_key"]})
        if err:
            return self._finish_op(op, "failed", f"I couldn't cancel booking {d['booking_id']} ({res.get('message')}). "
                                   "It is still active.", WorkflowState.FAILED, outcome="failed")
        if d["booking_id"] in self.bookings:
            self.bookings[d["booking_id"]]["status"] = "cancelled"
        return self._finish_op(op, "done", f"Cancelled ✔ your {d['service']} booking on {_fmt_slot(d['slot'])} "
                               f"(booking {d['booking_id']}).", WorkflowState.COMPLETED, outcome="cancelled", result=res)

    # ----------------------------------------------------------- reschedule
    def _execute_reschedule(self, req: ApprovalRequest) -> AgentReply:
        d = req.details
        ok, why = self.gate.authorize("reschedule", {"booking_id": d["booking_id"], "slot": d["slot"]})
        self.tracer.event("gate_check", action="reschedule", ok=ok, reason=why)
        if not ok:
            self._transition(WorkflowState.FAILED, reason=why)
            return self._reply_and_record(f"I can't reschedule: {why}.", outcome="failed")
        op = PendingOperation(op="reschedule", idempotency_key=d["idempotency_key"],
                              args={"booking_id": d["booking_id"], "slot": d["slot"]}, approval_id=req.approval_id)
        self.pending_ops.append(op)
        self._persist()
        res, err = self._call_env("reschedule", {"booking_id": d["booking_id"], "slot": d["slot"],
                                                 "idempotency_key": d["idempotency_key"]})
        if err:
            return self._finish_op(op, "failed", f"I couldn't move the booking ({res.get('message')}). "
                                   f"It stays at {_fmt_slot(d['previous_slot'])}.", WorkflowState.FAILED, outcome="failed")
        if res.get("status") == "slot_conflict":
            self.task.target_slot = None
            return self._finish_op(op, "failed", f"{_fmt_slot(d['slot'])} isn't available for {d.get('provider_name')} - "
                                   f"your booking stays at {_fmt_slot(d['previous_slot'])}. What other time would work?",
                                   WorkflowState.FAILED, outcome="slot_conflict")
        if res.get("status") != "rescheduled":
            return self._finish_op(op, "failed", f"The booking couldn't be moved ({res.get('status')}). "
                                   f"It stays at {_fmt_slot(d['previous_slot'])}.", WorkflowState.FAILED, outcome="failed")
        if d["booking_id"] in self.bookings:
            self.bookings[d["booking_id"]].update({"slot": d["slot"], "status": "rescheduled"})
        return self._finish_op(op, "done", f"Moved ✔ your {d['service']} booking to {_fmt_slot(d['slot'])} "
                               f"(booking {d['booking_id']}).", WorkflowState.COMPLETED, outcome="rescheduled", result=res)

    # ------------------------------------------------------- proactive events
    def _handle_provider_cancellation(self, ev: dict) -> AgentReply:
        b = ev["booking"]
        rec = self.bookings.get(b["booking_id"], {})
        if b["booking_id"] in self.bookings:
            self.bookings[b["booking_id"]]["status"] = "cancelled_by_provider"
        if self.gate.pending:
            self.gate.invalidate("provider cancellation changed the situation")
        head = (f"Heads up: {b.get('provider_name') or b['provider_id']} cancelled your {b['service']} visit on "
                f"{_fmt_slot(b['slot'])} ({ev.get('reason')}).")
        area = rec.get("area") or self.memory.value("preferred_area")
        budget = rec.get("budget_max") or _to_int(self.memory.value("default_budget"))
        tw = rec.get("time_window") or _tw_from_memory(self.memory.value("earliest_time"))
        if not area:
            self._transition(WorkflowState.COLLECTING, reason="provider cancelled; need area to find replacement")
            self.task = Task(intent="book", service=b["service"], budget_max=budget, time_window=tw)
            return self._reply_and_record(head + " I can find a replacement - which area should I search in?", outcome="info")
        day = b["slot"][:10]
        search, err = self._call_env("search_providers", {"service": b["service"], "area": area,
                                                          "date_range": {"start": day, "end": day}})
        candidates = [] if err else [p for p in search.get("providers", []) if p["provider_id"] != b["provider_id"]]
        pick, slot = _best(candidates, budget, tw)
        if not pick:
            self._transition(WorkflowState.IDLE, reason="no replacement found")
            return self._reply_and_record(head + f" I looked for a replacement in {area} on {day} within your preferences "
                                          "but found nothing suitable. Tell me if you'd like a different day.", outcome="info")
        hold, err = self._call_env("hold_slot", {"provider_id": pick["provider_id"], "slot": slot})
        if err:
            self._transition(WorkflowState.IDLE, reason="replacement hold failed")
            return self._reply_and_record(head + " I found an alternative but couldn't hold it. Want me to try again?", outcome="info")
        self.task = Task(intent="book", service=b["service"], area=area, budget_max=budget, time_window=tw,
                         date_range={"start": day, "end": day})
        details = {"provider_id": hold["provider_id"], "provider_name": hold.get("provider_name"), "service": b["service"],
                   "slot": hold["slot"], "price": hold["quoted_price"], "search_price": hold.get("search_price"),
                   "hold_id": hold["hold_id"], "idempotency_key": f"book:{hold['hold_id']}"}
        req = self.gate.request("book", details, self._book_summary(details))
        self.tracer.event("approval_requested", approval_id=req.approval_id, action="book", details=details,
                          fingerprint=req.fingerprint, proactive=True)
        self._transition(WorkflowState.AWAITING_APPROVAL, reason="proactive replacement proposed")
        text = head + f" I found a replacement within your preferences: {pick['name']} (rating {pick['rating']}) at " \
                      f"{_fmt_slot(slot)} for ₹{hold['quoted_price']}. " + req.summary
        return self._reply_and_record(text, approval_id=req.approval_id, outcome="approval_required")

    # ------------------------------------------------------- restart recovery
    def _recover_in_flight(self) -> None:
        for op in list(self.pending_ops):
            if op.status != "in_flight":
                continue
            self.tracer.event("recovery_started", op=op.op, idempotency_key=op.idempotency_key)
            if op.op == "book":
                status, err = self._call_env("get_booking_status", {"idempotency_key": op.idempotency_key,
                                                                    "hold_id": op.args.get("hold_id")})
                if not err and status.get("booking"):
                    bk = status["booking"]
                    op.status, op.result = "done", bk
                    self.bookings[bk["booking_id"]] = {**bk, "area": self.task.area}
                    self.gate.consume()
                    self.task = Task()  # fulfilled; same reset as _finish_op (BUGS.md #2)
                    self.state = WorkflowState.COMPLETED
                    self.recovery_notices.append(
                        f"While I was restarting, I verified that your {bk['service']} booking with "
                        f"{bk.get('provider_name')} on {_fmt_slot(bk['slot'])} did go through (booking {bk['booking_id']}).")
                    self.tracer.event("recovery_resolved", op="book", result="booked", booking_id=bk["booking_id"])
                else:
                    op.status = "failed"
                    self.gate.invalidate("process restarted before the booking completed")
                    self.state = WorkflowState.FAILED
                    self.recovery_notices.append(
                        "I restarted while a booking was in progress. The booking service confirms nothing was booked. "
                        "Nothing was charged; tell me if you'd like me to try again.")
                    self.tracer.event("recovery_resolved", op="book", result="not_booked")
            else:
                # cancel / reschedule are idempotent by key; re-issue once.
                res, err = self._call_env(op.op, {**op.args, "idempotency_key": op.idempotency_key})
                op.status = "failed" if err else "done"
                op.result = res
                self.tracer.event("recovery_resolved", op=op.op, result=res)
        self._persist()

    # ================================================================ helpers
    def _finish_op(self, op: PendingOperation, status: str, text: str, state: WorkflowState, outcome: str,
                   result: Optional[dict] = None, **extra) -> AgentReply:
        op.status, op.result = status, result
        if status == "done":
            self.gate.consume()
            # The task is fulfilled. Clearing it stops a later unrelated message from
            # re-entering the booking flow with stale details (BUGS.md #2).
            self.task = Task()
        else:
            self.gate.invalidate(f"operation {status}: {outcome}")
        self._transition(state, reason=outcome)
        self._persist()
        return self._reply_and_record(text, outcome=outcome, **extra)

    def _exhausted(self, reason: str, model_turns: int, tool_calls: int) -> AgentReply:
        self.tracer.event("limit_exhausted", reason=reason, model_turns=model_turns, tool_calls=tool_calls)
        self._transition(WorkflowState.FAILED, reason=f"limit exhausted: {reason}")
        human = {"model_turns": "I went back and forth too many times", "tool_calls": "I ran out of tool-call budget",
                 "wall_clock": "this took longer than my time limit", "model_error": "my language model is unavailable"}[reason]
        if reason == "model_error":
            err = getattr(self, "_last_model_error", None)
            ra = getattr(err, "retry_after", None)
            if getattr(err, "code", "") == "rate_limited":
                human = "the language model's rate limit was hit"
                if ra:
                    human += f" (the provider asks for a wait of about {int(ra // 60)} min {int(ra % 60)} s)"
        summary = ", ".join(f"{k}={v}" for k, v in self.task.to_dict().items() if v not in (None, {}, []))
        return self._reply_and_record(
            f"I had to stop: {human}. Nothing was booked, cancelled or moved. "
            f"What I have so far: {summary or 'nothing yet'}. You can retry or adjust the request.",
            outcome="limit_exhausted")

    def _reply(self, text: str, approval_id: Optional[str] = None, outcome: Optional[str] = None, **data) -> AgentReply:
        self.tracer.event("agent_message", text=text, state=self.state.value, outcome=outcome, approval_id=approval_id)
        return AgentReply(text=text, state=self.state.value, approval_id=approval_id, outcome=outcome, data=data)

    def _reply_and_record(self, text: str, **kw) -> AgentReply:
        self._append_assistant_text(text)
        return self._reply(text, **kw)

    def _append_user(self, text: str) -> None:
        self.history.append({"role": "user", "content": text})

    def _append_assistant_text(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": [{"type": "text", "text": text}]})

    def _transition(self, new: WorkflowState, reason: str = "") -> None:
        if new not in TRANSITIONS[self.state]:
            raise IllegalTransition(f"{self.state.value} -> {new.value} ({reason})")
        if new != self.state:
            self.tracer.event("state_transition", from_state=self.state.value, to_state=new.value, reason=reason)
        self.state = new

    def _hold_expired(self, hold: dict) -> bool:
        return self.clock.now() >= datetime.strptime(hold["expires_at"], "%Y-%m-%dT%H:%M:%S")

    def _booking_from_env(self, booking_id: Optional[str]) -> Optional[dict]:
        if not booking_id:
            return None
        res, err = self._call_env("list_bookings", {"user_id": self.user_id}, retries=1)
        if err:
            return None
        for b in res.get("bookings", []):
            if b["booking_id"] == booking_id:
                return b
        return None

    def _last_completed(self) -> str:
        for op in reversed(self.pending_ops):
            if op.status == "done" and op.result:
                r = op.result
                if op.op == "book":
                    return f"booking {r.get('booking_id')} ({r.get('service')} on {_fmt_slot(r.get('slot', ''))}) is confirmed."
                if op.op == "cancel":
                    return f"booking {r.get('booking_id')} is cancelled."
                return f"booking {r.get('booking_id')} was rescheduled."
        return ""

    def _drain_notices(self) -> list[str]:
        n, self.recovery_notices = self.recovery_notices, []
        return n

    def _persist(self) -> None:
        self.store.save({
            "task": self.task.to_dict(), "state": self.state.value, "history": self.history,
            "pending_approval": self.gate.pending.to_dict() if self.gate.pending else None,
            "pending_ops": [asdict(p) for p in self.pending_ops], "holds": self.holds, "bookings": self.bookings,
            "turn": self.turn, "last_user_text": self.last_user_text, "user_id": self.user_id,
        })


# ---------------------------------------------------------------- utilities
def _compact_search(res: dict) -> dict:
    """What the model needs from a search, at a fraction of the tokens (the trace keeps the full result)."""
    out = {k: v for k, v in res.items() if k != "providers"}
    out["providers"] = [{
        "provider_id": p["provider_id"], "name": p["name"], "price": p["price"], "rating": p["rating"],
        "distance_km": p.get("distance_km"), "verified": p.get("attributes", {}).get("verified"),
        "morning_available": p.get("attributes", {}).get("morning_available"),
        "slots": p.get("available_slots", [])[:5],
    } for p in res.get("providers", [])]
    return out


def _fmt_slot(slot: str) -> str:
    try:
        dt = datetime.strptime(slot, "%Y-%m-%dT%H:%M")
        return dt.strftime("%a %d %b, %H:%M")
    except (ValueError, TypeError):
        return str(slot)


def _to_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(re.sub(r"\D", "", str(v)))
    except ValueError:
        return None


def _tw_from_memory(v: Any) -> Optional[dict]:
    if not v:
        return None
    m = re.search(r"(\d{2}:\d{2})", str(v))
    if m:
        return {"earliest": m.group(1)}
    if "morning" in str(v).lower():
        return {"earliest": "08:00", "latest": "12:00"}
    return None


def _best(providers: list[dict], budget: Optional[int], tw: Optional[dict]) -> tuple[Optional[dict], Optional[str]]:
    ranked = []
    for p in providers:
        if budget and p["price"] > budget:
            continue
        for s in p.get("available_slots", []):
            t = s[11:16]
            if tw and tw.get("earliest") and t < tw["earliest"]:
                continue
            if tw and tw.get("latest") and t > tw["latest"]:
                continue
            ranked.append((-p["rating"], p["price"], p["provider_id"], p, s))
            break
    if not ranked:
        return None, None
    ranked.sort(key=lambda r: r[:3])
    return ranked[0][3], ranked[0][4]
