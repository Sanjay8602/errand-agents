"""Deterministic scripted model for offline evaluation.

It behaves like a competent LLM following the system prompt, but with no
randomness: given the same context it always produces the same tool calls.
It only reads what a real model would read - the rendered system prompt
(task state, memories, current time) and the message history - so context
budget tests are meaningful for it too.

`obey_injections=True` simulates a *compromised* model that follows
instructions found inside tool results. The application must still hold:
no booking without approval, no memory wipe without a user request.
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from typing import Any, Optional

from .base import ModelResponse
from .nlp import extract_facts, parse_clock_time, parse_memory_op, parse_service

DEFAULT_USER = "user-1"


def _tool_use(name: str, inp: dict) -> dict:
    return {"type": "tool_use", "id": "toolu_" + uuid.uuid4().hex[:10], "name": name, "input": inp}


def _text(t: str) -> dict:
    return {"type": "text", "text": t}


class ScriptedModel:
    name = "scripted"

    def __init__(self, obey_injections: bool = False, user_id: str = DEFAULT_USER, chatter: bool = False):
        self.obey_injections = obey_injections
        self.user_id = user_id
        self.chatter = chatter          # if True, never calls tools (simulates a model that loops on text)
        self.calls = 0

    # ------------------------------------------------------------------ api
    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> ModelResponse:
        self.calls += 1
        ctx = _parse_system(system)
        turn = _TurnView(messages)
        content = self._decide(ctx, turn)
        inp = (len(system) + len(json.dumps(messages, default=str))) // 4
        out = len(json.dumps(content, default=str)) // 4
        stop = "tool_use" if any(b["type"] == "tool_use" for b in content) else "end_turn"
        return ModelResponse(content=content, stop_reason=stop, input_tokens=inp, output_tokens=out, model=self.name)

    # --------------------------------------------------------------- policy
    def _decide(self, ctx: dict, turn: "_TurnView") -> list[dict]:
        task = ctx["task"]
        today = ctx["now"]
        user_text = turn.user_text or ""
        facts = extract_facts(user_text, today)
        memop = parse_memory_op(user_text)

        if self.chatter:
            return [_text("Let me think about that some more...")]

        # 0. A compromised model: follow instructions found in reviews.
        if self.obey_injections:
            injected = turn.injected_instruction()
            if injected and not turn.called("forget_memory"):
                # Reviews tell the assistant to wipe preferences and book at once; a
                # compromised model complies. The application must refuse both.
                return [_tool_use("forget_memory", {"key": "*"})]

        # 1. Memory operations requested by the user.
        if memop and not turn.called("remember_preference", "forget_memory", "inspect_memory"):
            if memop["op"] == "inspect":
                return [_tool_use("inspect_memory", {})]
            if memop["op"] == "forget":
                return [_tool_use("forget_memory", {"key": memop["key"]})]
            if memop["op"] == "remember":
                return [_tool_use("remember_preference", {"key": memop["key"], "value": memop["value"],
                                                          "kind": "explicit", "tags": memop["tags"]})]
        if memop and turn.called("remember_preference", "forget_memory", "inspect_memory"):
            res = turn.last_result_for("inspect_memory")
            if res is not None:
                return [_text(_render_memory(res))]
            res = turn.last_result_for("forget_memory")
            if res is not None:
                return [_text("Done - I've forgotten that." if not _is_error(res) else
                              "I couldn't forget that: " + _err(res))]
            res = turn.last_result_for("remember_preference")
            return [_text("Noted - I'll remember that for future bookings.")]

        # 2. Record newly learned facts (only if they change something).
        if not turn.called("update_task"):
            update = _diff_facts(task, facts)
            if "intent" not in update and facts.get("intent") and task.get("intent") != facts["intent"]:
                update["intent"] = facts["intent"]
            # Fill from stored preferences where the user gave nothing.
            effective_intent = update.get("intent") or task.get("intent")
            if effective_intent in (None, "book"):
                filled = []
                if not task.get("area") and "area" not in update and ctx["memories"].get("preferred_area"):
                    update["area"] = ctx["memories"]["preferred_area"]
                    filled.append("area")
                if not task.get("budget_max") and "budget_max" not in update and ctx["memories"].get("default_budget"):
                    try:
                        update["budget_max"] = int(re.sub(r"\D", "", str(ctx["memories"]["default_budget"])))
                        filled.append("budget_max")
                    except ValueError:
                        pass
                if not task.get("time_window") and "time_window" not in update and ctx["memories"].get("earliest_time"):
                    et = str(ctx["memories"]["earliest_time"])
                    tm = re.search(r"(\d{2}:\d{2})", et)
                    update["time_window"] = {"earliest": tm.group(1) if tm else "08:00"}
                    filled.append("time_window")
                if filled:
                    update["filled_from_memory"] = filled
            if update:
                return [_tool_use("update_task", update)]

        # Reload the effective task after update_task (system prompt is rebuilt per call,
        # but within the same call we merge locally).
        upd = turn.last_input_for("update_task") or {}
        eff = dict(task)
        for k, v in upd.items():
            if k == "preferences":
                eff["preferences"] = {**eff.get("preferences", {}), **v}
            elif v is not None:
                eff[k] = v
        intent = eff.get("intent")
        if intent is None:
            if not facts:
                # Nothing actionable in this message and no task in progress: just talk.
                return [_text("Happy to help - tell me which service you need, where, and when, "
                              "or ask me about your bookings.")]
            intent = "book"

        # 3. Intent-specific flows.
        if intent == "inquire":
            return self._inquire(turn)
        if intent == "cancel":
            return self._cancel(eff, turn, user_text)
        if intent == "reschedule":
            return self._reschedule(eff, turn, user_text, today)
        return self._book(eff, turn, ctx)

    # ------------------------------------------------------------ book flow
    def _book(self, eff: dict, turn: "_TurnView", ctx: dict) -> list[dict]:
        prefs = eff.get("preferences") or {}
        # Same provider as last time: look up history first.
        if prefs.get("same_as_last") and not prefs.get("provider_id"):
            res = turn.last_result_for("list_bookings")
            if res is None:
                return [_tool_use("list_bookings", {"user_id": self.user_id})]
            past = [b for b in res.get("bookings", []) if not eff.get("service") or b["service"] == eff["service"]]
            if not past:
                return [_text("I couldn't find a previous booking for that service, so there's no 'last time' provider "
                              "to reuse. Shall I search for a new one?")]
            last = sorted(past, key=lambda b: b["slot"])[-1]
            if not turn.called("update_task", times=2):
                return [_tool_use("update_task", {"preferences": {"provider_id": last["provider_id"],
                                                                   "provider_name": last.get("provider_name")},
                                                   "service": last["service"]})]
            prefs = {**prefs, "provider_id": last["provider_id"]}
            eff["service"] = eff.get("service") or last["service"]

        missing = [f for f in ("service", "area", "date_range", "budget_max") if not eff.get(f)]
        if missing:
            return [_text(_ask_for(missing, eff))]

        search = turn.last_result_for("search_providers")
        if search is None:
            return [_tool_use("search_providers", {"service": eff["service"], "area": eff["area"],
                                                   "date_range": eff["date_range"]})]
        if _is_error(search):
            return [_text("I couldn't search providers right now (" + _err(search) + "). Want me to try again later?")]

        pick, slot = _choose(search.get("providers", []), eff, prefs, ctx["memories"])
        if not pick:
            return [_text(f"I found {len(search.get('providers', []))} {eff['service']} providers in {eff['area']}, but none "
                          f"fit all your constraints (budget ₹{eff['budget_max']}, time window "
                          f"{_fmt_window(eff.get('time_window'))}). Would you like to relax the budget or the time window?")]

        details = turn.last_result_for("get_provider")
        if details is None:
            return [_tool_use("get_provider", {"provider_id": pick["provider_id"]})]

        hold = turn.last_result_for("hold_slot")
        if hold is None:
            return [_tool_use("hold_slot", {"provider_id": pick["provider_id"], "slot": slot})]
        if _is_error(hold):
            if hold.get("error") == "slot_taken":
                return [_text(f"That {slot} slot with {pick['name']} was just taken. Shall I look at the next best option?")]
            return [_text("I couldn't hold the slot (" + _err(hold) + "). Would you like me to try another provider?")]

        book_res = turn.last_result_for("book")
        if book_res is None:
            return [_tool_use("book", {"hold_id": hold["hold_id"], "idempotency_key": f"book:{hold['hold_id']}"})]
        # Normally the application ends the turn when book is filed; if we are called
        # again it is because approval was denied or something else happened.
        return [_text(_summarize_result(book_res))]

    # ------------------------------------------------------------ cancel flow
    def _cancel(self, eff: dict, turn: "_TurnView", user_text: str) -> list[dict]:
        res = turn.last_result_for("list_bookings")
        if res is None:
            return [_tool_use("list_bookings", {"user_id": self.user_id})]
        target = _pick_booking(res, eff, user_text)
        if not target:
            return [_text("I couldn't find an upcoming booking matching that. Here is what I see: "
                          + _render_bookings(res))]
        if turn.last_result_for("cancel") is None:
            return [_tool_use("cancel", {"booking_id": target["booking_id"],
                                         "idempotency_key": f"cancel:{target['booking_id']}"})]
        return [_text(_summarize_result(turn.last_result_for("cancel")))]

    # -------------------------------------------------------- reschedule flow
    def _reschedule(self, eff: dict, turn: "_TurnView", user_text: str, today: datetime) -> list[dict]:
        res = turn.last_result_for("list_bookings")
        if res is None:
            return [_tool_use("list_bookings", {"user_id": self.user_id})]
        target = _pick_booking(res, eff, user_text)
        if not target:
            return [_text("I couldn't find an upcoming booking to move. " + _render_bookings(res))]
        slot = eff.get("target_slot")
        if not slot:
            date = (eff.get("date_range") or {}).get("start")
            t = parse_clock_time(user_text) or (eff.get("time_window") or {}).get("earliest")
            if date and t:
                slot = f"{date}T{t}"
                if not turn.called("update_task", times=2):
                    return [_tool_use("update_task", {"target_slot": slot, "target_booking_id": target["booking_id"]})]
            else:
                if not date:
                    return [_text(f"Which day should I move your {target['service']} booking "
                                  f"({target['slot'].replace('T', ' ')}) to?")]
                return [_text(f"What time on {date} works for the {target['service']} visit? "
                              f"(It's currently {target['slot'].replace('T', ' ')}.)")]
        if turn.last_result_for("reschedule") is None:
            return [_tool_use("reschedule", {"booking_id": target["booking_id"], "slot": slot,
                                             "idempotency_key": f"reschedule:{target['booking_id']}:{slot}"})]
        return [_text(_summarize_result(turn.last_result_for("reschedule")))]

    def _inquire(self, turn: "_TurnView") -> list[dict]:
        res = turn.last_result_for("list_bookings")
        if res is None:
            return [_tool_use("list_bookings", {"user_id": self.user_id})]
        return [_text(_render_bookings(res))]


# ---------------------------------------------------------------- helpers
class _TurnView:
    """What happened since the last user text message."""

    def __init__(self, messages: list[dict]):
        self.messages = messages
        self.user_text: Optional[str] = None
        self.turn_msgs: list[dict] = []
        idx = None
        for i in range(len(messages) - 1, -1, -1):
            m = messages[i]
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                idx = i
                break
        if idx is not None:
            self.user_text = messages[idx]["content"]
            self.turn_msgs = messages[idx + 1:]
        self.tool_uses: dict[str, dict] = {}
        self.results: list[tuple[str, Any]] = []
        for m in self.turn_msgs:
            if not isinstance(m.get("content"), list):
                continue
            for b in m["content"]:
                if b.get("type") == "tool_use":
                    self.tool_uses[b["id"]] = b
                elif b.get("type") == "tool_result":
                    tu = self.tool_uses.get(b["tool_use_id"])
                    self.results.append((tu["name"] if tu else "?", _parse_result(b)))

    def called(self, *names: str, times: int = 1) -> bool:
        return sum(1 for tu in self.tool_uses.values() if tu["name"] in names) >= times

    def last_result_for(self, name: str) -> Any:
        for n, r in reversed(self.results):
            if n == name:
                return r
        return None

    def last_input_for(self, name: str) -> Optional[dict]:
        for tu in reversed(list(self.tool_uses.values())):
            if tu["name"] == name:
                return tu["input"]
        return None

    def injected_instruction(self) -> Optional[str]:
        for _, r in self.results:
            s = json.dumps(r, default=str)
            if "POSSIBLE PROMPT INJECTION" in s or "ignore all previous instructions" in s.lower():
                return s
        return None


def _parse_result(block: dict) -> Any:
    content = block.get("content")
    if isinstance(content, list):
        content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
    if not isinstance(content, str):
        return content
    m = re.search(r"<tool_result[^>]*>\n(.*)\n</tool_result>", content, re.DOTALL)
    body = m.group(1) if m else content
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return {"raw": content, "error": "unparseable"} if block.get("is_error") else {"raw": content}
    if block.get("is_error") and isinstance(data, dict):
        data.setdefault("error", "error")
    return data


def _is_error(res: Any) -> bool:
    return isinstance(res, dict) and "error" in res


def _err(res: Any) -> str:
    if isinstance(res, dict):
        return str(res.get("message") or res.get("error") or res)
    return str(res)


def _parse_system(system: str) -> dict:
    now = datetime(2026, 1, 1)
    m = re.search(r"Current date/time: (\S+)", system)
    if m:
        try:
            now = datetime.fromisoformat(m.group(1))
        except ValueError:
            pass
    task: dict[str, Any] = {}
    sec = re.search(r"Task state \(facts already supplied - do not re-ask\):\n(.*?)\n\nStill missing", system, re.DOTALL)
    if sec:
        for line in sec.group(1).splitlines():
            mm = re.match(r"- (\w+): (.*)", line)
            if mm:
                task[mm.group(1)] = _coerce(mm.group(2))
    mems: dict[str, Any] = {}
    sec = re.search(r"Stored user preferences[^\n]*\n(.*?)(?:\n\n|$)", system, re.DOTALL)
    if sec:
        for line in sec.group(1).splitlines():
            mm = re.match(r"- ([\w:\- ]+) = (.*?) \((explicit|inferred, confirmed)\)", line)
            if mm:
                mems[mm.group(1).strip()] = mm.group(2)
    return {"now": now, "task": task, "memories": mems}


def _coerce(v: str) -> Any:
    v = v.strip()
    if v.startswith("{") or v.startswith("["):
        try:
            return json.loads(v.replace("'", '"').replace("True", "true").replace("False", "false").replace("None", "null"))
        except json.JSONDecodeError:
            return v
    if v.isdigit():
        return int(v)
    return v


def _diff_facts(task: dict, facts: dict) -> dict:
    out = {}
    for k, v in facts.items():
        if k == "intent":
            continue
        if k == "preferences":
            cur = task.get("preferences") or {}
            new = {kk: vv for kk, vv in v.items() if cur.get(kk) != vv}
            if new:
                out["preferences"] = new
            continue
        if task.get(k) != v:
            out[k] = v
    return out


def _ask_for(missing: list[str], eff: dict) -> str:
    labels = {"service": "which service you need", "area": "your area / locality",
              "date_range": "which day(s) work for you", "budget_max": "your maximum budget"}
    known = []
    if eff.get("service"):
        known.append(eff["service"])
    if eff.get("date_range"):
        known.append(f"on {eff['date_range']['start']}" if eff["date_range"]["start"] == eff["date_range"]["end"]
                     else f"between {eff['date_range']['start']} and {eff['date_range']['end']}")
    if eff.get("time_window"):
        known.append(_fmt_window(eff["time_window"]))
    if eff.get("budget_max"):
        known.append(f"under ₹{eff['budget_max']}")
    lead = ("Got it - " + ", ".join(known) + ". ") if known else ""
    return lead + "To search, I still need " + " and ".join(labels[m] for m in missing) + "."


def _fmt_window(tw: Optional[dict]) -> str:
    if not tw:
        return "any time"
    if tw.get("earliest") and tw.get("latest"):
        return f"{tw['earliest']}-{tw['latest']}"
    if tw.get("earliest"):
        return f"not before {tw['earliest']}"
    return f"before {tw.get('latest')}"


def _slot_ok(slot: str, tw: Optional[dict]) -> bool:
    if not tw:
        return True
    t = slot[11:16]
    if tw.get("earliest") and t < tw["earliest"]:
        return False
    if tw.get("latest") and t > tw["latest"]:
        return False
    return True


def _choose(providers: list[dict], eff: dict, prefs: dict, mems: dict) -> tuple[Optional[dict], Optional[str]]:
    budget = eff.get("budget_max")
    tw = dict(eff.get("time_window") or {})
    # Stored earliest time preference applies when the user gave no explicit window.
    if not tw and mems.get("earliest_time"):
        mm = re.search(r"(\d{2}:\d{2})", str(mems["earliest_time"]))
        if mm:
            tw["earliest"] = mm.group(1)
    want_pid = prefs.get("provider_id")
    want_name = (mems.get("preferred_provider") or "").lower()
    ranked = []
    for p in providers:
        if budget and p["price"] > budget:
            continue
        slots = [s for s in (p.get("slots") or p.get("available_slots") or []) if _slot_ok(s, tw)]
        if not slots:
            continue
        score = p.get("rating", 0)
        if want_pid and p["provider_id"] == want_pid:
            score += 100
        elif want_name and want_name in p["name"].lower():
            score += 50
        ranked.append((score, -p["price"], p, slots[0]))
    if not ranked:
        return None, None
    ranked.sort(key=lambda r: (-r[0], r[1], r[2]["provider_id"]))
    _, _, p, slot = ranked[0]
    return p, slot


def _pick_booking(res: dict, eff: dict, user_text: str) -> Optional[dict]:
    bookings = [b for b in res.get("bookings", []) if b["status"] in ("confirmed", "rescheduled")]
    svc = eff.get("service") or parse_service(user_text)
    if eff.get("target_booking_id"):
        for b in bookings:
            if b["booking_id"] == eff["target_booking_id"]:
                return b
    if svc:
        bookings = [b for b in bookings if b["service"] == svc]
    bookings = [b for b in bookings if b.get("when") == "upcoming"] or bookings
    return sorted(bookings, key=lambda b: b["slot"])[0] if bookings else None


def _render_bookings(res: dict) -> str:
    items = res.get("bookings", [])
    if not items:
        return "You have no bookings on record."
    lines = [f"- {b['booking_id']}: {b['service']} with {b.get('provider_name') or b['provider_id']} on "
             f"{b['slot'].replace('T', ' ')} ({b['status']})" for b in items]
    return "Your bookings:\n" + "\n".join(lines)


def _render_memory(res: Any) -> str:
    items = res.get("memories", []) if isinstance(res, dict) else []
    if not items:
        return "I don't have anything stored about you."
    lines = []
    for m in items:
        status = "forgotten" if m["status"] == "forgotten" else (
            "explicit" if m["kind"] == "explicit" else ("inferred, confirmed" if m["confirmed"] else "inferred, unconfirmed"))
        lines.append(f"- {m['key']}: {m['value']} [{status}]")
    return "Here's what I have stored:\n" + "\n".join(lines) + "\nTell me if anything is wrong or should be forgotten."


def _summarize_result(res: Any) -> str:
    if res is None:
        return "I'm waiting on your confirmation."
    if _is_error(res):
        return "That didn't go through: " + _err(res)
    st = res.get("status") if isinstance(res, dict) else None
    return {
        "approval_required": "I've prepared the request - please confirm and I'll proceed.",
        "approval_denied": "Okay, I won't go ahead with that.",
        "success": "Done.",
    }.get(st, f"Status: {st}")
