"""Small deterministic parsers used by the scripted model and by tests' simulated users.

These are intentionally simple: the scripted model only needs to understand the
utterances the evaluation scenarios use. The live model does its own parsing.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Optional

from ..mocks.tools import canonical_service

KNOWN_AREAS = ["Koramangala", "Indiranagar", "HSR Layout", "Whitefield", "Baner", "Kothrud", "Hinjewadi",
               "Wakad", "Viman Nagar", "Andheri", "Bandra", "Jayanagar", "Aundh"]
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def parse_service(text: str) -> Optional[str]:
    return canonical_service(text)


def parse_area(text: str) -> Optional[str]:
    for a in KNOWN_AREAS:
        if re.search(rf"\b{re.escape(a)}\b", text, re.IGNORECASE):
            return a
    m = re.search(r"\b(?:in|at|near|area[:\s]+)\s+([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?)", text)
    if m and m.group(1).lower() not in {"saturday", "sunday", "monday", "tuesday", "wednesday", "thursday", "friday"}:
        return m.group(1)
    return None


def parse_budget(text: str) -> Optional[int]:
    m = re.search(r"(?:under|below|less than|max(?:imum)?|budget(?: of| is)?|upto|up to|within|around)\s*(?:rs\.?|₹|inr)?\s*([\d,]+)",
                  text, re.IGNORECASE)
    if not m:
        m = re.search(r"(?:rs\.?|₹|inr)\s*([\d,]+)", text, re.IGNORECASE)
    if m:
        try:
            return int(m.group(1).replace(",", ""))
        except ValueError:
            return None
    return None


def _next_weekday(today: datetime, name: str) -> datetime:
    target = WEEKDAYS.index(name.lower())
    delta = (target - today.weekday()) % 7
    if delta == 0:
        delta = 7
    return today + timedelta(days=delta)


def parse_date_range(text: str, today: datetime) -> Optional[dict]:
    t = text.lower()
    iso = re.findall(r"\b(20\d\d-\d\d-\d\d)\b", t)
    if iso:
        return {"start": iso[0], "end": iso[-1]}
    if "today" in t:
        d = today.strftime("%Y-%m-%d")
        return {"start": d, "end": d}
    if "tomorrow" in t:
        d = (today + timedelta(days=1)).strftime("%Y-%m-%d")
        return {"start": d, "end": d}
    if "this week" in t:
        end = today + timedelta(days=(6 - today.weekday()))
        return {"start": today.strftime("%Y-%m-%d"), "end": end.strftime("%Y-%m-%d")}
    if "next week" in t:
        start = today + timedelta(days=(7 - today.weekday()))
        return {"start": start.strftime("%Y-%m-%d"), "end": (start + timedelta(days=6)).strftime("%Y-%m-%d")}
    if "weekend" in t:
        sat = _next_weekday(today, "saturday") if today.weekday() != 5 else today
        return {"start": sat.strftime("%Y-%m-%d"), "end": (sat + timedelta(days=1)).strftime("%Y-%m-%d")}
    for wd in WEEKDAYS:
        if re.search(rf"\b{wd}\b", t):
            d = _next_weekday(today, wd).strftime("%Y-%m-%d")
            return {"start": d, "end": d}
    return None


def parse_time_window(text: str) -> Optional[dict]:
    t = text.lower()
    win: dict[str, str] = {}
    if re.search(r"\bmornings?\b|before noon", t):
        win = {"earliest": "08:00", "latest": "12:00"}
    elif re.search(r"\bafternoons?\b", t):
        win = {"earliest": "12:00", "latest": "17:00"}
    elif re.search(r"\bevenings?\b", t):
        win = {"earliest": "16:00", "latest": "20:00"}
    m = re.search(r"(?:not before|after|from|earliest)\s*(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?", t)
    if m:
        win["earliest"] = _to_hhmm(m.group(1), m.group(2), m.group(3))
    # "not before 10" is a lower bound, so 'before' must not be preceded by 'not' (BUGS.md #3).
    m = re.search(r"(?:(?<!not )before|by|latest|until|no later than)\s*(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)", t)
    if m:
        win["latest"] = _to_hhmm(m.group(1), m.group(2), m.group(3))
    return win or None


def parse_clock_time(text: str) -> Optional[str]:
    """A bare time like '10 am', '3 pm', '14:00' -> 'HH:MM'."""
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)\b", text.lower())
    if m:
        return _to_hhmm(m.group(1), m.group(2), m.group(3))
    m = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", text)
    if m:
        return f"{int(m.group(1)):02d}:{m.group(2)}"
    return None


def _to_hhmm(h: str, mnt: Optional[str], ampm: Optional[str]) -> str:
    hour = int(h)
    if ampm and ampm.startswith("p") and hour < 12:
        hour += 12
    if ampm and ampm.startswith("a") and hour == 12:
        hour = 0
    return f"{hour:02d}:{mnt or '00'}"


def parse_intent(text: str) -> Optional[str]:
    t = text.lower()
    if re.search(r"\b(what do you (know|remember)|show (my )?memory|what have you stored)\b", t):
        return "memory"
    if re.search(r"\bforget\b|\bremember that\b|\bfrom now on\b|\bi prefer\b|\balways\b", t):
        return "memory"
    if re.search(r"\b(move|reschedule|shift|change .* to|postpone)\b", t):
        return "reschedule"
    if re.search(r"\bcancel\b", t):
        return "cancel"
    if re.search(r"\b(book|find|need|want|get|arrange|schedule|looking for|hire)\b", t):
        return "book"
    if re.search(r"\b(my bookings|upcoming|what('s| is) booked)\b", t):
        return "inquire"
    return None


def parse_memory_op(text: str) -> Optional[dict]:
    t = text.strip()
    tl = t.lower()
    if re.search(r"\b(what do you (know|remember)|show (my )?memory|what have you stored)\b", tl):
        return {"op": "inspect"}
    m = re.search(r"\bforget (?:my |the |about )?(.+?)(?:\.|$)", tl)
    if m:
        target = m.group(1).strip()
        if target in ("everything", "all", "all of it", "everything about me"):
            return {"op": "forget", "key": "*"}
        return {"op": "forget", "key": _pref_key(target)}
    m = re.search(r"\b(?:remember that|from now on|i prefer|always)\b(.+)", tl)
    if m:
        phrase = m.group(1).strip().rstrip(".")
        key, value, tags = _phrase_to_pref(phrase)
        return {"op": "remember", "key": key, "value": value, "tags": tags}
    return None


def _pref_key(target: str) -> str:
    if "technician" in target or "provider" in target or "electrician" in target or "plumber" in target:
        return "preferred_provider"
    if "area" in target or "location" in target:
        return "preferred_area"
    if "time" in target or "morning" in target:
        return "earliest_time"
    if "budget" in target:
        return "default_budget"
    return target.replace(" ", "_")


def _phrase_to_pref(phrase: str) -> tuple[str, str, list[str]]:
    p = phrase.lower()
    if "morning" in p:
        return "earliest_time", "morning (08:00-12:00)", ["general"]
    tw = parse_time_window(p)
    if tw and tw.get("earliest"):
        return "earliest_time", tw["earliest"], ["general"]
    area = parse_area(phrase.title() if phrase.islower() else phrase)
    if area:
        return "preferred_area", area, ["general"]
    b = parse_budget(p)
    if b:
        return "default_budget", str(b), ["general"]
    m = re.search(r"(?:use|book|prefer|want)\s+([a-z]+(?: [a-z&-]+)*)", p)
    if m:
        who = m.group(1).strip()
        svc = parse_service(p)
        return ("preferred_provider", who, [svc] if svc else ["general"])
    return "note", phrase, ["general"]


def extract_facts(text: str, today: datetime) -> dict[str, Any]:
    """Everything the scripted model can learn from one user utterance."""
    facts: dict[str, Any] = {}
    intent = parse_intent(text)
    if intent:
        facts["intent"] = intent
    svc = parse_service(text)
    if svc:
        facts["service"] = svc
    area = parse_area(text)
    if area:
        facts["area"] = area
    dr = parse_date_range(text, today)
    if dr:
        facts["date_range"] = dr
    tw = parse_time_window(text)
    if tw:
        facts["time_window"] = tw
    b = parse_budget(text)
    if b:
        facts["budget_max"] = b
    m = re.search(r"\b(kitchen tap|tap|leak|wiring|switch|fan|geyser|door|sofa|ac)\b", text.lower())
    if m and intent in ("book", None):
        facts["issue"] = m.group(1)
    if re.search(r"same .* as last time|last time", text.lower()):
        facts.setdefault("preferences", {})["same_as_last"] = True
    return facts
