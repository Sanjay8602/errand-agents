"""Produce the end-to-end trace deliverable: a full booking whose book() call
times out after actually succeeding, resolved via get_booking_status.

Writes traces/timeout_after_success.jsonl (raw) and prints an annotated excerpt.
Deterministic: scripted model, fake clock, forced outcome.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from errand_agent.agent import ErrandAgent  # noqa: E402
from errand_agent.clock import FakeClock  # noqa: E402
from errand_agent.llm import ScriptedModel  # noqa: E402
from errand_agent.mocks import MockToolEnvironment, ScenarioConfig  # noqa: E402
from errand_agent.trace import Tracer  # noqa: E402


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    clock = FakeClock(datetime(2026, 11, 9, 8, 0))
    env = MockToolEnvironment(tmp / "env.json", clock, ScenarioConfig(book_outcomes=["timeout_booked"]))
    out_dir = ROOT / "traces"
    out_dir.mkdir(exist_ok=True)
    tracer = Tracer(tmp, run_id="timeout_after_success", now=clock.now)
    agent = ErrandAgent(env, ScriptedModel(), tmp, clock, tracer=tracer)
    agent.memory.remember("preferred_area", "Koramangala", kind="explicit", tags=["general"],
                          source="user said so in an earlier session")
    transcript = []
    for msg in ["Find a plumber for my kitchen tap this week, mornings only, under ₹1,500.", "yes"]:
        r = agent.handle(msg)
        transcript.append((msg, r))
    dest = out_dir / "timeout_after_success.jsonl"
    shutil.copy(tracer.path, dest)
    print(f"trace written: {dest} ({len(tracer.events)} events)\n")
    for msg, r in transcript:
        print(f"USER : {msg}\nAGENT: [{r.state}/{r.outcome}] {r.text}\n")
    print("Annotated excerpt (the interesting part):")
    for e in tracer.events:
        if e["type"] in ("approval_granted", "gate_check", "tool_call", "state_transition", "verification",
                         "booking_verified", "retry") and e["seq"] >= [x for x in tracer.events if x["type"] == "approval_granted"][0]["seq"]:
            keep = {k: v for k, v in e.items() if k not in ("run_id", "ts")}
            if "result" in keep and isinstance(keep["result"], dict):
                keep["result"] = {k: keep["result"][k] for k in list(keep["result"])[:4]}
            print(json.dumps(keep, default=str, ensure_ascii=False)[:260])
    print("\ntotals:", tracer.totals())
    return 0


if __name__ == "__main__":
    sys.exit(main())
