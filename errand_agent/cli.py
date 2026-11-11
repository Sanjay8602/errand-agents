"""Interactive chat with the agent's trace shown side by side.

    python -m errand_agent.cli                      # scripted model, fresh world
    python -m errand_agent.cli --model groq         # Groq (GROQ_API_KEY from env or .env), default openai/gpt-oss-120b
    python -m errand_agent.cli --model groq --llm qwen/qwen3.8-27b   # another Groq model id
    python -m errand_agent.cli --model openai       # OpenAI (OPENAI_API_KEY)
    python -m errand_agent.cli --model anthropic    # Claude (ANTHROPIC_API_KEY)
    python -m errand_agent.cli --scenario timeout_after_success --seed

Slash commands inside the chat:
    /bookings            list bookings (straight from the mock backend)
    /memory              dump memory store
    /cancel-sim <id>     simulate the provider cancelling booking <id>, then poll events
    /tick                poll provider events (proactive follow-up)
    /clock +11m          advance the fake clock (e.g. expire a hold)
    /trace               print trace file path and totals
    /quit
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .agent import ErrandAgent, Limits
from .clock import Clock, FakeClock
from .context import ContextBuilder
from .llm import ScriptedModel
from .mocks import MockToolEnvironment, ScenarioConfig
from .trace import Tracer

ROOT = Path(__file__).resolve().parents[1]

SCENARIOS = {
    "default": ScenarioConfig(),
    "always_success": ScenarioConfig(book_outcomes=["success"] * 20),
    "timeout_after_success": ScenarioConfig(book_outcomes=["timeout_booked", "success"]),
    "timeout_then_retry": ScenarioConfig(book_outcomes=["timeout_not_booked", "success"]),
    "slot_taken": ScenarioConfig(book_outcomes=["slot_taken", "success"]),
    "price_drift": ScenarioConfig(book_outcomes=["success"] * 5, price_drift={"*": 200}, drift_only_after_first_hold=True),
    "slot_conflict": ScenarioConfig(book_outcomes=["success"] * 5, conflict_slots=["2026-11-14T10:00"]),
    "flaky_search": ScenarioConfig(book_outcomes=["success"] * 5, transient_failures={"search_providers": 1}),
}

TRACE_KEYS = {
    "state_transition": lambda e: f"{e['from_state']} -> {e['to_state']}  ({e.get('reason', '')})",
    "context_sent": lambda e: f"{e['total_tokens']}/{e['budget_tokens']} tok, mem={len(e['memory_ids_sent'])}, "
                              f"hist={e['history_messages_sent']} (dropped {e['history_messages_dropped']})",
    "model_call": lambda e: f"{e['model']} {e['latency_ms']}ms in={e['input_tokens']} out={e['output_tokens']} "
                            f"${e['estimated_cost_usd']} -> {e['tools_requested'] or 'text'}",
    "tool_call": lambda e: f"{e['tool']}({_short(e.get('args'))}) -> " + ("ERROR " + e['error']['error'] if e.get('error') else _short(e.get('result'), 70)),
    "approval_requested": lambda e: f"{e['action']} fp={e['fingerprint']} {e.get('approval_id')}",
    "approval_granted": lambda e: f"{e['action']} fp={e['fingerprint']}",
    "approval_invalidated": lambda e: f"{e.get('reason')}",
    "gate_check": lambda e: f"{e['action']} ok={e['ok']} ({e['reason']})",
    "injection_detected": lambda e: f"{e['tool']} {e['paths']}",
    "policy_denied": lambda e: f"{e['tool']}: {e['reason']}",
    "verification": lambda e: f"status={e['result'].get('status')}",
    "booking_verified": lambda e: f"{e['booking_id']}",
    "limit_exhausted": lambda e: f"{e['reason']} turns={e['model_turns']} tools={e['tool_calls']}",
    "event_detected": lambda e: f"{e['event']['type']} {e['event']['booking_id']}",
    "task_updated": lambda e: f"changed={e['changed']}",
    "recovery_resolved": lambda e: f"{e['op']} -> {e['result'] if isinstance(e['result'], str) else 'done'}",
    "hold_expired": lambda e: e["hold_id"],
    "retry": lambda e: f"{e['op']} key={e['idempotency_key']}",
}
SKIP = {"user_message", "agent_message", "session_start", "approval_reply"}


def _short(v, n: int = 60) -> str:
    s = json.dumps(v, default=str, ensure_ascii=False) if not isinstance(v, str) else v
    return s if len(s) <= n else s[:n] + "…"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Errand agent chat")
    ap.add_argument("--model", choices=["scripted", "groq", "openai", "openrouter", "gemini", "ollama", "anthropic"],
                    default="scripted",
                    help="scripted = offline deterministic; groq = Groq (GROQ_API_KEY); openai = OpenAI "
                         "(OPENAI_API_KEY); openrouter / gemini / ollama likewise; anthropic = Claude")
    ap.add_argument("--llm", default=None, help="model id for the live provider (overrides ERRAND_AGENT_MODEL)")
    ap.add_argument("--scenario", choices=sorted(SCENARIOS), default="default")
    ap.add_argument("--seed", action="store_true", help="load seeds/memories.json and seeds/long_conversation.json")
    ap.add_argument("--data-dir", default=None, help="persist state here (default: runtime/<session-id>)")
    ap.add_argument("--fake-clock", default="2026-11-09T08:00", help="ISO start time for the fake clock, or 'real'")
    ap.add_argument("--budget", type=int, default=None,
                    help="max context tokens (default 6000 offline, 4000 for live providers to stay inside free-tier limits)")
    ap.add_argument("--no-trace-pane", action="store_true")
    args = ap.parse_args(argv)

    console = Console()
    data_dir = Path(args.data_dir) if args.data_dir else ROOT / "runtime" / datetime.now().strftime("%Y%m%d-%H%M%S")
    data_dir.mkdir(parents=True, exist_ok=True)
    clock: Clock = Clock() if args.fake_clock == "real" else FakeClock(datetime.fromisoformat(args.fake_clock))
    env = MockToolEnvironment(data_dir / "env.json", clock, SCENARIOS[args.scenario])

    if args.seed:
        shutil.copy(ROOT / "seeds" / "memories.json", data_dir / "memory.json")
        env.seed_past_bookings([{"booking_id": "bk-old-7", "user_id": "user-1", "provider_id": "electrician-02",
                                 "provider_name": "Ramesh Fix-It", "service": "electrician", "slot": "2026-10-20T11:00",
                                 "price": 900, "status": "confirmed", "hold_id": None, "idempotency_key": "seed:bk-old-7",
                                 "created_at": "2026-10-01T10:00:00"},
                                {"booking_id": "bk-seed-1", "user_id": "user-1", "provider_id": "ac-servicing-01",
                                 "provider_name": "Kavya Home Care", "service": "ac servicing", "slot": "2026-11-12T15:00",
                                 "price": 1300, "status": "confirmed", "hold_id": None, "idempotency_key": "seed:bk-seed-1",
                                 "created_at": "2026-11-01T10:00:00"}])

    pending_trace: list[str] = []

    def on_event(e: dict) -> None:
        if e["type"] in SKIP:
            return
        fmt = TRACE_KEYS.get(e["type"])
        detail = fmt(e) if fmt else _short({k: v for k, v in e.items() if k not in ("seq", "run_id", "ts", "type")}, 80)
        pending_trace.append(f"[dim]{e['ts'][11:]}[/] [bold]{e['type']}[/] {detail}")

    tracer = Tracer(ROOT / "traces", run_id=f"cli-{uuid.uuid4().hex[:6]}", now=clock.now, listener=on_event)

    if args.model in ("groq", "openai", "openrouter", "gemini", "ollama"):
        from .llm.openai_model import OpenAICompatibleModel, load_dotenv
        load_dotenv(ROOT / ".env")
        try:
            model = OpenAICompatibleModel(model=args.llm, provider=args.model)
        except Exception as e:  # missing key etc.
            console.print(f"[red]{e}[/]")
            return 2
    elif args.model == "anthropic":
        from .llm.anthropic_model import AnthropicModel, DEFAULT_MODEL as ANT_DEFAULT
        model = AnthropicModel(model=args.llm or ANT_DEFAULT)
    else:
        model = ScriptedModel()

    # Live providers can queue for 10-30 s per call on free tiers; give them a wider wall-clock budget,
    # honour retry-after, and send fewer tokens per call (free tiers cap tokens per minute and per day).
    live = args.model != "scripted"
    limits = Limits(wall_clock_seconds=180.0, model_retries=3, model_retry_backoff_s=5.0) if live else Limits()
    budget = args.budget or (4000 if live else 6000)
    agent = ErrandAgent(env, model, data_dir, clock, tracer=tracer, limits=limits,
                        context=ContextBuilder(max_context_tokens=budget))
    if args.seed:
        conv = json.loads((ROOT / "seeds" / "long_conversation.json").read_text(encoding="utf-8"))
        agent.history = conv + agent.history

    console.print(Panel(f"Errand agent  |  provider={args.model}  llm={getattr(model, 'name', '-')}  "
                        f"scenario={args.scenario}  data={data_dir}\n"
                        f"trace -> {tracer.path}\nType /quit to exit, /help for commands.", title="ready"))

    def render(user: str, replies: list) -> None:
        chat = Text()
        if user:
            chat.append("you  ", style="bold cyan")
            chat.append(user + "\n\n")
        for r in replies:
            chat.append("agent", style="bold green")
            chat.append(f" [{r.state}{'/' + r.outcome if r.outcome else ''}]\n", style="dim")
            chat.append(r.text + "\n\n")
        trace = Text.from_markup("\n".join(pending_trace) if pending_trace else "[dim](no events)[/]")
        pending_trace.clear()
        if args.no_trace_pane:
            console.print(Panel(chat, title="chat"))
            return
        t = Table.grid(expand=True, padding=(0, 1))
        t.add_column(ratio=1)
        t.add_column(ratio=1)
        t.add_row(Panel(chat, title="chat"), Panel(trace, title="trace"))
        console.print(t)

    while True:
        try:
            line = console.input("[bold cyan]> [/]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line in ("/quit", "/exit"):
            break
        if line == "/help":
            console.print(__doc__)
            continue
        if line == "/bookings":
            console.print_json(json.dumps(env.list_bookings("user-1"), default=str))
            continue
        if line == "/memory":
            console.print_json(json.dumps(agent.memory.inspect(), default=str))
            continue
        if line == "/trace":
            console.print(tracer.path, tracer.totals())
            continue
        if line.startswith("/clock"):
            m = re.search(r"\+(\d+)([mhs])", line)
            if m and isinstance(clock, FakeClock):
                unit = {"m": "minutes", "h": "hours", "s": "seconds"}[m.group(2)]
                clock.advance(**{unit: int(m.group(1))})
                console.print(f"clock now {clock.now()}")
            else:
                console.print("usage: /clock +11m  (fake clock only)")
            continue
        if line.startswith("/cancel-sim"):
            parts = line.split()
            if len(parts) == 2:
                env.simulate_provider_cancellation(parts[1], reason="technician unavailable")
                render("", agent.check_events())
            else:
                console.print("usage: /cancel-sim <booking_id>")
            continue
        if line == "/tick":
            render("", agent.check_events())
            continue
        replies = agent.check_events()
        replies.append(agent.handle(line))
        render(line, replies)

    console.print(f"trace saved: {tracer.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
