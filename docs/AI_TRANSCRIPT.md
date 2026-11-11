# AI coding assistant transcript

This project was built in a single Claude Code session (model: Claude Fable 5.1) driven from the terminal. The full transcript is the session log itself; export it with `/export` in Claude Code and save it next to this file as `docs/claude-code-session.txt` before submitting.

## What the assistant did, in order

1. Read the assignment and chose the stack: Python 3.12, no framework, Anthropic SDK for the live model, a deterministic scripted model for offline evals, pytest for scenarios, `rich` for the side-by-side CLI.
2. Wrote the mock backend (`mocks/tools.py`) implementing the eight fixed tool contracts with seeded randomness, forced-outcome scenario config, persistence and provider-cancellation events.
3. Wrote the core modules: explicit state machine + task slots, approval gate with fingerprints, memory store with explicit/inferred/forgotten semantics, token-budgeted context builder, tracer, sanitizer.
4. Wrote the model layer (tool schemas, live Anthropic client, NL parsers, scripted model with a "compromised" mode) and the orchestrator (bounded loop, approval interception, execution with idempotency, timeout resolution via the source of truth, restart recovery, proactive follow-up).
5. Ran a smoke booking, then wrote 30+ scenario tests with an eval-table generator; iterated on failures.
6. Hit and fixed the three bugs in `BUGS.md` (alias substring match; stale completed task re-entering the booking flow via two code paths; "not before" parsed as an upper bound), each with a regression test.
7. Built the CLI, the trace generator and the seed generator; produced the committed trace and eval table; wrote README, DESIGN, DEMO.

## Prompts used (summarised)

* The assignment text, pasted verbatim, with "new task for you".
* No further prompts were needed; the assistant ran tests and the CLI itself to find defects.

## What was *not* done by the assistant

* The live model paths (OpenAI-compatible adapter, added on request to avoid depending on a Claude key, and the Anthropic client) were written against the SDK documentation and checked with offline conversion tests, but not executed against a real API (no key in the build environment).
* The screen recording must be made by a human following `docs/DEMO.md`.
