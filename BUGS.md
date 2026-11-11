# Three real bugs hit while building

All three were found by running the system (tests or the CLI), not by inspection. Each has a regression test in `tests/test_scenarios.py`.

## 1. "Actually" was parsed as an AC-servicing request

**What broke.** In the correction scenario the user had a plumber booking pending and said *"Actually make it Thursday"*. The agent changed the day **and silently changed the service to `ac servicing`**, then searched for AC technicians. `test_correction_preserves_unrelated_context` failed with `'ac servicing' == 'plumber'`.

**Why.** `canonical_service()` matched service aliases with a plain substring test (`alias in text`). The alias `"ac"` is a substring of `"actually"`. Any utterance containing *actually, back, place, contact, space...* would have flipped the service.

**Fix.** Aliases are matched on word boundaries (`\b ac \b`). `errand_agent/mocks/tools.py::canonical_service`.

**Regression test.** `test_regression_service_alias_word_boundary` (unit: "Actually make it Thursday" -> `None`; "my AC is leaking" -> `ac servicing`; "the switchboard sparks" -> `None`) plus the original correction scenario, which now asserts that service, area, budget and time window are all preserved after the day changes.

## 2. A finished task stayed "live" and a follow-up question started a second booking

**What broke.** After a booking completed (and, separately, after a restart recovered an in-flight booking), the user asked *"Did that go through?"*. The agent answered with the recovery notice **and immediately filed a new approval request for the same plumber at another slot**. The approval gate stopped an actual duplicate booking, but the behaviour was wrong: an informational question must not re-enter the booking flow.

**Why.** The `Task` (service, area, dates, budget, `intent=book`) was persisted and never cleared when the operation completed. The scripted model (and a real model reading the same "task state" block) saw a fully specified booking intent with nothing missing and proceeded to search -> hold -> book. There were two completion paths (`_finish_op` and `_recover_in_flight`) and neither reset the task.

**Fix.** Both completion paths now reset `self.task = Task()` once the side effect is verified; the completed action is still available to the prompt via "Last completed action". The scripted model additionally answers conversationally when a message carries no intent and no new facts instead of defaulting to `book`. `errand_agent/agent.py::_finish_op`, `::_recover_in_flight`; `errand_agent/llm/scripted.py::_decide`.

**Regression tests.** `test_regression_followup_after_booking_does_not_rebook` (exactly one `search_providers`, one `book`, task cleared, state not `AWAITING_APPROVAL`) and `test_restart_mid_booking_recovers_without_duplicate` (after recovery, the follow-up question must not produce an approval request).

## 3. "not before 10 a.m." became a one-minute window and dropped the preferred electrician

**What broke.** In the seeded CLI world, *"Book the same electrician as last time, but not before 10 a.m."* produced the clarifying line "electrician, **10:00-10:00**, under ₹2000" and then booked **Pooja Fix-It** instead of Ramesh Fix-It (`electrician-02`), the provider from the user's previous booking. The user's stated preference was silently violated.

**Why.** `parse_time_window()` extracted the lower bound from `not before 10` correctly, but the upper-bound regex `(before|by|latest|until|...)\s*(\d+)` also matched the `before 10` inside `not before 10`, setting `latest = 10:00`. With `earliest == latest`, only providers with an exact 10:00 slot survived the filter, and the preferred provider (who had 11:00 and later) was excluded before preference scoring ran. The existing scenario test had passed only because, in that date range, `electrician-02` happened to have a 10:00 slot.

**Fix.** Negative lookbehind so `before` is not treated as an upper bound when preceded by `not`: `(?<!not )before`. `errand_agent/llm/nlp.py::parse_time_window`.

**Regression test.** `test_regression_not_before_is_a_lower_bound` (unit assertions on the parser, plus the full scenario asserting `time_window == {"earliest": "10:00"}` and `provider_id == "electrician-02"`).

## 4. (found in live testing) The model invented a user id and "lost" the user's bookings

**What broke.** On Groq (`openai/gpt-oss-120b`), *"Move my AC servicing to Saturday"* made the model call `list_bookings(user_id="current_user")`. The mock backend correctly filtered on that id and returned an empty list, so the model told the user it had no AC booking and proceeded to arrange a **new** one. The approval gate still required confirmation, but the intent was wrong. In the same turn the model also mislabelled 12 Nov as a Saturday.

**Why.** The `list_bookings` contract takes a `user_id`, but nothing told the model what it was; the scripted model always used the right constant so the tests never caught it. Identity is something the application knows and the model does not. Weekday arithmetic from a bare ISO date is also unreliable for an LLM.

**Fix.** The orchestrator overrides `user_id` on every `list_bookings` call with the session's user and traces an `arg_override` event; the system prompt also states the user id and an eight-day calendar with weekday names. The `hold_slot` result now carries an application note telling the model to call `book` immediately, which removed the extra "shall I book?" round trip. `errand_agent/agent.py::_execute_tool`, `::_tool_result`; `errand_agent/context.py::_calendar`.

**Regression test.** `test_regression_list_bookings_uses_session_user` (a stub model passes a wrong id; the backend must receive `user-1` and the seeded booking must come back).

## 5. (found in live testing) History truncation could leave an orphaned tool result, which Groq rejects

**What broke.** Deep into a seeded live session (240 old messages, 6,000-token budget) every model call began failing with Groq's `400 ... HarmonyError: Tools should have a name!`. The agent degraded correctly ("my language model is unavailable, nothing was booked") but the session was unusable from then on.

**Why.** `ContextBuilder._fit_history` trims history from the front to fit the budget, then cleans the head in two separate passes: first drop leading `tool_result` messages, then drop leading non-user messages. When the cut landed exactly on an assistant `tool_use` message, pass one saw an assistant message (not a tool result) and kept it, pass two dropped it, and the following `tool_result` message, now at the head, was never re-checked. The request then contained a `tool` message with no matching tool call, which Groq's prompt template rejects. The Anthropic API would reject it too. The scripted model never noticed because it ignores orphaned results.

**Fix.** A single loop that pops until the head is a plain user turn: `while kept and (role != "user" or starts_with_tool_result)`. `errand_agent/context.py::_fit_history`.

**Regression test.** `test_regression_history_truncation_never_orphans_tool_results` sweeps every budget from 0 to the full history size and asserts the window always starts with a plain user turn and every `tool_result` has its `tool_use` inside the window.

## 6. (found in live testing) A tight context budget dropped the current turn's tool results and the model looped

**What broke.** Running the smaller `openai/gpt-oss-20b` with a 3,000-token budget, the first message burned all eight model turns: update_task -> search -> hold -> update_task -> search -> search -> search ... and ended in `limit_exhausted`. The 120b model at 6,000 tokens had been fine, which hid the problem.

**Why.** `ContextBuilder._fit_history` applied the budget to the whole history from the newest message backwards. The system prompt alone used ~1,300 tokens, a search result ~1,000, so after the first tool call the window no longer had room for the current turn's own tool_use / tool_result pairs. Each model call then saw only the user message and, quite reasonably, started the task from scratch. The scripted model never showed this because it re-derives its plan from the same inputs.

**Fix.** The current turn (everything since the last plain user message) is always sent in full and the budget trims older history only; if the current turn alone exceeds the budget this is recorded as `current_turn_over_budget_tokens` in the `context_sent` event rather than silently cut. Two reinforcing changes: search results are compacted for the model (ids, price, rating, five slots; the trace keeps the full payload), and an identical read-only call repeated within a turn returns the cached result with a nudge instead of hitting the backend again. `errand_agent/context.py::_fit_history`, `errand_agent/agent.py::_execute_tool`, `::_compact_search`.

**Regression tests.** `test_regression_current_turn_never_dropped_under_tight_budget` (1,200-token budget still completes a search -> hold -> approval with exactly one search) and `test_regression_duplicate_read_call_is_cached`.

---

### Also worth recording (not counted above)

* Tooling, not product: patching a regex via a shell heredoc turned `\b` into a backspace byte (`\x08`) in the source file, which made every scenario fail at once. Writing the file directly instead of through the shell fixed it. The lesson carried into the project: the regression test for bug 1 asserts behaviour, so a silent regex corruption would be caught.
* The scripted "compromised model" stub used in the prompt-injection scenario was initially non-deterministic (it only tried to wipe memory if the *specific* injected review mentioned forgetting). That was a test bug; the stub now always acts on a detected injection so the gate is exercised every run.
