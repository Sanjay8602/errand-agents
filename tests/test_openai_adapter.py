"""Offline tests for the OpenAI-compatible adapter: history/tool conversion and response parsing.

No network and no SDK import: the conversion functions are pure, and the
response parser is fed a stub shaped like the SDK's ChatCompletion object.
"""
from __future__ import annotations

import json
from types import SimpleNamespace as NS

from errand_agent.llm.base import TOOL_SCHEMAS
from errand_agent.llm.openai_model import from_openai_response, to_openai_messages, to_openai_tools


def test_tools_convert_to_function_schema(evalrow):
    evalrow["expected"] = "all agent tools become OpenAI function tools with their JSON schema"
    tools = to_openai_tools(TOOL_SCHEMAS)
    assert len(tools) == len(TOOL_SCHEMAS)
    assert all(t["type"] == "function" for t in tools)
    book = next(t for t in tools if t["function"]["name"] == "book")
    assert book["function"]["parameters"]["required"] == ["hold_id", "idempotency_key"]
    evalrow["actual"] = f"{len(tools)} function tools"


def test_history_round_trips_tool_calls(evalrow):
    evalrow["expected"] = "tool_use -> assistant.tool_calls; tool_result -> role=tool with matching id; system first"
    history = [
        {"role": "user", "content": "Find a plumber"},
        {"role": "assistant", "content": [{"type": "text", "text": "Checking."},
                                          {"type": "tool_use", "id": "call_1", "name": "search_providers",
                                           "input": {"service": "plumber", "area": "X", "date_range": {"start": "a", "end": "b"}}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "<tool_result>{}</tool_result>"}]},
    ]
    msgs = to_openai_messages("SYS", history)
    assert msgs[0] == {"role": "system", "content": "SYS"}
    assert msgs[1] == {"role": "user", "content": "Find a plumber"}
    assert msgs[2]["role"] == "assistant" and msgs[2]["content"] == "Checking."
    assert msgs[2]["tool_calls"][0]["id"] == "call_1"
    assert json.loads(msgs[2]["tool_calls"][0]["function"]["arguments"])["service"] == "plumber"
    assert msgs[3] == {"role": "tool", "tool_call_id": "call_1", "content": "<tool_result>{}</tool_result>"}
    evalrow["actual"] = f"{len(msgs)} OpenAI messages, ids preserved"


def test_retry_after_parsing(evalrow):
    from errand_agent.llm.openai_model import parse_retry_after
    evalrow["expected"] = "Groq's 'try again in 11m7.872s', '2.5s', '850ms' and a Retry-After header are parsed to seconds"
    assert abs(parse_retry_after("Rate limit ... Please try again in 11m7.872s. Need more?") - 667.872) < 1e-6
    assert parse_retry_after("please try again in 2.5s") == 2.5
    assert parse_retry_after("try again in 850ms") == 0.85
    assert parse_retry_after("no hint here") is None
    assert parse_retry_after("whatever", {"retry-after": "7"}) == 7.0
    evalrow["actual"] = "all five forms parsed"


def test_response_parsing(evalrow):
    evalrow["expected"] = "tool_calls parsed into tool_use blocks with JSON args; usage mapped; stop reasons mapped"
    resp = NS(model="gpt-4o-mini",
              usage=NS(prompt_tokens=120, completion_tokens=30),
              choices=[NS(finish_reason="tool_calls",
                          message=NS(content=None,
                                     tool_calls=[NS(id="call_9", function=NS(name="hold_slot",
                                                                           arguments='{"provider_id": "p1", "slot": "2026-11-11T09:00"}'))]))])
    r = from_openai_response(resp)
    assert r.stop_reason == "tool_use" and r.tool_uses[0]["name"] == "hold_slot"
    assert r.tool_uses[0]["input"]["provider_id"] == "p1" and r.input_tokens == 120 and r.output_tokens == 30
    text_resp = NS(model="gpt-4o-mini", usage=None,
                   choices=[NS(finish_reason="stop", message=NS(content="Hello", tool_calls=None))])
    t = from_openai_response(text_resp)
    assert t.stop_reason == "end_turn" and t.text == "Hello"
    evalrow["actual"] = "tool_use + end_turn parsed"
