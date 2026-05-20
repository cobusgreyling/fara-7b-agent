"""Parser regression tests.

These lock in the behaviour the parser was hand-tuned for: quantisation-induced
wrapper-name drift, action-verb aliasing, and graceful failure on malformed
tool_call payloads.
"""
from __future__ import annotations

from fara.parser import parse


def test_canonical_visit_url():
    raw = (
        "I'll navigate to Wikipedia.\n"
        "<tool_call>\n"
        '{"name": "computer_use", "arguments": {"action": "visit_url", "url": "https://en.wikipedia.org"}}\n'
        "</tool_call>"
    )
    t = parse(raw)
    assert t.thought == "I'll navigate to Wikipedia."
    assert t.action is not None
    assert t.action.name == "visit_url"
    assert t.action.arguments == {"url": "https://en.wikipedia.org"}


def test_wrapper_name_drift_does_not_matter():
    """Quantised Fara emits 'B' or 'Rasa' as the wrapper name; the parser
    must dispatch on `arguments.action`, not `name`."""
    raw = (
        "Clicking the search bar.\n"
        "<tool_call>\n"
        '{"name": "B", "arguments": {"action": "left_click", "coordinate": [640, 120]}}\n'
        "</tool_call>"
    )
    t = parse(raw)
    assert t.action is not None
    assert t.action.name == "left_click"
    assert t.action.arguments == {"coordinate": [640, 120]}


def test_alias_visit_to_visit_url():
    raw = '<tool_call>{"name":"x","arguments":{"action":"visit","url":"https://example.com"}}</tool_call>'
    t = parse(raw)
    assert t.action is not None
    assert t.action.name == "visit_url"


def test_alias_click_to_left_click():
    raw = '<tool_call>{"name":"x","arguments":{"action":"click","coordinate":[10,20]}}</tool_call>'
    t = parse(raw)
    assert t.action is not None
    assert t.action.name == "left_click"


def test_alias_search_to_web_search():
    raw = '<tool_call>{"name":"x","arguments":{"action":"search","query":"shannon"}}</tool_call>'
    t = parse(raw)
    assert t.action is not None
    assert t.action.name == "web_search"
    assert t.action.arguments == {"query": "shannon"}


def test_alias_terminate_synonyms():
    for verb in ("stop", "done", "finish", "terminate"):
        raw = f'<tool_call>{{"name":"x","arguments":{{"action":"{verb}"}}}}</tool_call>'
        t = parse(raw)
        assert t.action is not None
        assert t.action.name == "terminate"


def test_unknown_action_passes_through_named():
    raw = '<tool_call>{"name":"x","arguments":{"action":"do_a_barrel_roll"}}</tool_call>'
    t = parse(raw)
    assert t.action is not None
    assert t.action.name == "do_a_barrel_roll"


def test_missing_tool_call_returns_thought_only():
    raw = "I'm thinking out loud but never emitted a tool_call."
    t = parse(raw)
    assert t.action is None
    assert t.thought == "I'm thinking out loud but never emitted a tool_call."


def test_malformed_json_payload_returns_thought_only():
    raw = "Trying to click.\n<tool_call>\n{not valid json}\n</tool_call>"
    t = parse(raw)
    assert t.action is None
    assert t.thought == "Trying to click."


def test_action_strips_from_arguments():
    raw = '<tool_call>{"name":"x","arguments":{"action":"type","text":"hello","press_enter":true}}</tool_call>'
    t = parse(raw)
    assert t.action is not None
    assert "action" not in t.action.arguments
    assert t.action.arguments == {"text": "hello", "press_enter": True}
