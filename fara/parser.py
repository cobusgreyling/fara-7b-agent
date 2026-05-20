"""Parse Fara-7B raw output into a thought + structured action.

Fara emits:
    <free-text thought>
    <tool_call>
    {"name": "...", "arguments": {"action": "...", ...}}
    </tool_call>

Quantised builds sometimes drift the wrapper "name" field (e.g. emits
"B" or "Rasa" instead of "computer_use") and shorten action verbs
("visit" instead of "visit_url"). We dispatch on the inner `action`
field and normalise common aliases.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(?P<payload>\{.*?\})\s*</tool_call>",
    re.DOTALL,
)

ACTION_ALIASES = {
    "visit": "visit_url",
    "navigate": "visit_url",
    "open": "visit_url",
    "click": "left_click",
    "search": "web_search",
    "back": "history_back",
    "go_back": "history_back",
    "stop": "terminate",
    "done": "terminate",
    "finish": "terminate",
}

KNOWN_ACTIONS = {
    "key",
    "type",
    "mouse_move",
    "left_click",
    "scroll",
    "visit_url",
    "web_search",
    "history_back",
    "pause_and_memorize_fact",
    "wait",
    "terminate",
}


@dataclass
class Action:
    name: str
    arguments: dict[str, Any]


@dataclass
class FaraTurn:
    thought: str
    action: Action | None
    raw: str


def parse(raw_output: str) -> FaraTurn:
    """Parse the raw assistant text into thought + action."""
    match = TOOL_CALL_RE.search(raw_output)
    if not match:
        return FaraTurn(thought=raw_output.strip(), action=None, raw=raw_output)

    thought = raw_output[: match.start()].strip()
    payload_str = match.group("payload")

    try:
        payload = json.loads(payload_str)
    except json.JSONDecodeError:
        return FaraTurn(thought=thought, action=None, raw=raw_output)

    arguments = payload.get("arguments", {})
    action_name = arguments.get("action") or payload.get("name", "")
    action_name = ACTION_ALIASES.get(action_name, action_name)

    if action_name not in KNOWN_ACTIONS:
        action_name = action_name or "unknown"

    args = {k: v for k, v in arguments.items() if k != "action"}
    return FaraTurn(thought=thought, action=Action(name=action_name, arguments=args), raw=raw_output)
