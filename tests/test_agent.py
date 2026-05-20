"""Agent-loop regression tests covering the Critical-Point gate.

The agent loop is exercised via stub model and executor objects so the tests
run without llama.cpp or Playwright installed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from fara.agent import FaraAgent, _is_committing

# ---- _is_committing ------------------------------------------------------


def test_left_click_is_committing():
    assert _is_committing("left_click", {"coordinate": [10, 20]}) is True


def test_type_only_committing_with_enter():
    assert _is_committing("type", {"text": "hi"}) is False
    assert _is_committing("type", {"text": "hi", "press_enter": True}) is True


def test_key_enter_variants_are_committing():
    assert _is_committing("key", {"key": "Enter"}) is True
    assert _is_committing("key", {"key": "Return"}) is True
    assert _is_committing("key", {"key": "Tab"}) is False


def test_scroll_and_wait_are_not_committing():
    assert _is_committing("scroll", {}) is False
    assert _is_committing("wait", {"seconds": 1}) is False


# ---- Critical-Point gate end-to-end -------------------------------------


class _StubPage:
    def __init__(self, url: str = "https://example.com/"):
        self.url = url

    def goto(self, *_, **__):
        pass

    def wait_for_load_state(self, *_, **__):
        pass

    def wait_for_timeout(self, *_, **__):
        pass


class _StubExecutor:
    def __init__(self, url: str = "https://example.com/"):
        self.page = _StubPage(url)
        self.calls: list[tuple[str, dict]] = []

    def settle(self):
        pass

    def screenshot(self, path):
        Path(path).write_bytes(b"\x89PNG\r\n")
        return path

    def at_critical_url(self):
        return any(p in self.page.url.lower() for p in ("/checkout", "/payment", "/signup"))

    def execute(self, name, args):
        self.calls.append((name, args))
        return "ok"


class _StubModel:
    def __init__(self, replies: list[str]):
        self._replies = list(replies)

    def step(self, **_):
        return self._replies.pop(0)


def _click_tool_call(x: int = 100, y: int = 100) -> str:
    return (
        "Clicking the button.\n"
        "<tool_call>\n"
        f'{{"name": "computer_use", "arguments": {{"action": "left_click", "coordinate": [{x}, {y}]}}}}\n'
        "</tool_call>"
    )


def _terminate_call() -> str:
    return '<tool_call>{"name": "x", "arguments": {"action": "terminate"}}</tool_call>'


def test_pauses_on_committing_click_at_critical_url(tmp_path: Path):
    model = _StubModel([_click_tool_call()])
    executor = _StubExecutor(url="https://shop.example.com/checkout")
    agent = FaraAgent(model=model, executor=executor, run_dir=tmp_path, max_turns=3)

    record = agent.run("buy the shoes")

    assert record.finished.startswith("paused at Critical Point")
    assert executor.calls == []  # action never dispatched
    assert record.turns[-1].status == "critical_point_pause"

    transcript = json.loads((tmp_path / "transcript.json").read_text())
    assert transcript["finished"].startswith("paused at Critical Point")


def test_interactive_approval_continues(tmp_path: Path):
    model = _StubModel([_click_tool_call(), _terminate_call()])
    executor = _StubExecutor(url="https://shop.example.com/checkout")
    agent = FaraAgent(
        model=model,
        executor=executor,
        run_dir=tmp_path,
        max_turns=3,
        interactive=True,
        approval_fn=lambda *_: True,
    )

    record = agent.run("buy the shoes")

    assert record.finished == "model emitted terminate"
    # The click was approved and dispatched once, then the loop terminated.
    assert executor.calls == [("left_click", {"coordinate": [100, 100]})]


def test_interactive_decline_pauses(tmp_path: Path):
    model = _StubModel([_click_tool_call()])
    executor = _StubExecutor(url="https://shop.example.com/checkout")
    agent = FaraAgent(
        model=model,
        executor=executor,
        run_dir=tmp_path,
        max_turns=3,
        interactive=True,
        approval_fn=lambda *_: False,
    )

    record = agent.run("buy the shoes")

    assert record.finished.startswith("paused at Critical Point")
    assert executor.calls == []


def test_non_critical_url_runs_through(tmp_path: Path):
    model = _StubModel([_click_tool_call(), _terminate_call()])
    executor = _StubExecutor(url="https://example.com/home")
    agent = FaraAgent(model=model, executor=executor, run_dir=tmp_path, max_turns=3)

    record = agent.run("just browse")

    assert record.finished == "model emitted terminate"
    assert executor.calls == [("left_click", {"coordinate": [100, 100]})]
    # First turn has both timings populated
    assert record.turns[0].executor_ms >= 0
    assert record.turns[0].infer_ms >= 0


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://shop.example.com/checkout", True),
        ("https://shop.example.com/cart", False),
        ("https://example.com/signup?ref=x", True),
        ("https://example.com/", False),
    ],
)
def test_stub_critical_url_matcher(url: str, expected: bool):
    e = _StubExecutor(url=url)
    assert e.at_critical_url() is expected
