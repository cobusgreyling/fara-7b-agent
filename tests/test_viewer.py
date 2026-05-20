"""Viewer rendering tests."""
from __future__ import annotations

import json
from pathlib import Path

from fara.viewer import render, write_viewer


def _transcript() -> dict:
    return {
        "task": "Find <Claude> & Shannon",
        "finished": "model emitted terminate",
        "notes": ["he co-founded info theory"],
        "turns": [
            {
                "turn": 1,
                "screenshot": "runs/latest/turn_01.png",
                "thought": "I'll search.",
                "action": {"name": "web_search", "arguments": {"query": "Claude Shannon"}},
                "status": "ok",
                "infer_ms": 1234,
                "executor_ms": 56,
            },
            {
                "turn": 2,
                "screenshot": "runs/latest/turn_02.png",
                "thought": "Done.",
                "action": {"name": "terminate", "arguments": {}},
                "status": "terminated",
                "infer_ms": 800,
            },
        ],
    }


def test_render_escapes_html_in_task_and_thought():
    out = render(_transcript())
    assert "Find &lt;Claude&gt; &amp; Shannon" in out
    assert "<Claude>" not in out  # not present as raw HTML
    # html.escape with default quote=True escapes apostrophes too.
    assert "I&#x27;ll search." in out


def test_render_includes_timings_and_notes():
    out = render(_transcript())
    assert "infer 1234 ms" in out
    assert "exec 56 ms" in out
    assert "he co-founded info theory" in out


def test_write_viewer_creates_html(tmp_path: Path):
    (tmp_path / "transcript.json").write_text(json.dumps(_transcript()))
    out = write_viewer(tmp_path)
    assert out == tmp_path / "transcript.html"
    body = out.read_text()
    assert "<!doctype html>" in body
    # Screenshot reference uses basename
    assert 'src="turn_01.png"' in body
