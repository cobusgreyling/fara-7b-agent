"""Render transcript.json + screenshots into a static, scrollable HTML page.

Usage from the CLI:
    fara-agent "..." --viewer
    # writes runs/latest/transcript.html

Or post-hoc:
    python -m fara.viewer runs/latest
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any


def _row(turn: dict[str, Any]) -> str:
    shot = Path(turn.get("screenshot", "")).name
    thought = html.escape(turn.get("thought", "") or "")
    action = turn.get("action")
    status = html.escape(turn.get("status", "") or "")
    timings = []
    if "infer_ms" in turn:
        timings.append(f"infer {turn['infer_ms']} ms")
    if "executor_ms" in turn:
        timings.append(f"exec {turn['executor_ms']} ms")
    timing_html = (
        f'<div class="timings">{html.escape(" · ".join(timings))}</div>' if timings else ""
    )
    if action is None:
        action_html = '<pre class="action none">(no tool_call emitted)</pre>'
    else:
        action_html = (
            f'<pre class="action">{html.escape(json.dumps(action, indent=2))}</pre>'
        )
    return f"""
<section class="turn">
  <h2>Turn {turn.get("turn", "?")}</h2>
  <div class="grid">
    <img src="{html.escape(shot)}" alt="screenshot for turn {turn.get('turn', '?')}" loading="lazy" />
    <div class="cell">
      <h3>Thought</h3>
      <p class="thought">{thought}</p>
      <h3>Action</h3>
      {action_html}
      <div class="status">status: <code>{status}</code></div>
      {timing_html}
    </div>
  </div>
</section>
"""


def render(transcript: dict[str, Any]) -> str:
    task = html.escape(transcript.get("task", "") or "")
    finished = html.escape(transcript.get("finished", "") or "")
    notes = transcript.get("notes") or []
    notes_html = ""
    if notes:
        items = "\n".join(f"<li>{html.escape(str(n))}</li>" for n in notes)
        notes_html = f'<aside class="notes"><h3>Memorised facts</h3><ul>{items}</ul></aside>'

    turns_html = "\n".join(_row(t) for t in transcript.get("turns", []))

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>fara-7b-agent run — {task}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 0; padding: 24px; max-width: 1400px; }}
  header h1 {{ font-size: 1.2rem; margin: 0 0 4px 0; }}
  header p {{ margin: 0; color: #666; }}
  section.turn {{ margin: 32px 0; border-top: 1px solid #ddd; padding-top: 16px; }}
  section.turn h2 {{ font-size: 1rem; margin: 0 0 12px 0; }}
  .grid {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 24px; align-items: start; }}
  .grid img {{ width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
  .cell h3 {{ font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; color: #888; margin: 12px 0 4px 0; }}
  .thought {{ white-space: pre-wrap; line-height: 1.45; }}
  pre.action {{ background: #f5f5f5; padding: 10px 12px; border-radius: 4px; overflow-x: auto; font-size: 0.85rem; }}
  pre.action.none {{ color: #999; }}
  .status code {{ background: #eee; padding: 1px 6px; border-radius: 3px; }}
  .timings {{ color: #888; font-size: 0.85rem; margin-top: 4px; }}
  aside.notes {{ background: #fffbea; border: 1px solid #f0d36a; padding: 8px 16px; border-radius: 4px; margin-top: 16px; }}
  @media (prefers-color-scheme: dark) {{
    body {{ background: #111; color: #eee; }}
    section.turn {{ border-top-color: #333; }}
    .grid img {{ border-color: #333; }}
    pre.action {{ background: #1a1a1a; }}
    .status code {{ background: #222; }}
    aside.notes {{ background: #2a2410; border-color: #5a4d20; }}
  }}
</style>
</head>
<body>
<header>
  <h1>{task}</h1>
  <p>finished: {finished}</p>
</header>
{notes_html}
{turns_html}
</body>
</html>
"""


def write_viewer(run_dir: Path) -> Path:
    transcript_path = run_dir / "transcript.json"
    if not transcript_path.exists():
        raise FileNotFoundError(f"no transcript.json in {run_dir}")
    transcript = json.loads(transcript_path.read_text())
    out = run_dir / "transcript.html"
    out.write_text(render(transcript))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Render a Fara run as a static HTML page")
    p.add_argument("run_dir", type=Path, help="Directory containing transcript.json and turn_*.png")
    args = p.parse_args(argv)
    out = write_viewer(args.run_dir)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
