"""The Fara agent loop: screenshot -> infer -> parse -> execute -> repeat."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .executor import BrowserExecutor
from .model import FaraModel
from .parser import FaraTurn, parse

CRITICAL_KEYWORDS = (
    "checkout",
    "purchase",
    "place order",
    "confirm payment",
    "submit payment",
    "book now",
    "send email",
    "call now",
    "sign up",
)


@dataclass
class TurnRecord:
    turn: int
    screenshot: str
    thought: str
    action: dict[str, Any] | None
    status: str


@dataclass
class RunRecord:
    task: str
    turns: list[TurnRecord] = field(default_factory=list)
    finished: str = ""


def _is_critical(thought: str) -> bool:
    low = thought.lower()
    return any(k in low for k in CRITICAL_KEYWORDS)


class FaraAgent:
    def __init__(
        self,
        model: FaraModel,
        executor: BrowserExecutor,
        run_dir: Path | str = "runs/latest",
        start_url: str = "https://www.google.com",
        max_turns: int = 15,
    ):
        self.model = model
        self.executor = executor
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.start_url = start_url
        self.max_turns = max_turns

    def run(self, task: str) -> RunRecord:
        record = RunRecord(task=task)
        self.executor.page.goto(self.start_url, wait_until="domcontentloaded")
        self.executor.page.wait_for_timeout(1000)

        history: list[dict[str, Any]] = []

        for turn_idx in range(1, self.max_turns + 1):
            shot = self.run_dir / f"turn_{turn_idx:02d}.png"
            self.executor.screenshot(shot)

            raw = self.model.step(task=task, screenshot_path=shot, history=history)
            parsed: FaraTurn = parse(raw)

            print(f"\n=== TURN {turn_idx} ===")
            print(f"thought: {parsed.thought}")
            if parsed.action:
                print(f"action:  {parsed.action.name}  args={parsed.action.arguments}")
            else:
                print("action:  (no tool_call emitted)")

            if parsed.action is None:
                status = "no_action"
                record.turns.append(
                    TurnRecord(turn_idx, str(shot), parsed.thought, None, status)
                )
                record.finished = "model returned no tool_call"
                break

            if parsed.action.name == "terminate":
                status = "terminated"
                record.turns.append(
                    TurnRecord(turn_idx, str(shot), parsed.thought, _action_dict(parsed), status)
                )
                record.finished = "model emitted terminate"
                break

            if _is_critical(parsed.thought):
                status = "critical_point_pause"
                record.turns.append(
                    TurnRecord(turn_idx, str(shot), parsed.thought, _action_dict(parsed), status)
                )
                record.finished = "paused at Critical Point — human input required"
                print("\n*** Critical Point reached — pausing for human input ***")
                break

            status = self.executor.execute(parsed.action.name, parsed.action.arguments)
            print(f"status:  {status}")

            record.turns.append(
                TurnRecord(turn_idx, str(shot), parsed.thought, _action_dict(parsed), status)
            )

            history.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": self.model._image_to_data_url(shot)
                            },
                        },
                        {"type": "text", "text": task if turn_idx == 1 else ""},
                    ],
                }
            )
            history.append({"role": "assistant", "content": raw})

        else:
            record.finished = f"reached max_turns ({self.max_turns})"

        self._save(record)
        return record

    def _save(self, record: RunRecord) -> None:
        payload = {
            "task": record.task,
            "finished": record.finished,
            "turns": [t.__dict__ for t in record.turns],
        }
        (self.run_dir / "transcript.json").write_text(json.dumps(payload, indent=2))


def _action_dict(parsed: FaraTurn) -> dict[str, Any] | None:
    if parsed.action is None:
        return None
    return {"name": parsed.action.name, "arguments": parsed.action.arguments}
