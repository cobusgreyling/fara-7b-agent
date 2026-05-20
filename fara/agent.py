"""The Fara agent loop: screenshot -> infer -> parse -> execute -> repeat."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .executor import BrowserExecutor
from .model import FaraModel, image_to_data_url
from .parser import FaraTurn, parse

# Actions that commit a user-facing transaction. Combined with a URL match
# against `CRITICAL_URL_PATTERNS` in the executor, these are treated as a
# Critical Point and the loop pauses for human input.
COMMITTING_ACTIONS = {"left_click", "key"}
COMMITTING_KEYS = {"Enter", "Return"}

# Keep the last N screenshots in the history sent to the model. Older turns
# keep their assistant text (chain-of-thought + tool_call) but the images are
# dropped — the n_ctx budget is dominated by vision tokens.
HISTORY_IMAGE_WINDOW = 3


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
    notes: list[str] = field(default_factory=list)


def _is_committing(action_name: str, args: dict[str, Any]) -> bool:
    if action_name == "left_click":
        return True
    if action_name == "type" and args.get("press_enter"):
        return True
    if action_name == "key":
        key = (args.get("key") or "").strip()
        return any(k in key for k in COMMITTING_KEYS)
    return False


class FaraAgent:
    def __init__(
        self,
        model: FaraModel,
        executor: BrowserExecutor,
        run_dir: Path | str = "runs/latest",
        start_url: str = "https://www.google.com",
        max_turns: int = 15,
        history_image_window: int = HISTORY_IMAGE_WINDOW,
    ):
        self.model = model
        self.executor = executor
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.start_url = start_url
        self.max_turns = max_turns
        self.history_image_window = history_image_window

    def run(self, task: str) -> RunRecord:
        record = RunRecord(task=task)
        self.executor.page.goto(self.start_url, wait_until="domcontentloaded")
        self.executor.settle()

        # Each entry: (screenshot_path, assistant_raw_text)
        turn_log: list[tuple[Path, str]] = []
        notes: list[str] = []

        for turn_idx in range(1, self.max_turns + 1):
            shot = self.run_dir / f"turn_{turn_idx:02d}.png"
            self.executor.screenshot(shot)

            history = self._build_history(task, turn_log)
            raw = self.model.step(
                task=task,
                screenshot_path=shot,
                history=history,
                notes=notes,
            )
            parsed: FaraTurn = parse(raw)

            print(f"\n=== TURN {turn_idx} ===")
            print(f"thought: {parsed.thought}")
            if parsed.action:
                print(f"action:  {parsed.action.name}  args={parsed.action.arguments}")
            else:
                print("action:  (no tool_call emitted)")

            if parsed.action is None:
                record.turns.append(
                    TurnRecord(turn_idx, str(shot), parsed.thought, None, "no_action")
                )
                record.finished = "model returned no tool_call"
                self._save(record, notes)
                break

            if parsed.action.name == "terminate":
                record.turns.append(
                    TurnRecord(turn_idx, str(shot), parsed.thought, _action_dict(parsed), "terminated")
                )
                record.finished = "model emitted terminate"
                self._save(record, notes)
                break

            if self.executor.at_critical_url() and _is_committing(
                parsed.action.name, parsed.action.arguments
            ):
                record.turns.append(
                    TurnRecord(
                        turn_idx,
                        str(shot),
                        parsed.thought,
                        _action_dict(parsed),
                        "critical_point_pause",
                    )
                )
                record.finished = (
                    f"paused at Critical Point ({self.executor.page.url}) — human input required"
                )
                print("\n*** Critical Point reached — pausing for human input ***")
                self._save(record, notes)
                break

            status = self.executor.execute(parsed.action.name, parsed.action.arguments)
            print(f"status:  {status}")

            if parsed.action.name == "pause_and_memorize_fact":
                fact = (parsed.action.arguments.get("fact") or "").strip()
                if fact:
                    notes.append(fact)

            record.turns.append(
                TurnRecord(turn_idx, str(shot), parsed.thought, _action_dict(parsed), status)
            )
            turn_log.append((shot, raw))
            self._save(record, notes)

        else:
            record.finished = f"reached max_turns ({self.max_turns})"
            self._save(record, notes)

        record.notes = notes
        return record

    def _build_history(
        self, task: str, turn_log: list[tuple[Path, str]]
    ) -> list[dict[str, Any]]:
        """Reconstruct the chat history for the next inference turn.

        Past turns within the sliding image window contribute (user-image,
        assistant-text) pairs. Older turns contribute only the assistant text
        — dropping their images keeps n_ctx from blowing up while preserving
        the chain of reasoning.
        """
        history: list[dict[str, Any]] = []
        cutoff = len(turn_log) - self.history_image_window
        for i, (shot, raw) in enumerate(turn_log):
            if i < cutoff:
                history.append({"role": "assistant", "content": raw})
                continue
            history.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_to_data_url(shot)},
                        },
                        {"type": "text", "text": task},
                    ],
                }
            )
            history.append({"role": "assistant", "content": raw})
        return history

    def _save(self, record: RunRecord, notes: list[str]) -> None:
        payload = {
            "task": record.task,
            "finished": record.finished,
            "notes": list(notes),
            "turns": [t.__dict__ for t in record.turns],
        }
        (self.run_dir / "transcript.json").write_text(json.dumps(payload, indent=2))


def _action_dict(parsed: FaraTurn) -> dict[str, Any] | None:
    if parsed.action is None:
        return None
    return {"name": parsed.action.name, "arguments": parsed.action.arguments}
