"""CLI entry point for the `fara-agent` console script and `python run.py`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agent import FaraAgent
from .executor import BrowserExecutor
from .model import FaraModel


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run a Fara-7B agent task")
    p.add_argument("task", help="Natural-language task for the agent")
    p.add_argument("--start-url", default="https://www.google.com")
    p.add_argument("--max-turns", type=int, default=15)
    p.add_argument("--headless", action="store_true", help="Run browser headless")
    p.add_argument("--run-dir", default="runs/latest", type=Path)
    p.add_argument("--model", default=None, type=Path)
    p.add_argument("--mmproj", default=None, type=Path)
    p.add_argument(
        "--settle-ms",
        type=int,
        default=1500,
        help="Max time to wait for the page to reach networkidle after each action",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy, recommended for agentic runs)",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Max tokens per model turn — must hold thought + <tool_call>",
    )
    p.add_argument(
        "--history-images",
        type=int,
        default=3,
        help="Number of recent screenshots to keep in the prompt history",
    )
    p.add_argument(
        "--interactive",
        action="store_true",
        help="At a Critical Point, prompt y/n on the proposed action and continue if approved",
    )
    p.add_argument(
        "--viewer",
        action="store_true",
        help="After the run finishes, write a transcript.html alongside transcript.json",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    model_kwargs = {}
    if args.model:
        model_kwargs["model_path"] = args.model
    if args.mmproj:
        model_kwargs["mmproj_path"] = args.mmproj

    print("Loading Fara-7B...")
    model = FaraModel(**model_kwargs)
    print(f"Opening browser ({'headless' if args.headless else 'headed'})...")
    executor = BrowserExecutor(headless=args.headless, settle_ms=args.settle_ms)

    agent = FaraAgent(
        model=model,
        executor=executor,
        run_dir=args.run_dir,
        start_url=args.start_url,
        max_turns=args.max_turns,
        history_image_window=args.history_images,
        interactive=args.interactive,
    )

    _orig_step = model.step

    def step_with_overrides(*a, **kw):
        kw.setdefault("max_tokens", args.max_tokens)
        kw.setdefault("temperature", args.temperature)
        return _orig_step(*a, **kw)

    model.step = step_with_overrides  # type: ignore[assignment]

    try:
        record = agent.run(args.task)
        print(f"\n=== FINISHED: {record.finished} ===")
        print(f"transcript: {args.run_dir}/transcript.json")
        print(f"screenshots: {args.run_dir}/turn_*.png")
        if args.viewer:
            from .viewer import write_viewer

            html_path = write_viewer(Path(args.run_dir))
            print(f"viewer:     {html_path}")
        return 0
    finally:
        executor.close()


if __name__ == "__main__":
    sys.exit(main())
