"""CLI entry point: run a Fara-7B agent task in a real browser.

Usage:
    python run.py "Find the Wikipedia article about Claude Shannon."
    python run.py "Find a hotel in Cape Town for next weekend." --start-url https://www.google.com --max-turns 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fara.agent import FaraAgent
from fara.executor import BrowserExecutor
from fara.model import FaraModel


def main() -> int:
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
        help="Max tokens per model turn — must be large enough to contain thought + <tool_call>",
    )
    p.add_argument(
        "--history-images",
        type=int,
        default=3,
        help="Number of recent screenshots to keep in the prompt history",
    )
    args = p.parse_args()

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
    )

    # Per-turn sampling overrides — patched onto the model.step bound method
    # so the agent loop need not know about them.
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
        return 0
    finally:
        executor.close()


if __name__ == "__main__":
    sys.exit(main())
