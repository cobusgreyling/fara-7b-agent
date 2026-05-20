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
    args = p.parse_args()

    model_kwargs = {}
    if args.model:
        model_kwargs["model_path"] = args.model
    if args.mmproj:
        model_kwargs["mmproj_path"] = args.mmproj

    print(f"Loading Fara-7B...")
    model = FaraModel(**model_kwargs)
    print(f"Opening browser ({'headless' if args.headless else 'headed'})...")
    executor = BrowserExecutor(headless=args.headless)

    agent = FaraAgent(
        model=model,
        executor=executor,
        run_dir=args.run_dir,
        start_url=args.start_url,
        max_turns=args.max_turns,
    )

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
