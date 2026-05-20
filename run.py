"""Shim that invokes the fara-agent CLI.

For installed users:
    fara-agent "Find the Wikipedia article about Claude Shannon."

For source-tree users:
    python run.py "Find the Wikipedia article about Claude Shannon."
"""
from __future__ import annotations

import sys

from fara.cli import main

if __name__ == "__main__":
    sys.exit(main())
