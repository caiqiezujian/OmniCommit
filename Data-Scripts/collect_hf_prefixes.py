#!/usr/bin/env python3
"""Collect real batched HF Omni prefixes from switch-timestamp JSONL."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omnicommit.prefix_collection import main

if __name__ == "__main__":
    raise SystemExit(main())
