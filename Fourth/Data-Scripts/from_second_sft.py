#!/usr/bin/env python3
"""Compatibility entry point. Fourth recasts the original timestamped source."""
import sys

from build_sft import main


if __name__ == "__main__":
    print("Fourth needs the timestamped source JSONL with messages, output and timestamps. "
          "Use Data-Scripts/build_sft.py INPUT --output-dir DIR.", file=sys.stderr)
    raise SystemExit(main())
