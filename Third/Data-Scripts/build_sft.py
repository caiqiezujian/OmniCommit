#!/usr/bin/env python3
"""Build Third SFT rows from Second-compatible timestamped source records."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Second"))
sys.path.insert(0, str(ROOT / "Third"))

from second_asr.dataset import main as build_dataset
from third_asr import protocol


def main(argv=None):
    return build_dataset(argv, protocol=protocol, default_chunk_ms=800)


if __name__ == "__main__":
    raise SystemExit(main())
