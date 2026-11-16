#!/usr/bin/env python3
"""Build Third SFT rows from Second-compatible timestamped source records."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Second"))
sys.path.insert(0, str(ROOT / "Third"))

from second_asr.dataset import main as build_dataset
from third_asr.protocol import PROTOCOLS, get_protocol


def main(argv=None):
    return build_dataset(argv, protocol=get_protocol("explicit"), default_chunk_ms=800, protocol_choices=PROTOCOLS)


if __name__ == "__main__":
    raise SystemExit(main())
