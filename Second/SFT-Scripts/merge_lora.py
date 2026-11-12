#!/usr/bin/env python3
"""Validate and export a Second LoRA checkpoint for the Second ASR service."""
from pathlib import Path
import runpy
import sys

SECOND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SECOND))
from second_asr.service_backend import read_export_protocol


def main(argv=None):
    shared = runpy.run_path(str(SECOND.parent / "SFT-Scripts" / "merge_lora.py"))
    return shared["main"](argv, protocol_reader=read_export_protocol)


if __name__ == "__main__":
    sys.exit(main())
