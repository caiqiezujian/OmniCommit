#!/usr/bin/env python3
"""Validate Fourth streaming rows, audio references, targets and token inventory."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fourth_asr.sft_checks import main, read_tokens, validate_row


if __name__ == "__main__":
    sys.exit(main())
