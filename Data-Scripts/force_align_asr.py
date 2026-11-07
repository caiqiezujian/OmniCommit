#!/usr/bin/env python3
"""Optional real acoustic alignment; use a separate WhisperX environment."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omnicommit.forced_alignment import main

if __name__ == "__main__":
    raise SystemExit(main())
