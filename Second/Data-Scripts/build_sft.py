"""Build Second speaker-attributed streaming SFT JSONL files."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from second_asr.dataset import json_line, main, prepare_record


if __name__ == "__main__":
    raise SystemExit(main())
