"""The historical entry point cannot recover switch timing from compact training targets."""
import json
from pathlib import Path
import subprocess
import sys

FOURTH = Path(__file__).resolve().parents[1]


def test_old_entrypoint_redirects_and_requires_timestamped_source(tmp_path):
    source = tmp_path / "compact.jsonl"
    source.write_text(json.dumps({"audios": [], "messages": []}) + "\n", encoding="utf-8")
    output = tmp_path / "new"
    result = subprocess.run([sys.executable, str(FOURTH / "Data-Scripts/from_second_sft.py"),
                             str(source), "--output-dir", str(output)], capture_output=True, text=True)
    assert result.returncode == 1
    assert "timestamped source" in result.stderr
    assert (output / "train.1000ms.jsonl").read_bytes() == b""
    assert json.loads((output / "rejected.jsonl").read_text())["line"] == 1
