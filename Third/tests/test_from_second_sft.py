import json
import subprocess
import sys
import wave
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
THIRD = REPO / "Third"
WAIT = "<|WAIT|>"


def run_script(script, *args):
    return subprocess.run([sys.executable, str(script), *map(str, args)],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")


@pytest.mark.parametrize("number_format", ["normalized", "verbatim"])
def test_existing_second_dataset_matches_third_recast(tmp_path, number_format):
    wav = tmp_path / "talk.wav"
    with wave.open(str(wav), "wb") as f:
        f.setparams((1, 2, 16000, 51200, "NONE", ""))
        f.writeframes(bytes(102400))
    original = ["<zh><spk1>开始。", "<en><spk2>Let's", " meet", "<zh><spk1>，好的。\n"]
    intervals = [(300, 900), (920, 1510), (1700, 2200), (2260, 2900)]
    record = {
        "audios": [wav.name],
        "messages": [{"role": "assistant", "content": "".join(original)}],
        "output": {"segments": [{"source_text": s, "action": "WAIT"} for s in original]},
        "timestamps": [{"start_ms": start, "end_ms": end} for start, end in intervals],
    }
    timed = tmp_path / "timed.jsonl"
    timed.write_text(json.dumps(record) + "\n", encoding="utf-8")
    for name in ("Second", "Third"):
        result = run_script(REPO / name / "Data-Scripts" / "build_sft.py", timed,
                            "--chunk-ms", 800, "--output-dir", tmp_path / name,
                            "--number-format", number_format)
        assert result.returncode == 0, result.stderr
    name = "train.800ms.jsonl"
    sample = (tmp_path / "Second" / name).read_text(encoding="utf-8")
    wav.unlink()  # Conversion only changes the target protocol.
    source = tmp_path / "existing.jsonl"
    source.write_text("{invalid\n" + sample, encoding="utf-8")
    output = tmp_path / "converted" / name
    convert = THIRD / "Data-Scripts" / "from_second_sft.py"
    command = run_script(convert, source, "--output", output)
    assert command.returncode == 0, command.stderr
    converted = json.loads(output.read_text(encoding="utf-8"))
    assert converted == json.loads((tmp_path / "Third" / name).read_text(encoding="utf-8"))
    replies = [m["content"] for m in converted["messages"] if m["role"] == "assistant"]
    assert replies == [WAIT, "开始。Let's", " meet", "，好的。\n", WAIT]
    assert output.with_suffix(".jsonl.tokens.txt").read_text(encoding="utf-8").splitlines() == [WAIT]
    assert json.loads(output.with_suffix(".jsonl.rejected.jsonl").read_text(encoding="utf-8"))["line"] == 1
    assert run_script(convert, source, "--output", output).returncode != 0
