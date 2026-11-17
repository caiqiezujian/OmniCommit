import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import pytest


THIRD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THIRD))
from third_asr.chunk_refs import encode_chunk_ref
WAIT = "<|WAIT|>"
DIRECT = "<|DIRECT|>"


def run_script(script, *args, cwd=None):
    return subprocess.run([sys.executable, "-I", str(script), *map(str, args)], cwd=cwd,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")


@pytest.mark.parametrize("number_format", ["normalized", "verbatim"])
@pytest.mark.parametrize("protocol", ["plain", "explicit"])
def test_existing_second_dataset_matches_third_recast(tmp_path, number_format, protocol):
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
    result = run_script(THIRD / "Data-Scripts" / "build_sft.py", timed, "--chunk-ms", 800,
                        "--output-dir", tmp_path / "Third", "--number-format", number_format,
                        "--protocol", protocol)
    assert result.returncode == 0, result.stderr
    name = "train.800ms.jsonl"
    numbers = "Use digits for numbers." if number_format == "normalized" else "Preserve spoken numbers."
    messages = [{"role": "system", "content": (
        "Transcribe new speech. Mark first text and changes with <language><speaker>. "
        f"Emit {WAIT} if nothing is ready. {numbers}")}]
    refs = []
    for i, reply in enumerate([WAIT, original[0] + original[1], original[2], original[3]]):
        refs.append(encode_chunk_ref(str(wav), i * 800 / 1000, (i + 1) * 800 / 1000, .8))
        messages.extend([{"role": "user", "content": "<audio>"}, {"role": "assistant", "content": reply}])
    messages.extend([{"role": "user", "content": "<end_of_audio>"}, {"role": "assistant", "content": WAIT}])
    sample = json.dumps({"audios": refs, "messages": messages}) + "\n"
    wav.unlink()  # Conversion only changes the target protocol.
    source = tmp_path / "existing.jsonl"
    source.write_text("{invalid\n" + sample, encoding="utf-8")
    output = tmp_path / "converted" / name
    # Simulate copying just Third to a server. Keep its parent free of any
    # Second modules and start Python in an unrelated working directory.
    copied = tmp_path / "deployment" / "Third"
    shutil.copytree(THIRD / "Data-Scripts", copied / "Data-Scripts")
    shutil.copytree(THIRD / "third_asr", copied / "third_asr", ignore=shutil.ignore_patterns("__pycache__"))
    convert = copied / "Data-Scripts" / "from_second_sft.py"
    protocol_args = ["--protocol", "plain"] if protocol == "plain" else []
    command = run_script(convert, source, "--output", output, *protocol_args, cwd=tmp_path)
    assert command.returncode == 0, command.stderr
    converted = json.loads(output.read_text(encoding="utf-8"))
    assert converted == json.loads((tmp_path / "Third" / name).read_text(encoding="utf-8"))
    replies = [m["content"] for m in converted["messages"] if m["role"] == "assistant"]
    prefix = DIRECT if protocol == "explicit" else ""
    inventory = [WAIT, DIRECT] if protocol == "explicit" else [WAIT]
    assert replies == [WAIT, prefix + "开始。Let's", prefix + " meet", prefix + "，好的。\n", WAIT]
    assert output.with_suffix(".jsonl.tokens.txt").read_text(encoding="utf-8").splitlines() == inventory
    assert json.loads(output.with_suffix(".jsonl.rejected.jsonl").read_text(encoding="utf-8"))["line"] == 1
    assert run_script(convert, source, "--output", output).returncode != 0
    if protocol == "explicit":
        plain = tmp_path / "old-third.jsonl"
        upgraded = tmp_path / "upgraded.jsonl"
        assert run_script(convert, source, "--protocol", "plain", "--output", plain).returncode == 0
        assert run_script(convert, plain, "--output", upgraded).returncode == 0
        assert upgraded.read_bytes() == output.read_bytes()

    if (number_format, protocol) == ("normalized", "explicit"):
        broken = []
        for field, index, key, value in (
            ("messages", 2, "content", "speech before initial attribution"),
            ("messages", 3, "role", "assistant"),
            ("messages", 5, "content", "no audio placeholder"),
            ("messages", 6, "content", "<en><spk2>redundant header"),
            ("messages", -1, "content", "<|DIRECT|>text after EOS"),
        ):
            row = json.loads(sample)
            row[field][index][key] = value
            broken.append(row)
        overlap = json.loads(sample)
        overlap["audios"][1] = overlap["audios"][0]
        broken.append(overlap)
        invalid_source = tmp_path / "invalid.jsonl"
        invalid_source.write_text("".join(json.dumps(v) + "\n" for v in broken), encoding="utf8")
        invalid_output = tmp_path / "invalid-output.jsonl"
        result = run_script(convert, invalid_source, "--output", invalid_output, cwd=tmp_path)
        assert result.returncode == 1 and invalid_output.read_bytes() == b""
        rejected = Path(str(invalid_output) + ".rejected.jsonl")
        assert [json.loads(line)["line"] for line in rejected.read_text().splitlines()] == list(range(1, len(broken)+1))
