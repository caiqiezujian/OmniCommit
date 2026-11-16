"""Compare real Second and Third build commands on the same chunk grid."""
import json
from pathlib import Path
import subprocess
import sys
import wave

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Third"))
from third_asr.protocol import EOS, WAIT, parse_reply, system_prompt


@pytest.mark.parametrize("mode", ["semantic-unit", "timed-span"])
def test_third_preserves_second_turns_and_text(tmp_path, mode):
    path = tmp_path / "switch.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", ""))
        wav.writeframes(bytes(2 * 37760))  # 2.36 s, including an incomplete last grid cell
    head, tail = "<zh><spk1>发给你了。", "<en><spk2>Let's"
    record = {
        "audios": [path.name],
        "messages": [{"role": "assistant", "content": head + tail + " meet tomorrow."}],
        "output": json.dumps({"segments": [{"source_text": head + tail, "action": "WAIT"},
                                          {"source_text": " meet tomorrow.", "action": "DIRECT"}]}),
        "timestamps": [
            {"start_ms": 0, "end_ms": 1350, "spans": [
                {"start_ms": 0, "end_ms": 800, "text": head, "language": "zh", "speaker": "spk1"},
                {"start_ms": 990, "end_ms": 1350, "text": tail, "language": "en", "speaker": "spk2"}]},
            {"start_ms": 1400, "end_ms": 2300}],
    }
    source = tmp_path / "data.jsonl"
    source.write_text("invalid\n" + json.dumps(record) + "\n", encoding="utf-8")
    outputs = {}
    for family in ("Second", "Third"):
        out = tmp_path / family
        cmd = [sys.executable, str(ROOT / family / "Data-Scripts" / "build_sft.py"), str(source),
               "--output-dir", str(out), "--release-on", mode]
        if family == "Second":
            cmd.extend(["--chunk-ms", "800"])
        subprocess.run(cmd, check=True, capture_output=True)
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        assert (summary["kept"], summary["rejected"], summary["chunk_ms"]) == (1, 1, [800])
        outputs[family] = json.loads((out / "train.800ms.jsonl").read_text(encoding="utf-8"))
    assert outputs["Second"]["audios"] == outputs["Third"]["audios"]
    prompts = outputs["Third"]["messages"]
    assert prompts[0]["content"] == system_prompt("normalized")
    assert prompts[-2:] == [{"role": "user", "content": EOS}, {"role": "assistant", "content": WAIT}]
    replies = [m["content"] for m in prompts if m["role"] == "assistant"]
    expected = ([WAIT, "发给你了。Let's"] if mode == "semantic-unit" else ["发给你了。", "Let's"])
    assert replies == [*expected, " meet tomorrow.", WAIT]
    second = [m["content"] for m in outputs["Second"]["messages"] if m["role"] == "assistant"]
    assert [text == WAIT for text in second] == [text == WAIT for text in replies]
    assert (tmp_path / "Third" / "train.800ms.jsonl.tokens.txt").read_text().splitlines() == [WAIT]


@pytest.mark.parametrize("invalid", ["", " ", " <|WAIT|>", "<|WAIT|>x", "<zh>好", "<spk1>hey", "<|DIRECT|>x"])
def test_third_rejects_mixed_or_tagged_replies(invalid):
    with pytest.raises(ValueError):
        parse_reply(invalid)


def test_third_preserves_word_separator():
    assert parse_reply(" meet") == " meet"
    assert parse_reply(WAIT) == ""
