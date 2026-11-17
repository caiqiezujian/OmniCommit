"""Check the Third build command against expected audio intervals and transcript turns."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import wave

import pytest

THIRD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THIRD))
from third_asr.chunk_refs import decode_chunk_ref
from third_asr.protocol import DIRECT, EOS, WAIT, get_protocol, parse_reply


@pytest.mark.parametrize("mode", ["explicit", "plain"])
def test_standalone_third_build_on_one_second_grid(tmp_path, mode):
    # Deploy only Third, allowing both its location and its folder name to change.
    deployment = tmp_path / "standalone-asr"
    shutil.copytree(THIRD / "Data-Scripts", deployment / "Data-Scripts")
    shutil.copytree(THIRD / "third_asr", deployment / "third_asr", ignore=shutil.ignore_patterns("__pycache__"))
    work = tmp_path / "input"
    work.mkdir()
    with wave.open(str(work / "talk.wav"), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", ""))
        wav.writeframes(bytes(2 * 37600))  # 2.35 s
    pieces = ["<zh><spk1>好。", "<en><spk2>See you", " tomorrow."]
    record = {
        "audios": ["talk.wav"],
        "messages": [{"role": "assistant", "content": "".join(pieces)}],
        "output": {"segments": [{"source_text": text} for text in pieces]},
        "timestamps": [{"start_ms": 50, "end_ms": 1160}, {"start_ms": 1160, "end_ms": 1990},
                       {"start_ms": 2010, "end_ms": 2350}],
    }
    (work / "input.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    script = deployment / "Data-Scripts" / "build_sft.py"
    run = subprocess.run([sys.executable, "-I", str(script), "input.jsonl", "--output-dir", "output",
                          "--chunk-ms", "1000", "--protocol", mode], cwd=work, capture_output=True)
    assert run.returncode == 0, run.stderr.decode(errors="replace")
    out = work / "output"
    train = json.loads((out / "train.1000ms.jsonl").read_text(encoding="utf-8"))
    prefix = DIRECT if mode == "explicit" else ""
    assert len(train["audios"]) == 3
    turns = train["messages"]
    assert [m["content"] for m in turns if m["role"] == "assistant"] == [
        WAIT, prefix + "好。See you", prefix + " tomorrow.", WAIT,
    ]
    assert turns[-2] == {"role": "user", "content": EOS}
    tokens = (out / "train.1000ms.jsonl.tokens.txt").read_text().splitlines()
    assert tokens == ([WAIT, DIRECT] if mode == "explicit" else [WAIT])


@pytest.mark.parametrize("mode", ["semantic-unit", "timed-span"])
@pytest.mark.parametrize("protocol_name", ["plain", "explicit"])
def test_third_preserves_intervals_and_text(tmp_path, mode, protocol_name):
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
    out = tmp_path / "training"
    cmd = [sys.executable, str(THIRD / "Data-Scripts" / "build_sft.py"), str(source),
           "--output-dir", str(out), "--release-on", mode, "--protocol", protocol_name]
    subprocess.run(cmd, check=True, capture_output=True)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert (summary["kept"], summary["rejected"], summary["chunk_ms"]) == (1, 1, [800])
    row = json.loads((out / "train.800ms.jsonl").read_text(encoding="utf-8"))
    refs = [decode_chunk_ref(ref) for ref in row["audios"]]
    assert [(ref["start"], ref["end"]) for ref in refs] == [(0, 0.8), (0.8, 1.6), (1.6, 2.36)]
    assert all(Path(ref["path"]) == path for ref in refs)
    assert [ref["input_seconds"] for ref in refs] == pytest.approx([.8, .8, .76])
    prompts = row["messages"]
    protocol = get_protocol(protocol_name)
    assert prompts[0]["content"] == protocol.system_prompt("normalized")
    assert summary["protocol"] == protocol_name
    assert prompts[-2:] == [{"role": "user", "content": EOS}, {"role": "assistant", "content": WAIT}]
    replies = [m["content"] for m in prompts if m["role"] == "assistant"]
    expected = ([WAIT, "发给你了。Let's"] if mode == "semantic-unit" else ["发给你了。", "Let's"])
    assert replies == [text if text == WAIT or protocol_name == "plain" else DIRECT + text
                       for text in [*expected, " meet tomorrow.", WAIT]]
    assert (out / "train.800ms.jsonl.tokens.txt").read_text().splitlines() == list(protocol.tokens)


@pytest.mark.parametrize("invalid", ["", " ", " <|WAIT|>", "<|WAIT|>x", "<zh>好", "<spk1>hey", "<|DIRECT|>x"])
def test_third_rejects_mixed_or_tagged_replies(invalid):
    with pytest.raises(ValueError):
        parse_reply(invalid)


def test_third_preserves_word_separator():
    assert parse_reply(" meet") == " meet"
    assert parse_reply(WAIT) == ""


@pytest.mark.parametrize("invalid", [DIRECT, DIRECT + " ", DIRECT + WAIT, DIRECT + DIRECT + "yes",
                                     WAIT + DIRECT + "yes", DIRECT + "<en>yes", "yes"])
def test_explicit_requires_one_direct_and_new_text(invalid):
    with pytest.raises(ValueError):
        get_protocol("explicit").parse_reply(invalid)


def test_explicit_preserves_spaces():
    assert get_protocol("explicit").parse_reply(DIRECT + " meet\n") == " meet\n"
