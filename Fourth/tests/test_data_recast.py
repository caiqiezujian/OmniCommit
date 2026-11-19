"""Check the Fourth build command against expected audio intervals and transcript turns."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import wave

import pytest

FOURTH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FOURTH))
from fourth_asr.chunk_refs import decode_chunk_ref
from fourth_asr.protocol import DIRECT, EOS, SPK_CHANGE, WAIT, get_protocol, parse_reply


@pytest.mark.parametrize("mode", ["speaker", "explicit"])
def test_standalone_fourth_build_on_one_second_grid(tmp_path, mode):
    # Deploy only Fourth, allowing both its location and its folder name to change.
    deployment = tmp_path / "standalone-asr"
    shutil.copytree(FOURTH / "Data-Scripts", deployment / "Data-Scripts")
    shutil.copytree(FOURTH / "fourth_asr", deployment / "fourth_asr", ignore=shutil.ignore_patterns("__pycache__"))
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
    assert len(train["audios"]) == 3
    turns = train["messages"]
    assert [m["content"] for m in turns if m["role"] == "assistant"] == [
        WAIT, DIRECT + "好。", DIRECT + SPK_CHANGE + "See you tomorrow.", WAIT,
    ]
    assert turns[-2] == {"role": "user", "content": EOS}
    tokens = (out / "train.1000ms.jsonl.tokens.txt").read_text().splitlines()
    assert tokens == [WAIT, DIRECT, SPK_CHANGE]
    assert (out / "token.txt").read_bytes() == (out / "train.1000ms.jsonl.tokens.txt").read_bytes()
    assert train["recast"]["events"] == [{"turn": 2, "kind": "switch_deferred"},
                                         {"turn": 3, "kind": "speaker_change"},
                                         {"turn": 3, "kind": "audio_end"}]
    assert train["recast"]["source_audio"] == str((work / "talk.wav").resolve())


@pytest.mark.parametrize("mode", ["semantic-unit", "timed-span"])
def test_fourth_preserves_intervals_and_text(tmp_path, mode):
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
    cmd = [sys.executable, str(FOURTH / "Data-Scripts" / "build_sft.py"), str(source),
           "--output-dir", str(out), "--release-on", mode, "--chunk-ms", "800"]
    subprocess.run(cmd, check=True, capture_output=True)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert (summary["kept"], summary["rejected"], summary["chunk_ms"]) == (1, 1, [800])
    row = json.loads((out / "train.800ms.jsonl").read_text(encoding="utf-8"))
    refs = [decode_chunk_ref(ref) for ref in row["audios"]]
    assert [(ref["start"], ref["end"]) for ref in refs] == [(0, 0.8), (0.8, 1.6), (1.6, 2.36)]
    assert all(Path(ref["path"]) == path for ref in refs)
    assert [ref["input_seconds"] for ref in refs] == pytest.approx([.8, .8, .76])
    prompts = row["messages"]
    protocol = get_protocol("speaker")
    assert prompts[0]["content"] == protocol.system_prompt("normalized")
    assert summary["protocol"] == "speaker"
    assert prompts[-2:] == [{"role": "user", "content": EOS}, {"role": "assistant", "content": WAIT}]
    replies = [m["content"] for m in prompts if m["role"] == "assistant"]
    assert replies == [DIRECT + "发给你了。", DIRECT + SPK_CHANGE + "Let's", DIRECT + " meet tomorrow.", WAIT]
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
