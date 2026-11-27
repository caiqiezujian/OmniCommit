"""Check known-silence recordings through Fourth's existing SFT reader and preflight."""
from copy import deepcopy
import json
import os
from pathlib import Path
import runpy
import sys
import wave

import pytest

FOURTH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FOURTH))
from fourth_asr.chunk_refs import decode_chunk_ref
from fourth_asr.protocol import DIRECT, EOS, WAIT, get_protocol
from fourth_asr.recast import build_sft_row
from fourth_asr.sft_checks import main as check_sft, validate_row
from fourth_asr.timestamped_source import Unit

generate = runpy.run_path(str(FOURTH / "Data-Scripts" / "build_silence_sft.py"))["generate"]


def wav(path, frames, rate=16000, channels=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setparams((channels, 2, rate, 0, "NONE", ""))
        stream.writeframes(bytes(frames * channels * 2))
    return path


def test_whole_recording_all_wait_and_mixed_training_preflight(tmp_path):
    folder = tmp_path / "sources"
    first = wav(folder / "nested" / "安静.WAV", 37600)
    second = wav(folder / "exact.wav", 44100, 22050, 2)
    tiny = wav(folder / "tiny.wav", 1, 48000)
    out = generate(folder, tmp_path / "negative", 3, 42, 1000, "normalized")
    assert out.name == "train.1000ms.jsonl"
    by_path = {Path(row["recast"]["source_audio"]): row for row in map(json.loads, out.read_text("utf-8").splitlines())}
    assert set(by_path) == {first, second, tiny}
    assert [len(by_path[p]["audios"]) for p in (first, second, tiny)] == [3, 2, 1]
    refs = [decode_chunk_ref(ref) for ref in by_path[first]["audios"]]
    assert [(r["start"], r["end"], r["input_seconds"]) for r in refs] == [(0, 1, 1), (1, 2, 1), (2, 2.35, .5)]
    profile = get_protocol("speaker")
    for row in by_path.values():
        turns = len(row["audios"])
        assert row["messages"][0]["content"] == profile.system_prompt("normalized")
        assert [m["content"] for m in row["messages"][1::2]] == ["<audio>"] * turns + [EOS]
        assert [m["content"] for m in row["messages"][2::2]] == [WAIT] * (turns + 1)
        assert row["recast"]["events"] == [{"turn": 1, "kind": "silence_only"}, {"turn": turns, "kind": "audio_end"}]

    speech = build_sft_row([Unit(0, 1000, (("en", "spk1", "Hello."),))], first, 37600, 16000, 1000,
                           "normalized", protocol=profile)
    mixed = tmp_path / "mixed.jsonl"
    mixed.write_bytes((json.dumps(speech) + "\n").encode() + out.read_bytes())
    sidecar = Path(str(out) + ".tokens.txt")
    assert sidecar.read_text("utf-8").splitlines() == list(profile.tokens)
    assert (out.parent / "token.txt").read_bytes() == sidecar.read_bytes()
    assert check_sft([str(mixed), "--tokens-file", str(sidecar), "--decode"]) == 0

    row = deepcopy(by_path[first])
    row["recast"]["events"].pop(0)
    with pytest.raises(ValueError, match="no transcribed speech"):
        validate_row(row)
    row = deepcopy(by_path[first])
    row["messages"][2]["content"] = DIRECT + "invented"
    with pytest.raises(ValueError, match="standalone WAIT"):
        validate_row(row)
    row = deepcopy(by_path[first])
    row["recast"]["events"].append({"turn": 3, "kind": "annotation_end"})
    with pytest.raises(ValueError):
        validate_row(row)


def test_sample_without_replacement_is_reproducible_and_never_publishes_a_short_set(tmp_path):
    folder = tmp_path / "sources"
    files = [wav(folder / f"part{i}.wav", 16032 + i) for i in range(5)]
    os.link(files[0], folder / "alias.wav")
    (folder / "broken.wav").write_bytes(b"not audio")
    wav(folder / "empty.wav", 0)
    output_dir = tmp_path / "negative"
    with pytest.raises(ValueError, match="Only 5 usable distinct recordings"):
        generate(folder, output_dir, 6, 7, 800, "verbatim")
    assert list(output_dir.iterdir()) == []
    out = generate(folder, output_dir, 3, 7, 800, "verbatim")
    repeat = generate(folder, tmp_path / "repeat", 3, 7, 800, "verbatim")
    assert out.name == "train.800ms.jsonl"
    assert out.read_bytes() == repeat.read_bytes()
    rows = list(map(json.loads, out.read_text("utf-8").splitlines()))
    assert len({os.stat(r["recast"]["source_audio"]).st_ino for r in rows}) == len(rows) == 3
    assert all(len(r["audios"]) == 2 and r["recast"]["chunk_ms"] == 800 for r in rows)
    with pytest.raises(FileExistsError):
        generate(folder, output_dir, 1, 42, 800, "normalized")
