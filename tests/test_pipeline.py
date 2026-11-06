"""End-to-end CPU checks of the public CLIs. No model or API is contacted.

Audio is a synthetic waveform, not a recognition-quality fixture. Teacher
responses are replayed to exercise the whole-output commitment algorithm,
speaker/language attribution, resume, and conversion to interleaved SFT.
"""
from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]


def run_cli(script, *args, expected=0):
    result = subprocess.run(
        [sys.executable, "-X", "utf8", str(ROOT / script), *map(str, args)],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert result.returncode == expected, result.stdout + "\n" + result.stderr
    return result


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def segment(text, language="zh", speaker="spk1"):
    return {"language": language, "speaker": speaker, "text": text}


@pytest.fixture
def fixture_files(tmp_path):
    audio = tmp_path / "synthetic.wav"
    # Deliberately use 44.1 kHz stereo to exercise source slicing before resample.
    rate = 44100
    wave = np.sin(np.arange(4 * rate, dtype=np.float32) * 0.03) * 0.1
    sf.write(audio, np.stack([wave, wave * 0.5], axis=1), rate, subtype="FLOAT")
    budget = segment("预算是二十一万元。")
    question = segment("Any questions?", "en")
    answer = segment("没有问题。", speaker="spk2")
    full = [budget, question, answer]
    reference = "<zh><spk1>预算是二十一万元。<en><spk1>Any questions?<zh><spk2>没有问题。"
    source = {"id": "fixture-1", "audios": [str(audio)], "messages": [
        {"role": "system", "content": "Source instructions must not enter teacher calls."},
        {"role": "user", "content": "<audio>**Entity Reference:** []"},
        {"role": "assistant", "content": reference},
    ]}
    replay = {"id": source["id"], "full": {"segments": full}, "prefixes": [
        # A matching head on an incorrect whole hypothesis earns no partial credit.
        {"end_ms": 500, "segments": [segment("预算是二十二万元。")]},
        {"end_ms": 1000, "segments": [segment("预算是")]},
        {"end_ms": 1500, "segments": [budget]},
        # Text alone matches. Wrong speaker attribution must prevent submission.
        {"end_ms": 2000, "segments": [budget, segment("Any questions?", "en", "spk2")]},
        {"end_ms": 2500, "segments": [budget, question]},
        {"end_ms": 3000, "segments": [budget, question, segment("没有", speaker="spk2")]},
        {"end_ms": 3500, "segments": full},
    ]}
    paths = {name: tmp_path / (name + ".jsonl") for name in ["source", "replay", "trace", "trajectory", "train"]}
    write_jsonl(paths["source"], [source])
    write_jsonl(paths["replay"], [replay])
    return paths, audio, full


def collect(paths, *extra):
    return run_cli("Data-Scripts/collect_prefixes.py", paths["source"],
                   "--output", paths["trace"], "--replay-responses", paths["replay"],
                   "--chunk-seconds", "0.5", *extra)


def build(paths, *extra):
    return run_cli("Data-Scripts/build_trajectories.py", paths["trace"],
                   "--output", paths["trajectory"], *extra)


@pytest.mark.parametrize("wrong_attribute", ["speaker", "language"])
def test_replay_to_interleaved_sft_with_attribute_stability(fixture_files, wrong_attribute):
    paths, source_audio, full = fixture_files
    if wrong_attribute == "language":
        replay, = read_jsonl(paths["replay"])
        replay["prefixes"][3]["segments"][1] = segment("Any questions?", "zh", "spk1")
        write_jsonl(paths["replay"], [replay])
    collect(paths)
    build(paths)
    trajectory, = read_jsonl(paths["trajectory"])
    turns = trajectory["turns"]
    assert len(turns) == 8
    assert [t["action"] for t in turns] == ["WAIT", "DIRECT", "DIRECT", "WAIT", "DIRECT", "DIRECT", "DIRECT", "WAIT"]
    assert [(t["start_ms"], t["end_ms"]) for t in turns] == [(n * 500, (n + 1) * 500) for n in range(8)]
    assert turns[1]["segments"] == [segment("预算是")]
    assert turns[2]["segments"] == [segment("二十一万元。")]
    assert turns[4]["segments"] == [segment("Any questions?", "en")]
    assert turns[5]["segments"] == [segment("没有", speaker="spk2")]
    assert turns[6]["segments"] == [segment("问题。", speaker="spk2")]
    reconstructed = "".join(s["text"] for t in turns for s in t["segments"])
    assert reconstructed == "".join(s["text"] for s in full)

    run_cli("SFT-Scripts/prepare_sft.py", paths["trajectory"], "--output", paths["train"])
    trained, = read_jsonl(paths["train"])
    assert len(trained["audios"]) == 8
    messages = trained["messages"]
    assert messages[0]["role"] == "system"
    assert trained["policy"]["number_format"] == "verbatim"
    assert "do not convert them to digits" in messages[0]["content"].lower()
    assert len(messages) == 17
    assert [m["role"] for m in messages[1:]] == ["user", "assistant"] * 8
    assert all(m["content"].count("<audio>") == 1 for m in messages[1::2])
    answers = [m["content"] for m in messages[2::2]]
    assert answers == [
        "<|WAIT|>", "<|DIRECT|><zh><spk1>预算是",
        "<|DIRECT|><zh><spk1>二十一万元。", "<|WAIT|>",
        "<|DIRECT|><en><spk1>Any questions?",
        "<|DIRECT|><zh><spk2>没有", "<|DIRECT|><zh><spk2>问题。", "<|WAIT|>",
    ]
    assert "<|FLEX|>" not in json.dumps(trained)
    tokens = Path(str(paths["train"]) + ".tokens.txt").read_text(encoding="utf-8").splitlines()
    assert tokens[:2] == ["<|WAIT|>", "<|DIRECT|>"]
    assert set(tokens) == {"<|WAIT|>", "<|DIRECT|>", "<zh>", "<en>", "<spk1>", "<spk2>"}

    # Exercise the actual training-time virtual reader, not just metadata.
    spec = importlib.util.spec_from_file_location("pipeline_audio_reader", ROOT / "SFT-Scripts/omnicommit_audio.py")
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    assert all(ref.startswith("omnicommitaudio:v1:") for ref in trained["audios"])
    first = trained["audios"][0]
    details = reader.decode_chunk_ref(first)
    assert details["path"] == str(source_audio)
    assert details["start"] == 0 and details["end"] == 0.5
    samples, rate = sf.read(io.BytesIO(reader.chunk_wav_bytes(first)), dtype="float32")
    assert rate == 16000 and samples.shape == (8000,)
    run_cli("SFT-Scripts/check_sft.py", paths["train"], "--rows", "1")
    run_cli("SFT-Scripts/audit_sft.py", paths["train"], "--decode")


def test_completed_pipeline_resume_does_not_duplicate(fixture_files):
    paths, _, _ = fixture_files
    collect(paths)
    build(paths)
    trace_before = paths["trace"].read_bytes()
    trajectory_before = paths["trajectory"].read_bytes()
    collect(paths, "--resume")
    build(paths, "--resume")
    assert paths["trace"].read_bytes() == trace_before
    assert paths["trajectory"].read_bytes() == trajectory_before


def test_bad_source_record_does_not_stop_next_session(fixture_files):
    paths, _, _ = fixture_files
    valid, = read_jsonl(paths["source"])
    bad = {"id": "missing-audio", "audios": [str(paths["source"].parent / "absent.wav")], "messages": valid["messages"]}
    write_jsonl(paths["source"], [bad, valid])
    collect(paths)
    rows = read_jsonl(paths["trace"])
    assert len(rows) == 2
    assert rows[0]["status"] == "error" and rows[0]["output"] == ""
    assert rows[1]["status"] == "ok"
    build(paths)
    trajectories = read_jsonl(paths["trajectory"])
    assert len(trajectories) == 2
    assert trajectories[0]["status"] == "error" and trajectories[0]["output"] == ""
    assert len(trajectories[1]["turns"]) == 8
    run_cli("SFT-Scripts/prepare_sft.py", paths["trajectory"], "--output", paths["train"])
    assert len(read_jsonl(paths["train"])) == 1


def test_normalized_numeric_boundary_survives_all_stages(fixture_files):
    paths, _, _ = fixture_files
    source, = read_jsonl(paths["source"])
    source["messages"][-1]["content"] = "<en><spk1>Budget is 100000000."
    write_jsonl(paths["source"], [source])
    full = [segment("Budget is 100000000.", "en")]
    replay = {"id": source["id"], "full": {"segments": full}, "prefixes": [
        {"end_ms": 500, "segments": [segment("Budget is ", "en")]},
        # Although this is a literal prefix, do not commit part of the final
        # normalized digit string as if its magnitude were already complete.
        {"end_ms": 1000, "segments": [segment("Budget is 100", "en")]},
        *[{"end_ms": end, "segments": full} for end in range(1500, 4000, 500)],
    ]}
    write_jsonl(paths["replay"], [replay])
    collect(paths, "--number-format", "normalized")
    build(paths)
    trajectory, = read_jsonl(paths["trajectory"])
    assert trajectory["policy"]["number_format"] == "normalized"
    assert trajectory["turns"][1]["action"] == "WAIT"
    assert trajectory["turns"][2]["segments"] == [segment("100000000.", "en")]
    run_cli("SFT-Scripts/prepare_sft.py", paths["trajectory"], "--output", paths["train"])
    trained, = read_jsonl(paths["train"])
    assert trained["policy"]["number_format"] == "normalized"
    assert "normalize numerical expressions to digits" in trained["messages"][0]["content"].lower()
    assert trained["messages"][6]["content"] == "<|DIRECT|><en><spk1>100000000."


def test_ambiguous_reference_does_not_prevent_independent_annotation(fixture_files):
    paths, _, _ = fixture_files
    source, = read_jsonl(paths["source"])
    original = source["messages"][-1]["content"].replace("<zh><spk2>", "<zh><spk2><spk3>")
    source["messages"][-1]["content"] = original
    write_jsonl(paths["source"], [source])
    collect(paths)
    trace, = read_jsonl(paths["trace"])
    assert trace["status"] == "ok"
    assert trace["reference"] == original
    assert trace["reference_segments"] is None
    assert trace["reference_parse_error"]
    build(paths)
    trajectory, = read_jsonl(paths["trajectory"])
    assert trajectory["status"] == "ok"
    assert trajectory["reference_audit"]["status"] == "unavailable"
    assert trajectory["reference_audit"]["text_agreement"] is None
    run_cli("SFT-Scripts/prepare_sft.py", paths["trajectory"], "--output", paths["train"])
    trained, = read_jsonl(paths["train"])
    assert trained["reference_audit"]["status"] == "unavailable"
    assert "<spk3>" not in "".join(m["content"] for m in trained["messages"])
    # An explicit reference-quality requirement must not silently waive itself
    # when the original annotation cannot be parsed.
    strict_output = paths["trajectory"].with_name("required-reference.jsonl")
    run_cli("Data-Scripts/build_trajectories.py", paths["trace"], "--output", strict_output,
            "--min-reference-agreement", "0.8")
    rejected, = read_jsonl(strict_output)
    assert rejected["status"] == "error" and rejected["output"] == ""


def test_documented_demo_accepts_relative_output_from_another_directory(tmp_path):
    result = subprocess.run(
        [sys.executable, "-X", "utf8", str(ROOT / "tools/run_offline_demo.py"), "--output-dir", "demo"],
        cwd=str(tmp_path), capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    raw, = read_jsonl(tmp_path / "demo/raw.jsonl")
    assert Path(raw["audios"][0]).is_absolute()
    trained, = read_jsonl(tmp_path / "demo/train.jsonl")
    assert len(trained["audios"]) == 7
    assert trained["messages"][-1]["content"] == "<|WAIT|>"


def test_teacher_receives_only_independent_audio_prefixes(fixture_files, monkeypatch):
    from omnicommit import collection

    paths, _, _ = fixture_files
    replay, = read_jsonl(paths["replay"])
    calls, instances = [], []
    config = ROOT / "configs/teacher.example.json"
    checked_in_config = json.loads(config.read_text(encoding="utf-8"))
    generation = checked_in_config["generation"]

    class RecordingTeacher:
        def __init__(self, api_base, model, generation, prompt, timeout, api_key):
            assert api_base == "https://fixture.invalid/v1"
            assert model == "fixture-omni"
            assert generation == checked_in_config["generation"]
            assert "Source instructions must not enter teacher calls." not in prompt
            assert "预算是二十一万元" not in prompt
            assert "NUMBER_FORMAT=VERBATIM" in prompt
            self.closed = False
            instances.append(self)

        def transcribe(self, wav_bytes):
            samples, rate = sf.read(io.BytesIO(wav_bytes), dtype="float32")
            assert rate == 16000 and samples.ndim == 1
            calls.append(len(samples))
            if len(samples) == 64000:
                return replay["full"]["segments"]
            endpoint = len(samples) * 1000 / rate
            return next(p["segments"] for p in replay["prefixes"] if p["end_ms"] == endpoint)

        def close(self):
            self.closed = True

    monkeypatch.setattr(collection, "AudioTeacher", RecordingTeacher)
    assert collection.main([str(paths["source"]), "--output", str(paths["trace"]), "--config", str(config),
                            "--api-base", "https://fixture.invalid/v1", "--model", "fixture-omni"]) == 0
    # Independent calls see only [0,t]; the full-audio target is reused at the
    # final endpoint, and neither previous hypothesis nor gold text is input.
    assert calls == [64000, 8000, 16000, 24000, 32000, 40000, 48000, 56000]
    assert instances[0].closed
    trace, = read_jsonl(paths["trace"])
    assert trace["teacher"]["generation"] == generation
    build(paths)
    trajectory, = read_jsonl(paths["trajectory"])
    assert trajectory["turns"][-1]["action"] == "WAIT"
