"""Local collection-to-SFT integration with deterministic CPU teacher doubles.

These tests exercise real waveform reads, batching, durable resume, switch
gates and the existing SFT audio decoder. They do not measure model quality.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from omnicommit.prefix_collection import parser, run
from omnicommit.prefix_labels import build_trajectory


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sft_modules(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "SFT-Scripts"))
    modules = []
    for name in ("prepare_sft", "check_sft"):
        spec = importlib.util.spec_from_file_location(name, ROOT / "SFT-Scripts" / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


def source_record(audio, text="<en><spk1>Hello world.", timestamps=None):
    result = {"id": "test-record", "audios": [str(audio)], "messages": [
        {"role": "system", "content": "REFERENCE_SECRET_SYSTEM_DO_NOT_LEAK"},
        {"role": "user", "content": "<audio>**Entity Reference:** [REFERENCE_SECRET_ENTITY]"},
        {"role": "assistant", "content": text}]}
    if timestamps is not None:
        result["timestamps"] = timestamps
    return result


def make_job(tmp_path, *, rate=16000, seconds=2.0, text=None, timestamps=None, stereo=False):
    path = tmp_path / "audio.wav"
    count = round(rate * seconds)
    t = np.arange(count, dtype=np.float32)
    wave = np.sin(t * .012) * .15
    if stereo:
        wave = np.stack([wave, wave * .5], axis=1)
    sf.write(path, wave, rate, subtype="FLOAT")
    row = source_record(path, text or "<en><spk1>Hello world.", timestamps)
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    model = tmp_path / "offline-model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"dummy-test"}', encoding="utf-8")
    args = parser().parse_args([str(source), "--output", str(tmp_path / "trace.jsonl"),
                               "--model", str(model), "--batch-size", "2"])
    return args, row


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fake_teacher(script, *, interrupt_on=None):
    """Capture the complete backend boundary: only waves/languages, no answers."""
    calls, loads = [], []

    class Teacher:
        def __init__(self, model_path, **kwargs):
            loads.append((model_path, kwargs))

        def transcribe_batch(self, waves, languages):
            assert len(waves) == len(languages)
            assert all(isinstance(wave, np.ndarray) and wave.ndim == 1 for wave in waves)
            calls.append([(language, wave.copy()) for language, wave in zip(languages, waves)])
            if interrupt_on is not None and len(calls) == interrupt_on:
                raise KeyboardInterrupt("intentional interruption between durable batches")
            return [{"text": script(language, len(wave)), "token_ids": [], "prompt_tokens": 10,
                     "error": None} for language, wave in zip(languages, waves)]

    return Teacher, calls, loads


def basic_script(language, count):
    assert language == "en"
    return "" if count <= 8000 else "Hello" if count <= 16000 else "Hello world."


def test_batched_teacher_to_existing_prepare_and_decode_audit(tmp_path, sft_modules):
    args, row = make_job(tmp_path)
    teacher, calls, loads = fake_teacher(basic_script)
    summary = run(args, teacher_factory=teacher)
    assert summary["ok"] == 1 and summary["queries_inferred"] == 4
    assert summary["real_hf_inference_executed"] is False
    assert len(loads) == 1 and [len(batch) for batch in calls] == [2, 2]
    # The backend interface never receives source prompt, reference answer, or
    # Entity Reference. This does not substitute for the HF backend unit tests.
    assert "REFERENCE_SECRET" not in repr(loads)
    trace, = read_rows(args.output)
    assert trace["teacher"]["reference_in_prompt"] is False
    trajectory = build_trajectory(trace)
    assert trajectory["turns"][0]["action"] == "WAIT"
    assert trajectory["turns"][1]["segments"][0]["text"] == "Hello "
    prepare, check = sft_modules
    prepared, tokens = prepare.prepare_row(trajectory)
    compact = {key: prepared[key] for key in ("audios", "messages")}
    audit = check.check_row(compact, decode=True, token_inventory=tokens)
    assert audit["audio_turns"] == 4 and audit["eos_turns"] == 1
    assert audit["sample_counts"] == [8000] * 4
    targets = [m["content"] for m in compact["messages"] if m["role"] == "assistant"]
    assert targets[0] == "<|WAIT|>" and targets[-1] == "<|WAIT|>"
    assert targets[1] == "<|DIRECT|><en><spk1>Hello "


def test_complete_resume_does_not_load_teacher(tmp_path):
    args, _ = make_job(tmp_path)
    teacher, _, _ = fake_teacher(basic_script)
    run(args, teacher_factory=teacher)
    original = args.output.read_bytes()
    args.resume = True

    def forbidden_factory(*args, **kwargs):
        raise AssertionError("A fully resumed job must not load the HF model")

    counts = run(args, teacher_factory=forbidden_factory)
    assert counts["resumed"] == 1 and counts["queries_inferred"] == 0
    assert args.output.read_bytes() == original


def test_resume_rejects_changed_model_configuration(tmp_path):
    args, _ = make_job(tmp_path)
    teacher, _, _ = fake_teacher(basic_script)
    run(args, teacher_factory=teacher)
    (args.model / "config.json").write_text('{"model_type":"different-model"}', encoding="utf-8")
    args.resume = True
    with pytest.raises(ValueError, match="configuration differs"):
        run(args, teacher_factory=teacher)


def test_resume_reuses_successful_queries_from_unfinished_record(tmp_path):
    args, _ = make_job(tmp_path)
    interrupted, first_calls, _ = fake_teacher(basic_script, interrupt_on=2)
    with pytest.raises(KeyboardInterrupt):
        run(args, teacher_factory=interrupted)
    assert len(first_calls) == 2
    assert not args.output.exists() or not args.output.read_bytes()
    args.resume = True
    teacher, next_calls, _ = fake_teacher(basic_script)
    counts = run(args, teacher_factory=teacher)
    assert counts["query_cache_hits"] == 2 and counts["queries_inferred"] == 2
    assert [len(wave) for _, wave in next_calls[0]] == [24000, 32000]
    trace, = read_rows(args.output)
    assert trace["status"] == "ok" and len(trace["queries"]) == 4


def test_malformed_json_row_does_not_stop_following_record(tmp_path):
    args, row = make_job(tmp_path)
    args.input.write_text('{"broken":\n' + json.dumps(row) + "\n", encoding="utf-8")
    teacher, _, loads = fake_teacher(basic_script)
    counts = run(args, teacher_factory=teacher)
    assert counts["failed"] == 1 and counts["ok"] == 1 and len(loads) == 1
    bad, good = read_rows(args.output)
    assert bad["status"] == "error" and bad["output"] == ""
    assert good["status"] == "ok" and good["source_line"] == 2
    assert build_trajectory(good)["status"] == "ok"


def test_invalid_unicode_row_does_not_stop_following_record(tmp_path):
    args, row = make_job(tmp_path)
    malformed = {"id": "bad\ud800", "audios": row["audios"], "messages": row["messages"]}
    args.input.write_text(json.dumps(malformed, ensure_ascii=True) + "\n" + json.dumps(row) + "\n",
                          encoding="utf-8")
    teacher, _, _ = fake_teacher(basic_script)
    counts = run(args, teacher_factory=teacher)
    assert counts["failed"] == 1 and counts["ok"] == 1
    assert read_rows(args.output)[0]["status"] == "error"


def test_fatal_cuda_error_is_not_converted_to_training_wait(tmp_path):
    args, _ = make_job(tmp_path)

    class FatalTeacher:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe_batch(self, waves, languages):
            raise RuntimeError("CUDA error: device-side assert triggered")

    with pytest.raises(RuntimeError, match="device-side assert"):
        run(args, teacher_factory=FatalTeacher)
    assert not args.output.exists() or not args.output.read_bytes()


def test_resumed_failures_are_counted_separately(tmp_path):
    args, _ = make_job(tmp_path)
    args.input.write_text('{"broken":\n', encoding="utf-8")
    run(args)
    args.resume = True
    counts = run(args)
    assert counts["resumed_failed"] == 1 and counts["resumed_ok"] == 0


def test_44100_stereo_cropped_before_resampling_and_decodes_final_short_chunk(tmp_path, sft_modules):
    args, _ = make_job(tmp_path, rate=44100, seconds=2.1, stereo=True)
    teacher, calls, _ = fake_teacher(basic_script)
    run(args, teacher_factory=teacher)
    from scipy.signal import resample_poly
    pcm, rate = sf.read(str(tmp_path / "audio.wav"), dtype="float32", always_2d=True)
    first = calls[0][0][1]
    expected = resample_poly(pcm[:22050].mean(axis=1, dtype=np.float32), 160, 441)
    np.testing.assert_array_equal(first, expected)
    trace, = read_rows(args.output)
    assert trace["queries"][-1]["end_sample"] == 92610
    prepare, check = sft_modules
    prepared, inventory = prepare.prepare_row(build_trajectory(trace))
    audit = check.check_row(prepared, decode=True, token_inventory=inventory)
    assert audit["audio_turns"] == 5
    assert audit["sample_counts"] == [8000] * 5  # .1s real tail is padded only at materialization.


def test_language_switch_restarts_teacher_but_keeps_mixed_student_chunks(tmp_path, sft_modules):
    args, _ = make_job(tmp_path, seconds=2.5, text="<zh><spk1>甲乙<en><spk2>Please check.",
                       timestamps=[{"time_ms": 1300, "changed": ["language", "speaker"]}])

    def script(language, count):
        if language == "zh":
            return "甲" if count <= 16000 else "甲乙"
        return "Please" if count < 11200 else "Please check."

    teacher, calls, _ = fake_teacher(script)
    run(args, teacher_factory=teacher)
    en_lengths = sorted(len(wave) for batch in calls for language, wave in batch if language == "en")
    assert en_lengths == [3200, 11200, 19200]  # .2, .7, 1.2 seconds after 1.3s switch.
    trace, = read_rows(args.output)
    en_queries = [q for q in trace["queries"] if q["run_index"] == 1]
    assert {q["start_sample"] for q in en_queries} == {20800}
    trajectory = build_trajectory(trace)
    crossing = trajectory["turns"][2]
    assert (crossing["start_ms"], crossing["end_ms"]) == (1000, 1500)
    assert crossing["action"] == "DIRECT" and crossing["provenance"]["switch_gate"]
    assert crossing["segments"] == [{"language": "zh", "speaker": "spk1", "text": "乙"}]
    after = trajectory["turns"][3]
    assert after["segments"][0]["language"] == "en" and after["segments"][0]["speaker"] == "spk2"
    prepare, check = sft_modules
    prepared, inventory = prepare.prepare_row(trajectory)
    check.check_row(prepared, decode=True, token_inventory=inventory)
    assert prepared["audio_segments"][2]["start"] == 1.0
    assert prepared["audio_segments"][2]["end"] == 1.5


def test_same_language_speaker_switch_retains_teacher_context_and_defers_new_speaker(tmp_path):
    args, _ = make_job(tmp_path, seconds=2.5,
                       text="<en><spk1>Hello there.<en><spk2> Good day.",
                       timestamps=[{"time_ms": 1300, "changed": ["speaker"]}])

    def script(language, count):
        assert language == "en"
        if count <= 16000:
            return "Hello"
        if count <= 20800:
            return "Hello there."
        if count <= 32000:
            return "Hello there. Good"
        return "Hello there. Good day."

    teacher, calls, _ = fake_teacher(script)
    run(args, teacher_factory=teacher)
    trace, = read_rows(args.output)
    assert len(trace["source"]["language_runs"]) == 1
    assert {q["start_sample"] for q in trace["queries"]} == {0}
    assert sorted(len(wave) for batch in calls for _, wave in batch) == [8000, 16000, 20800, 24000, 32000, 40000]
    trajectory = build_trajectory(trace)
    cross = trajectory["turns"][2]
    assert cross["action"] == "DIRECT" and cross["segments"] == [
        {"language": "en", "speaker": "spk1", "text": "there."}]
    assert trajectory["turns"][3]["segments"] == [
        {"language": "en", "speaker": "spk2", "text": " Good "}]
    assert "".join(s["text"] for turn in trajectory["turns"] for s in turn["segments"]) == "Hello there. Good day."


def test_builder_cli_rejects_failed_trace_and_exports_compact_sft_sidecars(tmp_path, sft_modules):
    args, row = make_job(tmp_path)
    args.input.write_text('not-json\n' + json.dumps(row) + "\n", encoding="utf-8")
    teacher, _, _ = fake_teacher(basic_script)
    run(args, teacher_factory=teacher)
    train = tmp_path / "train.500ms.jsonl"
    process = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "Data-Scripts" / "build_prefix_sft.py"),
                              str(args.output), "--output", str(train)], cwd=ROOT,
                             capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr
    compact, = read_rows(train)
    assert set(compact) == {"audios", "messages"}
    assert "REFERENCE_SECRET" not in repr(compact)
    rejected, = read_rows(Path(str(train) + ".rejected.jsonl"))
    assert rejected["line"] == 1
    assert Path(str(train) + ".review.jsonl").exists()
    trajectory, = read_rows(Path(str(train) + ".trajectory.jsonl"))
    assert trajectory["status"] == "ok"
    inventory = Path(str(train) + ".tokens.txt").read_text(encoding="utf-8").splitlines()
    assert inventory == ["<|WAIT|>", "<|DIRECT|>", "<en>", "<spk1>"]
    report = json.loads(Path(str(train) + ".summary.json").read_text(encoding="utf-8"))
    assert report["kept_rows"] == report["rejected_rows"] == 1
    assert report["language_rows"] == {"en": 1}
    assert report["full_structural_check"] is True and report["acoustic_quality_verified"] is False
    assert sft_modules[1].check_row(compact, decode=True, token_inventory=set(inventory))["decoded"] is True
