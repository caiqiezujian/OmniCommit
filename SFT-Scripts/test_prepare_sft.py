"""Validate real audio preparation and failure isolation without API/GPU calls."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_sft import prepare_row, main, sort_tokens
from check_sft import check_row, check_content, read_tokens
from audit_sft import main as audit_main
from omnicommit_audio import decode_chunk_ref


def trajectory(tmp_path):
    audio = tmp_path / "speech.wav"
    sf.write(audio, np.zeros(24000, dtype=np.float32), 16000, subtype="FLOAT")
    return {"schema_version": 1, "id": "demo", "audio": str(audio), "duration_ms": 1500,
            "turns": [
                {"start_ms": 0, "end_ms": 500, "action": "DIRECT", "segments": [
                    {"language": "zh", "speaker": "spk1", "text": "我同意。"}]},
                {"start_ms": 500, "end_ms": 1000, "action": "DIRECT", "segments": [
                    {"language": "en", "speaker": "spk1", "text": "Any questions?"},
                    {"language": "zh", "speaker": "spk2", "text": "没有。"}]},
                {"start_ms": 1000, "end_ms": 1500, "action": "WAIT", "segments": []}]}


def test_interleaved_chunks_language_changes_and_final_wait(tmp_path):
    row, tokens = prepare_row(trajectory(tmp_path))
    assert [m["role"] for m in row["messages"]] == ["system"] + ["user", "assistant"] * 3
    assert row["messages"][2]["content"] == "<|DIRECT|><zh><spk1>我同意。"
    assert row["messages"][4]["content"] == "<|DIRECT|><en><spk1>Any questions?\n<zh><spk2>没有。"
    assert row["messages"][6]["content"] == "<|WAIT|>"
    assert [(decode_chunk_ref(a)["start"], decode_chunk_ref(a)["end"]) for a in row["audios"]] == [(0, .5), (.5, 1), (1, 1.5)]
    assert sort_tokens(tokens) == ["<|WAIT|>", "<|DIRECT|>", "<en>", "<zh>", "<spk1>", "<spk2>"]
    assert check_row(row, token_inventory=tokens)["wait_turns"] == 1
    assert "verbatim" in row["messages"][0]["content"]
    assert "Normalize numerical" not in row["messages"][0]["content"]


def test_numeric_policy_changes_prompt_but_never_rewrites_supervision(tmp_path):
    row = trajectory(tmp_path)
    row["policy"] = {"number_format": "normalized"}
    row["turns"][0]["segments"][0]["text"] = "预算是210000元。"
    prepared, _ = prepare_row(row)
    assert "Normalize numerical" in prepared["messages"][0]["content"]
    assert prepared["messages"][2]["content"].endswith("预算是210000元。")
    row["policy"]["number_format"] = "guess"
    with pytest.raises(ValueError, match="number_format"):
        prepare_row(row)


@pytest.mark.parametrize("mutation", [
    lambda r: r["turns"][1].update(start_ms=490),
    lambda r: r["turns"][1].update(start_ms=510),
    lambda r: r["turns"][2].update(end_ms=1600),
    lambda r: r["turns"][2].update(end_ms=1400),
    lambda r: r["turns"][0].update(action="FLEX"),
    lambda r: r["turns"][2].update(segments=[{"text": "bad"}]),
    lambda r: r["turns"][0].update(segments=[]),
    lambda r: r["turns"][0]["segments"][0].update(speaker="spk0"),
    lambda r: r["turns"][0]["segments"][0].update(language="中文"),
    lambda r: r["turns"][0]["segments"][0].update(text="<spk2>猜测"),
    lambda r: r.update(duration_ms=float("nan")),
    lambda r: r.update(duration_ms=True),
    lambda r: r.update(status="error"),
])
def test_rejects_invalid_trajectory(tmp_path, mutation):
    row = trajectory(tmp_path)
    mutation(row)
    with pytest.raises(ValueError):
        prepare_row(row)


def test_no_silent_all_wait_training(tmp_path):
    row = trajectory(tmp_path)
    for turn in row["turns"]:
        turn.update(action="WAIT", segments=[])
    with pytest.raises(ValueError, match="All-WAIT"):
        prepare_row(row)


def test_failed_rows_do_not_stop_preparation_and_inventory_excludes_failed_rows(tmp_path):
    good = trajectory(tmp_path)
    bad = copy.deepcopy(good)
    bad["turns"][0]["segments"][0]["language"] = "ja"
    bad["turns"][2]["action"] = "FLEX"
    nonfinite = copy.deepcopy(good)
    nonfinite["provenance"] = {"score": float("nan")}
    source, output = tmp_path / "source.jsonl", tmp_path / "train.jsonl"
    source.write_text(json.dumps(bad) + "\nnot json\n" + json.dumps(nonfinite) + "\n" + json.dumps(good), encoding="utf-8")
    assert main([str(source), "--output", str(output)]) == 0
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    rejects = Path(str(output) + ".rejected.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rejects) == 3
    tokens = read_tokens(Path(str(output) + ".tokens.txt"))
    assert "<ja>" not in tokens
    assert check_row(rows[0], token_inventory=tokens)["direct_turns"] == 2
    assert audit_main([str(output), "--decode", "--report", str(tmp_path / "audit.json")]) == 0
    with pytest.raises(SystemExit):
        main([str(source), "--output", str(output)])


def test_tiny_tail_is_padded_without_reusing_future_audio(tmp_path):
    row = trajectory(tmp_path)
    audio = Path(row["audio"])
    sf.write(audio, np.zeros(16080, dtype=np.float32), 16000, subtype="FLOAT")
    row["duration_ms"] = 1005
    row["turns"][2]["end_ms"] = 1005
    prepared, tokens = prepare_row(row)
    last = decode_chunk_ref(prepared["audios"][-1])
    assert last["start"] == 1 and last["end"] == 1.005
    assert last["input_seconds"] == .5
    assert check_row(prepared, token_inventory=tokens)["sample_counts"][-1] == 8000


@pytest.mark.parametrize("content", ["<|FLEX|><zh><spk1>你好", "<|DIRECT|>",
                                    "<|DIRECT|><zh><spk1><spk2>你好", "<|WAIT|>你好",
                                    "<|DIRECT|><zh><spk1>你好\n", "<|DIRECT|><zh><spk1> "])
def test_invalid_assistant_outputs_are_not_trainable(content):
    with pytest.raises(ValueError):
        check_content(content)


def test_missing_token_inventory_and_cumulative_audio_ranges_rejected(tmp_path):
    prepared, tokens = prepare_row(trajectory(tmp_path))
    with pytest.raises(ValueError, match="Missing special"):
        check_row(prepared, token_inventory=set())
    prepared["audios"][1] = prepared["audios"][0]
    with pytest.raises(ValueError, match="contiguous"):
        check_row(prepared, token_inventory=tokens)


def test_language_codes_with_speaker_prefix_remain_language_tokens(tmp_path):
    row = trajectory(tmp_path)
    row["turns"][0]["segments"][0]["language"] = "spk"
    _, tokens = prepare_row(row)
    assert sort_tokens(tokens) == ["<|WAIT|>", "<|DIRECT|>", "<en>", "<spk>", "<zh>", "<spk1>", "<spk2>"]
