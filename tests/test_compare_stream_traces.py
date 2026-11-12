import json
from pathlib import Path
import runpy

import pytest


cli = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools" / "compare_stream_traces.py"))["main"]


def dump(path, word=" Hallo.", *, failure=None, incomplete=False, step=0.5, protocol="first", cached=0, dtype=None):
    rows = [
        {"type": "ready", "protocol": protocol, "input": "/audio.wav", "model": "/model", "chunk_ms": int(step * 1000)},
        {"type": "result", "event": "audio", "turn": 1, "start_seconds": 0, "end_seconds": step,
         "action": "WAIT", "raw_model_text": "<|WAIT|>", "num_cached_tokens": cached},
    ]
    if dtype is not None:
        rows[0]["requested_dtype"] = dtype
    if failure:
        rows.append({"type": "error", "stage": "audio", "error_type": "ContextBudgetError", "error": failure,
                     "heard_seconds": step, "attempted_end_seconds": 2 * step, "backend_called": False})
    elif not incomplete:
        rows.extend([{"type": "result", "event": "audio", "turn": 2, "start_seconds": step, "end_seconds": 2 * step,
                      "action": "DIRECT", "raw_model_text": "<|DIRECT|><de><spk1>" + word},
                     {"type": "result", "event": "end_of_audio", "turn": 3,
                      "start_seconds": 2 * step, "end_seconds": 2 * step, "action": "WAIT", "raw_model_text": "<|WAIT|>"}])
    path.write_text(json.dumps(rows) if path.suffix == ".json" else "\n".join(map(json.dumps, rows)), encoding="utf-8")
    return str(path)


def test_first_pairwise_difference_preserves_whitespace_and_ignores_cache_statistics(tmp_path, capsys):
    paths = [dump(tmp_path / "hf.jsonl"), dump(tmp_path / "on.jsonl", word="Hallo."),
             dump(tmp_path / "off.json", cached=144)]
    assert cli(paths) == 0
    out = capsys.readouterr().out
    assert out.count("First difference at turn 2: model output differs") == 2
    assert "audio [0.5, 1.0] s" in out and "<de><spk1> Hallo." in out
    assert "Matched all 3 turns through end_of_audio" in out


def test_identical_early_errors_distinguished_from_success(tmp_path, capsys):
    paths = [dump(tmp_path / "hf.jsonl", failure="budget"), dump(tmp_path / "on.jsonl"),
             dump(tmp_path / "off.jsonl", failure="budget")]
    assert cli(paths) == 0
    out = capsys.readouterr().out
    assert "both stopped at turn 2" in out and "budget" in out
    assert out.count("First difference at turn 2: execution outcome differs") == 2


@pytest.mark.parametrize(("variants", "message"), [({"incomplete": True}, "incomplete trace"),
                                                 ({"protocol": "second"}, "ready.protocol differs"),
                                                 ({"step": .2}, "ready.chunk_ms differs")])
def test_incomplete_or_incompatible_trace(tmp_path, capsys, variants, message):
    paths = [dump(tmp_path / "one.jsonl"), dump(tmp_path / "two.jsonl", **variants), dump(tmp_path / "three.jsonl")]
    with pytest.raises(SystemExit) as exc:
        cli(paths)
    assert exc.value.code == 2 and message in capsys.readouterr().err


def test_compares_requested_dtype_when_present_in_both_files(tmp_path, capsys):
    paths = [dump(tmp_path / "old.jsonl"), dump(tmp_path / "on.jsonl", dtype="bfloat16"),
             dump(tmp_path / "off.jsonl", dtype="bfloat16")]
    assert cli(paths) == 0
    dump(tmp_path / "off.jsonl", dtype="float16")
    with pytest.raises(SystemExit) as exc:
        cli(paths)
    assert exc.value.code == 2 and "requested_dtype differs" in capsys.readouterr().err
