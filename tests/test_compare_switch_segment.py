import json

from tools.compare_switch_segment import main


WAIT = "<|WAIT|>"


def dump(path, replies, *, terminal=WAIT, step=.5):
    rows = [{"type": "ready", "chunk_ms": round(1000 * step)}]
    for turn, raw in enumerate(replies, 1):
        rows.append({"type": "result", "event": "audio", "turn": turn, "start_seconds": (turn-1)*step,
                     "end_seconds": turn*step, "action": "WAIT" if raw == WAIT else "DIRECT", "raw_model_text": raw})
    if isinstance(terminal, str):
        rows.append({"type": "result", "event": "end_of_audio", "turn": len(replies)+1,
                     "start_seconds": len(replies)*step, "end_seconds": len(replies)*step,
                     "action": "WAIT" if terminal == WAIT else "DIRECT", "raw_model_text": terminal})
    elif terminal is not None:
        rows.append(terminal)
    path.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
    return str(path)


def args(hf, on, off):
    return ["--hf", hf, "--vllm-on", on, "--vllm-off", off, "--start-seconds", "58", "--end-seconds", "59.5"]


def test_rebase_original_times_count_raw_explicit_tagged_text_separately_from_first_direct(tmp_path, capsys):
    ja = "<|DIRECT|><ja><spk1>Ich bin "
    de = "<|DIRECT|><ja><spk1>hier.<de><spk2>Hallo!\nJa."
    fresh = dump(tmp_path / "fresh.jsonl", [WAIT, ja, de])
    full = dump(tmp_path / "full.jsonl", [WAIT] * 116 + [de, WAIT, WAIT])
    assert main(args(fresh, fresh, fresh) + ["--full-hf", full]) == 0
    out = capsys.readouterr().out
    assert '1\t117\t0.5\t58.5\t"<|WAIT|>"' in out
    assert '1\t117\t0.5\t58.5\t"<|DIRECT|><ja><spk1>hier.<de><spk2>Hallo!\\nJa."' in out
    assert "initial observed WAITs=1, first text=59s (+1s), first explicit <de> with text=59.5s (+1.5s)" in out
    assert "initial observed WAITs=0, first text=58.5s (+0.5s), first explicit <de> with text=58.5s" in out


def test_rejected_replies_and_terminal_text_do_not_count_as_audio_turns(tmp_path, capsys):
    raw = "<|DIRECT|><de><spk1>Hallo"
    failed = dump(tmp_path / "failed.jsonl", [WAIT], terminal={
        "type": "error", "stage": "audio", "error": "unknown speaker", "attempted_end_seconds": 1,
        "raw_model_text": raw})
    eos = dump(tmp_path / "eos.jsonl", [WAIT] * 3, terminal=raw)
    incomplete = dump(tmp_path / "truncated.jsonl", [WAIT, WAIT], terminal=None)
    assert main(args(eos, failed, incomplete)) == 1
    out = capsys.readouterr().out
    assert f'terminal: ERROR audio: unknown speaker (original 59s) raw="{raw}"' in out
    assert f'terminal: EOS at original 59.5s: "{raw}"' in out
    assert "incomplete: no EOS or error record" in out and "(no result)" in out
    assert out.count("first explicit <de> with text=unobserved") == 3


def test_failure_before_ready_preserves_the_other_backends(tmp_path, capsys):
    failed = tmp_path / "failed.jsonl"
    failed.write_text(json.dumps({"type": "error", "stage": "load_model", "error": "out of memory"}))
    good = dump(tmp_path / "good.jsonl", ["<de><spk1>Hallo", " ihr", " alle."])
    assert main(args(str(failed), good, str(tmp_path / "not-created.jsonl"))) == 1
    out = capsys.readouterr().out
    assert "ERROR load_model: out of memory" in out and "not-created.jsonl" in out
    assert "first explicit <de> with text=58.5s" in out


def test_reject_wrong_fresh_duration_and_keep_early_results(tmp_path, capsys):
    overlong = dump(tmp_path / "long.jsonl", [WAIT] * 4)
    early = dump(tmp_path / "short.jsonl", [WAIT] * 2)
    assert main(args(overlong, early, early)) == 1
    out = capsys.readouterr().out
    assert "exceeds the selected interval" in out
    assert "ended before requested interval was covered" in out
