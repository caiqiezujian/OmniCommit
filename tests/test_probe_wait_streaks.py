import json

import numpy as np
import pytest

from omnicommit.streaming import WAIT
from Second.second_asr.service_session import SecondStreamOptions, SecondStreamSession
from tools import probe_wait_streaks as probe


def baseline(tmp_path, changed_hash=False):
    actions = ["DIRECT", "WAIT", "WAIT", "WAIT", "WAIT", "DIRECT", "WAIT", "WAIT", "DIRECT", "WAIT", "WAIT"]
    ready = {"type": "ready", "backend": "VLLMBackend", "input": "original.wav", "model": "mock",
             "protocol": "second", "chunk_ms": 500, "automatic_reference": "disabled"}
    turns, packets = [], []
    for t, action in enumerate(actions, 1):
        pcm = np.full(8000, t / 100., dtype="float32")
        packets.append((pcm, t == len(actions)))
        turns.append({"type": "result", "event": "audio", "turn": t, "action": action,
                      "start_seconds": (t-1)/2, "end_seconds": t/2,
                      "raw_model_text": WAIT if action == "WAIT" else "<zh><spk3>好",
                      "pcm_sha256": "changed" if changed_hash and t == 2 else probe.pcm_sha256(pcm)})
    eos = dict(turns[-1], event="end_of_audio", turn=len(actions)+1, start_seconds=len(actions)/2)
    path = tmp_path / "warm.jsonl"
    path.write_text("".join(json.dumps(r)+"\n" for r in [ready, *turns, eos]), encoding="utf8")
    return path, packets


@pytest.mark.parametrize("invalid_reply", [False, True])
def test_paired_fresh_sessions_once_per_streak(tmp_path, monkeypatch, invalid_reply):
    path, packets = baseline(tmp_path)
    original = path.read_bytes()
    loads, contexts = [], []

    class Backend:
        def generate(self, messages, audios):
            traces = [round(float(pcm[0])*100) for pcm in audios]
            contexts.append(traces)
            if invalid_reply and traces == [1]:
                return "missing initial language tag"
            return "<de><spk1>Ja." if len(audios) == 1 else WAIT

    def runtime(header):
        def factory():
            loads.append(1)
            return Backend()
        return "second", SecondStreamOptions(), SecondStreamSession, factory

    monkeypatch.setattr(probe, "engine_from_trace", runtime)
    monkeypatch.setattr(probe, "iter_media_chunks", lambda *_: (p for p in packets))
    out = tmp_path / "cold.jsonl"
    status = probe.main([str(path), "--output", str(out), "--wait-turns", "2", "--max-probes", "2"])
    rows = list(probe.records(out))
    assert path.read_bytes() == original
    assert loads == [1] and status == int(invalid_reply)
    a, b = rows[1:3]
    assert [(r["first_turn"], r["last_turn"]) for r in (a, b)] == [(1, 3), (6, 8)]
    assert [x["action"] for x in a["original"]] == ["DIRECT", "WAIT", "WAIT"]
    assert b["candidate"][0]["raw_model_text"] == "<de><spk1>Ja."
    assert b["all_packet_hashes_verified"] and rows[-1]["failed"] == int(invalid_reply)
    assert contexts[-3:] == [[6], [6, 7], [6, 7, 8]]
    if invalid_reply:
        assert a["error"]["turn"] == 1 and not a["candidate"]
    else:
        assert contexts[:3] == [[1], [1, 2], [1, 2, 3]]
        assert a["elapsed_seconds"] >= 0


def test_probe_checks_packet_provenance_and_skips_model_if_no_windows(tmp_path, monkeypatch):
    path, packets = baseline(tmp_path, changed_hash=True)
    loads = []
    monkeypatch.setattr(probe, "engine_from_trace", lambda _: (
        "second", SecondStreamOptions(), SecondStreamSession, lambda: loads.append(1)))
    monkeypatch.setattr(probe, "iter_media_chunks", lambda *_: (p for p in packets))
    out = tmp_path / "skipped.jsonl"
    assert probe.main([str(path), "--output", str(out), "--start-seconds", "100"]) == 0
    assert not loads and list(probe.records(out))[-1]["probed"] == 0
    with pytest.raises(SystemExit):
        probe.main([str(path), "--output", str(tmp_path/"fail.jsonl")])
    assert "PCM content differs" in list(probe.records(tmp_path/"fail.jsonl"))[-1]["error"]


def test_probe_uses_settings_recorded_by_baseline(monkeypatch):
    header = {"protocol": "second", "chunk_ms": 500, "backend": "VLLMBackend", "model": "/models/m",
              "requested_dtype": "bfloat16", "enable_prefix_caching": False, "mm_processor_cache_gb": .2,
              "max_new_tokens": 48, "max_context_tokens": 2048, "automatic_reference": "disabled",
              "engine_options": {"tensor_parallel_size": 2, "gpu_memory_utilization": .66},
              **vars(SecondStreamOptions())}
    monkeypatch.setattr(probe, "resolve_runtime", lambda args: args)
    args = probe.engine_from_trace(header)
    assert args.tensor_parallel_size == 2 and not args.enable_prefix_caching
    assert args.dtype == "bfloat16" and args.gpu_memory_utilization == .66
    assert args.mm_processor_cache_gb == .2 and args.chunk_seconds == .5 and args.max_new_tokens == 48
