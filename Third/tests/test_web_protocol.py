"""Verify server startup negotiation and the exact wire transcript on recovery."""
from pathlib import Path
import json
import runpy
import sys

import numpy as np
import pytest

THIRD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THIRD))
from third_asr.runtime.streaming import ReplayBackend
from third_asr.protocol import DIRECT, WAIT, get_protocol
from third_asr.service_backend import read_export_protocol
from third_asr.service_session import ThirdStreamOptions
from third_asr.service_web import create_app


@pytest.mark.parametrize("mode", ["explicit", "plain"])
def test_cli_selects_checkpoint_protocol_before_model_load(tmp_path, monkeypatch, mode):
    inventory = tmp_path / "tokens.txt"
    inventory.write_text("\n".join(get_protocol(mode).tokens), encoding="utf8")
    tokens, metadata = read_export_protocol(inventory)
    (tmp_path / "omnicommit_tokens.json").write_text(json.dumps({**metadata, "tokens": tokens}))
    main = runpy.run_path(str(THIRD / "tools/serve_web.py"))["main"]
    captured = []
    monkeypatch.setitem(main.__globals__, "serve_web", lambda p, args, factory, opts, app: captured.append(opts))
    monkeypatch.setattr("third_asr.recovery.check_recovery_dependencies", lambda: None)
    main(["--model", str(tmp_path)])
    assert captured[0].protocol == mode
    assert captured[0].enable_recovery == (mode == "explicit")
    main(["--model", str(tmp_path), "--no-recovery"])
    assert not captured[1].enable_recovery
    other = "plain" if mode == "explicit" else "explicit"
    with pytest.raises(SystemExit):
        main(["--model", str(tmp_path), "--protocol", other])


def test_websocket_silence_then_voiced_wait_recovery():
    pytest.importorskip("webrtcvad")
    from fastapi.testclient import TestClient

    rest = bytes(12800 * 2)
    t = np.arange(12800) / 16000
    tone = (.20 * np.sin(2*np.pi*155*t + .8*np.sin(2*np.pi*7*t))
            + .07 * np.sin(2*np.pi*680*t) + .04 * np.sin(2*np.pi*1100*t))
    tone = np.rint(tone * 32768).astype("<i2").tobytes()
    # Script primary replies for each packet, the shadow candidate, then EOS.
    backend = ReplayBackend([DIRECT + "你好。", WAIT, WAIT, WAIT, WAIT, DIRECT + " Guten Morgen.", WAIT])
    app = create_app(lambda: backend, ThirdStreamOptions(protocol="explicit"))
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["action_protocol"] == "explicit" and health["gap_recovery"]
        with client.websocket_connect("/stream") as ws:
            ws.send_json({"type": "start", "sample_rate": 16000})
            assert ws.receive_json()["action_protocol"] == "explicit"
            events = []
            for packet in [rest, rest, tone, tone, tone]:
                ws.send_bytes(packet)
                events.append(ws.receive_json())
            assert all("recovery" not in e for e in events[:-1])
            recovered = events[-1]
            assert recovered["primary_text"] == WAIT and recovered["text"] == DIRECT + " Guten Morgen."
            assert recovered["recovery"]["status"] == "accepted"
            assert "".join(e["display_text"] for e in events) == "你好。 Guten Morgen."
            ws.send_json({"type": "end"})
            final = ws.receive_json()
            assert final["text"] == WAIT and final["display_text"] == "" and "recovery" not in final


def test_vllm_reserves_slots_for_page_choices(monkeypatch):
    main = runpy.run_path(str(THIRD / "tools/serve_web.py"))["main"]
    monkeypatch.setitem(main.__globals__, "detect_model_protocol", lambda model: "plain")
    monkeypatch.setattr("third_asr.runtime.vllm_backend.VLLMBackend", lambda model, **kw: kw)
    calls = []
    def serve(parser, args, backend_factory, options, app_factory):
        calls.append((options.window_seconds, options.discard_seconds, args.max_window_seconds,
                      backend_factory()["max_audio_turns"]))
    monkeypatch.setitem(main.__globals__, "serve_web", serve)
    main(["--model", "checkpoint"])
    main(["--model", "checkpoint", "--window-seconds", "75", "--discard-seconds", "45",
          "--max-window-seconds", "150", "--chunk-ms", "1000"])
    assert calls == [(60, 50, 120, 150), (75, 45, 150, 150)]
    with pytest.raises(SystemExit):
        main(["--model", "checkpoint", "--max-window-seconds", "20"])


def test_sessions_negotiate_independent_windows_and_apply_them_during_wait():
    from fastapi.testclient import TestClient
    app = create_app(ReplayBackend, ThirdStreamOptions(chunk_seconds=1), max_sessions=2)
    with TestClient(app) as client, client.websocket_connect("/stream") as mic, client.websocket_connect("/stream") as system:
        defaults = client.get("/health")
        assert defaults.headers["cache-control"] == "no-store"
        assert (defaults.json()["window_seconds"], defaults.json()["discard_seconds"]) == (60, 50)
        for ws, kind, window, discard in ((mic, "microphone", 3, 2), (system, "system", 2, 1)):
            ws.send_json(dict(type="start", sample_rate=16000, source_kind=kind,
                              window_seconds=window, discard_seconds=discard))
            ready = ws.receive_json()
            assert (ready["window_seconds"], ready["discard_seconds"]) == (window, discard)
        for starts in [(0, 0), (0, 0), (0, 1), (2, 2)]:
            for ws, start in zip((mic, system), starts):
                ws.send_bytes(bytes(32000))
                row = ws.receive_json()
                assert row["window_start_seconds"] == start and row["action"] == "WAIT"
        system.send_json({"type": "end", "window_seconds": 3})
        assert system.receive_json()["type"] == "error"
        mic.send_json({"type": "end"})
        assert mic.receive_json()["event"] == "end_of_audio"


@pytest.mark.parametrize("window,discard", [(0, 1), (2, 3), (False, 1), (3, "1"), (20, 10), (.2, .1)])
def test_reject_invalid_or_over_capacity_window_before_decoding(window, discard):
    from fastapi.testclient import TestClient
    backend = ReplayBackend([])
    backend.max_audio_turns = 4
    opts = ThirdStreamOptions(chunk_seconds=1, window_seconds=4, discard_seconds=3)
    with TestClient(create_app(lambda: backend, opts)) as client, client.websocket_connect("/stream") as ws:
        assert client.get("/health").json()["max_window_seconds"] == 4
        ws.send_json(dict(type="start", sample_rate=16000, window_seconds=window, discard_seconds=discard))
        error = ws.receive_json()
        assert error["type"] == "error" and error["heard_seconds"] == 0
