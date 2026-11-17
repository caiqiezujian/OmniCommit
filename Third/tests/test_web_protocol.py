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
