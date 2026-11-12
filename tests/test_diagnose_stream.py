"""Compare the standalone trace with the real web file path using deterministic replies."""
import json
from pathlib import Path
import runpy
import shutil
import wave

import pytest
from omnicommit.streaming import DIRECT, WAIT, ReplayBackend, StreamOptions
from omnicommit.web_service import create_app
from Second.second_asr.service_session import SecondStreamOptions
from Second.second_asr.service_web import create_app as second_app


cli = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools" / "diagnose_stream.py"))["main"]
pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="requires ffmpeg")


def wav(path, frames):
    with wave.open(str(path), "wb") as out:
        out.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        out.writeframes(bytes(frames * 2))
    return path


def trace(tmp_path, audio, replies, ms, protocol, extension):
    scripted = tmp_path / "replies.json"
    scripted.write_text(json.dumps(replies), encoding="utf-8")
    output = tmp_path / ("trace." + extension)
    status = cli([str(audio), "--chunk-ms", str(ms), "--backend", "replay",
                  "--protocol", protocol, "--replies", str(scripted), "--output", str(output)])
    data = output.read_text(encoding="utf-8")
    rows = json.loads(data) if extension == "json" else [json.loads(line) for line in data.splitlines()]
    return status, rows


@pytest.mark.parametrize("protocol,extension", [("first", "jsonl"), ("second", "json")])
def test_cli_file_matches_websocket_turn_for_turn(tmp_path, protocol, extension):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    audio = wav(tmp_path / "test.wav", 7200)
    prefix = DIRECT if protocol == "first" else ""
    replies = [prefix + "<zh><spk1>请联系", WAIT, prefix + "<en><spk2>Sure.", WAIT]
    code, rows = trace(tmp_path, audio, replies, 200, protocol, extension)
    assert code == 0 and rows[0]["type"] == "ready"
    assert rows[0]["protocol"] == protocol and rows[0]["chunk_ms"] == 200
    assert rows[0]["context_policy"] == "rolling" and rows[0]["number_format"] == "normalized"
    if protocol == "first":
        app_factory, options_type = create_app, StreamOptions
    else:
        app_factory, options_type = second_app, SecondStreamOptions
    app = app_factory(lambda: ReplayBackend(replies), options_type(context_policy="rolling", number_format="normalized"))
    with TestClient(app) as client, client.websocket_connect("/stream") as socket:
        upload = client.post("/api/media", content=audio.read_bytes()).json()
        socket.send_json({"type": "start", "sample_rate": 16000, "source_kind": "file",
                          "chunk_ms": 200, "media_id": upload["media_id"]})
        assert socket.receive_json()["type"] == "ready"
        for cli_row in rows[1:]:
            if cli_row["event"] == "end_of_audio":
                socket.send_json({"type": "end"})
            else:
                socket.send_json({"type": "file_next", "release_seconds": .45, "final": True})
            web = socket.receive_json()
            for key in ("type", "event", "turn", "action", "text", "segments", "start_seconds",
                        "end_seconds", "retained_audio_seconds", "evicted_this_turn", "inferred_commit_seconds"):
                assert web[key] == cli_row[key]
    assert [r["raw_model_text"] for r in rows[1:]] == replies
    assert [r["end_seconds"] for r in rows[1:]] == pytest.approx([.2, .4, .45, .45])


def test_late_wait_budget_matches_failed_meeting(tmp_path):
    audio = wav(tmp_path / "meeting.wav", 86 * 16000)
    replies = [DIRECT + "<en><spk1>Hello."] * 57 + [WAIT] * 28
    code, rows = trace(tmp_path, audio, replies, 1000, "first", "jsonl")
    assert code == 1
    assert len(rows) == 1 + 85 + 1
    assert any("trimmed_interval_seconds" in r for r in rows if r["type"] == "result")
    last, err = rows[-2:]
    assert last["action"] == "WAIT" and last["consecutive_audio_waits"] == 28
    assert err["type"] == "error" and err["error_type"] == "ContextBudgetError"
    assert err["heard_seconds"] == err["attempted_start_seconds"] == 85
    assert err["attempted_end_seconds"] == 86 and err["commit_watermark_seconds"] == 55
    assert err["last_direct_seconds"] == 57 and err["history_start_seconds"] == 55
    assert not err["backend_called"] and err["raw_model_text"] is None


def test_malformed_reply_preserved_in_closed_json(tmp_path):
    audio = wav(tmp_path / "bad-reply.wav", 16000)
    code, rows = trace(tmp_path, audio, ["<en>missing speaker"], 1000, "second", "json")
    assert code == 1 and [r["type"] for r in rows] == ["ready", "error"]
    err = rows[-1]
    assert err["raw_model_text"] == "<en>missing speaker" and err["backend_called"]
    assert err["heard_seconds"] == 0 and err["attempted_end_seconds"] == 1
    assert err["error_type"] == "SecondReplyError"
