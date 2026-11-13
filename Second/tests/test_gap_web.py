"""Exercise recovery through the same WebSocket used by files and the microphone."""
from pathlib import Path
import sys
import threading

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omnicommit.streaming import ReplayBackend
from second_asr.protocol import WAIT
from second_asr.service_session import SecondStreamOptions
from second_asr.service_web import create_app


VOICE = (6000 * np.sin(2 * np.pi * 220 * np.arange(12800) / 16000)).astype("<i2").tobytes()


class GapBackend:
    def __init__(self, words, hold=False, fail=False, speech_ranges=None):
        self.words = words
        self.fail = fail
        self.ranges = speech_ranges
        self.decodes = 0
        self.release = threading.Event()
        if not hold:
            self.release.set()

    def speech_ranges(self, pcm):
        return [(0, len(pcm))] if self.ranges is None else self.ranges

    def transcribe(self, pcm):
        self.decodes += 1
        assert self.release.wait(timeout=10)
        if self.fail:
            raise RuntimeError("offline decode failed")
        return [dict(text=t, start=s, end=e, probability=.95, language_probability=.9, language="de")
                for t, s, e in self.words]


class Primary(ReplayBackend):
    def generate(self, messages, audios):
        self.prompt = messages[0]["content"]
        return super().generate(messages, audios)


def app(backend, offline):
    return create_app(lambda: backend, SecondStreamOptions(chunk_seconds=.8),
                      gap_backend_factory=lambda: offline)


def start(ws):
    ws.send_json({"type": "start", "sample_rate": 16000})
    assert ws.receive_json()["gap_recovery"]


def packet(ws):
    ws.send_bytes(VOICE)
    return ws.receive_json()


def test_idle_delivery_and_delayed_primary_replace():
    main = Primary(["<ja><spk1>はい。", WAIT, WAIT, WAIT, "<de><spk2>Guten Morgen.", WAIT])
    offline = GapBackend([("はい。", .1, .7), ("Guten", .8, 1.2), (" Morgen.", 1.3, 1.9)], hold=True)
    with TestClient(app(main, offline)) as client, client.websocket_connect("/stream") as ws:
        start(ws)
        assert client.get("/health").json()["gap_recovery"]
        assert packet(ws)["new_segments"][0]["language"] == "ja"
        assert packet(ws)["text"] == WAIT
        pending = packet(ws)
        assert pending["text"] == WAIT and pending["gap_events"][0]["status"] == "pending"
        offline.release.set()
        update = ws.receive_json()  # Browser may be paused with no next audio packet.
        assert update["type"] == "gap_update"
        recovered, = update["new_segments"]
        assert recovered["origin"] == "recovery" and recovered["model_speaker"] is None
        assert recovered["language"] is None  # even a backend that supplies an estimate cannot label a gap
        assert recovered["text"].strip() == "Guten Morgen."
        assert packet(ws)["text"] == WAIT
        assert "Guten Morgen." in main.prompt
        late = packet(ws)
        assert late["removed_segment_ids"] == [recovered["segment_id"]]
        assert late["new_segments"][0]["display_speaker"] == "spk2"
        assert late["new_segments"][0]["language"] == "de"
        ws.send_json({"type": "end"})
        final = ws.receive_json()
        while final["type"] == "gap_update":
            final = ws.receive_json()
        assert final["event"] == "end_of_audio"


@pytest.mark.parametrize("failed", [False, True])
def test_end_drains_short_voiced_wait_even_when_backend_fails(failed):
    main = Primary(["<ja><spk1>はい。", WAIT, WAIT])
    offline = GapBackend([("Danke.", .9, 1.4)], fail=failed)
    with TestClient(app(main, offline)) as client, client.websocket_connect("/stream") as ws:
        start(ws)
        packet(ws)
        assert packet(ws)["action"] == "WAIT"
        ws.send_json({"type": "end"})
        result = ws.receive_json()
        assert result["type"] == "result" and result["text"] == WAIT
        assert result["event"] == "end_of_audio"
        if failed:
            assert not result["new_segments"] and result["gap_events"][-1]["status"] == "error"
        else:
            assert result["new_segments"][0]["text"] == "Danke."
            assert result["gap_events"][-1]["status"] == "recovered"


def test_noise_during_wait_cannot_trigger_decode_via_speech_in_direct_lead():
    main = Primary(["<ja><spk1>はい。", WAIT, WAIT, WAIT])
    # The energy detector sees loud sound throughout. Independent VAD only
    # finds speech in the preceding Japanese phrase (0.0–0.6).
    offline = GapBackend([], speech_ranges=[(0, 9600)])
    with TestClient(app(main, offline)) as client, client.websocket_connect("/stream") as ws:
        start(ws)
        assert packet(ws)["new_segments"][0]["language"] == "ja"
        assert packet(ws)["action"] == "WAIT"
        row = packet(ws)
        events = list(row.get("gap_events", []))
        ws.send_json({"type": "end"})
        while True:
            row = ws.receive_json()
            events.extend(row.get("gap_events", []))
            assert not row.get("new_segments")
            if row.get("event") == "end_of_audio":
                break
        assert any(e["status"] == "skipped" for e in events)
        assert not offline.decodes
