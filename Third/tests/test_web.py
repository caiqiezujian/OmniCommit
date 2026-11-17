"""Third model/tokenizer, streaming state, and browser transport contracts."""
from io import BytesIO
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import wave

import numpy as np
import pytest

THIRD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THIRD))

from third_asr.runtime.media import MediaCursor
from third_asr.runtime.streaming import ContextBudgetError, ReplayBackend
from third_asr.protocol import EOS, WAIT, system_prompt
from third_asr.service_backend import read_export_protocol, validate_manifest
from third_asr.service_session import ThirdStreamOptions, ThirdStreamSession
from third_asr.service_web import create_app


class Recorder(ReplayBackend):
    def generate(self, messages, audios):
        self.messages = messages
        self.lengths = list(map(len, audios))
        return super().generate(messages, audios)


def client_for(backend):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    return TestClient(create_app(lambda: backend, ThirdStreamOptions()))


def start(ws, kind, **kwargs):
    ws.send_json({"type": "start", "sample_rate": 16000, "source_kind": kind, **kwargs})
    assert ws.receive_json()["protocol"] == "third"


@pytest.mark.parametrize("kind", ["microphone", "system"])
def test_web_incremental_text_and_eos(kind):
    backend = Recorder(["中文。Let's", " meet.", WAIT, WAIT])
    with client_for(backend) as client:
        assert client.get("/health").json()["chunk_seconds"] == .8
        assert 'value="system"' in client.get("/").text
        with client.websocket_connect("/stream") as ws:
            start(ws, kind)
            for op in ("reference", "correct_segment", "merge_speakers", "rename_speaker"):
                ws.send_json({"type": "control", "op": op})
                assert ws.receive_json()["type"] == "control_error"
            for expected, frames in [("中文。Let's", 12800), (" meet.", 12800), (WAIT, 16)]:
                ws.send_bytes(bytes(frames * 2))
                result = ws.receive_json()
                assert result["type"] == "result" and result["text"] == expected
                assert result["action"] == ("WAIT" if expected == WAIT else "DIRECT")
                assert result["reference_speakers"] == result["new_segments"] == []
            assert backend.lengths[-1] == 8000
            ws.send_json({"type": "end"})
            assert ws.receive_json()["text"] == WAIT
            assert backend.messages[0]["content"] == system_prompt("normalized")
            assert [m["content"] for m in backend.messages[1:]] == [
                "<audio>", "中文。Let's", "<audio>", " meet.", "<audio>", WAIT, EOS]


def test_file_chunks_release_only_after_the_full_source_frame():
    # At 44.1 kHz the first 101 ms ends between frames 4454 and 4455.
    upload = SimpleNamespace(ready=True, claimed=True, sample_rate=44100, samples=44100)
    cursor = MediaCursor(upload)
    assert cursor.next_chunk(4454 / 44100, .101) is None
    assert cursor._chunk_end_offset(.101) == 4455


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="requires FFmpeg")
def test_file_upload_playback_gate_and_stop_at_zero():
    backend = Recorder(["日语。", "Über.", WAIT])
    with client_for(backend) as client:
        assert client.post("/api/media", content=b"").status_code == 400
        wav = BytesIO()
        with wave.open(wav, "wb") as audio:
            audio.setparams((1, 2, 16000, 0, "NONE", ""))
            audio.writeframes(bytes(16000 * 2))
        for stop_at_zero in (True, False):
            upload = client.post("/api/media", content=wav.getvalue())
            assert upload.status_code == 200
            with client.websocket_connect("/stream") as ws:
                start(ws, "file", media_id=upload.json()["media_id"])
                if stop_at_zero:
                    ws.send_json({"type": "file_next", "release_seconds": 0, "final": True})
                    assert ws.receive_json()["type"] == "file_done"
                    ws.send_json({"type": "end"})
                    assert ws.receive_json()["type"] == "file_end_empty"
                else:
                    ws.send_json({"type": "file_next", "release_seconds": .79})
                    assert ws.receive_json()["type"] == "file_wait"
                    for release, expected in [(.8, "日语。"), (1., "Über.")]:
                        ws.send_json({"type": "file_next", "release_seconds": release})
                        assert ws.receive_json()["text"] == expected
                    ws.send_json({"type": "end"})
                    assert ws.receive_json()["text"] == WAIT


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="requires FFmpeg")
def test_file_range_uses_negotiated_window_relative_to_range_start():
    wav = BytesIO()
    with wave.open(wav, "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", ""))
        audio.writeframes(bytes(10 * 16000 * 2))
    from fastapi.testclient import TestClient
    with TestClient(create_app(ReplayBackend, ThirdStreamOptions(chunk_seconds=1))) as client:
        upload = client.post("/api/media", content=wav.getvalue()).json()
        with client.websocket_connect("/stream") as ws:
            ws.send_json(dict(type="start", sample_rate=16000, source_kind="file", media_id=upload["media_id"],
                              file_start_seconds=4, file_end_seconds=9, window_seconds=3, discard_seconds=2))
            ready = ws.receive_json()
            assert (ready["file_start_seconds"], ready["window_seconds"], ready["discard_seconds"]) == (4, 3, 2)
            for turn, head in enumerate([0, 0, 0, 2, 2], 1):
                ws.send_json(dict(type="file_next", release_seconds=4 + turn))
                row = ws.receive_json()
                assert (row["window_start_seconds"], row["end_seconds"], row["source_end_seconds"]) == (head, turn, 4 + turn)
            ws.send_json({"type": "end"})
            assert ws.receive_json()["window_start_seconds"] == 2


def test_silence_compaction_and_voiced_pending_guard():
    opts = ThirdStreamOptions(context_policy="rolling", recent_seconds=.8, max_context_seconds=2.4, max_pending_seconds=2.4)
    session = ThirdStreamSession(ReplayBackend(), opts)
    for _ in range(2250):  # 30 minutes of digital silence
        event = session.step(np.zeros(12800, dtype=np.float32))
    assert event["retained_audio_seconds"] <= 2.4 + 1e-8
    assert session.step(end_of_audio=True)["text"] == WAIT
    voiced = ThirdStreamSession(ReplayBackend(), opts)
    for _ in range(3):
        voiced.step(np.full(12800, .05, dtype=np.float32))
    with pytest.raises(ContextBudgetError):
        voiced.step(np.full(12800, .05, dtype=np.float32))


def test_wrong_protocol_tokenizer_is_rejected(tmp_path):
    class Tokenizer:
        all_special_tokens = [WAIT]

        def __len__(self): return 16

        def encode(self, *args, **kwargs): return [15]

    inventory = tmp_path / "tokens.txt"
    inventory.write_text(WAIT + "\n", encoding="utf8")
    tokens, metadata = read_export_protocol(inventory)
    manifest = {**metadata, "tokens": tokens, "token_ids": {WAIT: 15}, "tokenizer_length": 16}
    assert validate_manifest(Tokenizer(), manifest) == {WAIT}
    with pytest.raises(ValueError, match="Expected a Third"):
        validate_manifest(Tokenizer(), {**manifest, "schema_version": 2})
    with pytest.raises(ValueError, match="mismatch"):
        validate_manifest(Tokenizer(), {**manifest, "token_ids": {WAIT: 14}})
    inventory.write_text(WAIT + "\n<en>\n<spk1>\n", encoding="utf8")
    with pytest.raises(ValueError, match="exactly"):
        read_export_protocol(inventory)
