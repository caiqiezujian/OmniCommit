"""Pre-decode silence decisions keep one clock for the model and the Web client."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fourth_asr.protocol import DIRECT, WAIT
from fourth_asr.runtime.streaming import ReplayBackend, SAMPLE_RATE
from fourth_asr.service_session import FourthStreamOptions, FourthStreamSession


class Detector:
    def __init__(self, kinds):
        self.kinds = iter(kinds)
        self.sizes = []

    def observe(self, audio, *, final):
        self.sizes.append((len(audio), final))
        kind = next(self.kinds)
        if isinstance(kind, Exception):
            raise kind
        return SimpleNamespace(kind=kind, peak_probability=0.8 if kind == "speech" else 0.02)


class Recorder(ReplayBackend):
    def __init__(self, replies):
        super().__init__(replies)
        self.calls = []

    def generate(self, messages, audios):
        self.calls.append(list(audios))
        return super().generate(messages, audios)


def session_with(detector, replies=(), **options):
    model = Recorder(replies)
    return FourthStreamSession(model, FourthStreamOptions(enable_recovery=False, **options),
                               silence_detector=detector), model


def test_long_silence_advances_clock_and_resumes_with_preroll_and_whole_first_word():
    quiet_chunks = 1800
    detector = Detector(["quiet"] * quiet_chunks + ["uncertain"])
    session, model = session_with(detector, [DIRECT + "はい。", WAIT])
    quiet = np.zeros(SAMPLE_RATE, np.float32)
    for i in range(quiet_chunks):
        row = session.step(quiet)
        assert row["text"] == WAIT and row["turn"] == row["end_seconds"] == i + 1
    assert not model.calls and len(session.history) == 1
    assert row["silence_gate"]["decision"] == "skip" and row["generation_seconds"] == 0
    onset = quiet.copy()
    onset[-1200:] = .02
    row = session.step(onset)
    assert row["display_text"] == "はい。" and row["silence_gate"]["decision"] == "uncertain"
    assert len(model.calls) == 1 and len(model.calls[0]) == 2
    np.testing.assert_array_equal(model.calls[0][0], quiet)
    np.testing.assert_array_equal(model.calls[0][1], onset)
    assert session.step(end_of_audio=True)["text"] == WAIT


def test_one_quiet_turn_releases_delayed_tail_then_skips_quiet_and_eos():
    session, model = session_with(Detector(["speech", "quiet", "quiet", "quiet"]),
                                  [WAIT, DIRECT + "到了。"])
    voice = np.ones(SAMPLE_RATE, np.float32) * 0.1
    quiet = np.zeros_like(voice)
    assert session.step(voice)["text"] == WAIT
    tail = session.step(quiet)
    assert tail["display_text"] == "到了。" and tail["silence_gate"]["decision"] == "tail"
    for _ in range(2):
        assert session.step(quiet)["text"] == WAIT
    end = session.step(end_of_audio=True)
    assert end["text"] == WAIT and end["end_seconds"] == 4
    assert end["silence_gate"]["decision"] == "end_quiet" and session.closed
    assert len(model.calls) == 2


def test_window_rotation_keeps_preroll_after_a_compacted_silence_interval():
    session, model = session_with(Detector(["speech", "quiet", "quiet", "quiet", "speech"]),
                                  [DIRECT + "Hi", WAIT, DIRECT + "Bonjour"],
                                  window_seconds=4., discard_seconds=3.)
    for turn in range(1, 6):
        row = session.step(np.full(SAMPLE_RATE, turn / 100, dtype=np.float32))
    assert row["window_start_seconds"] == 3 and row["display_text"] == "Bonjour"
    assert [round(float(a[0]) * 100) for a in model.calls[-1]] == [4, 5]


def test_detector_receives_true_tail_length_and_can_fail_open():
    detector = Detector([RuntimeError("local detector failed")])
    session, model = session_with(detector, [DIRECT + "OK.", DIRECT + "好。", WAIT])
    row = session.step(np.zeros(SAMPLE_RATE))
    assert row["display_text"] == "OK." and row["silence_gate"]["decision"] == "bypass_error"
    assert session.step(np.zeros(100), final_chunk=True)["display_text"] == "好。"
    assert detector.sizes == [(SAMPLE_RATE, False)] and len(model.calls) == 2

    detector = Detector(["uncertain"])
    session, model = session_with(detector, [DIRECT + "是。", WAIT])
    assert session.step(np.zeros(100), final_chunk=True)["end_seconds"] == 100 / SAMPLE_RATE
    assert detector.sizes == [(100, True)] and len(model.calls[0][-1]) == SAMPLE_RATE // 2
    assert session.step(end_of_audio=True)["text"] == WAIT


def test_gate_precedes_both_primary_and_voiced_wait_recovery():
    class AllVoiced:
        def observe(self, audio):
            return 0, len(audio)

    options = FourthStreamOptions(recovery_voiced_seconds=.1)
    session = FourthStreamSession(Recorder([]), options, activity=AllVoiced(),
                                  silence_detector=Detector(["quiet", "quiet"]))
    for _ in range(2):
        row = session.step(np.zeros(SAMPLE_RATE))
        assert row["text"] == WAIT and "recovery" not in row
    assert not session.backend.calls


def test_verified_mode_keeps_quiet_audio_until_trusted_watermark():
    session, model = session_with(Detector(["quiet"] * 4), context_policy="verified")
    for _ in range(4):
        session.step(np.full(SAMPLE_RATE, .00001))
    assert len(session.history) == 4 and session.verified_commit_seconds == 0
    assert not model.calls


def test_web_loads_one_cpu_model_and_starts_distinct_streams(monkeypatch):
    from fastapi.testclient import TestClient
    from fourth_asr import silero_vad
    from fourth_asr.service_web import create_app

    loaded, streams = [], []
    class LocalModel:
        def __init__(self, path, *, quiet_threshold):
            loaded.append((path, quiet_threshold))

        def new_stream(self):
            stream = Detector(["quiet"])
            streams.append(stream)
            return stream

    monkeypatch.setattr(silero_vad, "SileroVadModel", LocalModel)
    options = FourthStreamOptions(enable_recovery=False)
    app = create_app(lambda: Recorder([]), options, max_sessions=2, silero_vad_model="local.onnx")
    with TestClient(app) as client:
        assert client.get("/health").json()["silence_gate"] == "silero"
        with client.websocket_connect("/stream") as a, client.websocket_connect("/stream") as b:
            for sock in (a, b):
                sock.send_json({"type": "start", "sample_rate": 16000})
                assert sock.receive_json()["silence_gate"] == "silero"
                sock.send_bytes(bytes(2 * SAMPLE_RATE))
                assert sock.receive_json()["silence_gate"]["decision"] == "skip"
                sock.send_json({"type": "end"})
                assert sock.receive_json()["event"] == "end_of_audio"
    assert loaded == [("local.onnx", .10)]
    assert len(streams) == 2 and streams[0] is not streams[1]
