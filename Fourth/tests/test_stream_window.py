"""Inspect the audio and assistant replies Fourth hands to its inference backend."""
from itertools import repeat
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fourth_asr.protocol import DIRECT, WAIT
from fourth_asr.runtime.streaming import ReplayBackend, SAMPLE_RATE
from fourth_asr.service_session import FourthStreamOptions, FourthStreamSession


class Recorder(ReplayBackend):
    def generate(self, messages, audios):
        self.audios = audios
        self.messages = messages
        return super().generate(messages, audios)


@pytest.mark.parametrize("protocol", ["plain", "explicit", "speaker"])
def test_window_crossings_keep_exact_turns_even_after_a_minute_of_wait(protocol):
    replies = [WAIT] * 112
    replies[53] = ("" if protocol == "plain" else DIRECT) + "Guten Morgen."
    backend = Recorder(replies)
    session = FourthStreamSession(backend, FourthStreamOptions(protocol=protocol, chunk_seconds=1, enable_recovery=False))
    for end in range(1, 112):
        start = 0 if end <= 60 else 50 if end <= 110 else 100
        event = session.step(np.full(SAMPLE_RATE, end / 128, dtype=np.float32))
        assert (event["window_start_seconds"], event["retained_audio_seconds"]) == (start, end - start)
        assert event["discarded_seconds"] == (50 if end in (61, 111) else 0)
        assert [a[0] * 128 for a in backend.audios] == list(range(start + 1, end + 1))
        assert [m["content"] for m in backend.messages if m["role"] == "assistant"] == replies[start:end-1]
    event = session.step(end_of_audio=True)
    assert event["window_start_seconds"] == 100 and event["discarded_seconds"] == 0
    assert len(backend.audios) == 11


def test_nondivisible_discard_and_partial_tail_use_real_audio_duration():
    opts = FourthStreamOptions(chunk_seconds=.7, window_seconds=2.2, discard_seconds=1, enable_recovery=False)
    session = FourthStreamSession(Recorder(repeat(WAIT)), opts)
    pcm = np.zeros(11200, dtype=np.float32)
    for _ in range(3):
        session.step(pcm)
    # 2.1 seconds already fit. Exactly 0.1 remains for a short last packet.
    event = session.step(pcm[:1600], final_chunk=True)
    assert event["discarded_seconds"] == 0 and event["end_seconds"] == pytest.approx(2.2)
    assert len(session.backend.audios[-1]) == 8000
    session = FourthStreamSession(Recorder(repeat(WAIT)), opts)
    for _ in range(4):
        event = session.step(pcm)
    assert event["discarded_seconds"] == pytest.approx(1.4)
    assert len(session.backend.audios) == 2


def test_rejected_reply_does_not_commit_window_rotation():
    session = FourthStreamSession(Recorder([WAIT, WAIT, DIRECT]), FourthStreamOptions(
        protocol="explicit", enable_recovery=False, chunk_seconds=1, window_seconds=2, discard_seconds=2))
    pcm = np.ones(SAMPLE_RATE, dtype=np.float32)
    session.step(pcm)
    session.step(pcm)
    with pytest.raises(ValueError):
        session.step(pcm)
    assert session.failed and session.heard_seconds == 2
    assert [turn.start for turn in session.history] == [0, 1]


@pytest.mark.parametrize(("window", "discard"), [
    (float("nan"), 1), (float("inf"), 1), (60, False), ("60", 50), (0, 1), (.3, .1), (60, 0), (2, 3)
])
def test_invalid_window_options(window, discard):
    with pytest.raises(ValueError):
        FourthStreamOptions(window_seconds=window, discard_seconds=discard)
