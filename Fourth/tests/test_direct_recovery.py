"""A failed local decode retains the primary WAIT, audio and commit state."""
from pathlib import Path
import sys

import numpy as np
import pytest

FOURTH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FOURTH))
from fourth_asr.runtime.streaming import GenerationTruncated, SAMPLE_RATE
from fourth_asr.protocol import DIRECT, SPK_CHANGE, WAIT
from fourth_asr.recovery import SpeechActivity, novel_continuation
from fourth_asr.service_session import FourthStreamOptions, FourthStreamSession


class Activity:
    def observe(self, pcm):
        voiced = np.flatnonzero(np.abs(pcm) > .01)
        return (int(voiced[0]), len(voiced)) if len(voiced) else (None, 0)


class Backend:
    def __init__(self, replies, recoveries):
        self.replies = iter(replies)
        self.recoveries = iter(recoveries)
        self.calls = []
        self.history = None

    def generate(self, messages, audios):
        self.history = [m["content"] for m in messages]
        return {"text": next(self.replies)}

    def generate_with_prefix(self, messages, audios, prefix):
        assert [m["role"] for m in messages] == ["system", "user"]
        assert messages[1]["content"] == "<audio>" and len(audios) == 1
        assert prefix == DIRECT
        self.calls.append(audios[0])
        raw = next(self.recoveries)
        if isinstance(raw, Exception):
            raise raw
        return {"text": raw}


def block(value, seconds=.8):
    return np.full(round(seconds * SAMPLE_RATE), value, dtype=np.float32)


def make_session(replies, recoveries, **opts):
    backend = Backend(replies, recoveries)
    options = FourthStreamOptions(**{"protocol": "explicit", "chunk_seconds": .8, **opts})
    return FourthStreamSession(backend, options, activity=Activity()), backend


def test_crossing_recovery_updates_model_history_and_never_duplicates_prefix():
    session, backend = make_session([DIRECT + "こんにちは。", WAIT, WAIT, WAIT, WAIT],
                                    [DIRECT + "こんにちは。Guten Morgen zusammen."])
    assert session.step(block(.04))["display_text"] == "こんにちは。"
    for value in [.12, .25]:
        assert session.step(block(value))["text"] == WAIT
    result = session.step(block(.35))
    assert result["primary_text"] == WAIT and result["action"] == "DIRECT"
    assert result["text"] == DIRECT + "Guten Morgen zusammen."
    assert result["display_text"] == "Guten Morgen zusammen."
    assert result["recovery"]["raw_text"].startswith(DIRECT + "こんにちは。")
    np.testing.assert_array_equal(backend.calls[0], np.concatenate([block(v) for v in [.04,.12,.25,.35]]))
    assert session.verified_commit_seconds == session.inferred_commit_seconds == 0
    assert session.step(end_of_audio=True)["text"] == WAIT
    assert backend.history[-2] == result["text"] and len(backend.calls) == 1


@pytest.mark.parametrize("candidate,status", [
    (DIRECT, "rejected"), (WAIT, "rejected"), (DIRECT + "!", "rejected"),
    (DIRECT + "Thank you!", "rejected"), (RuntimeError("retry failed"), "error"),
    (GenerationTruncated("vLLM", 256, 400, 256, 4096), "truncated")])
def test_rejection_keeps_primary_and_does_not_retry_per_chunk(candidate, status):
    session, backend = make_session([DIRECT + "Thank you.", WAIT, WAIT, WAIT, WAIT],
                                    [candidate], recovery_voiced_seconds=.8)
    session.step(block(.1))
    result = session.step(block(.5))
    assert result["recovery"]["status"] == status and result["text"] == WAIT
    if status == "truncated":
        assert result["recovery"]["generated_tokens"] == 256 and "truncation" not in result
    assert session.history[-1].reply == WAIT and session.inferred_commit_seconds == 0
    assert "recovery" not in session.step(block(.5))
    assert "recovery" not in session.step(end_of_audio=True)
    assert len(backend.calls) == 1


def test_background_pause_compacts_and_short_voice_is_not_lost():
    count = 65
    session, backend = make_session([WAIT] * (count + 3), [DIRECT + "Bonjour."],
                                    context_policy="rolling", recovery_voiced_seconds=1.2, max_pending_seconds=3.2,
                                    recent_seconds=1.6, max_context_seconds=3.2)
    for _ in range(count):
        result = session.step(block(.0002))
    assert result["vad_quiet_evicted_seconds"] > 0 and not backend.calls
    assert result["retained_audio_seconds"] <= 3.2 + 1e-8
    assert len(session.quiet_waits) <= 4
    assert session.step(block(.1))["text"] == WAIT
    final = session.step(block(.12, .4), final_chunk=True)
    assert final["text"] == DIRECT + "Bonjour."
    np.testing.assert_array_equal(backend.calls[0], np.concatenate([block(.0002), block(.1), block(.12, .4)]))
    assert session.step(end_of_audio=True)["text"] == WAIT


def test_overlong_window_preserves_pending_without_tail_only_decode():
    session, backend = make_session([WAIT] * 4, [], recovery_voiced_seconds=1.6,
                                    recovery_overlap_seconds=0., recovery_window_seconds=1.6)
    session.step(block(.1))
    session.step(block(0))
    result = session.step(block(.1))
    assert result["recovery"]["status"] == "window_exceeded"
    assert not backend.calls and result["text"] == WAIT
    assert len(session.history) == 3 and session.verified_commit_seconds == 0


def test_consecutive_successes_keep_new_interval_but_allow_long_rolling_sessions():
    count = 60
    session, _ = make_session([WAIT] * count, [DIRECT + " "+str(i) for i in range(count)],
                              context_policy="rolling", recovery_voiced_seconds=.8, recent_seconds=1.6,
                              max_context_seconds=4.8, max_pending_seconds=4.8)
    for _ in range(count):
        result = session.step(block(.1))
        assert result["recovery"]["status"] == "accepted"
        assert result["inferred_commit_seconds"] <= result["recovery"]["start_seconds"] + 1e-8
    assert session.verified_commit_seconds == 0 and session.heard_seconds > 45


def test_no_recovery_when_disabled():
    session, backend = make_session([WAIT] * 6, [], enable_recovery=False)
    for _ in range(6):
        assert "recovery" not in session.step(block(.1))
    assert not backend.calls


def test_recovery_after_slice_uses_only_retained_voiced_evidence():
    session, backend = make_session([WAIT] * 5 + [WAIT], [DIRECT + "Hallo"],
                                    chunk_seconds=1., window_seconds=3., discard_seconds=2.,
                                    recovery_voiced_seconds=2., recovery_overlap_seconds=1.)
    for v in (.2, 0., 0.):
        assert session.step(block(v, 1.))["text"] == WAIT
    # Voice from the discarded first second must not trigger recovery here.
    row = session.step(block(.3, 1.))
    assert row["window_start_seconds"] == 2 and "recovery" not in row
    row = session.step(block(.4, 1.))
    assert row["primary_text"] == WAIT and row["text"] == DIRECT + "Hallo"
    assert (row["recovery"]["start_seconds"], row["recovery"]["end_seconds"]) == (2, 5)
    np.testing.assert_array_equal(backend.calls[0], np.concatenate([block(v, 1.) for v in (0, .3, .4)]))


def test_carry_vad_frames_across_odd_chunk_and_exclude_literal_silence():
    class Vad:
        def is_speech(self, frame, rate):
            assert len(frame) == 320 and rate == 16000
            return True

    activity = SpeechActivity.__new__(SpeechActivity)
    activity.vad, activity.tail = Vad(), np.zeros(0, dtype=np.int16)
    assert activity.observe(block(.1, .009)) == (None, 0)
    assert activity.observe(block(.1, .011)) == (-144, 320)
    assert activity.observe(block(0, .01)) == (None, 0)


def test_overlap_is_only_removed_with_acoustic_and_lexical_evidence():
    assert novel_continuation("End now. Guten Morgen!", "End now.", has_overlap=True) == (" Guten Morgen!", None)
    assert novel_continuation("End now. Guten Morgen!", "End now.", has_overlap=False)[0] == ""
    assert novel_continuation("now, maybe", "End now.", has_overlap=True)[0] == ""


def test_speaker_recovery_preserves_only_markers_after_overlap_and_keeps_raw_history():
    old = "Hello there."
    change = SPK_CHANGE + "好的。"
    recovered = DIRECT + old + change
    session, backend = make_session([DIRECT + old, WAIT, WAIT], [recovered],
                                    protocol="speaker", recovery_voiced_seconds=.8)
    session.step(block(.1))
    result = session.step(block(.3))
    assert result["primary_text"] == WAIT and result["recovery"]["raw_text"] == recovered
    assert (result["text"], result["display_text"]) == (DIRECT + change, "好的。")
    assert result["transcript_parts"] == [{"text": "好的。", "speaker_change": True}]
    session.step(end_of_audio=True)
    assert backend.history[-2] == DIRECT + change
