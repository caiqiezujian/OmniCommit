"""Exercise length-stop handling through the Fourth streaming session entrypoint."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fourth_asr.protocol import DIRECT, EOS, WAIT
from fourth_asr.runtime.streaming import ContextBudgetError, GenerationTruncated, SAMPLE_RATE
from fourth_asr.service_session import FourthStreamOptions, FourthStreamSession


def truncated():
    return GenerationTruncated("vLLM", 256, 1200, 256, 4096)


class ScriptedBackend:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.audio_ids = []
        self.prompts = []

    def generate(self, messages, audios):
        self.audio_ids.append([round(float(waveform[0]) * 128) for waveform in audios])
        self.prompts.append(messages)
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return {"text": reply}


def session(replies, **options):
    return FourthStreamSession(ScriptedBackend(replies), FourthStreamOptions(
        chunk_seconds=1, protocol="explicit", enable_recovery=False, **options))


def send(stream, audio_id):
    return stream.step(np.full(SAMPLE_RATE, audio_id / 128, dtype=np.float32))


@pytest.mark.parametrize("policy", ["window", "rolling", "verified"])
def test_retry_keeps_last_commit_and_all_pending_audio(policy):
    s = session([DIRECT + "Hello.", DIRECT + " Contact", WAIT, truncated(), DIRECT + " Jane.", WAIT],
                context_policy=policy)
    rows = [send(s, i) for i in range(1, 5)]
    recovered = rows[-1]
    assert recovered["truncation"]["status"] == "recovered" and recovered["action"] == "DIRECT"
    assert recovered["turn"] == 4 and recovered["end_seconds"] == 4
    assert "".join(r["display_text"] for r in rows) == "Hello. Contact Jane."
    assert s.backend.audio_ids[-2:] == [[1, 2, 3, 4], [2, 3, 4]]
    assert [m["content"] for m in s.backend.prompts[-1] if m["role"] == "assistant"] == [DIRECT + " Contact", WAIT]
    send(s, 5)
    if policy == "window":
        assert recovered["discarded_seconds"] == 1 and recovered["window_start_seconds"] == 1
        assert s.backend.audio_ids[-1] == [2, 3, 4, 5]
    else:
        assert s.backend.audio_ids[-1] == [1, 2, 3, 4, 5]


def test_second_length_stop_marks_gap_preserves_pending_wait_and_starts_clean():
    s = session([DIRECT + "Go", WAIT, truncated(), truncated(), DIRECT + "Hello", WAIT])
    first, waiting, failed = [send(s, i) for i in range(1, 4)]
    assert first["display_text"] == "Go" and waiting["action"] == "WAIT"
    assert failed["action"] is None and failed["text"] == failed["display_text"] == ""
    gap = failed["truncation"]
    assert (gap["status"], gap["start_seconds"], gap["end_seconds"]) == ("gap", 1, 3)
    assert gap["primary"]["generated_tokens"] == gap["retry"]["generated_tokens"] == 256
    assert s.backend.audio_ids[-2:] == [[1, 2, 3], [2, 3]]
    assert not s.history and not s.failed
    resumed = send(s, 4)
    assert s.backend.audio_ids[-1] == [4]
    assert (resumed["turn"], resumed["start_seconds"], resumed["end_seconds"]) == (4, 3, 4)
    assert resumed["display_text"] == "Hello" and s.step(end_of_audio=True)["action"] == "WAIT"


def test_rotated_waits_are_included_in_gap_and_eos_does_not_repeat_it():
    s = session([WAIT, WAIT, WAIT, truncated()], window_seconds=2, discard_seconds=1)
    for i in range(1, 4):
        send(s, i)
    failed = send(s, 4)
    assert s.backend.audio_ids == [[1], [1, 2], [2, 3], [3, 4]]
    assert (failed["truncation"]["start_seconds"], failed["truncation"]["end_seconds"]) == (0, 4)
    end = s.step(end_of_audio=True)
    assert end["event"] == "end_of_audio" and end["action"] is None
    assert end["end_reason"] == "gap_already_reported" and "truncation" not in end
    assert len(s.backend.prompts) == 4 and s.closed


def test_eos_exhaustion_keeps_audio_for_retry_and_reports_pending_interval():
    s = session([DIRECT + "Hello.", WAIT, truncated(), truncated()])
    send(s, 1)
    send(s, 2)
    end = s.step(end_of_audio=True)
    assert s.backend.audio_ids[-2:] == [[1, 2], [2]]
    assert all(p[-1]["content"] == EOS for p in s.backend.prompts[-2:])
    assert end["event"] == "end_of_audio" and end["action"] is None and s.closed
    assert (end["truncation"]["start_seconds"], end["truncation"]["end_seconds"]) == (1, 2)


def test_new_gap_excludes_verified_audio_and_an_earlier_reported_gap():
    s = session([WAIT, truncated(), WAIT, truncated()], context_policy="verified", max_pending_seconds=2)
    send(s, 1)
    s.verify_commit(0.75, verified=True)
    first = send(s, 2)["truncation"]
    assert (first["start_seconds"], first["end_seconds"]) == (0.75, 2)
    send(s, 3)
    second = send(s, 4)["truncation"]
    assert (second["start_seconds"], second["end_seconds"]) == (2, 4)
    assert s.verified_commit_seconds == 0.75


def test_length_stop_on_first_default_window_rotation_and_later_gap():
    # 53 s of WAIT, then a commit, six pending seconds and a new chunk at t=61.
    s = session([WAIT] * 53 + [DIRECT + "Hello"] + [WAIT] * 6
                + [truncated(), DIRECT + " again.", WAIT, truncated(), truncated(), DIRECT + "Goodbye."])
    for i in range(1, 61):
        send(s, i)
    recovered = send(s, 61)
    assert recovered["display_text"] == " again." and recovered["truncation"]["status"] == "recovered"
    assert s.backend.audio_ids[-2:] == [list(range(51, 62)), list(range(54, 62))]
    assert recovered["discarded_seconds"] == recovered["window_start_seconds"] == 53
    assert send(s, 62)["action"] == "WAIT"
    gap = send(s, 63)["truncation"]
    assert s.backend.audio_ids[-1] == [61, 62, 63]
    assert (gap["start_seconds"], gap["end_seconds"]) == (61, 63)
    resumed = send(s, 64)
    assert s.backend.audio_ids[-1] == [64] and resumed["display_text"] == "Goodbye."
    assert (resumed["start_seconds"], resumed["end_seconds"]) == (63, 64)


def test_gap_starts_new_voiced_wait_streak_and_drops_vad_carry():
    class AlwaysVoiced:
        def __init__(self):
            self.resets = 0

        def observe(self, pcm):
            return 0, len(pcm)

        def reset(self):
            self.resets += 1

    activity = AlwaysVoiced()
    s = FourthStreamSession(ScriptedBackend([WAIT, WAIT, truncated(), WAIT]),
                           FourthStreamOptions(chunk_seconds=1, protocol="explicit"), activity=activity)
    send(s, 1)
    send(s, 2)
    assert s.recovery.voiced_samples == 2 * SAMPLE_RATE
    assert send(s, 3)["truncation"]["status"] == "gap" and activity.resets == 1
    assert send(s, 4)["action"] == "WAIT"
    assert s.recovery.first_voice == 3 and s.recovery.voiced_samples == SAMPLE_RATE


@pytest.mark.parametrize("error", [ContextBudgetError("context"), ValueError("bad protocol")])
def test_unrelated_error_during_retry_is_fatal_and_never_advances_failed_chunk(error):
    s = session([DIRECT + "Start", truncated(), error])
    send(s, 1)
    with pytest.raises(type(error), match=str(error)):
        send(s, 2)
    assert s.backend.audio_ids == [[1], [1, 2], [2]]
    assert s.failed and s.turn_count == 1 and s.heard_seconds == 1
