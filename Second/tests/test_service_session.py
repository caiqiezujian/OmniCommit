"""Exercise the Second wire protocol on real PCM arrays with scripted model replies."""
from itertools import chain, repeat
from pathlib import Path
import sys

import numpy as np
import pytest

SECOND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SECOND))
sys.path.insert(0, str(SECOND.parent))
from omnicommit.session_identity import SessionLedger
from omnicommit.streaming import ContextBudgetError
from second_asr.protocol import EOS, WAIT, system_prompt
from second_asr.service_session import SecondReplyError, SecondStreamOptions, SecondStreamSession


class Script:
    allowed_tokens = {WAIT, "<zh>", "<en>", "<spk3>", "<spk7>"}

    def __init__(self, outputs):
        self.outputs = iter(outputs)

    def generate(self, messages, audios):
        self.messages, self.audios = messages, audios
        return {"text": next(self.outputs)}


HALF_SECOND = np.full(8000, .1, dtype=np.float32)


def test_changes_in_same_reply_and_inherited_attribution_after_wait():
    answers = [WAIT, "<zh><spk7>请联系", WAIT, "张伟。<en><spk7> OK?<en><spk3>Sure,",
               " I will.\n", WAIT]
    model = Script(answers)
    s = SecondStreamSession(model)
    events = [s.step(HALF_SECOND) for _ in answers[:-1]]
    eos = s.step(end_of_audio=True)
    assert eos["text"] == WAIT and eos["event"] == "end_of_audio" and s.closed
    assert model.messages[0]["content"] == system_prompt("normalized")
    assert model.messages[-1]["content"] == EOS and len(model.audios) == 5
    assert [msg["content"] for msg in model.messages if msg["role"] == "assistant"] == answers[:-1]
    assert [e["text"] for e in events] == answers[:-1]
    assert events[3]["segments"] == [
        {"language": "zh", "speaker": "spk7", "text": "张伟。"},
        {"language": "en", "speaker": "spk7", "text": " OK?"},
        {"language": "en", "speaker": "spk3", "text": "Sure,"},
    ]
    assert events[4]["segments"] == [{"language": "en", "speaker": "spk3", "text": " I will.\n"}]
    assert s.history[3].attributions == (("zh", "spk7"), ("en", "spk7"), ("en", "spk3"))


def test_invalid_reply_then_retry_same_audio_and_eos_without_state_leak():
    s = SecondStreamSession(Script([
        "missing first tags", "<|DIRECT|><zh><spk7>旧", "<zh><spk7>好",
        "<en><spk3>Yes.<fr><spk7>Oui.", "。", "hallucinated tail", WAIT,
    ]))
    for _ in range(2):
        with pytest.raises(SecondReplyError):
            s.step(HALF_SECOND)
        assert not s.failed and not s.speakers and s.heard_seconds == 0 and not s.history
    s.step(HALF_SECOND)
    with pytest.raises(SecondReplyError, match="inventory"):
        s.step(HALF_SECOND)
    assert s.speakers == ["spk7"] and s.heard_seconds == .5
    assert s.step(HALF_SECOND)["segments"][0]["speaker"] == "spk7"
    with pytest.raises(SecondReplyError, match="end_of_audio"):
        s.step(end_of_audio=True)
    assert not s.closed and s.turn_count == 2
    assert s.step(end_of_audio=True)["segments"] == []


@pytest.mark.parametrize("chunk_ms", [1, 5, 80, 499, 500, 700, 800, 1000, 8000])
def test_audio_padding_matches_dataset_and_tracks_only_heard_samples(chunk_ms):
    step = chunk_ms / 1000
    s = SecondStreamSession(Script([WAIT, WAIT, WAIT]), SecondStreamOptions(chunk_seconds=step))
    real_frames = chunk_ms * 16
    x = np.full(real_frames, .2, dtype=np.float32)
    s.step(x)
    assert len(s.backend.audios[-1]) == max(real_frames, 160)
    s.step(x, final_chunk=True)
    assert len(s.backend.audios[-1]) == max(real_frames, 8000)
    assert not np.any(s.backend.audios[-1][real_frames:])
    assert s.heard_seconds == 2 * step
    s.step(end_of_audio=True)


@pytest.mark.parametrize("seconds", [True, float("nan"), 0, .0009, 8.001, .12345])
def test_step_must_be_a_valid_dataset_grid(seconds):
    with pytest.raises(ValueError):
        SecondStreamOptions(chunk_seconds=seconds)


def test_short_final_chunk_and_pending_wait_budget():
    s = SecondStreamSession(Script([WAIT]), SecondStreamOptions(chunk_seconds=.7))
    s.step(np.zeros(123, dtype=np.float32))
    assert s.heard_seconds == 123 / 16000 and len(s.history[-1].audio) == 8000
    with pytest.raises(ValueError, match="followed by EOS"):
        s.step(np.zeros(123, dtype=np.float32))
    s = SecondStreamSession(Script([WAIT]), SecondStreamOptions(max_pending_seconds=.8, context_policy="rolling"))
    s.step(HALF_SECOND)
    with pytest.raises(ContextBudgetError, match="pending"):
        s.step(HALF_SECOND)
    assert s.heard_seconds == .5 and s.last_committed_language is None


def test_reference_and_display_corrections_preserve_live_model_state():
    s = SecondStreamSession(Script(["<en><spk7>Hi.", "<en><spk3>Bye.", " See you."]))
    ledger = SessionLedger(s)
    ledger.record(s.step(HALF_SECOND))
    s.register_reference("spk7", "en", "Hi.", 0, .5, verified=True, clean=True, overlap=False)
    ledger.record(s.step(HALF_SECOND))
    assert s.backend.messages[1]["content"].count("<audio>") == 2
    assert len(s.backend.audios) == 3 and "<en><spk7>Hi." in s.backend.messages[1]["content"]
    ledger.merge_speakers("spk3", "spk7")
    assert not s.references
    line = ledger.record(s.step(HALF_SECOND))[0]
    assert (line["model_speaker"], line["display_speaker"]) == ("spk3", "spk7")
    assert line["text"] == " See you."


@pytest.mark.parametrize("pending_speech", [False, True])
def test_rolling_minute_of_digital_silence_then_speech(pending_speech):
    # Quiet recorded speech is deliberately far below usual VAD thresholds.
    # Storage compaction must not erase a single nonzero audio sample.
    silence = np.zeros(12800, dtype=np.float32)
    quiet = silence.copy()
    quiet[len(quiet) // 2] = 1 / 32768.
    count = 75
    model = Script(chain(repeat(WAIT, count + int(pending_speech)), ["<en><spk7>Back.", WAIT]))
    s = SecondStreamSession(model, SecondStreamOptions(chunk_seconds=.8, context_policy="rolling"))
    if pending_speech:
        s.step(quiet)
    for _ in range(count):
        row = s.step(silence)
        assert row["text"] == WAIT and not row["segments"]
    assert s.heard_seconds == pytest.approx(60 + .8 * pending_speech)
    assert sum(t.end-t.start for t in s.history) < 15
    assert s.verified_commit_seconds == s.inferred_commit_seconds == 0
    assert not s.heuristic_eviction and not s.last_committed_language
    event = s.step(HALF_SECOND, final_chunk=True)
    assert event["start_seconds"] == pytest.approx(60 + .8 * pending_speech)
    assert event["segments"] == [{"language": "en", "speaker": "spk7", "text": "Back."}]
    if pending_speech:
        assert np.array_equal(model.audios[0], quiet)
        assert (s.history[0].start, s.history[0].end) == (0., .8)
    assert s.step(end_of_audio=True)["text"] == WAIT


def test_quiet_nonzero_wait_still_consumes_pending_budget():
    quiet = np.full(8000, 1 / 32768., dtype=np.float32)
    model = Script(repeat(WAIT))
    s = SecondStreamSession(model, SecondStreamOptions(max_pending_seconds=1., context_policy="rolling"))
    s.step(quiet)
    for _ in range(4):
        s.step(np.zeros_like(quiet))
    s.step(quiet)
    with pytest.raises(ContextBudgetError, match="pending"):
        s.step(quiet)
    assert s.verified_commit_seconds == s.inferred_commit_seconds == 0
    assert sum(np.any(t.audio) for t in s.history) == 2


def test_30_minute_rolling_meeting_keeps_reference_and_reconstructs_sparse_history():
    # Speaker 7 returns after nearly 30 minutes of speech by speaker 3.
    # The replies are scripted to test context and identity plumbing, not ASR accuracy.
    replies = chain(["<zh><spk7>开始", "<en><spk3>Hi"], repeat(" yes", 3597),
                    ["<en><spk7>Back.", WAIT])
    s = SecondStreamSession(Script(replies), SecondStreamOptions(
        context_policy="rolling", recent_seconds=2, rolling_tail_seconds=1,
        max_context_seconds=3, max_pending_seconds=3))
    ledger = SessionLedger(s)
    ledger.record(s.step(HALF_SECOND))
    s.register_reference("spk7", "zh", "开始", 0, .5, verified=True, clean=True, overlap=False)
    for _ in range(3599):
        last = ledger.record(s.step(HALF_SECOND))[0]
        assert len(s.history) <= 5
    assert last["model_speaker"] == "spk7" and last["display_speaker"] == "spk7"
    assert last["text"] == "Back." and s.heard_seconds == 1800 and s.heuristic_eviction
    assert s._last_attribution_before_history == ("en", "spk3")
    assert "<en><spk3>" in s.backend.messages[0]["content"]
    assert "spk7" in s.backend.messages[0]["content"] and "<zh><spk7>开始" in s.backend.messages[1]["content"]
    assert len(s.references["spk7"].audio) == 8000 and s.speakers == ["spk7", "spk3"]
    assert s.step(end_of_audio=True)["text"] == WAIT
