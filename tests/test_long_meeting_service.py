"""Long-session protocol regression with synthetic audio, not acoustic validation."""
import numpy as np

from omnicommit.streaming import DIRECT, WAIT, ReplayBackend, StreamOptions, StreamSession


def test_30_minute_bounded_context_retains_returning_speaker_reference():
    chunks = 30 * 60 * 2
    replies = [DIRECT + "<zh><spk1>开始。"]
    replies += [DIRECT + "<en><spk2>Speaking. "] * (chunks - 2)
    replies += [DIRECT + "<en><spk1>I'm back.", WAIT]
    options = StreamOptions(context_policy="rolling", recent_seconds=3,
                            rolling_tail_seconds=2, max_context_seconds=10,
                            max_pending_seconds=10)
    session = StreamSession(ReplayBackend(replies), options)
    audio = np.zeros(8000, dtype=np.float32)
    session.step(audio)
    # Synthetic reference is supplied by the test; no acoustic truth is implied.
    session.register_reference("spk1", "zh", "开始。", 0, .5,
                               verified=True, clean=True, overlap=False)
    for _ in range(1, chunks):
        result = session.step(audio)
    assert result["end_seconds"] == 1800
    assert result["segments"][0]["speaker"] == "spk1"
    assert result["reference_speakers"] == ["spk1"]
    assert result["heuristic_eviction"] is True
    assert result["retained_audio_seconds"] <= options.max_context_seconds
    messages, _ = session._context(audio)
    assert "<zh><spk1>开始。" in messages[1]["content"]
    assert session.step(end_of_audio=True)["action"] == "WAIT"
    session.close()
