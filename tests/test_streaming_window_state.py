"""The system message summarizes only replies preceding the live window."""
import numpy as np
import pytest

from omnicommit.runtime_protocol import END_OF_AUDIO, last_committed_text, speaker_registry_text, system_prompt
from omnicommit.streaming import DIRECT, WAIT, ReplayBackend, StreamOptions, StreamSession


@pytest.mark.parametrize("policy", ["verified", "rolling"])
def test_conditioning_at_window_start_across_returning_language(policy):
    first = DIRECT + "<ja><spk1>はい。<de><spk2>Ja."
    second = DIRECT + "<fr><spk3>Oui."
    returning = DIRECT + "<ja><spk1>続けます。"

    class RecordingBackend(ReplayBackend):
        def generate(self, messages, audios):
            self.messages = messages
            return super().generate(messages, audios)

    backend = RecordingBackend([first, second, WAIT, WAIT, returning, WAIT, WAIT])
    options = StreamOptions(context_policy=policy, recent_seconds=.5, rolling_tail_seconds=.5)
    session = StreamSession(backend, options)
    pcm = np.zeros(8000, dtype=np.float32)
    initial = system_prompt()

    session.step(pcm)
    session.step(pcm)
    assert backend.messages[0]["content"] == initial
    assert backend.messages[2]["content"] == first
    if policy == "verified":
        session.verify_commit(.5, verified=True)

    after_first = initial + speaker_registry_text(["spk1", "spk2"]) + last_committed_text("de", "spk2")
    for _ in range(3):
        session.step(pcm)
        assert backend.messages[0]["content"] == after_first
        assert backend.messages[2]["content"] == second
    assert (session.last_committed_language, session.last_committed_speaker) == ("ja", "spk1")
    assert session.speakers == ["spk1", "spk2", "spk3"]
    if policy == "verified":
        session.verify_commit(2, verified=True)

    # The French reply and the two WAITs leave the window together. Its new
    # prefix ends in French and the visible dialogue starts with Japanese.
    after_second = initial + speaker_registry_text(session.speakers) + last_committed_text("fr", "spk3")
    session.step(pcm)
    assert backend.messages[0]["content"] == after_second
    assert backend.messages[2]["content"] == returning
    session.step(end_of_audio=True)
    assert backend.messages[0]["content"] == after_second
    assert backend.messages[-2]["content"] == WAIT
    assert backend.messages[-1]["content"] == END_OF_AUDIO
