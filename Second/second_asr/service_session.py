"""Inference state for Second's sparse, speaker-attributed transcript protocol."""
from __future__ import annotations

from dataclasses import dataclass

from omnicommit.runtime_protocol import speaker_registry_text
from omnicommit.streaming import StreamOptions, StreamSession

from .protocol import WAIT, parse_reply, system_prompt


class SecondReplyError(ValueError):
    """Invalid model output. No part of this turn has been committed."""


@dataclass
class SecondStreamOptions(StreamOptions):
    number_format: str = "normalized"


class SecondStreamSession(StreamSession):
    # Source annotations may retain a global or cropped meeting's speaker IDs.
    sequential_speaker_ids = False
    retryable_errors = (SecondReplyError,)

    def __init__(self, backend, options=None):
        super().__init__(backend, options or SecondStreamOptions())

    def _context_system_prompt(self):
        prompt = system_prompt(self.options.number_format)
        lang, speaker = self._last_attribution_before_history
        if speaker is not None:
            prompt += speaker_registry_text(self._speakers_before_history)
            prompt += (f"\nAt the start of the retained live dialogue the last committed voice was "
                       f"<{lang}><{speaker}>. Subsequent untagged text inherits the preceding live attribution.")
        return prompt

    def _model_reply(self, reply, end_of_audio):
        state = None
        if self.last_committed_speaker is not None:
            state = self.last_committed_language, self.last_committed_speaker
        try:
            spans, _ = parse_reply(reply, state)
        except ValueError as exc:
            raise SecondReplyError(str(exc)) from exc
        if end_of_audio and spans:
            raise SecondReplyError("Second requires a standalone WAIT at end_of_audio")
        tokens = getattr(self.backend, "allowed_tokens", None)
        segments = []
        for lang, speaker, text in spans:
            if tokens is not None and (f"<{lang}>" not in tokens or f"<{speaker}>" not in tokens):
                raise SecondReplyError("Generated language/speaker tag is absent from the trained inventory")
            segments.append({"language": lang, "speaker": speaker, "text": text})
        # action is event metadata for the existing transport and ledger. Only
        # the unmodified model reply goes into the transcript history.
        return {"action": "WAIT" if reply == WAIT else "DIRECT", "segments": segments, "text": reply}
