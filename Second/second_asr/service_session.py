"""Inference state for Second's sparse, speaker-attributed transcript protocol."""
from __future__ import annotations

from dataclasses import dataclass
import math

from omnicommit.runtime_protocol import LIVE_CHUNK_MS, speaker_registry_text
from omnicommit.streaming import SAMPLE_RATE, StreamOptions, StreamSession

from .protocol import WAIT, parse_reply, system_prompt


def chunk_ms_choices(default_seconds):
    """Millisecond steps available for each new meeting on this server."""
    return tuple(sorted({*LIVE_CHUNK_MS, round(default_seconds * 1000)}))


class SecondReplyError(ValueError):
    """Invalid model output. No part of this turn has been committed."""


@dataclass
class SecondStreamOptions(StreamOptions):
    number_format: str = "normalized"

    def _validate_chunk_seconds(self):
        value = self.chunk_seconds
        if type(value) not in (int, float) or not math.isfinite(value) or not .001 <= value <= 8:
            raise ValueError("chunk_seconds must be between 0.001 and 8 seconds")
        milliseconds = round(value * 1000)
        if milliseconds / 1000 != value:
            raise ValueError("chunk_seconds must be a whole number of milliseconds")


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

    def _minimum_input_samples(self, final_chunk):
        return SAMPLE_RATE // 2 if final_chunk else SAMPLE_RATE // 100

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
