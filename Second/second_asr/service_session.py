"""Inference state for Second's sparse, speaker-attributed transcript protocol."""
from __future__ import annotations

from dataclasses import dataclass
import json

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
        self._recovered_text = {}

    def remember_recovery(self, item):
        self._recovered_text[item["segment_id"]] = (item["audio_end_seconds"], item["text"])
        if self.options.context_policy == "rolling":
            # Same heuristic budget as a live DIRECT. This is not a verified watermark.
            self.inferred_commit_seconds = max(self.inferred_commit_seconds,
                item["audio_end_seconds"] - self.options.rolling_tail_seconds)

    def forget_recovery(self, segment_id):
        self._recovered_text.pop(segment_id, None)

    def _context_system_prompt(self):
        prompt = system_prompt(self.options.number_format)
        lang, speaker = self._last_attribution_before_history
        if speaker is not None:
            prompt += speaker_registry_text(self._speakers_before_history)
            prompt += (f"\nAt the start of the retained live dialogue the last committed voice was "
                       f"<{lang}><{speaker}>. Subsequent untagged text inherits the preceding live attribution.")
        cutoff = self.history[0].start if self.history else self.heard_seconds
        self._recovered_text = {key: value for key, value in self._recovered_text.items()
                                if value[0] > cutoff}
        if self._recovered_text:
            # An outside recognizer has no meeting speaker ID. Do not invent an
            # assistant header or change attribution of subsequent bare text.
            recent = sorted(self._recovered_text.values())[-8:]
            text = json.dumps([value[:320] for _, value in recent], ensure_ascii=False)
            prompt += "\nRecent words already transcribed from earlier audio; continue after them: " + text
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
