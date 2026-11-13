"""Inference state for Second's sparse, speaker-attributed transcript protocol."""
from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np

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

    def _pending_seconds(self, commit):
        if self.options.context_policy != "rolling":
            return super()._pending_seconds(commit)
        # Source times stay absolute: compacted silence creates holes in the
        # retained history, even when speech before those holes is still pending.
        return sum(max(0., t.end - max(commit, t.start)) for t in self.history)

    def _trim(self):
        super()._trim()
        options = self.options
        if options.context_policy != "rolling" or not self.history:
            return
        commit = max(self.verified_commit_seconds, self.inferred_commit_seconds)
        pending = self._pending_seconds(commit)
        retained = sum(t.end - t.start for t in self.history)
        cutoff = self.heard_seconds - options.recent_seconds
        retained_turns = []
        for turn in self.history:
            need_space = (retained + options.chunk_seconds > options.max_context_seconds + 1e-8
                          or pending + options.chunk_seconds > options.max_pending_seconds + 1e-8)
            if (turn.end <= cutoff or need_space) and turn.reply == WAIT and not np.any(turn.audio):
                # A literal all-zero input with no text can be omitted without
                # inferring a speech watermark or discarding an earlier voiced WAIT.
                retained -= turn.end - turn.start
                pending -= max(0., turn.end - max(commit, turn.start))
            else:
                retained_turns.append(turn)
        self.history[:] = retained_turns

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
