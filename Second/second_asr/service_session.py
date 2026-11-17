"""Inference state for Second's sparse, speaker-attributed transcript protocol."""
from __future__ import annotations

from dataclasses import dataclass
import json
import sys

import numpy as np

from omnicommit.runtime_protocol import speaker_registry_text
from omnicommit.streaming import StreamOptions, StreamSession

from .protocol import WAIT, parse_reply, system_prompt


class SecondReplyError(ValueError):
    """Invalid model output. No part of this turn has been committed."""


@dataclass
class SecondStreamOptions(StreamOptions):
    context_policies = ("window", "verified", "rolling")
    number_format: str = "normalized"
    context_policy: str = "window"
    window_seconds: float = 60.0
    discard_seconds: float = 50.0

    def __post_init__(self):
        super().__post_init__()
        if self.context_policy == "window":
            for name in ("window_seconds", "discard_seconds"):
                value = getattr(self, name)
                if type(value) not in (float, int) or not 0 < value <= sys.float_info.max:
                    raise ValueError(f"{name} must be positive and finite")
            if self.chunk_seconds > self.window_seconds:
                raise ValueError("window_seconds must be at least chunk_seconds")
            if self.discard_seconds > self.window_seconds:
                raise ValueError("discard_seconds must not exceed window_seconds")


class SecondStreamSession(StreamSession):
    # Source annotations may retain a global or cropped meeting's speaker IDs.
    sequential_speaker_ids = False
    retryable_errors = (SecondReplyError,)

    def __init__(self, backend, options=None):
        super().__init__(backend, options or SecondStreamOptions())
        self._recovered_text = {}
        self._before_rotation = None

    def _window_start(self):
        return self.history[0].start if self.history else self.heard_seconds

    def _prepare_audio_context(self, duration):
        if self.options.context_policy != "window":
            return super()._prepare_audio_context(duration)
        start = self._window_start()
        end = self.heard_seconds + duration
        if end - start <= self.options.window_seconds + 1e-8:
            return

        # Keep audio/assistant pairs indivisible. If a boundary falls inside
        # one pair, drop through its end; never replay a reply with half its audio.
        cutoff = max(start + self.options.discard_seconds, end - self.options.window_seconds)
        count = len(self.history)
        for index, turn in enumerate(self.history, 1):
            if turn.end + 1e-8 >= cutoff:
                count = index
                break
        removed = self.history[:count]
        # Hold the old state until this decode succeeds. A rejected answer
        # can be retried with the same (or a shorter final) piece of audio.
        self._before_rotation = (self.history, self._speakers_before_history,
                                 self._last_attribution_before_history, self._recovered_text,
                                 self.heuristic_eviction)
        self.history = self.history[count:]
        self._speakers_before_history = self._speakers_before_history.copy()
        for turn in removed:
            for attribution in turn.attributions:
                self._speakers_before_history.add(attribution[1])
                self._last_attribution_before_history = attribution
        self.evicted_this_turn = True
        if removed[-1].end > self.verified_commit_seconds:
            self.heuristic_eviction = True

    def step(self, audio=None, *, end_of_audio=False, final_chunk=False):
        if self.options.context_policy != "window":
            return super().step(audio, end_of_audio=end_of_audio, final_chunk=final_chunk)
        start = self._window_start()
        try:
            result = super().step(audio, end_of_audio=end_of_audio, final_chunk=final_chunk)
        except Exception:
            if self._before_rotation is not None:
                (self.history, self._speakers_before_history, self._last_attribution_before_history,
                 self._recovered_text, self.heuristic_eviction) = self._before_rotation
                self.evicted_this_turn = False
            raise
        finally:
            self._before_rotation = None
        result["window_start_seconds"] = self._window_start()
        result["discarded_seconds"] = result["window_start_seconds"] - start
        # A bounded audio window has no estimate of what the model consumed.
        del result["unverified_audio_seconds"], result["inferred_commit_seconds"]
        return result

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
