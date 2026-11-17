"""Incremental text-only ASR with bounded acoustic history."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .runtime.streaming import StreamOptions, StreamSession
from .protocol import WAIT, get_protocol
from .recovery import VoicedWaitRecovery


@dataclass
class ThirdStreamOptions(StreamOptions):
    chunk_seconds: float = .8
    context_policy: str = "window"
    number_format: str = "normalized"
    protocol: str = "plain"
    enable_recovery: bool = True
    recovery_voiced_seconds: float = 2.4
    recovery_overlap_seconds: float = .8
    recovery_window_seconds: float = 6.4

    def __post_init__(self):
        super().__post_init__()
        get_protocol(self.protocol)
        for name in ("recovery_voiced_seconds", "recovery_overlap_seconds", "recovery_window_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        if round(self.recovery_voiced_seconds * 16000) < 1:
            raise ValueError("recovery_voiced_seconds must be positive")
        if self.recovery_voiced_seconds + self.recovery_overlap_seconds > self.recovery_window_seconds + 1e-8:
            raise ValueError("recovery_window_seconds must cover the voiced threshold and overlap")


class ThirdStreamSession(StreamSession):
    def __init__(self, backend, options=None, *, activity=None):
        super().__init__(backend, options or ThirdStreamOptions())
        self.protocol = get_protocol(self.options.protocol)
        self.recovery = None
        self.committed_tail = ""
        self.last_text_end = 0.
        self.quiet_waits = set()
        self.quiet_evicted_seconds = 0.
        if self.options.protocol == "explicit" and self.options.enable_recovery:
            self.recovery = VoicedWaitRecovery(self.options, activity)

    def _context_system_prompt(self):
        return self.protocol.system_prompt(self.options.number_format)

    def _model_reply(self, reply, end_of_audio):
        text = self.protocol.parse_reply(reply)
        if end_of_audio and text:
            raise ValueError("Third expects standalone WAIT at end_of_audio")
        return {"action": "DIRECT" if text else "WAIT", "segments": [],
                "text": reply, "display_text": text}

    def step(self, audio=None, *, end_of_audio=False, final_chunk=False):
        self.quiet_evicted_seconds = 0.
        result = super().step(audio, end_of_audio=end_of_audio, final_chunk=final_chunk)
        if self.recovery is not None and not end_of_audio:
            self.recovery.observe(self, result)
            seconds = result.get("recovery", {}).get("seconds", 0.)
            if seconds:
                result["generation_seconds"] += seconds
                result["realtime_factor"] = result["generation_seconds"] / (
                    result["end_seconds"] - result["start_seconds"])
        if result["action"] == "DIRECT":
            self.committed_tail = (self.committed_tail + result["display_text"])[-2048:]
            self.last_text_end = result["end_seconds"]
        if self.quiet_evicted_seconds:
            result["vad_quiet_evicted_seconds"] = self.quiet_evicted_seconds
        return result

    def _pending_seconds(self, commit):
        if self.options.context_policy == "verified":
            return super()._pending_seconds(commit)
        return sum(max(0., t.end - max(commit, t.start)) for t in self.history)

    def _trim(self):
        super()._trim()
        if self.options.context_policy != "rolling":
            return
        # Digital silence is exact. For low-background VAD silence, also guard
        # the recent boundary and all audio after the first pending voiced WAIT.
        commit = max(self.verified_commit_seconds, self.inferred_commit_seconds)
        pending = self._pending_seconds(commit)
        retained = sum(t.end - t.start for t in self.history)
        keep = []
        quiet_cutoff = self.heard_seconds - self.options.recovery_overlap_seconds
        if self.recovery and self.recovery.first_voice is not None:
            quiet_cutoff = min(quiet_cutoff, self.recovery.first_voice - self.options.recovery_overlap_seconds)
        for turn in self.history:
            step = self.options.chunk_seconds
            over_budget = (retained + step > self.options.max_context_seconds + 1e-8 or
                           pending + step > self.options.max_pending_seconds + 1e-8)
            old = turn.end <= self.heard_seconds - self.options.recent_seconds
            quiet = turn.start in self.quiet_waits and turn.end <= quiet_cutoff
            if (old or over_budget) and turn.reply == WAIT and (quiet or not np.any(turn.audio)):
                if quiet and np.any(turn.audio):
                    self.quiet_evicted_seconds += turn.end - turn.start
                    self.heuristic_eviction = self.evicted_this_turn = True
                retained -= turn.end - turn.start
                pending -= max(0., turn.end - max(commit, turn.start))
            else:
                keep.append(turn)
        self.history[:] = keep
        if self.quiet_waits:
            self.quiet_waits.intersection_update(t.start for t in keep)
