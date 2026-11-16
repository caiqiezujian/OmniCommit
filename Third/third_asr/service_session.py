"""Incremental text-only ASR with bounded acoustic history."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omnicommit.streaming import StreamOptions, StreamSession
from .protocol import WAIT, parse_reply, system_prompt


@dataclass
class ThirdStreamOptions(StreamOptions):
    chunk_seconds: float = .8
    context_policy: str = "rolling"
    number_format: str = "normalized"


class ThirdStreamSession(StreamSession):
    def __init__(self, backend, options=None):
        super().__init__(backend, options or ThirdStreamOptions())

    def _context_system_prompt(self):
        return system_prompt(self.options.number_format)

    def _model_reply(self, reply, end_of_audio):
        text = parse_reply(reply)
        if end_of_audio and text:
            raise ValueError("Third expects standalone WAIT at end_of_audio")
        # action is WebSocket metadata. Only the exact reply enters model history.
        return {"action": "DIRECT" if text else "WAIT", "segments": [], "text": reply}

    def _pending_seconds(self, commit):
        if self.options.context_policy == "verified":
            return super()._pending_seconds(commit)
        return sum(max(0., t.end - max(commit, t.start)) for t in self.history)

    def _trim(self):
        super()._trim()
        if self.options.context_policy != "rolling":
            return
        # A long digital pause may exceed the context budget without committing
        # text. Drop only zero-valued WAIT audio; a voiced WAIT can still be pending.
        commit = max(self.verified_commit_seconds, self.inferred_commit_seconds)
        pending = self._pending_seconds(commit)
        retained = sum(t.end - t.start for t in self.history)
        keep = []
        for turn in self.history:
            step = self.options.chunk_seconds
            over_budget = (retained + step > self.options.max_context_seconds + 1e-8 or
                           pending + step > self.options.max_pending_seconds + 1e-8)
            old = turn.end <= self.heard_seconds - self.options.recent_seconds
            if (old or over_budget) and turn.reply == WAIT and not np.any(turn.audio):
                retained -= turn.end - turn.start
                pending -= max(0., turn.end - max(commit, turn.start))
            else:
                keep.append(turn)
        self.history[:] = keep
