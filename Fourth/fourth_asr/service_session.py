"""Incremental text-only ASR with bounded acoustic history."""
from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import time

import numpy as np

from .runtime.streaming import StreamOptions, StreamSession
from .packet_punctuation import replace_terminal_period
from .protocol import SPK_CHANGE, WAIT, get_protocol
from .recovery import VoicedWaitRecovery


@dataclass
class FourthStreamOptions(StreamOptions):
    chunk_seconds: float = 1.
    context_policy: str = "window"
    number_format: str = "normalized"
    protocol: str = "speaker"
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


class FourthStreamSession(StreamSession):
    def __init__(self, backend, options=None, *, activity=None, silence_detector=None):
        super().__init__(backend, options or FourthStreamOptions())
        self.protocol = get_protocol(self.options.protocol)
        self.recovery = None
        self.committed_tail = ""
        self.quiet_waits = set()
        self.quiet_evicted_seconds = 0.
        self.silence_detector = silence_detector
        self.allow_silence_tail = False
        self.last_gated_turn = None
        if self.protocol.prefix and self.options.enable_recovery:
            self.recovery = VoicedWaitRecovery(self.options, activity)

    def _context_system_prompt(self):
        return self.protocol.system_prompt(self.options.number_format)

    def _model_reply(self, reply, end_of_audio):
        text = self.protocol.parse_reply(reply)
        if end_of_audio and text:
            raise ValueError("Fourth expects standalone WAIT at end_of_audio")
        parts = []
        heard_speaker = self.last_text_end > 0
        for i, part in enumerate(text.split(SPK_CHANGE)):
            if part:
                parts.append({"text": part, "speaker_change": i > 0 and heard_speaker})
                heard_speaker = heard_speaker or bool(part.strip())
        return {"action": "DIRECT" if text else "WAIT", "segments": [],
                "text": reply, "display_text": text.replace(SPK_CHANGE, ""), "transcript_parts": parts}

    def _decode_audio_turn(self, history, pcm, source_samples, final_chunk):
        if self.silence_detector is None:
            return super()._decode_audio_turn(history, pcm, source_samples, final_chunk)
        started = time.perf_counter()
        try:
            observation = self.silence_detector.observe(pcm[:source_samples], final=final_chunk)
        except Exception as exc:
            # An optional detector failure must never discard the current audio.
            logging.getLogger(__name__).exception("Silero disabled for this Web session")
            self.silence_detector = None
            gate = {"decision": "bypass_error", "error": str(exc)}
        else:
            decision = observation.kind
            if decision == "quiet":
                decision = "tail" if self.allow_silence_tail else "skip"
            gate = {"decision": decision, "peak_probability": observation.peak_probability}
        gate["seconds"] = time.perf_counter() - started
        if gate["decision"] == "skip":
            # Keep the latest quiet audio for a weak onset. A long pause should
            # not teach the next decode to imitate dozens of generated WAITs.
            if (self.options.context_policy != "verified" and history
                    and history[-1] is self.last_gated_turn):
                history = history[:-1]
            parsed = self._model_reply(WAIT, False)
            parsed["silence_gate"] = gate
            return None, parsed, history
        generated, parsed, history = super()._decode_audio_turn(history, pcm, source_samples, final_chunk)
        parsed["silence_gate"] = gate
        return generated, parsed, history

    def _decode_turn(self, history, pcm, end_of_audio):
        if end_of_audio and history and history[-1] is self.last_gated_turn:
            parsed = self._model_reply(WAIT, True)
            parsed["silence_gate"] = {"decision": "end_quiet", "seconds": 0.}
            return None, parsed, history
        return super()._decode_turn(history, pcm, end_of_audio)

    def step(self, audio=None, *, end_of_audio=False, final_chunk=False):
        self.quiet_evicted_seconds = 0.
        result = super().step(audio, end_of_audio=end_of_audio, final_chunk=final_chunk)
        gate = result.get("silence_gate")
        skipped = gate is not None and gate["decision"] == "skip"
        if gate is not None:
            self.allow_silence_tail = (gate["decision"] in ("speech", "uncertain")
                                       and result.get("truncation", {}).get("status") != "gap")
            elapsed = (0. if skipped or end_of_audio else
                       max(0., result["generation_seconds"] - gate["seconds"]))
            result["generation_seconds"] = elapsed
            result["realtime_factor"] = None if end_of_audio else elapsed / (result["end_seconds"] - result["start_seconds"])
        if result.get("truncation", {}).get("status") == "gap":
            self.quiet_waits.clear()
            self.committed_tail = ""
            if self.recovery is not None:
                self.recovery.on_gap()
        elif skipped:
            if self.recovery is not None and self.last_gated_turn is None:
                self.recovery.on_gap()  # Separate voiced WAIT streaks across a pause.
        elif self.recovery is not None and not end_of_audio:
            self.recovery.observe(self, result)
            seconds = result.get("recovery", {}).get("seconds", 0.)
            if seconds:
                result["generation_seconds"] += seconds
                result["realtime_factor"] = result["generation_seconds"] / (
                    result["end_seconds"] - result["start_seconds"])
        self.last_gated_turn = self.history[-1] if skipped else None
        if result["action"] == "DIRECT":
            # A speaker boundary separates words for recovery overlap detection.
            marked = result["text"].removeprefix(self.protocol.prefix)
            self.committed_tail = (self.committed_tail + marked.replace(SPK_CHANGE, " "))[-2048:]
            self.last_text_end = result["end_seconds"]
        if self.quiet_evicted_seconds:
            result["vad_quiet_evicted_seconds"] = self.quiet_evicted_seconds
        return result

    def apply_packet_comma(self, result, previous_text):
        """Edit the packet's last commit before the next request builds its prompt."""
        marked = self.protocol.parse_reply(result["text"])
        corrected = replace_terminal_period(marked, previous_text)
        if marked == corrected:
            return False
        reply = self.protocol.format_text(corrected)
        result.update(text=reply, display_text=corrected.replace(SPK_CHANGE, ""),
                      terminal_period_to_comma=True)
        # Preserve each speaker-change flag computed at the original commit.
        result["transcript_parts"][-1]["text"] = corrected.rsplit(SPK_CHANGE, 1)[-1]
        self.committed_tail = (previous_text + corrected.replace(SPK_CHANGE, " "))[-2048:]
        for turn in reversed(self.history):
            if turn.start == result["start_seconds"]:
                turn.reply = reply
                break
        return True

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
