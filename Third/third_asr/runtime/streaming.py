"""Bounded, causal streaming state for the Third web service."""
from __future__ import annotations

from dataclasses import dataclass
import math
import sys
import time

import numpy as np

from ..protocol import EOS, WAIT

SAMPLE_RATE = 16000
MAX_CHUNK_MS = 8000


def chunk_milliseconds(seconds):
    if type(seconds) not in (int, float) or not .001 <= seconds <= MAX_CHUNK_MS / 1000:
        raise ValueError(f"chunk_seconds must represent 1 to {MAX_CHUNK_MS} milliseconds")
    ms = round(seconds * 1000)
    if ms / 1000 != seconds:
        raise ValueError("chunk_seconds must be a whole number of milliseconds")
    return ms


class ContextBudgetError(RuntimeError):
    """No safe eviction remains. Do not drop pending speech and continue."""


class GenerationTruncated(RuntimeError):
    """A single decode exhausted its length budget before the reply completed."""

    def __init__(self, backend, generated_tokens, prompt_tokens, max_new_tokens, max_context_tokens):
        self.backend = backend
        self.generated_tokens = generated_tokens
        self.prompt_tokens = prompt_tokens
        self.max_new_tokens = max_new_tokens
        self.max_context_tokens = max_context_tokens
        super().__init__(
            f"{backend} generation truncated (generated_tokens={generated_tokens}, "
            f"max_new_tokens={max_new_tokens}, prompt_tokens={prompt_tokens}, "
            f"max_context_tokens={max_context_tokens}). Partial output discarded."
        )

    def as_dict(self):
        return {"backend": self.backend, "generated_tokens": self.generated_tokens,
                "prompt_tokens": self.prompt_tokens, "max_new_tokens": self.max_new_tokens,
                "max_context_tokens": self.max_context_tokens}


@dataclass
class StreamOptions:
    chunk_seconds: float = .5
    recent_seconds: float = 12.0
    max_context_seconds: float = 30.0
    max_pending_seconds: float = 30.0
    max_speakers: int = 16
    max_reference_seconds: float = 5.0
    max_total_reference_seconds: float = 30.0
    number_format: str = "verbatim"
    context_policy: str = "verified"
    rolling_tail_seconds: float = 2.0
    window_seconds: float = 60.0
    discard_seconds: float = 50.0

    def __post_init__(self):
        chunk_milliseconds(self.chunk_seconds)
        for key in ("recent_seconds", "max_context_seconds", "max_pending_seconds",
                    "max_reference_seconds", "max_total_reference_seconds", "rolling_tail_seconds"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(key + " must be positive and finite")
        if self.recent_seconds > self.max_context_seconds:
            raise ValueError("recent_seconds exceeds context budget")
        if type(self.max_speakers) is not int or self.max_speakers < 1:
            raise ValueError("max_speakers must be a positive integer")
        if self.number_format not in ("normalized", "verbatim"):
            raise ValueError("number_format must be normalized or verbatim")
        if self.context_policy not in ("verified", "rolling", "window"):
            raise ValueError("context_policy must be verified, rolling or window")
        if self.context_policy == "window":
            for key in ("window_seconds", "discard_seconds"):
                value = getattr(self, key)
                if type(value) not in (float, int) or not 0 < value <= sys.float_info.max:
                    raise ValueError(key + " must be positive and finite")
            if self.chunk_seconds > self.window_seconds:
                raise ValueError("window_seconds must be at least chunk_seconds")
            if self.discard_seconds > self.window_seconds:
                raise ValueError("discard_seconds must not exceed window_seconds")


@dataclass
class AudioTurn:
    start: float
    end: float
    audio: np.ndarray
    reply: str


class StreamSession:
    retryable_errors = ()

    def __init__(self, backend, options=None):
        self.backend = backend
        self.options = options or StreamOptions()
        self.history = []
        self.heard_seconds = 0.0
        self.verified_commit_seconds = 0.0
        self.inferred_commit_seconds = 0.0
        self.heuristic_eviction = False
        self.evicted_this_turn = False
        self.closed = False
        self.failed = False
        self.turn_count = 0
        self._short_tail = False
        self._last_gap_end = 0.0
        self.last_text_end = 0.0

    def _check_open(self):
        if self.closed or self.failed:
            raise RuntimeError("Session is closed or failed; start a new session")

    def _trim(self):
        # Audio newer than the verified watermark is never removed, even after
        # DIRECT. Whole audio/assistant pairs remain indivisible.
        commit = self.verified_commit_seconds
        if self.options.context_policy == "rolling":
            commit = max(commit, self.inferred_commit_seconds)
        cutoff = min(commit,
                     self.heard_seconds - self.options.recent_seconds)
        while self.history and self.history[0].end <= cutoff + 1e-8:
            if self.history[0].end > self.verified_commit_seconds:
                self.heuristic_eviction = True
                self.evicted_this_turn = True
            self.history.pop(0)

    def _context_system_prompt(self):
        raise NotImplementedError

    def _model_reply(self, reply, end_of_audio):
        raise NotImplementedError

    def _context(self, history, current_audio=None, end_of_audio=False):
        messages = [{"role": "system", "content": self._context_system_prompt()}]
        audios = []
        for turn in history:
            messages.extend([{"role": "user", "content": "<audio>"},
                             {"role": "assistant", "content": turn.reply}])
            audios.append(turn.audio)
        messages.append({"role": "user", "content": EOS if end_of_audio else "<audio>"})
        if current_audio is not None:
            audios.append(current_audio)
        return messages, audios

    def _minimum_input_samples(self, final_chunk):
        return SAMPLE_RATE // 2 if final_chunk else SAMPLE_RATE // 100

    def _pending_seconds(self, commit):
        return self.heard_seconds - max(commit, self._last_gap_end)

    def _generate_turn(self, history, pcm, end_of_audio):
        messages, audios = self._context(history, pcm, end_of_audio)
        generated = self.backend.generate(messages, audios)
        reply = generated["text"] if isinstance(generated, dict) else generated
        return generated, self._model_reply(reply, end_of_audio)

    def _decode_turn(self, history, pcm, end_of_audio):
        try:
            generated, parsed = self._generate_turn(history, pcm, end_of_audio)
            return generated, parsed, history
        except GenerationTruncated as first:
            diagnostic = {"status": "gap", "primary": first.as_dict()}

        # Change the greedy prompt for at most one retry; retain trailing WAIT
        # audio. A DIRECT is a conversational anchor, not a verified watermark.
        for index in range(len(history) - 1, -1, -1):
            if history[index].reply == WAIT:
                continue
            # When the anchor is already first, drop just the anchor so the
            # request differs from the one that just exhausted its budget.
            suffix = history[max(1, index):]
            if end_of_audio and not suffix:
                break
            diagnostic["retry_history_turns"] = len(suffix)
            try:
                generated, parsed = self._generate_turn(suffix, pcm, end_of_audio)
            except GenerationTruncated as second:
                diagnostic["retry"] = second.as_dict()
                break
            diagnostic["status"] = "recovered"
            parsed["truncation"] = diagnostic
            # Extend the successful prompt on the next windowed decode. Other
            # policies retain their established watermark-based eviction rules.
            return generated, parsed, suffix if self.options.context_policy == "window" else history

        diagnostic["start_seconds"] = max(self.last_text_end, self._last_gap_end, self.verified_commit_seconds)
        return None, {"action": None, "text": "", "display_text": "", "segments": [], "truncation": diagnostic}, history

    def _history_for_audio(self, duration):
        if self.options.context_policy == "window":
            end = self.heard_seconds + duration
            if self.history and end - self.history[0].start > self.options.window_seconds + 1e-8:
                # Both targets round up to a whole audio/assistant pair. An
                # unusually small discard must still leave room for this chunk.
                cutoff = max(self.history[0].start + self.options.discard_seconds,
                             end - self.options.window_seconds)
                for index, turn in enumerate(self.history, 1):
                    if turn.end + 1e-8 >= cutoff:
                        return self.history[index:]
                return []
            return self.history
        self._trim()
        commit = self.verified_commit_seconds
        if self.options.context_policy == "rolling":
            commit = max(commit, self.inferred_commit_seconds)
        pending = self._pending_seconds(commit) + duration
        retained = duration + sum(t.end - t.start for t in self.history)
        if pending > self.options.max_pending_seconds + 1e-8:
            raise ContextBudgetError(
                f"Unverified pending audio exceeded budget ({pending:.2f}s > "
                f"{self.options.max_pending_seconds:g}s; {self.options.context_policy} policy, "
                f"commit watermark {commit:.2f}s, attempted audio end {self.heard_seconds + duration:.2f}s)"
            )
        if retained > self.options.max_context_seconds + 1e-8:
            raise ContextBudgetError("Context audio budget exhausted; cannot safely evict uncommitted speech")
        return self.history

    def step(self, audio=None, *, end_of_audio=False, final_chunk=False):
        self._check_open()
        self.evicted_this_turn = False
        history = self.history
        previous_start = history[0].start if history else self.heard_seconds
        if end_of_audio:
            if audio is not None or (not self.history and not self._last_gap_end):
                raise ValueError("EOS is text-only and requires preceding audio")
            duration, pcm = 0.0, None
        else:
            if self._short_tail:
                raise ValueError("A partial final chunk must be followed by EOS")
            pcm = np.array(audio, dtype=np.float32, copy=True)
            if pcm.ndim != 1 or not len(pcm) or not np.isfinite(pcm).all():
                raise ValueError("Audio must be finite, nonempty, mono float32 at 16000 Hz")
            if len(pcm) > round(self.options.chunk_seconds * SAMPLE_RATE):
                raise ValueError("Audio exceeds configured chunk step")
            if np.max(np.abs(pcm)) > 1.001:
                raise ValueError("Audio samples must be in [-1,1]")
            duration = len(pcm) / SAMPLE_RATE
            history = self._history_for_audio(duration)
            # Match SFT's last short-tail padding without inventing source time.
            short_chunk = len(pcm) < round(self.options.chunk_seconds * SAMPLE_RATE)
            minimum = self._minimum_input_samples(final_chunk or short_chunk)
            if len(pcm) < minimum:
                pcm = np.pad(pcm, (0, minimum - len(pcm)))
        started = time.perf_counter()
        try:
            if end_of_audio and not history:
                # The final audio already has a reported gap; there is no audio
                # prompt left to send. Finish without fabricating an EOS reply.
                generated = None
                parsed = {"action": None, "text": "", "display_text": "", "segments": [],
                          "end_reason": "gap_already_reported"}
            else:
                generated, parsed, history = self._decode_turn(history, pcm, end_of_audio)
        except Exception as exc:
            # State remains uncommitted. No fabricated WAIT hides model errors.
            self.failed = not isinstance(exc, self.retryable_errors)
            raise
        elapsed = time.perf_counter() - started
        start = self.heard_seconds
        if history is not self.history:
            # Only commit a proposed window rotation once the reply is valid.
            self.history = history
            self.evicted_this_turn = True
        self.turn_count += 1
        self.heard_seconds += duration
        if parsed["action"] == "DIRECT" and self.options.context_policy == "rolling":
            # Explicit experimental heuristic, not evidence that all earlier
            # speech was emitted. A long-held entity can exceed this guard.
            self.inferred_commit_seconds = max(self.inferred_commit_seconds,
                self.heard_seconds - self.options.rolling_tail_seconds)
        if parsed.get("truncation", {}).get("status") == "gap":
            parsed["truncation"]["end_seconds"] = self.heard_seconds
            self._last_gap_end = self.heard_seconds
            self.history.clear()
            self.evicted_this_turn = True
        elif not end_of_audio:
            self.history.append(AudioTurn(start, self.heard_seconds, pcm, parsed["text"]))
        if not end_of_audio:
            self._short_tail = short_chunk or final_chunk
        else:
            self.closed = True
        if self.options.context_policy == "window":
            head = self.history[0].start if self.history else self.heard_seconds
            policy_stats = dict(window_start_seconds=head, discarded_seconds=head - previous_start)
        else:
            policy_stats = dict(unverified_audio_seconds=self.heard_seconds - self.verified_commit_seconds,
                                heuristic_eviction=self.heuristic_eviction, inferred_commit_seconds=self.inferred_commit_seconds)
        return dict(parsed, turn=self.turn_count, event="end_of_audio" if end_of_audio else "audio",
                    start_seconds=start, end_seconds=self.heard_seconds,
                    generation_seconds=elapsed, realtime_factor=elapsed / duration if duration else None,
                    retained_audio_seconds=sum(t.end - t.start for t in self.history),
                    reference_speakers=[],
                    context_policy=self.options.context_policy,
                    evicted_this_turn=self.evicted_this_turn,
                    **policy_stats,
                    prompt_tokens=generated.get("prompt_tokens") if isinstance(generated, dict) else None,
                    num_cached_tokens=generated.get("num_cached_tokens") if isinstance(generated, dict) else None)

    def verify_commit(self, through_seconds, *, verified=False):
        """Trusted aligner/operator API, never inferred from DIRECT alone.

        The caller certifies ALL speech through this position has been committed,
        not merely that one later word aligned there. In particular no held name
        or numerical expression may remain before the watermark.
        """
        self._check_open()
        if verified is not True:
            raise ValueError("A commit watermark requires explicit reliable verification")
        if (isinstance(through_seconds, bool) or not isinstance(through_seconds, (int, float))
                or not math.isfinite(through_seconds)
                or not self.verified_commit_seconds <= through_seconds <= self.heard_seconds):
            raise ValueError("Commit watermark must be causal and monotonic")
        self.verified_commit_seconds = float(through_seconds)

    def close(self):
        self.closed = True
        self.history.clear()


class ReplayBackend:
    """Scripted transport/state test. Not an ASR model and never an evaluation."""
    def __init__(self, replies=None):
        self.replies = iter(replies) if replies is not None else None
        self.allowed_tokens = None

    def generate(self, messages, audios):
        return {"text": next(self.replies) if self.replies is not None else WAIT,
                "prompt_tokens": None}

    def generate_with_prefix(self, messages, audios, prefix):
        # Each scripted entry represents one entire backend response, including
        # the supplied prefix for a successful recovery.
        return self.generate(messages, audios)
