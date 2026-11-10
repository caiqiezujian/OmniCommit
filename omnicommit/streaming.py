"""Bounded, causal streaming state. No acoustic identity is guessed in Python.

DIRECT is a text commitment, not a trustworthy consumed-audio timestamp. Only
verified commit watermarks authorize eviction; an exhausted budget is explicit.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
import time

import numpy as np

from .runtime_protocol import (system_prompt, END_OF_AUDIO,
                               speaker_reference_text, LIVE_AUDIO_PREFIX,
                               speaker_registry_text, last_committed_text)

SAMPLE_RATE = 16000
WAIT = "<|WAIT|>"
DIRECT = "<|DIRECT|>"
HEADER = re.compile(r"<([a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*)><(spk[1-9][0-9]*)>")


class ContextBudgetError(RuntimeError):
    """No safe eviction remains. Do not drop pending speech and continue."""


def parse_reply(reply, allowed_tokens=None):
    if not isinstance(reply, str):
        raise ValueError("Model response must be text")
    if reply.strip() == WAIT:
        return {"action": "WAIT", "segments": [], "text": WAIT}
    # Discard wrapper whitespace only before the action. Trailing spaces and
    # newlines belong to incremental text (e.g. 'Please ' + 'confirm').
    reply = reply.lstrip()
    if not reply.startswith(DIRECT):
        raise ValueError("Expected exactly WAIT or DIRECT followed by attributed new text")
    body = reply[len(DIRECT):]
    matches = list(HEADER.finditer(body))
    if not matches or matches[0].start() != 0:
        raise ValueError("Every DIRECT needs an initial complete <language><speaker> header")
    segments = []
    for i, match in enumerate(matches):
        value = body[match.end():matches[i + 1].start() if i + 1 < len(matches) else len(body)]
        # Preserve transcript whitespace; literal newlines are not speaker tags.
        if not value.strip() or "<" in value or ">" in value:
            raise ValueError("Empty or malformed attributed segment")
        lang, speaker = match.groups()
        if allowed_tokens is not None and any(t not in allowed_tokens for t in
                                              ("<%s>" % lang, "<%s>" % speaker)):
            raise ValueError("Generated language/speaker tag is absent from the trained inventory")
        segments.append({"language": lang, "speaker": speaker, "text": value})
    return {"action": "DIRECT", "segments": segments, "text": reply}


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

    def __post_init__(self):
        if isinstance(self.chunk_seconds, bool) or self.chunk_seconds not in (.5, .8, 1.0):
            raise ValueError("chunk_seconds must be 0.5, 0.8 or 1.0")
        for key in ("recent_seconds", "max_context_seconds", "max_pending_seconds",
                    "max_reference_seconds", "max_total_reference_seconds", "rolling_tail_seconds"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(key + " must be positive and finite")
        if self.recent_seconds > self.max_context_seconds:
            raise ValueError("recent_seconds exceeds context budget")
        if type(self.max_speakers) is not int or self.max_speakers < 1:
            raise ValueError("max_speakers must be a positive integer")
        system_prompt(self.number_format)
        if self.context_policy not in ("verified", "rolling"):
            raise ValueError("context_policy must be verified or rolling")


@dataclass
class AudioTurn:
    start: float
    end: float
    audio: np.ndarray
    reply: str


@dataclass
class SpeakerReference:
    speaker: str
    language: str
    text: str
    start: float
    end: float
    audio: np.ndarray


class StreamSession:
    def __init__(self, backend, options=None):
        self.backend = backend
        self.options = options or StreamOptions()
        self.history = []
        self.references = {}
        self.speakers = []
        self.last_committed_language = None
        self.last_committed_speaker = None
        self.heard_seconds = 0.0
        self.verified_commit_seconds = 0.0
        self.inferred_commit_seconds = 0.0
        self.heuristic_eviction = False
        self.evicted_this_turn = False
        self.closed = False
        self.failed = False
        self.turn_count = 0
        self._short_tail = False

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

    def _context(self, current_audio=None, end_of_audio=False):
        messages = [{"role": "system", "content": system_prompt(self.options.number_format)}]
        messages[0]["content"] += speaker_registry_text(self.speakers)
        messages[0]["content"] += last_committed_text(self.last_committed_language,
                                                     self.last_committed_speaker)
        audios = []
        ref_prefix = ""
        for speaker in sorted(self.references, key=lambda name: int(name[3:])):
            ref = self.references[speaker]
            ref_prefix += speaker_reference_text(ref.speaker, ref.language, ref.text)
            audios.append(ref.audio)
        first = True
        for turn in self.history:
            content = "<audio>"
            if first and ref_prefix:
                content = ref_prefix + LIVE_AUDIO_PREFIX + content
            messages.extend([{"role": "user", "content": content},
                             {"role": "assistant", "content": turn.reply}])
            audios.append(turn.audio)
            first = False
        content = END_OF_AUDIO if end_of_audio else "<audio>"
        if first and ref_prefix:
            content = ref_prefix + LIVE_AUDIO_PREFIX + content
        messages.append({"role": "user", "content": content})
        if current_audio is not None:
            audios.append(current_audio)
        return messages, audios

    def _validate_speakers(self, segments):
        seen = list(self.speakers)
        for segment in segments:
            speaker = segment["speaker"]
            if speaker not in seen:
                if len(seen) >= self.options.max_speakers:
                    raise ValueError("Speaker registry budget exhausted; never recycle an ID")
                if speaker != "spk%d" % (len(seen) + 1):
                    raise ValueError("New speakers must use the next meeting-wide ID")
                seen.append(speaker)
        return seen

    def step(self, audio=None, *, end_of_audio=False):
        self._check_open()
        self.evicted_this_turn = False
        if end_of_audio:
            if audio is not None or not self.history:
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
            self._trim()
            commit = self.verified_commit_seconds
            if self.options.context_policy == "rolling":
                commit = max(commit, self.inferred_commit_seconds)
            pending = self.heard_seconds + duration - commit
            retained = duration + sum(t.end - t.start for t in self.history)
            if pending > self.options.max_pending_seconds + 1e-8:
                raise ContextBudgetError("Unverified pending audio exceeded budget; supply a reliable commit watermark")
            if retained > self.options.max_context_seconds + 1e-8:
                raise ContextBudgetError("Context audio budget exhausted; cannot safely evict uncommitted speech")
            # Match SFT's last short-tail padding without inventing source time.
            if len(pcm) < SAMPLE_RATE // 2:
                pcm = np.pad(pcm, (0, SAMPLE_RATE // 2 - len(pcm)))
        messages, audios = self._context(pcm, end_of_audio)
        started = time.perf_counter()
        try:
            generated = self.backend.generate(messages, audios)
            reply = generated["text"] if isinstance(generated, dict) else generated
            parsed = parse_reply(reply, getattr(self.backend, "allowed_tokens", None))
            new_speakers = self._validate_speakers(parsed["segments"])
        except Exception:
            # State remains uncommitted. No fabricated WAIT hides model errors.
            self.failed = True
            raise
        elapsed = time.perf_counter() - started
        start = self.heard_seconds
        self.speakers = new_speakers
        if parsed["segments"]:
            final_segment = parsed["segments"][-1]
            self.last_committed_language = final_segment["language"]
            self.last_committed_speaker = final_segment["speaker"]
        self.turn_count += 1
        self.heard_seconds += duration
        if parsed["action"] == "DIRECT" and self.options.context_policy == "rolling":
            # Explicit experimental heuristic, not evidence that all earlier
            # speech was emitted. A long-held entity can exceed this guard.
            self.inferred_commit_seconds = max(self.inferred_commit_seconds,
                self.heard_seconds - self.options.rolling_tail_seconds)
        if not end_of_audio:
            self.history.append(AudioTurn(start, self.heard_seconds, pcm, parsed["text"]))
            self._short_tail = duration < self.options.chunk_seconds - 1e-8
        else:
            self.closed = True
        return dict(parsed, turn=self.turn_count, event="end_of_audio" if end_of_audio else "audio",
                    start_seconds=start, end_seconds=self.heard_seconds,
                    generation_seconds=elapsed, realtime_factor=elapsed / duration if duration else None,
                    retained_audio_seconds=sum(t.end - t.start for t in self.history),
                    unverified_audio_seconds=self.heard_seconds - self.verified_commit_seconds,
                    reference_speakers=list(self.references),
                    context_policy=self.options.context_policy,
                    heuristic_eviction=self.heuristic_eviction,
                    evicted_this_turn=self.evicted_this_turn,
                    inferred_commit_seconds=self.inferred_commit_seconds,
                    prompt_tokens=generated.get("prompt_tokens") if isinstance(generated, dict) else None)

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

    def register_reference(self, speaker, language, text, start, end, *,
                           verified=False, clean=False, overlap=True):
        self._check_open()
        if verified is not True or clean is not True or overlap is not False:
            raise ValueError("Speaker reference requires verified, clean, nonoverlapping audio")
        if speaker not in self.speakers:
            raise ValueError("Reference speaker must already occur in this meeting")
        parse_reply(DIRECT + "<%s><%s>%s" % (language, speaker, text),
                    getattr(self.backend, "allowed_tokens", None))
        speaker_reference_text(speaker, language, text)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in (start, end)) or not 0 <= start < end <= self.heard_seconds:
            raise ValueError("Reference must be entirely inside already heard audio")
        if end - start > self.options.max_reference_seconds:
            raise ValueError("Reference exceeds per-speaker duration budget")
        if end - start < .5:
            raise ValueError("Reference shorter than 0.5s is not admitted")
        total = sum(r.end - r.start for s, r in self.references.items() if s != speaker) + end - start
        if total > self.options.max_total_reference_seconds:
            raise ValueError("Total speaker reference audio budget exceeded")
        pieces, cursor = [], start
        for turn in self.history:
            low, high = max(start, turn.start), min(end, turn.end)
            if high <= low:
                continue
            if abs(low - cursor) > 1 / SAMPLE_RATE:
                raise ValueError("Requested reference audio was already evicted")
            pieces.append(turn.audio[round((low-turn.start)*SAMPLE_RATE):round((high-turn.start)*SAMPLE_RATE)])
            cursor = high
        if abs(cursor - end) > 1 / SAMPLE_RATE:
            raise ValueError("Reference audio is unavailable; register before eviction")
        self.references[speaker] = SpeakerReference(speaker, language, text, start, end,
                                                    np.concatenate(pieces).copy())

    def close(self):
        self.closed = True
        self.history.clear()
        self.references.clear()


class ReplayBackend:
    """Scripted transport/state test. Not an ASR model and never an evaluation."""
    def __init__(self, replies=None):
        self.replies = iter(replies) if replies is not None else None
        self.allowed_tokens = None

    def generate(self, messages, audios):
        return {"text": next(self.replies) if self.replies is not None else WAIT,
                "prompt_tokens": None}
