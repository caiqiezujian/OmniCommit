"""Bounded, session-local audio evidence for a missing stretch of transcription."""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
import unicodedata

import numpy as np


RATE = 16_000
FRAME = 320
LATE_MAIN_SECONDS = 4.


@dataclass(frozen=True)
class GapRequest:
    audio: np.ndarray
    start_seconds: float
    end_seconds: float
    first_turn: int
    last_turn: int
    reason: str
    lead_text: str
    lead_end_seconds: float
    eligible_after_seconds: float


@dataclass
class _Packet:
    start: float
    pcm: np.ndarray
    turn: int

    @property
    def end(self):
        return self.start + len(self.pcm) / RATE

    def tail(self, seconds):
        n = min(len(self.pcm), round(seconds * RATE))
        return _Packet(self.end - n / RATE, self.pcm[-n:].copy(), self.turn)


def _activity(pcm):
    """A conservative energy gate (20 ms frames); it does not establish language."""
    pad = (-len(pcm)) % FRAME
    frames = np.pad(pcm, (0, pad)).reshape(-1, FRAME)
    # Ignore a DC offset. Correctly center the short last frame before padding.
    if pad:
        frames[-1, -pad:] = frames[-1, :-pad].mean()
    centered = frames - frames.mean(axis=1, keepdims=True)
    active = np.sqrt(np.mean(centered * centered, axis=1)) >= .007
    counts = np.full(len(frames), FRAME)
    counts[-1] -= pad
    ends = np.cumsum(counts)
    last = int(ends[active][-1]) if np.any(active) else 0
    return int(counts[active].sum()) / RATE, last / RATE


class GapTracker:
    """Observe the main stream in order; each request owns its PCM snapshot."""
    def __init__(self, wait_seconds=1.6):
        if not math.isfinite(wait_seconds) or not 0 < wait_seconds <= 30:
            raise ValueError("wait_seconds must be in (0, 30]")
        self.wait_seconds = wait_seconds
        self._lead = None
        self._lead_text = ""
        self._language = None
        self._pending = []
        self._voiced = 0.
        self._last_voice = None
        self._eligible = 0.

    def _reset(self):
        self._pending.clear()
        self._voiced = 0.
        self._last_voice = None

    def _request(self, reason):
        if not self._pending:
            return None
        if self._voiced + 1e-8 < min(.10, self.wait_seconds):
            self._lead = self._pending[-1].tail(.35)
            self._lead_text = ""
            self._reset()
            return None
        packets = ([self._lead] if self._lead else []) + self._pending
        audio = np.concatenate([p.pcm for p in packets])
        audio.setflags(write=False)
        first, last = packets[0], packets[-1]
        request = GapRequest(audio, first.start, last.end, first.turn, last.turn,
                             reason, self._lead_text,
                             self._lead.end if self._lead else first.start, self._eligible)
        self._eligible = last.end
        self._lead = last.tail(.50)
        self._lead_text = ""
        self._reset()
        return request

    def _direct_tail(self, reason):
        # A successful DIRECT can cover only the earlier part of the packet.
        # Probe only at a pause/end, once, when its last 0.5s contained speech.
        if self._pending or self._lead is None or not self._lead_text:
            return None
        packet = self._lead
        lead_text, self._lead_text = self._lead_text, ""
        if (packet.end <= self._eligible + 1e-7
                or _activity(packet.tail(.5).pcm)[0] < .12):
            return None
        audio = packet.pcm.copy()
        audio.setflags(write=False)
        request = GapRequest(audio, packet.start, packet.end, packet.turn, packet.turn,
                             reason, lead_text, packet.end, self._eligible)
        self._eligible = packet.end
        return request

    def finish(self):
        return self._request("end_of_audio") if self._pending else self._direct_tail("direct_end")

    def observe(self, pcm, result):
        if pcm is None:
            return self.finish()
        samples = np.asarray(pcm, dtype=np.float32)
        if samples.ndim != 1 or not len(samples):
            raise ValueError("Expected one nonempty 16 kHz PCM packet")
        packet = _Packet(float(result["start_seconds"]), samples.copy(), result["turn"])
        action = result["action"]
        if action == "DIRECT":
            spans = result["segments"]
            new_lang = spans[0]["language"]
            req = None
            if self._pending and self._language is not None and new_lang != self._language:
                # A short voiced WAIT followed by another language may be an insertion.
                # Retain this packet as right context in case the missing word spans it.
                self._pending.append(packet)
                req = self._request("language_change")
            self._reset()
            self._language = spans[-1]["language"]
            self._lead_text = "".join(span["text"] for span in spans)
            self._lead = packet.tail(1.)
            self._eligible = req.end_seconds if req is not None else self._lead.start
            return req
        if action != "WAIT":
            raise ValueError("Unexpected streaming action")
        voiced, voice_end = _activity(samples)
        if not self._pending and not voiced:
            # No speech in this complete packet; do not bridge it with stale PCM.
            req = self._direct_tail("direct_pause")
            self._lead = packet.tail(.35)
            self._lead_text = ""
            return req
        self._pending.append(packet)
        self._voiced += voiced
        if voice_end:
            self._last_voice = packet.start + voice_end
        if self._voiced + 1e-8 >= self.wait_seconds:
            return self._request("voiced_wait")
        if self._last_voice is not None and packet.end - self._last_voice >= .25:
            return self._request("pause")
        # Also bound sparse noise that only intermittently crosses the gate.
        if packet.end - self._pending[0].start >= self.wait_seconds + 2:
            return self._request("window_limit")
        return None


# Fold case and width for matching, but keep original indices and emitted text.
_TOKEN = re.compile(
    r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]|"
    r"(?:(?![\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af])[^\W_])+",
    re.UNICODE)


def _tokens(text):
    return [(unicodedata.normalize("NFKC", m.group()).casefold(), m.start(), m.end())
            for m in _TOKEN.finditer(text)]


@dataclass
class _Unit:
    value: str
    owner: int
    left: int
    right: int
    start: float
    end: float
    word_index: int


def _units(spans, request):
    units = []
    for i, span in enumerate(spans):
        tokens = _tokens(span["text"])
        words = span.get("words", ())
        if words:
            aligned = [(value, w["start_seconds"], w["end_seconds"], j)
                       for j, w in enumerate(words) for value, _, _ in _tokens(w["text"])]
            if [v[0] for v in aligned] != [t[0] for t in tokens]:
                raise ValueError("Word timestamps do not match recovery text")
        else:
            aligned = [(t[0], span.get("start_seconds", request.start_seconds),
                        span.get("end_seconds", request.end_seconds), -1) for t in tokens]
        for (token, left, right), (_, start, end, j) in zip(tokens, aligned):
            if not (math.isfinite(start) and math.isfinite(end) and start <= end):
                raise ValueError("Invalid recovery timestamps")
            units.append(_Unit(token, i, left, right, start, end, j))
    return units


def _overlap(left, right):
    """Length of the exact normalized suffix of left that begins right."""
    for count in range(min(len(left), len(right)), 0, -1):
        if left[-count:] == right[:count]:
            return count
    return 0


def _slice_spans(spans, units, low, high):
    selected = units[low:high]
    pieces = []
    for owner in dict.fromkeys(u.owner for u in selected):
        kept = [u for u in selected if u.owner == owner]
        first, last = kept[0], kept[-1]
        span = spans[owner]
        source = span["text"]
        left = first.left
        # Preserve original joining whitespace and capitalization.
        while left and source[left-1].isspace():
            left -= 1
        previous = any(u.owner == owner for u in units[:low])
        if not previous:
            left = 0
        following = next((u for u in units[high:] if u.owner == owner), None)
        right = following.left if following else len(source)
        if following:
            while right > last.right and source[right-1].isspace():
                right -= 1
        piece = dict(span, text=source[left:right],
                     start_seconds=first.start, end_seconds=last.end)
        if span.get("words"):
            piece["words"] = span["words"][first.word_index:last.word_index+1]
        pieces.append(piece)
    return pieces


def reconcile(request, spans, main_turns=()):
    """Remove already covered overlap at this window's edges, preserving attribution.

    All span and optional word timestamps are absolute session seconds.
    A decoder's clip-local speaker ID must never be mapped to a meeting ID here.
    """
    units = _units(spans, request)
    values = [u.value for u in units]
    low = 0
    while low < len(units) and units[low].end <= request.eligible_after_seconds + 1e-7:
        low += 1
    high = len(units)
    lead = [v for v, _, _ in _tokens(request.lead_text)]
    if lead:
        # Identical words spoken after the lead interval are legitimate repetitions.
        head = low
        while head < high and units[head].start < request.lead_end_seconds:
            head += 1
        lead_overlap = _overlap(lead, values[low:head])
        if request.reason in ("direct_pause", "direct_end"):
            # This whole interval was possibly transcribed. Never emit a competing
            # paraphrase or translation for already published text. Accept only
            # a timed suffix after we locate the published words inside the clip.
            if not lead_overlap:
                return []
            if low + lead_overlap < high:
                last_known, next_word = units[low + lead_overlap-1:low + lead_overlap+1]
                if next_word.start < last_known.end - .04:
                    return []
        low += lead_overlap
    before = after = None
    for turn in main_turns:
        n = turn["turn"]
        if n < request.first_turn or turn["action"] != "DIRECT":
            continue
        if request.lead_text and n == request.first_turn:
            continue
        post_window = n > request.last_turn
        delayed = n > request.last_turn + 1
        if post_window and turn["end_seconds"] - request.end_seconds > LATE_MAIN_SECONDS + 1e-7:
            break
        segments = [(_tokens(s["text"]), i) for i, s in enumerate(turn["segments"])]
        emitted = [v for tokens, _ in segments for v, _, _ in tokens]
        if low >= high:
            return []
        remaining = values[low:high]
        # A lone "yes" or "Hi" could easily be a new utterance after more WAITs.
        min_match = 2 if delayed else 1
        if min_match <= len(remaining) <= len(emitted) and any(
                emitted[k:k+len(remaining)] == remaining
                for k in range(len(emitted) - len(remaining) + 1)):
            return []
        match = _overlap(emitted, remaining)
        if match >= min_match:
            low += match
            after = n, next(i for tokens, i in reversed(segments) if tokens)
        else:
            right_match = _overlap(remaining, emitted)
            if right_match >= min_match:
                high -= right_match
                if (request.reason == "language_change" and n == request.last_turn
                        and low < high and units[high-1].end <= units[high].start + .04):
                    first_index = next(i for tokens, i in segments if tokens)
                    before = n, first_index
        if post_window:
            break  # Only the first subsequent emission can plausibly discharge this gap.
    pieces = _slice_spans(spans, units, low, high)
    for piece in pieces:
        if before is not None:
            piece["sort_before_turn"], piece["sort_before_segment"] = before
        if after is not None:
            piece["sort_after_turn"], piece["sort_after_segment"] = after
    return pieces
