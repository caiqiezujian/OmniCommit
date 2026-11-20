"""Place completed transcript units onto the actual source audio sample grid."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from .chunk_refs import MAX_CHUNK_SECONDS, encode_chunk_ref
from .protocol import EOS, WAIT, FourthProtocol
from .semantic_preparation import STANDALONE, closes_clause, unfinished_tail
from .timestamped_source import Unit


def validate_chunk_ms(chunk_ms: int) -> None:
    if type(chunk_ms) is not int or not 1 <= chunk_ms <= MAX_CHUNK_SECONDS * 1000:
        raise ValueError(f"chunk-ms must be an integer between 1 and {MAX_CHUNK_SECONDS * 1000}")


def end_frame(end_ms: float, frames: int, rate: int) -> int:
    if end_ms == frames * 1000 / rate:
        return frames
    frame = Fraction(str(end_ms)) * rate / 1000
    return -(-frame.numerator // frame.denominator)


def _switch_cuts(units, ends, frames, rate, chunk_ms):
    """First switch with old and new annotated speech inside the same real grid cell."""
    cuts = {}
    covered_until = units[0].end_ms
    for i in range(1, len(units)):
        old, new = units[i-1], units[i]
        overlaps = covered_until > new.start_ms
        covered_until = max(covered_until, new.end_ms)
        if old.spans[-1][:2] == new.spans[0][:2] or overlaps:
            continue
        new_start = end_frame(new.start_ms, frames, rate)
        turn = new_start * 1000 // (chunk_ms * rate) + 1
        window_start = ((turn-1) * chunk_ms * rate + 999) // 1000
        if ends[i-1] > window_start:
            cuts.setdefault(turn, i)
    return cuts


def _punctuation_cut(units, cursor, ready, protected, chunk_ms):
    """Delay a fresh unfinished suffix, only when an earlier clause can be published."""
    if ready <= cursor+1 or ready == len(units) or ready <= protected:
        return ready
    last, following = units[ready-1], units[ready]
    if (len(last.spans) != 1 or last.spans[-1][:2] != following.spans[0][:2]
            or following.start_ms - last.end_ms > chunk_ms
            or last.text.strip().casefold() in STANDALONE or closes_clause(last.text, following.text)):
        return ready
    if last.phrase_end and not unfinished_tail(last, following):
        return ready
    for i in range(ready-1, max(cursor+1, protected)-1, -1):
        if (units[i].spans[0][:2] != last.spans[0][:2] or len(units[i].spans) != 1
                or last.end_ms - units[i].start_ms > chunk_ms):
            break
        # A decimal comma and an abbreviation period are not clause boundaries.
        if closes_clause(units[i-1].text, units[i].text):
            return i
    return ready


def build_sft_row(units: list[Unit], path: Path, frames: int, rate: int,
                  chunk_ms: int, number_format: str, *, protocol: FourthProtocol, audit=None,
                  release_completed=False, open_end=False) -> dict:
    validate_chunk_ms(chunk_ms)
    if not units or frames <= 0 or rate <= 0:
        raise ValueError("A record needs positive-duration audio and aligned text")
    if chunk_ms * rate < 1000:
        raise ValueError("The selected chunk size contains less than one source sample")
    ends = [end_frame(unit.end_ms, frames, rate) for unit in units]
    if any(end > frames for end in ends):
        raise ValueError("Unit ends must fit the audio")
    defer = protocol.name == "speaker" and not release_completed
    cuts = _switch_cuts(units, ends, frames, rate, chunk_ms) if defer else {}
    if audit is not None:
        audit.update(switch_deferred_turns=[], punctuation_deferred_turns=[])

    refs = []
    messages = [{"role": "system", "content": protocol.system_prompt(number_format)}]
    cursor, start, turn = 0, 0, 1
    ready = 0
    state = None
    while start < frames:
        # Round absolute grid boundaries up to whole source samples. Each sample
        # belongs to exactly one turn, including the last incomplete grid cell.
        end = min(frames, (turn * chunk_ms * rate + 999) // 1000)
        # Prior ready text is due now, including on silent turns. Prefix order
        # also keeps a shorter, overlapping interjection behind its parent turn.
        previously_ready = ready
        while ready < len(units) and ends[ready] <= end:
            ready += 1
        cutoff = ready
        if end < frames and defer:
            switch = cuts.get(turn)
            if switch is not None and max(cursor+1, previously_ready) <= switch <= ready:
                cutoff = switch
                reason = "switch_deferred_turns"
            else:
                cutoff = _punctuation_cut(units, cursor, ready, previously_ready, chunk_ms)
                reason = "punctuation_deferred_turns"
            if cutoff < ready and audit is not None:
                audit[reason].append(turn)
        spans = [span for unit in units[cursor:cutoff] for span in unit.spans]
        cursor = cutoff
        reply, state = protocol.render_reply(spans, state)
        span_seconds = (end - start) / rate
        # Tail padding is generated after cropping. It cannot release more text.
        input_seconds = max(span_seconds, 0.5 if end == frames else 0.01)
        refs.append(encode_chunk_ref(str(path), start / rate, end / rate, input_seconds))
        messages.extend([{"role": "user", "content": "<audio>"},
                         {"role": "assistant", "content": reply}])
        start = end
        turn += 1
    if cursor != len(units):
        raise ValueError("Audio ended with uncommitted transcript units")
    if not open_end:
        messages.extend([{"role": "user", "content": EOS}, {"role": "assistant", "content": WAIT}])
    return {"audios": refs, "messages": messages}
