"""Place completed transcript units onto the actual source audio sample grid."""
from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

from .protocol import EOS, WAIT, render_reply, system_prompt
from .source import Unit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "SFT-Scripts"))
from omnicommit_audio import MAX_CHUNK_SECONDS, encode_chunk_ref


def validate_chunk_ms(chunk_ms: int) -> None:
    if type(chunk_ms) is not int or not 1 <= chunk_ms <= MAX_CHUNK_SECONDS * 1000:
        raise ValueError(f"chunk-ms must be an integer between 1 and {MAX_CHUNK_SECONDS * 1000}")


def _end_frame(end_ms: float, frames: int, rate: int) -> int:
    if end_ms == frames * 1000 / rate:
        return frames
    frame = Fraction(str(end_ms)) * rate / 1000
    return -(-frame.numerator // frame.denominator)


def build_sft_row(units: list[Unit], path: Path, frames: int, rate: int,
                  chunk_ms: int, number_format: str = "normalized") -> dict:
    validate_chunk_ms(chunk_ms)
    if not units or frames <= 0 or rate <= 0:
        raise ValueError("A record needs positive-duration audio and aligned text")
    if chunk_ms * rate < 1000:
        raise ValueError("The selected chunk size contains less than one source sample")
    ends = [_end_frame(unit.end_ms, frames, rate) for unit in units]
    if ends[-1] > frames or any(b < a for a, b in zip(ends, ends[1:])):
        raise ValueError("Unit ends must follow transcript order and fit the audio")

    refs = []
    messages = [{"role": "system", "content": system_prompt(number_format)}]
    cursor, start, turn = 0, 0, 1
    state = None
    while start < frames:
        # Round absolute grid boundaries up to whole source samples. Each sample
        # belongs to exactly one turn, including the last incomplete grid cell.
        end = min(frames, (turn * chunk_ms * rate + 999) // 1000)
        spans = []
        while cursor < len(units) and ends[cursor] <= end:
            spans.extend(units[cursor].spans)
            cursor += 1
        reply, state = render_reply(spans, state)
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
    messages.extend([{"role": "user", "content": EOS}, {"role": "assistant", "content": WAIT}])
    return {"audios": refs, "messages": messages}
