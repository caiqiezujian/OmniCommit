"""Project a potentially clipped sentence ending onto the training target."""
from dataclasses import replace
import re
import unicodedata

from .semantic_preparation import ABBREVIATION, TRAILING


def simple_period_offset(text):
    """Return the offset of one terminal period, preserving quotes and whitespace."""
    end = len(text)
    while end and (text[end-1].isspace() or text[end-1] in TRAILING):
        end -= 1
    if end < 2 or text[end-1] not in ".。．":
        return None
    if not (text[end-2].isalnum() or unicodedata.category(text[end-2]).startswith("M")):
        return None
    if text[end-1] == ".":
        stem = text[:end]
        if ABBREVIATION.search(stem) or re.search(
                r"\b(?:[a-z]|etc|inc|ltd|jr|sr|sra|mme|mlle|art|sig|nr|bzw)\.$", stem, re.I):
            return None
    return end - 1


def project_open_tail(units, final_state_text, expected):
    """Change at most one character in the final attributed run of an intact reference."""
    offset = simple_period_offset(final_state_text)
    if offset is None:
        return units, expected, False
    distance = len(final_state_text) - offset
    target_index = len(expected) - distance
    if expected[target_index] != final_state_text[offset]:
        raise ValueError("Final attributed run disagrees with rendered reference")
    for i in range(len(units)-1, -1, -1):
        unit = units[i]
        if distance > len(unit.text):
            distance -= len(unit.text)
            continue
        spans = list(unit.spans)
        for j in range(len(spans)-1, -1, -1):
            lang, speaker, body = spans[j]
            if distance > len(body):
                distance -= len(body)
                continue
            k = len(body) - distance
            if body[k] != final_state_text[offset]:
                raise ValueError("Terminal period does not match timed text")
            new_text = body[:k] + body[k+1:]
            spans[j] = (lang, speaker, new_text)
            projected = list(units)
            if any(text.strip() for _, _, text in spans):
                end = unit.end_ms
                # Step 1 can already have attached a separately timed period.
                # In that case use the remaining children's actual end times.
                if (len(unit.children) >= 2 and j == len(spans)-1
                        and unit.children[-1].text.strip() == body[k]
                        and not body[k+1:].strip()):
                    end = max(child.end_ms for child in unit.children[:-1])
                projected[i] = replace(unit, spans=tuple(spans), end_ms=end, children=())
            else:
                # A separately timestamped period contributes no trainable speech.
                if i == 0 or units[i-1].spans[-1][:2] != (lang, speaker):
                    return units, expected, False
                del projected[i]
                if new_text:
                    prev = projected[i-1]
                    a, b, t = prev.spans[-1]
                    projected[i-1] = replace(prev, spans=(*prev.spans[:-1], (a, b, t + new_text)), children=())
            return projected, expected[:target_index] + expected[target_index+1:], True
    raise ValueError("Terminal period is outside the supervised transcript")
