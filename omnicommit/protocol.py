"""Canonical, lossless representation of language/speaker attributed text.

The LSP comparison operates on (language, speaker, character) units, not on
literal header strings: segment boundaries alone must not cause a disagreement.
Speaker IDs are deliberately never reassigned based on guessed identities.
"""
from __future__ import annotations

import re

LANGUAGE = re.compile(r"[a-z]{2,3}(?:-[a-z0-9]{2,8})*\Z")
SPEAKER = re.compile(r"spk[1-9][0-9]*\Z")
TAG = re.compile(r"<([^<>]+)>")


def canonical_segments(segments):
    if not isinstance(segments, list):
        raise ValueError("segments must be an array")
    result = []
    for segment in segments:
        if not isinstance(segment, dict) or set(segment) != {"language", "speaker", "text"}:
            raise ValueError("Each segment must contain exactly language, speaker, text")
        lang, speaker, value = (segment[k] for k in ("language", "speaker", "text"))
        if not isinstance(lang, str) or not LANGUAGE.fullmatch(lang):
            raise ValueError("Invalid lowercase language code")
        if not isinstance(speaker, str) or not SPEAKER.fullmatch(speaker):
            raise ValueError("Invalid speaker ID; expected spk1, spk2, ...")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Empty attributed text segment")
        if "<" in value or ">" in value:
            raise ValueError("Text must not contain protocol tags or angle brackets")
        if result and (result[-1]["language"], result[-1]["speaker"]) == (lang, speaker):
            result[-1]["text"] += value
        else:
            result.append(dict(segment))
    return result


def parse_teacher(value):
    if not isinstance(value, dict) or set(value) != {"segments"}:
        raise ValueError("Teacher response must be an object with only segments")
    segments = canonical_segments(value["segments"])
    seen = []
    for s in segments:
        if s["speaker"] not in seen:
            expected = "spk%d" % (len(seen) + 1)
            if s["speaker"] != expected:
                raise ValueError("Teacher speaker IDs must follow first occurrence: " + expected)
            seen.append(s["speaker"])
    return segments


def parse_tagged_text(text):
    """Accept repeated full headers and change-only headers, reject empty turns."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Reference assistant text is empty")
    lang = speaker = None
    pending_speaker = pending_language = False
    result, position = [], 0

    def append_text(value):
        nonlocal pending_speaker, pending_language
        if not value.strip():
            if value and result and not pending_speaker and not pending_language:
                result[-1]["text"] += value
            return
        if lang is None or speaker is None:
            raise ValueError("Reference text needs language and speaker headers")
        result.append({"language": lang, "speaker": speaker, "text": value})
        pending_speaker = pending_language = False

    for match in TAG.finditer(text):
        append_text(text[position:match.start()])
        tag = match.group(1)
        if SPEAKER.fullmatch(tag):
            if pending_speaker:
                raise ValueError("Consecutive speaker headers without text; inspect the audio")
            speaker, pending_speaker = tag, True
        elif LANGUAGE.fullmatch(tag):
            if pending_language:
                raise ValueError("Consecutive language headers without text")
            lang, pending_language = tag, True
        else:
            raise ValueError("Unknown reference tag: " + tag)
        position = match.end()
    append_text(text[position:])
    if pending_speaker or pending_language:
        raise ValueError("Reference ends with an empty tagged segment")
    return canonical_segments(result)


def to_units(segments):
    return [(s["language"], s["speaker"], c)
            for s in canonical_segments(segments) for c in s["text"]]


def from_units(units):
    result = []
    for language, speaker, char in units:
        if result and (result[-1]["language"], result[-1]["speaker"]) == (language, speaker):
            result[-1]["text"] += char
        else:
            result.append({"language": language, "speaker": speaker, "text": char})
    return result


def render_segments(segments):
    return "".join("<%s><%s>%s" % (s["language"], s["speaker"], s["text"])
                   for s in canonical_segments(segments))


def plain_text(segments):
    return "".join(s["text"] for s in canonical_segments(segments))
