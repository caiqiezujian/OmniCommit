"""State carried between assistant turns of one incremental ASR recording."""
from __future__ import annotations

import re

WAIT = "<|WAIT|>"
EOS = "<end_of_audio>"
LANGUAGE = r"[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*"
SPEAKER = r"spk[1-9][0-9]*"
HEADER = re.compile(rf"<({LANGUAGE})><({SPEAKER})>")


def system_prompt(number_format="verbatim"):
    numbers = {"verbatim": "Preserve spoken numbers.", "normalized": "Use digits for numbers."}
    if number_format not in numbers:
        raise ValueError("number_format must be verbatim or normalized")
    return ("Transcribe new speech. Mark first text and changes with <language><speaker>. "
            f"Emit {WAIT} if nothing is ready. {numbers[number_format]}")


def _require_text(text):
    if not isinstance(text, str) or not text:
        raise ValueError("Expected nonempty transcript text")
    if "<" in text or ">" in text:
        raise ValueError("Unexpected tag or reserved angle bracket in transcript")


def render_reply(spans, state=None):
    """Render attributed (language, speaker, exact text) spans, updating state."""
    pieces = []
    has_speech = False
    for lang, speaker, text in spans:
        if not isinstance(lang, str) or not re.fullmatch(LANGUAGE, lang):
            raise ValueError(f"Invalid language {lang!r}")
        if not isinstance(speaker, str) or not re.fullmatch(SPEAKER, speaker):
            raise ValueError(f"Invalid speaker {speaker!r}")
        _require_text(text)
        has_speech = has_speech or bool(text.strip())
        attribution = (lang, speaker)
        if attribution != state:
            pieces.append(f"<{lang}><{speaker}>")
        pieces.append(text)
        state = attribution
    if pieces and not has_speech:
        raise ValueError("A text turn must contain more than whitespace")
    return ("".join(pieces) if pieces else WAIT), state


def parse_reply(reply, state=None):
    """Recover exact text and attribution. A WAIT preserves all incoming state."""
    if reply == WAIT:
        return [], state
    if not isinstance(reply, str) or not reply:
        raise ValueError("Expected transcript or standalone WAIT")
    spans = []
    pos = 0
    while pos < len(reply):
        if reply[pos] == "<":
            header = HEADER.match(reply, pos)
            if header is None:
                raise ValueError(f"Invalid language/speaker header at offset {pos}")
            attribution = header.groups()
            if attribution == state:
                raise ValueError("Unchanged language and speaker must inherit their tags")
            state = attribution
            pos = header.end()
        if state is None:
            raise ValueError("First text in this recording requires language and speaker")
        next_header = reply.find("<", pos)
        end = len(reply) if next_header == -1 else next_header
        text = reply[pos:end]
        _require_text(text)
        spans.append((*state, text))
        pos = end
    if not any(text.strip() for _, _, text in spans):
        raise ValueError("A text turn must contain more than whitespace")
    return spans, state


def sort_tokens(tokens):
    """Stable tokenizer extension order, keeping WAIT available in every dataset."""
    languages, speakers = set(), set()
    for token in tokens:
        if token == WAIT:
            continue
        if isinstance(token, str) and re.fullmatch(rf"<{LANGUAGE}>", token):
            languages.add(token)
        elif isinstance(token, str) and re.fullmatch(rf"<{SPEAKER}>", token):
            speakers.add(token)
        else:
            raise ValueError(f"Unexpected training token {token!r}")
    return [WAIT, *sorted(languages), *sorted(speakers, key=lambda t: (len(t), t))]
