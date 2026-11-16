"""Text-only streaming supervision shared by data preparation and inference."""

WAIT = "<|WAIT|>"
EOS = "<end_of_audio>"


def system_prompt(number_format="verbatim"):
    numbers = {"verbatim": "Preserve spoken numbers.", "normalized": "Use digits for numbers."}
    if number_format not in numbers:
        raise ValueError("number_format must be verbatim or normalized")
    return f"Transcribe new speech in its original language. Emit {WAIT} if nothing is ready. {numbers[number_format]}"


def parse_reply(reply):
    """Return exact new text, or an empty string for a standalone WAIT."""
    if reply == WAIT:
        return ""
    if not isinstance(reply, str) or not reply.strip():
        raise ValueError("Expected transcript text or a standalone WAIT")
    if "<" in reply or ">" in reply:
        raise ValueError("Unexpected tag or reserved angle bracket in Third transcript")
    return reply


def render_reply(spans, state=None):
    """Join newly completed source spans; preserve separators across turns."""
    if not spans:
        return WAIT, None
    reply = "".join(text for _, _, text in spans)
    if not parse_reply(reply):
        raise ValueError("Source transcript contains a reserved WAIT token")
    return reply, None


def training_tokens(units):
    return {WAIT}
