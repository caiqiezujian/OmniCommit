"""Shared conditioning contract for SFT and streaming inference.

Reference blocks belong to user input, never assistant supervision. The end
marker is plain text: it is not a third action or a newly registered token.
"""
import re

LIVE_CHUNK_MS = (500, 700, 800, 1000)
LIVE_CHUNK_SECONDS = tuple(ms / 1000 for ms in LIVE_CHUNK_MS)
MAX_CHUNK_MS = 8000  # Same upper bound as the virtual SFT audio reader.
END_OF_AUDIO = "<end_of_audio>"
LIVE_AUDIO_PREFIX = "[Current live stream]\n"
SPEAKER_REFERENCE_INTRO = "[Speaker reference: identity only; never transcribe as new speech]"
SYSTEM_PROMPT = """Transcribe streaming speech in its original language.
Output <|WAIT|> or <|DIRECT|><language><speaker>new text. Label each language/speaker span.
Wait on uncertainty or a detected language/speaker change. Keep speaker IDs consistent and never repeat committed text.
{number_instruction}
At <end_of_audio>, output remaining text or <|WAIT|>. Speaker references are for identification only."""


def chunk_milliseconds(seconds):
    if type(seconds) not in (int, float) or not .001 <= seconds <= MAX_CHUNK_MS / 1000:
        raise ValueError(f"chunk_seconds must represent 1 to {MAX_CHUNK_MS} milliseconds")
    ms = round(seconds * 1000)
    if ms / 1000 != seconds:
        raise ValueError("chunk_seconds must be a whole number of milliseconds")
    return ms


def web_chunk_ms_bounds(default_seconds, max_reference_seconds):
    """Reserve slots down to 200 ms unless the operator requests a shorter default."""
    default_ms = chunk_milliseconds(default_seconds)
    # Use the server's actual 16 kHz sample budget before taking whole milliseconds.
    reference_samples = round(min(MAX_CHUNK_MS / 1000, max_reference_seconds) * 16000)
    reference_ms = reference_samples // 16
    return min(200, default_ms), max(LIVE_CHUNK_MS[-1], default_ms, reference_ms)


def system_prompt(number_format="verbatim"):
    if number_format == "verbatim":
        instruction = "Transcribe numbers verbatim; do not convert them to digits."
    elif number_format == "normalized":
        instruction = "Normalize numerical expressions to digits."
    else:
        raise ValueError("number_format must be verbatim or normalized")
    return SYSTEM_PROMPT.format(number_instruction=instruction)


def speaker_reference_text(speaker, language, text, allowed_tokens=None):
    if not isinstance(speaker, str) or not re.fullmatch(r"spk[1-9][0-9]*", speaker):
        raise ValueError("Invalid reference speaker")
    if not isinstance(language, str) or not re.fullmatch(r"[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*", language):
        raise ValueError("Invalid reference language")
    if not isinstance(text, str) or not text.strip() or any(c in text for c in "<>\r\n"):
        raise ValueError("Reference text must be nonempty plain single-line transcription")
    if allowed_tokens is not None and (f"<{language}>" not in allowed_tokens or f"<{speaker}>" not in allowed_tokens):
        raise ValueError("Reference language/speaker tag is absent from the trained inventory")
    return (SPEAKER_REFERENCE_INTRO + "\n<audio>\n" + f"<{language}><{speaker}>{text}"
            + "\n[End speaker reference]\n")


def speaker_registry_text(speakers):
    values = sorted(set(speakers), key=lambda value: int(value[3:]) if re.fullmatch(r"spk[1-9][0-9]*", value) else -1)
    if any(not re.fullmatch(r"spk[1-9][0-9]*", value) for value in values):
        raise ValueError("Invalid registered speaker ID")
    if not values:
        return ""
    return "\nPreviously assigned meeting speaker IDs: " + ", ".join(values) + ". Never reuse these IDs for new voices."


def last_committed_text(language, speaker):
    """Persist output attribution across context trims, independently of SPC.

    This is NOT read-state: a retained WAIT may already contain a change whose
    text has not been released. Never use this helper to replay that gate.
    """
    if language is None and speaker is None:
        return ""
    if not isinstance(language, str) or not re.fullmatch(r"[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*", language):
        raise ValueError("Invalid last committed language")
    if not isinstance(speaker, str) or not re.fullmatch(r"spk[1-9][0-9]*", speaker):
        raise ValueError("Invalid last committed speaker")
    return (f"\nLast committed live attribution: <{language}><{speaker}>. "
            "This is committed-output state, not necessarily the latest heard voice/language. "
            "Speaker references do not change it. Do not repeat a change deferral already represented in retained live history.")
