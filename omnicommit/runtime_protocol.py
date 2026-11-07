"""Shared conditioning contract for SFT and streaming inference.

Reference blocks belong to user input, never assistant supervision. The end
marker is plain text: it is not a third action or a newly registered token.
"""
import re

END_OF_AUDIO = "<end_of_audio>"
LIVE_AUDIO_PREFIX = "[Current live stream]\n"
SPEAKER_REFERENCE_INTRO = "[Speaker reference: identity only; never transcribe as new speech]"
SYSTEM_PROMPT = """You transcribe speech incrementally in its original language and identify its speaker.
Each user turn provides only the next audio chunk; use earlier audio and text as context.
Return only newly committed text, never repeat or revise previously committed text.
If new text and its language/speaker attribution are sufficiently supported, begin with <|DIRECT|>.
For every emitted segment, write <language><speaker>text, for example <zh><spk1>你好。
Use the same speaker ID across this entire meeting; changing language does not change speaker identity.
Assign new speaker IDs in first-appearance order. Never reuse an existing ID for a different voice.
If no new segment can be committed, return exactly <|WAIT|>. Do not output empty speaker/language segments.
Keep names, entities and numerical expressions whole; do not commit an incomplete protected expression.
When a protected expression becomes complete, defer that audio round with WAIT and release it with later supported text on a following round.
When the first complete source phrase after a real live language or speaker change becomes available, defer that audio round with WAIT; a following supported round may release the held text.
Initialization and repeated identical headers are not changes. Multiple events in one round require one WAIT, not repeated waits for the same event.
Ordinary supported rounds may be consecutive DIRECT responses; do not alternate WAIT and DIRECT mechanically.
At end_of_audio, release supported pending content without requiring nonexistent future audio.
Preserve the spoken language and disfluencies. {number_instruction}
Use punctuation supported by the available audio. A chunk boundary alone is not a sentence boundary.
Within a DIRECT response, start each changed attribution with its complete language/speaker pair. Preserve spoken text formatting; do not insert a newline solely for a metadata change. Do not translate, explain or include timestamps.
Speaker reference blocks contain earlier voices and their known identities. They are conditioning only, never new speech.
Do not transcribe their audio again or treat changes between reference blocks as live speaker/language changes.
An absent recent speaker may return with the same ID. Insufficient identity evidence calls for WAIT, not a guessed new ID.
At <end_of_audio>, no more audio will arrive. Release any remaining supported transcription, or return WAIT if nothing remains.
Never invent missing speech to fill a wait or finish the meeting."""


def system_prompt(number_format="verbatim"):
    if number_format == "verbatim":
        instruction = "Transcribe numbers in their spoken form (verbatim); do not convert them to digits."
    elif number_format == "normalized":
        instruction = "Normalize numerical expressions to digits only when their complete value is determined."
    else:
        raise ValueError("number_format must be verbatim or normalized")
    return SYSTEM_PROMPT.format(number_instruction=instruction)


def speaker_reference_text(speaker, language, text):
    if not isinstance(speaker, str) or not re.fullmatch(r"spk[1-9][0-9]*", speaker):
        raise ValueError("Invalid reference speaker")
    if not isinstance(language, str) or not re.fullmatch(r"[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*", language):
        raise ValueError("Invalid reference language")
    if not isinstance(text, str) or not text.strip() or any(c in text for c in "<>\r\n"):
        raise ValueError("Reference text must be nonempty plain single-line transcription")
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
