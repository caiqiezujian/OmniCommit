"""Fourth streaming-ASR protocols shared by data construction, SFT and serving."""

from dataclasses import dataclass

WAIT = "<|WAIT|>"
DIRECT = "<|DIRECT|>"
SPK_CHANGE = "<|SPK_CHANGE|>"
EOS = "<end_of_audio>"


def _check_text(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Expected nonempty Fourth transcript text")
    if "<" in text or ">" in text:
        raise ValueError("Unexpected tag or reserved angle bracket in Fourth transcript")


@dataclass(frozen=True)
class FourthProtocol:
    name: str
    prefix: str
    speaker_changes: bool = False

    @property
    def tokens(self):
        if self.speaker_changes:
            return (WAIT, DIRECT, SPK_CHANGE)
        return (WAIT, self.prefix) if self.prefix else (WAIT,)

    def system_prompt(self, number_format="verbatim"):
        numbers = {"verbatim": "Preserve spoken numbers.", "normalized": "Use digits for numbers."}
        if number_format not in numbers:
            raise ValueError("number_format must be verbatim or normalized")
        action = f" Begin new text with {self.prefix}." if self.prefix else ""
        if self.speaker_changes:
            return (f"Transcribe new speech. Use {DIRECT} for text, {WAIT} to wait, "
                    f"and insert {SPK_CHANGE} when the speaker changes. {numbers[number_format]}")
        return f"Transcribe new speech in its original language.{action} Emit {WAIT} if nothing is ready. {numbers[number_format]}"

    def parse_reply(self, reply):
        """Return exact new text, or an empty string for a standalone WAIT."""
        if reply == WAIT:
            return ""
        if not isinstance(reply, str) or not reply.startswith(self.prefix):
            raise ValueError(f"Expected {self.prefix} and transcript text, or standalone {WAIT}")
        text = reply[len(self.prefix):]
        self._validate_body(text)
        return text

    def _validate_body(self, text):
        if self.speaker_changes and isinstance(text, str) and SPK_CHANGE in text:
            opening, *changes = text.split(SPK_CHANGE)
            if opening.strip():
                _check_text(opening)
            for new_speech in changes:
                _check_text(new_speech)
        else:
            _check_text(text)

    def format_text(self, text):
        self._validate_body(text)
        return self.prefix + text

    def render_reply(self, spans, state=None):
        """Join newly completed source spans; preserve separators across turns."""
        if not spans:
            return WAIT, state
        if self.speaker_changes:
            parts = []
            for _, speaker, text in spans:
                if text.strip():
                    if state is not None and speaker != state:
                        parts.append(SPK_CHANGE)
                    state = speaker
                parts.append(text)
            return self.format_text("".join(parts)), state
        return self.format_text("".join(text for _, _, text in spans)), None

    def training_tokens(self, units):
        return set(self.tokens)

    def sort_tokens(self, inventory):
        if set(inventory) != set(self.tokens):
            raise ValueError(f"Unexpected token inventory for Fourth {self.name}")
        return self.tokens


PROTOCOLS = {
    "speaker": FourthProtocol("speaker", DIRECT, speaker_changes=True),
    "explicit": FourthProtocol("explicit", DIRECT),
    "plain": FourthProtocol("plain", ""),
}


def get_protocol(name):
    try:
        return PROTOCOLS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown Fourth protocol: {name}") from exc


def protocol_for_tokens(tokens):
    ordered = tuple(tokens)
    for protocol in PROTOCOLS.values():
        if ordered == protocol.tokens:
            return protocol
    raise ValueError(f"Fourth token inventory must exactly match one of {[p.tokens for p in PROTOCOLS.values()]}")


# The original module-level API names the plain protocol used by existing callers.
system_prompt = PROTOCOLS["plain"].system_prompt
parse_reply = PROTOCOLS["plain"].parse_reply
render_reply = PROTOCOLS["plain"].render_reply
training_tokens = PROTOCOLS["plain"].training_tokens
sort_tokens = PROTOCOLS["plain"].sort_tokens
