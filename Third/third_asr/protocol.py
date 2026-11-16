"""Third streaming-ASR protocols shared by data construction, SFT and serving."""

from dataclasses import dataclass

WAIT = "<|WAIT|>"
DIRECT = "<|DIRECT|>"
EOS = "<end_of_audio>"


def _check_text(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Expected nonempty Third transcript text")
    if "<" in text or ">" in text:
        raise ValueError("Unexpected tag or reserved angle bracket in Third transcript")


@dataclass(frozen=True)
class ThirdProtocol:
    name: str
    prefix: str

    @property
    def tokens(self):
        return (WAIT, self.prefix) if self.prefix else (WAIT,)

    def system_prompt(self, number_format="verbatim"):
        numbers = {"verbatim": "Preserve spoken numbers.", "normalized": "Use digits for numbers."}
        if number_format not in numbers:
            raise ValueError("number_format must be verbatim or normalized")
        action = f" Begin new text with {self.prefix}." if self.prefix else ""
        return f"Transcribe new speech in its original language.{action} Emit {WAIT} if nothing is ready. {numbers[number_format]}"

    def parse_reply(self, reply):
        """Return exact new text, or an empty string for a standalone WAIT."""
        if reply == WAIT:
            return ""
        if not isinstance(reply, str) or not reply.startswith(self.prefix):
            raise ValueError(f"Expected {self.prefix} and transcript text, or standalone {WAIT}")
        text = reply[len(self.prefix):]
        _check_text(text)
        return text

    def format_text(self, text):
        _check_text(text)
        return self.prefix + text

    def render_reply(self, spans, state=None):
        """Join newly completed source spans; preserve separators across turns."""
        if not spans:
            return WAIT, None
        return self.format_text("".join(text for _, _, text in spans)), None

    def training_tokens(self, units):
        return set(self.tokens)

    def sort_tokens(self, inventory):
        if set(inventory) != set(self.tokens):
            raise ValueError(f"Unexpected token inventory for Third {self.name}")
        return self.tokens


PROTOCOLS = {"explicit": ThirdProtocol("explicit", DIRECT), "plain": ThirdProtocol("plain", "")}


def get_protocol(name):
    try:
        return PROTOCOLS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown Third protocol: {name}") from exc


def protocol_for_tokens(tokens):
    ordered = tuple(tokens)
    for protocol in PROTOCOLS.values():
        if ordered == protocol.tokens:
            return protocol
    raise ValueError(f"Third token inventory must contain exactly {WAIT} or {WAIT}, {DIRECT} in that order")


# The original module-level API names the plain protocol used by existing callers.
system_prompt = PROTOCOLS["plain"].system_prompt
parse_reply = PROTOCOLS["plain"].parse_reply
render_reply = PROTOCOLS["plain"].render_reply
training_tokens = PROTOCOLS["plain"].training_tokens
sort_tokens = PROTOCOLS["plain"].sort_tokens
