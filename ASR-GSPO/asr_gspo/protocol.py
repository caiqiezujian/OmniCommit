"""Actor prompts and reply grammar shared with the Third explicit SFT model."""
from third_asr.protocol import DIRECT, EOS, WAIT, get_protocol

PROTOCOL = get_protocol("explicit")
parse_reply = PROTOCOL.parse_reply


def initial_messages():
    return [{"role": "system", "content": PROTOCOL.system_prompt("normalized")},
            {"role": "user", "content": "<audio>"}]
