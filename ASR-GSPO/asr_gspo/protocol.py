"""Actor prompts and reply grammar shared with the Fourth speaker-change SFT model."""
from fourth_asr.protocol import DIRECT, EOS, SPK_CHANGE, WAIT, get_protocol

PROTOCOL = get_protocol("speaker")
parse_reply = PROTOCOL.parse_reply


def total_turns(task):
    policy = task.get("end_policy", "eof")
    if policy not in ("eof", "annotation_end"):
        raise ValueError("Unknown end_policy")
    return len(task["chunks"]) + int(policy == "eof")


def initial_messages():
    return [{"role": "system", "content": PROTOCOL.system_prompt("normalized")},
            {"role": "user", "content": "<audio>"}]
