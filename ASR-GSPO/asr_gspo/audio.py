"""Use the same virtual ranges, source cropping and tail padding as Fourth SFT."""
import base64
import json

from fourth_asr.chunk_refs import PREFIX, decode_chunk_ref, encode_chunk_ref as encode_range
from fourth_asr.training_audio import SAMPLE_RATE, materialize_chunk_refs


def encode_chunk_ref(task, index):
    chunk = task["chunks"][index]
    ref = encode_range(task["audio_path"], chunk["start_seconds"], chunk["end_seconds"],
                       chunk["input_seconds"])
    inventory = {k: task[k] for k in ("audio_size_bytes", "audio_mtime_ns") if k in task}
    if inventory:
        # Fourth's loader checks these optional fields even on a waveform-cache
        # hit. Fail if a dataset file has been replaced since preparation.
        payload = {**decode_chunk_ref(ref), **inventory}
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ref = PREFIX + base64.urlsafe_b64encode(raw).decode("ascii")
    return ref
