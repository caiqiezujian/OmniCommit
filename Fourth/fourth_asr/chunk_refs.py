"""Version 1 audio references shared by Fourth preparation and its training reader."""
import base64
import json
import math

PREFIX = "omnicommitaudio:v1:"
MAX_CHUNK_SECONDS = 8  # Project recipe limit, not a universal model limit.


def encode_chunk_ref(audio_path, start_seconds, end_seconds, input_seconds):
    payload = {"path": str(audio_path), "start": start_seconds,
               "end": end_seconds, "input_seconds": input_seconds}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    ref = PREFIX + base64.urlsafe_b64encode(raw).decode("ascii")
    decode_chunk_ref(ref)
    return ref


def decode_chunk_ref(value):
    if not isinstance(value, str) or not value.startswith(PREFIX):
        raise ValueError("Expected omnicommitaudio:v1: virtual audio reference")
    try:
        raw = base64.b64decode(value[len(PREFIX):], altchars=b"-_", validate=True)
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Malformed virtual audio reference") from exc
    if not isinstance(data, dict) or not isinstance(data.get("path"), str) or not data["path"]:
        raise ValueError("Virtual audio requires a local path")
    for name in ("start", "end", "input_seconds"):
        number = data.get(name)
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
            raise ValueError(f"Invalid virtual audio {name}")
    span = data["end"] - data["start"]
    if data["start"] < 0 or span <= 0:
        raise ValueError("Invalid virtual audio time range")
    if not 0 < data["input_seconds"] <= MAX_CHUNK_SECONDS or data["input_seconds"] + 1e-8 < span:
        raise ValueError("Chunk must fit its input duration (project maximum 8 seconds)")
    return data
