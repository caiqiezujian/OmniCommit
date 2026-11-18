"""Read virtual SFT chunks without creating WAV files on disk.

Adapted from FLEX virtual SFT audio reading; no RL or SWIFT dependency. Crop source samples BEFORE resampling or feature extraction;
processing the whole recording first can leak future audio across a boundary.
"""
from __future__ import annotations

from functools import lru_cache
import io
import math
from pathlib import Path

from .chunk_refs import decode_chunk_ref

SAMPLE_RATE = 16000


def chunk_wav_bytes(ref):
    """16 kHz mono FLOAT WAV in memory; never a temporary audio file.

    File inventory is checked on every call, including cache hits. The cache
    contains at most 64 small chunks per worker, not complete source recordings.
    """
    data = decode_chunk_ref(ref)
    path = Path(data["path"])
    if not path.is_absolute():
        raise ValueError("Audio path must be absolute on the training server")
    info = path.stat()
    for name, actual in (("audio_size_bytes", info.st_size), ("audio_mtime_ns", info.st_mtime_ns)):
        if name in data and data[name] != actual:
            raise ValueError(f"Source audio changed after preparation ({name}): {path}")
    return _read_chunk(ref, info.st_size, info.st_mtime_ns)


@lru_cache(maxsize=64)
def _read_chunk(ref, size, mtime_ns):
    import numpy as np
    import soundfile as sf

    data = decode_chunk_ref(ref)
    with sf.SoundFile(data["path"]) as stream:
        sr = stream.samplerate
        start = round(data["start"] * sr)
        end = round(data["end"] * sr)
        if data["end"] > len(stream) / sr + 1e-8 or start < 0 or end > len(stream) or end <= start:
            raise ValueError("Chunk bounds exceed the actual source audio")
        stream.seek(start)
        pcm = stream.read(end - start, dtype="float32", always_2d=True)
    if len(pcm) != end - start or not np.isfinite(pcm).all():
        raise ValueError("Truncated or non-finite audio samples")
    mono = pcm.mean(axis=1, dtype=np.float32)
    if sr != SAMPLE_RATE:
        from scipy.signal import resample_poly
        divisor = math.gcd(sr, SAMPLE_RATE)
        mono = resample_poly(mono, SAMPLE_RATE // divisor, sr // divisor).astype(np.float32)
    # Retain one resampled sample even when an original tail is shorter than
    # 1/16000 s; rounding to zero would silently drop that real final sample.
    actual_count = max(1, round((data["end"] - data["start"]) * SAMPLE_RATE))
    output_count = round(data["input_seconds"] * SAMPLE_RATE)
    if output_count < actual_count or output_count < 160:
        raise ValueError("Invalid padded chunk length")
    result = np.zeros(output_count, dtype=np.float32)
    count = min(actual_count, len(mono))
    result[:count] = mono[:count]
    out = io.BytesIO()
    sf.write(out, result, SAMPLE_RATE, format="WAV", subtype="FLOAT")
    return out.getvalue()


def materialize_chunk_refs(refs):
    if not isinstance(refs, (list, tuple)) or not refs:
        raise ValueError("A virtual session needs at least one audio chunk")
    return [chunk_wav_bytes(ref) for ref in refs]
