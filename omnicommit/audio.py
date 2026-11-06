"""Audio prefixes are cropped at source sample boundaries BEFORE resampling."""
from __future__ import annotations

import io
import math
from pathlib import Path


def inspect_audio(path, max_seconds=300.0):
    import soundfile as sf
    stat = Path(path).stat()
    with sf.SoundFile(path) as stream:
        frames, rate = len(stream), stream.samplerate
    if frames <= 0 or rate <= 0:
        raise ValueError("Audio is empty")
    duration_ms = frames * 1000.0 / rate
    if duration_ms > max_seconds * 1000:
        raise ValueError("Audio exceeds --max-audio-seconds; no silent truncation")
    return {"frames": frames, "sample_rate": rate, "duration_ms": duration_ms,
            "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def prefix_endpoints(info, chunk_seconds):
    if not math.isfinite(chunk_seconds) or chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive and finite")
    step = round(chunk_seconds * info["sample_rate"])
    if step <= 0:
        raise ValueError("chunk_seconds shorter than one source sample")
    return [frame * 1000.0 / info["sample_rate"]
            for frame in range(step, info["frames"], step)] + [info["duration_ms"]]


def prefix_wav(path, end_ms, sample_rate=16000):
    import numpy as np
    import soundfile as sf
    with sf.SoundFile(path) as stream:
        source_rate = stream.samplerate
        count = round(end_ms * source_rate / 1000)
        if count <= 0 or count > len(stream):
            raise ValueError("Prefix is outside source audio")
        pcm = stream.read(count, dtype="float32", always_2d=True)
    if len(pcm) != count or not np.isfinite(pcm).all():
        raise ValueError("Corrupt or non-finite audio")
    pcm = pcm.mean(axis=1, dtype=np.float32)
    if source_rate != sample_rate:
        from scipy.signal import resample_poly
        divisor = math.gcd(source_rate, sample_rate)
        pcm = resample_poly(pcm, sample_rate // divisor, source_rate // divisor)
    out = io.BytesIO()
    sf.write(out, pcm, sample_rate, format="WAV", subtype="PCM_16")
    return out.getvalue()
