"""Read local audio/video using the same model-input chunks as Web file playback."""
import math
from pathlib import Path

import numpy as np

from .media import MediaCursor, MediaStore
from .runtime_protocol import chunk_milliseconds


def iter_media_chunks(path, chunk_seconds, protocol):
    """Yield (mono float32 PCM at 16 kHz, final_chunk), including a short tail."""
    chunk_milliseconds(chunk_seconds)
    if protocol == "first":
        cursor_class = MediaCursor
    elif protocol == "second":
        from Second.second_asr.service_web import SecondMediaCursor
        cursor_class = SecondMediaCursor
    else:
        raise ValueError("protocol must be first or second")
    source = Path(path).resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"Expected a local audio/video file: {source}")

    store = MediaStore()
    try:
        item = store.reserve()
        metadata = item.decode(local_source=source, timeout=None)
        cursor = cursor_class(store.claim(item.media_id))
        # Make the final source sample available even when duration * rate
        # rounds just below its integer frame count on a long recording.
        release = math.nextafter(metadata["duration_seconds"], math.inf)
        while not cursor.finished:
            raw = cursor.next_chunk(release, chunk_seconds, final=True)
            if not raw:
                raise RuntimeError("No audio returned before the end of the decoded file")
            yield np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0, cursor.finished
    finally:
        store.close()
