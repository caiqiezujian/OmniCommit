"""Stable vLLM content IDs with the lifetime of a streaming audio block."""
import hashlib

import numpy as np


class _StreamPCM(bytes):
    # The samples are immutable. Only the digest is filled lazily by vLLM.
    uuid = None


def snapshot_audio(audio):
    """Own the input samples while exposing a standard, read-only mono ndarray."""
    samples = np.asarray(audio, dtype="<f4")
    if samples.ndim != 1:
        raise ValueError("Audio must be finite, nonempty, mono float32 at 16000 Hz")
    return np.frombuffer(_StreamPCM(samples), dtype="<f4")


def _compute_uuid(pcm_bytes):
    return "omnicommit-16k-" + hashlib.sha256(pcm_bytes).hexdigest()


def audio_uuid(audio):
    samples = np.asarray(audio)
    owner = samples.base
    # Only our complete snapshot may reuse the stored ID. In particular, a
    # read-only view of a caller's writable array can still change underneath.
    if (isinstance(owner, _StreamPCM) and samples.dtype == np.dtype("<f4")
            and samples.flags.c_contiguous and samples.nbytes == len(owner)):
        if owner.uuid is None:
            owner.uuid = _compute_uuid(owner)
        return owner.uuid
    return _compute_uuid(np.asarray(samples, dtype="<f4").tobytes())
