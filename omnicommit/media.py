"""Bounded server-side uploads and playback-gated audio chunks.

This module never hands future samples to the model: a cursor reads only
samples already released by the client's playback clock. That clock is a
client-provided boundary, not a trusted wall clock. Uploaded filenames are
never used as paths or shell arguments.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import uuid

import numpy as np
import soundfile as sf

from .runtime_protocol import chunk_milliseconds

RATE = 16000
MAX_UPLOAD_BYTES = 1024 * 1024 * 1024
MAX_DURATION_SECONDS = 4 * 60 * 60


class MediaError(ValueError):
    pass


def _container(path):
    with path.open('rb') as stream:
        h = stream.read(64)
    if h[:4] in (b'RIFF', b'RF64') and h[8:12] == b'WAVE': return 'wav'
    if h[:4] == b'fLaC': return 'flac'
    if h[:4] == b'OggS': return 'ogg'
    if h[:4] == b'\x1aE\xdf\xa3': return 'matroska'
    if h[:16] == bytes.fromhex('3026b2758e66cf11a6d900aa0062ce6c'): return 'asf'
    if h[4:8] in (b'ftyp', b'moov', b'mdat', b'wide', b'free'): return 'mov'
    if h[:4] == b'RIFF' and h[8:12] == b'AVI ': return 'avi'
    if h[:4] == b'FORM' and h[8:12] in (b'AIFF', b'AIFC'): return 'aiff'
    if len(h) > 1 and h[0] == 255 and h[1] & 0xf6 == 0xf0: return 'aac'
    if h[:3] == b'ID3' or (len(h) > 1 and h[0] == 255 and h[1] & 0xe0 == 0xe0): return 'mp3'
    raise MediaError('Unsupported audio/video format; playlists are not accepted')


@dataclass
class MediaUpload:
    media_id: str
    directory: Path
    created: float
    max_bytes: int
    max_seconds: float
    size: int = 0
    samples: int = 0
    sample_rate: int = 0
    ready: bool = False
    claimed: bool = False

    @property
    def raw_path(self): return self.directory / 'upload.bin'

    @property
    def pcm_path(self): return self.directory / 'audio.wav'

    def append(self, data):
        if self.ready or self.claimed or not isinstance(data, bytes):
            raise MediaError('Invalid upload state')
        if self.size + len(data) > self.max_bytes:
            raise MediaError('Upload exceeds configured byte limit')
        with self.raw_path.open('ab') as output:
            output.write(data)
        self.size += len(data)

    def decode(self, ffmpeg='ffmpeg', timeout=3600, *, local_source=None):
        """Decode an upload, or an explicitly supplied local CLI file, into temporary PCM."""
        if self.ready or self.claimed or (local_source is None and not self.size):
            raise MediaError('Upload is empty or already decoded')
        source = self.raw_path if local_source is None else Path(local_source)
        fmt = _container(source)
        command = [ffmpeg, '-nostdin', '-hide_banner', '-v', 'error',
                   '-protocol_whitelist', 'file,pipe', '-f', fmt, '-i', str(source),
                   '-map', '0:a:0', '-vn', '-sn', '-dn']
        if local_source is None:
            command.extend(['-t', str(self.max_seconds + 1)])
        command.extend(['-ac', '1', '-c:a', 'pcm_s16le', '-rf64', 'auto', '-f', 'wav', '-y', str(self.pcm_path)])
        try:
            result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.PIPE, timeout=timeout, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise MediaError(f'Audio decoding failed: {exc}') from exc
        if result.returncode != 0 or not self.pcm_path.is_file():
            raise MediaError('Audio decoding failed: ' + result.stderr.decode('utf-8', 'replace')[-1000:])
        info = sf.info(str(self.pcm_path))
        if (info.channels != 1 or info.samplerate <= 0 or info.frames <= 0
                or (local_source is None and info.frames / info.samplerate > self.max_seconds + 1e-9)):
            raise MediaError('Audio is empty, corrupt, or exceeds configured duration')
        self.samples = info.frames
        self.sample_rate = info.samplerate
        self.created = time.monotonic()
        self.ready = True
        # Playback uses the user's original local file; source bytes need not be retained.
        if local_source is None:
            self.raw_path.unlink()
        return self.public()

    def public(self):
        return {'media_id': self.media_id, 'sample_rate': self.sample_rate, 'samples': self.samples,
                'duration_seconds': self.samples / self.sample_rate}


class MediaCursor:
    def __init__(self, upload):
        if not upload.ready or not upload.claimed:
            raise MediaError('A ready claimed upload is required')
        self.upload = upload
        self.position = 0
        self.last_release = 0
        self.final_release = None
        self.chunk_index = 0
        self.chunk_seconds = None

    @property
    def done(self): return self.position == self.upload.samples

    @property
    def finished(self):
        return self.done or (self.final_release is not None and self.position >= self.final_release)

    def _next_source_boundary(self, chunk_seconds):
        try:
            chunk_ms = chunk_milliseconds(chunk_seconds)
        except ValueError as exc:
            raise MediaError(str(exc)) from exc
        if chunk_ms * self.upload.sample_rate < 1000:
            raise MediaError("The selected step contains less than one source sample")
        return round((self.chunk_index + 1) * chunk_seconds * self.upload.sample_rate)

    def next_chunk(self, release_seconds, chunk_seconds, *, final=False):
        if isinstance(release_seconds, bool) or not isinstance(release_seconds, (int, float)) or not math.isfinite(release_seconds):
            raise MediaError('Invalid playback position')
        nominal_end = self._next_source_boundary(chunk_seconds)
        if self.chunk_seconds is not None and chunk_seconds != self.chunk_seconds:
            raise MediaError('Chunk length cannot change within a session')
        self.chunk_seconds = chunk_seconds
        if not isinstance(final, bool):
            raise MediaError('final must be boolean')
        source_rate = self.upload.sample_rate
        release = math.floor(release_seconds * source_rate + 1e-7)
        if release < self.last_release or release < 0 or release > self.upload.samples + 1:
            raise MediaError('Playback position must be monotonic and within media')
        release = min(release, self.upload.samples)
        if self.final_release is not None and release != self.final_release:
            raise MediaError('Cannot change playback position after final release')
        if final:
            self.final_release = release
        self.last_release = release
        if self.finished: return None
        end = min(nominal_end, self.upload.samples)
        if end > release:
            if self.final_release is None or release <= self.position:
                return None
            end = release
        # Crop raw source samples BEFORE any sample-rate conversion: whole-file
        # resampling would mix samples from after the released chunk boundary.
        with sf.SoundFile(str(self.upload.pcm_path)) as audio:
            audio.seek(self.position)
            chunk = audio.read(end - self.position, dtype='float32', always_2d=False)
        if len(chunk) != end - self.position or not np.isfinite(chunk).all():
            raise MediaError('Truncated decoded audio')
        full_chunk = end == nominal_end
        target = round(chunk_seconds * RATE) if full_chunk else max(1, round((end - self.position) * RATE / source_rate))
        if source_rate != RATE:
            from scipy.signal import resample_poly
            divisor = math.gcd(source_rate, RATE)
            chunk = resample_poly(chunk, RATE // divisor, source_rate // divisor)
        # Preserve the real duration rounded to the closest 16 kHz sample.
        if len(chunk) < target:
            chunk = np.pad(chunk, (0, target-len(chunk)))
        chunk = np.clip(np.rint(chunk[:target] * 32768), -32768, 32767).astype('<i2').tobytes()
        self.position = end
        if full_chunk:
            self.chunk_index += 1
        return chunk


class MediaStore:
    """Temporary uploads; use from a single event loop, decode in a worker."""
    def __init__(self, max_slots=4, ttl_seconds=600, *, temp_dir=None,
                 max_bytes=MAX_UPLOAD_BYTES, max_seconds=MAX_DURATION_SECONDS):
        if max_slots < 1 or ttl_seconds <= 0 or max_bytes < 1 or max_seconds <= 0:
            raise ValueError('Invalid media limits')
        self.temporary = tempfile.TemporaryDirectory(prefix='omnicommit-media-', dir=temp_dir)
        self.root = Path(self.temporary.name)
        self.uploads = {}
        self.max_slots, self.ttl_seconds = max_slots, ttl_seconds
        self.max_bytes, self.max_seconds = max_bytes, max_seconds

    def expire(self):
        for key, item in list(self.uploads.items()):
            if item.ready and not item.claimed and time.monotonic() - item.created > self.ttl_seconds:
                self.discard(key)

    def reserve(self):
        self.expire()
        if len(self.uploads) >= self.max_slots:
            raise MediaError('Too many uploads or active file sessions')
        key = uuid.uuid4().hex
        directory = self.root / key
        directory.mkdir()
        upload = MediaUpload(key, directory, time.monotonic(), self.max_bytes, self.max_seconds)
        self.uploads[key] = upload
        return upload

    def claim(self, key):
        self.expire()
        item = self.uploads.get(key)
        if item is None or not item.ready or item.claimed:
            raise MediaError('Upload missing, unready or already in use')
        item.claimed = True
        return item

    def discard(self, key):
        item = self.uploads.pop(key, None)
        if item is not None:
            shutil.rmtree(item.directory)

    def close(self):
        self.uploads.clear()
        self.temporary.cleanup()
