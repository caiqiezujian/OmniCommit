import io
import shutil
import wave

import numpy as np
import pytest

from omnicommit.media import MediaCursor, MediaError, MediaStore


def wav_bytes(samples, sample_rate=16000):
    out = io.BytesIO()
    with wave.open(out, 'wb') as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(samples.astype('<i2').tobytes())
    return out.getvalue()


@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='requires ffmpeg')
def test_file_release_never_reads_future_and_flushes_real_tail(tmp_path):
    store = MediaStore(temp_dir=tmp_path, max_seconds=5)
    try:
        samples = np.concatenate([np.ones(8000), np.full(8000, 2), np.full(3200, 3)]).astype('<i2')
        item = store.reserve()
        item.append(wav_bytes(samples))
        assert item.decode()['samples'] == len(samples)
        assert not item.raw_path.exists()
        cursor = MediaCursor(store.claim(item.media_id))
        assert cursor.next_chunk(.49, .5) is None
        assert np.array_equal(np.frombuffer(cursor.next_chunk(.5, .5), dtype='<i2'), samples[:8000])
        assert cursor.next_chunk(.99, .5) is None
        assert np.array_equal(np.frombuffer(cursor.next_chunk(1., .5), dtype='<i2'), samples[8000:16000])
        assert cursor.next_chunk(1.19, .5) is None
        assert np.array_equal(np.frombuffer(cursor.next_chunk(1.2, .5), dtype='<i2'), samples[16000:])
        assert cursor.done and cursor.next_chunk(1.2, .5) is None
        with pytest.raises(MediaError, match='monotonic'):
            cursor.next_chunk(.5, .5)
        with pytest.raises(MediaError, match='already in use'):
            store.claim(item.media_id)
        store.discard(item.media_id)
        assert not item.directory.exists()
    finally:
        store.close()


def test_upload_limits_invalid_formats_and_cleanup(tmp_path):
    store = MediaStore(max_slots=1, max_bytes=20, temp_dir=tmp_path)
    try:
        item = store.reserve()
        with pytest.raises(MediaError, match='Too many'):
            store.reserve()
        item.append(b'#EXTM3U\n')
        with pytest.raises(MediaError, match='Unsupported'):
            item.decode()
        with pytest.raises(MediaError, match='byte limit'):
            item.append(b'x' * 20)
        store.discard(item.media_id)
        assert not item.directory.exists()
    finally:
        store.close()


@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='requires ffmpeg')
def test_early_end_drains_released_audio_including_short_tail(tmp_path):
    store = MediaStore(temp_dir=tmp_path)
    try:
        samples = np.arange(19200, dtype='<i2')
        item = store.reserve()
        item.append(wav_bytes(samples))
        item.decode()
        cursor = MediaCursor(store.claim(item.media_id))
        first = cursor.next_chunk(.77, .5, final=True)
        tail = cursor.next_chunk(.77, .5, final=True)
        assert np.array_equal(np.frombuffer(first + tail, dtype='<i2'), samples[:12320])
        assert cursor.finished and not cursor.done
        assert cursor.next_chunk(.77, .5, final=True) is None
        with pytest.raises(MediaError, match='after final release'):
            cursor.next_chunk(1.0, .5)
    finally:
        store.close()


@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='requires ffmpeg')
@pytest.mark.parametrize('rate', [44100, 48000])
def test_resampling_does_not_leak_future_audio_across_chunk_boundary(tmp_path, rate):
    store = MediaStore(temp_dir=tmp_path, max_seconds=2)
    try:
        outputs = []
        for future in (False, True):
            samples = np.zeros(rate, dtype='<i2')
            if future:
                samples[rate//2:rate//2+10] = 30000
            item = store.reserve()
            item.append(wav_bytes(samples, rate))
            assert item.decode()['sample_rate'] == rate
            cursor = MediaCursor(store.claim(item.media_id))
            outputs.append(cursor.next_chunk(.5, .5))
            store.discard(item.media_id)
        assert len(outputs[0]) == 16000
        assert outputs[0] == outputs[1]
    finally:
        store.close()


@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='requires ffmpeg')
@pytest.mark.parametrize('rate', [44100, 48000])
def test_one_second_chunks_and_final_partial_preserve_duration(tmp_path, rate):
    store = MediaStore(temp_dir=tmp_path, max_seconds=4)
    try:
        samples = np.full(round(rate*1.25), 5000, dtype='<i2')
        item = store.reserve()
        item.append(wav_bytes(samples, rate))
        item.decode()
        cursor = MediaCursor(store.claim(item.media_id))
        assert len(cursor.next_chunk(1.0, 1.0)) == 2*16000
        assert cursor.next_chunk(1.24, 1.0) is None
        assert len(cursor.next_chunk(1.25, 1.0, final=True)) == 2*4000
        assert cursor.finished
    finally:
        store.close()
