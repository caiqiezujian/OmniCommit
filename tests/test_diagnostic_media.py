from contextlib import closing
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from omnicommit import diagnostic_media
from omnicommit.media import MediaCursor, MediaStore
from Second.second_asr.service_web import SecondMediaCursor


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="requires FFmpeg")
@pytest.mark.parametrize(("protocol", "cursor_type"), [("first", MediaCursor), ("second", SecondMediaCursor)])
def test_local_video_delivers_web_file_packets_and_true_tail(tmp_path, protocol, cursor_type):
    source = tmp_path / "source.wav"
    video = tmp_path / "meeting.mkv"
    rate = 22050
    samples = np.random.default_rng(7).integers(-14000, 14000, size=(rate // 2, 2), dtype=np.int16)
    sf.write(source, samples, rate, subtype="PCM_16")
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi", "-i",
                    "color=s=16x16:r=2:d=1", "-i", str(source), "-c:v", "ffv1",
                    "-c:a", "copy", str(video)], check=True)

    # Drive the real web upload and release one source sample grid at a time.
    step = .201
    store = MediaStore(temp_dir=tmp_path)
    try:
        item = store.reserve()
        item.append(video.read_bytes())
        duration = item.decode()["duration_seconds"]
        cursor = cursor_type(store.claim(item.media_id))
        expected = []
        for position in [.21, .41, duration]:
            raw = cursor.next_chunk(position, step, final=position == duration)
            expected.append(np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0)
        assert cursor.done
    finally:
        store.close()
    actual = list(diagnostic_media.iter_media_chunks(video, step, protocol))
    assert [final for _, final in actual] == [False, False, True]
    assert len(actual[-1][0]) < round(step * 16000)
    for (pcm, _), reference in zip(actual, expected):
        assert pcm.dtype == np.float32
        np.testing.assert_array_equal(pcm, reference)
    assert video.is_file()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="requires FFmpeg")
def test_early_stop_cleans_decode_without_touching_local_file(tmp_path, monkeypatch):
    source = tmp_path / "original.wav"
    sf.write(source, np.zeros(6400, dtype=np.float32), 16000)
    original = source.read_bytes()
    store = MediaStore(temp_dir=tmp_path, max_bytes=16, max_seconds=.1)
    monkeypatch.setattr(diagnostic_media, "MediaStore", lambda: store)
    with closing(diagnostic_media.iter_media_chunks(source, .2, "first")) as chunks:
        pcm, final = next(chunks)
        assert len(pcm) == 3200 and not final
        assert store.root.is_dir()
    assert not store.root.exists()
    assert source.read_bytes() == original
