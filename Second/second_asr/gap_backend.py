"""Local CPU transcription for one pending window of 16 kHz microphone/file audio."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def local_whisper_directory(value):
    path = Path(value).expanduser()
    if not path.is_dir():
        raise ValueError(f"--gap-asr-model needs a local CTranslate2 Whisper directory: {path}")
    # faster-whisper otherwise tries to fetch its tokenizer from the Hub.
    for name in ("model.bin", "config.json", "tokenizer.json"):
        if not (path / name).is_file():
            raise ValueError(f"Local CTranslate2 Whisper model is missing {name}: {path}")
    return path.resolve()


class WhisperGapBackend:
    def __init__(self, model_path):
        directory = local_whisper_directory(model_path)
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                'Install the optional gap ASR dependency: python -m pip install "faster-whisper>=1.1,<2"'
            ) from exc
        self.model = WhisperModel(str(directory), device="cpu", compute_type="int8",
                                  cpu_threads=2, num_workers=1, local_files_only=True)
        if not self.model.model.is_multilingual:
            raise ValueError("--gap-asr-model must be multilingual; English-only Whisper cannot recover language switches")

    def transcribe(self, pcm):
        """Return timed words. The language is Whisper's estimate for the window."""
        pcm = np.asarray(pcm, dtype=np.float32)
        if pcm.ndim != 1 or not pcm.size or not np.isfinite(pcm).all():
            raise ValueError("Gap ASR expects nonempty finite mono 16 kHz PCM")
        duration = len(pcm) / 16000
        segments, info = self.model.transcribe(
            pcm, task="transcribe", language=None, condition_on_previous_text=False,
            word_timestamps=True, beam_size=1, temperature=0.0,
        )
        words = []
        for segment in segments:
            for word in segment.words or ():
                start, end = float(word.start), float(word.end)
                if not (math.isfinite(start) and math.isfinite(end)):
                    continue
                start, end = max(0.0, start), min(duration, end)
                if start < end and word.word.strip():
                    words.append({"text": word.word, "start": start, "end": end,
                                  "language": info.language, "probability": float(word.probability),
                                  "language_probability": float(info.language_probability)})
        return words
