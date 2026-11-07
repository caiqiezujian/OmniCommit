"""Optional WhisperX/Qwen3 forced alignment of known monolingual transcripts.

Install WhisperX in a SEPARATE alignment environment. This module never changes
the transcription or calls a text teacher. Mixed-language recordings and
unsupported lexical characters require externally reviewed acoustic timings.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import sys
import unicodedata

from .alignment import annotation_units, lexical_chars, alignment_words, source_rows
from .io import Journal, fingerprint, row_id, write_row

QWEN_LANGUAGES = {"zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean",
                  "de": "German", "fr": "French", "ru": "Russian", "pt": "Portuguese",
                  "es": "Spanish", "it": "Italian"}


def strict_qwen_timestamps(data):
    """Reject upstream timestamp anomalies instead of interpolating them."""
    values = data.tolist()
    if not values or any(not math.isfinite(float(x)) or x < 0 for x in values):
        raise ValueError("Qwen3 returned invalid timestamp predictions")
    if any(a > b for a, b in zip(values, values[1:])):
        raise ValueError("Qwen3 timestamps are not monotonic; interpolation is disabled")
    return [int(value) for value in values]


def restore_qwen_spans(text, items):
    """Undo only the verified Qwen0.0.6 processor's deterministic punctuation cleaning.

    Items contain words from the GIVEN transcript, not ASR hypotheses. Match
    their entire cleaned sequence exactly and restore each original substring.
    Numbers retain decimal/grouping punctuation in downstream timestamp files.
    """
    positions = [i for i, char in enumerate(text)
                 if char == "'" or unicodedata.category(char)[0] in "LN"]
    cleaned = "".join(text[i] for i in positions)
    if "".join(item.text for item in items) != cleaned:
        raise ValueError("Qwen3 cleaned tokens differ from original text; no fuzzy matching")
    words, lexical_offset, raw_offset = [], 0, 0
    for index, item in enumerate(items):
        if not item.text:
            raise ValueError("Qwen3 returned an empty aligned item")
        lexical_offset += len(item.text)
        raw_end = len(text) if index == len(items) - 1 else positions[lexical_offset - 1] + 1
        words.append({"text": text[raw_offset:raw_end], "start": item.start_time, "end": item.end_time})
        raw_offset = raw_end
    return words


class Qwen3Aligner:
    def __init__(self, device="cuda:0", model_name="Qwen/Qwen3-ForcedAligner-0.6B",
                 dtype="bfloat16", max_seconds=60.0):
        self.device, self.model_name, self.dtype, self.max_seconds = device, model_name, dtype, max_seconds
        self.model = None

    def align(self, row, audio_root=None):
        import soundfile as sf
        _, units, _ = annotation_units(row)
        languages = {language.split("-", 1)[0] for language, _, _ in units}
        if len(languages) != 1:
            raise ValueError("Automatic alignment is monolingual only; mixed speech needs real external timestamps")
        language = next(iter(languages))
        if language not in QWEN_LANGUAGES:
            raise ValueError("Qwen3 ForcedAligner does not cover this configured language; use external or WhisperX timings")
        paths = row.get("audios")
        if not isinstance(paths, list) or len(paths) != 1 or not isinstance(paths[0], str):
            raise ValueError("Exactly one audio path is required")
        audio = Path(paths[0]).expanduser()
        if not audio.is_absolute():
            audio = Path(audio_root or Path.cwd()) / audio
        audio = audio.resolve(strict=True)
        info = sf.info(str(audio))
        seconds = info.frames / info.samplerate
        if not 0 < seconds <= self.max_seconds:
            raise ValueError("Audio exceeds forced-aligner length bound; provide externally timed subspans")
        if self.model is None:
            from qwen_asr import Qwen3ForcedAligner
            import torch
            self.model = Qwen3ForcedAligner.from_pretrained(
                self.model_name, dtype=getattr(torch, self.dtype), device_map=self.device)
            # Version-pinned adapter hook; prevents the upstream processor's
            # own anomaly interpolation from masquerading as measured timing.
            self.model.aligner_processor.fix_timestamp = strict_qwen_timestamps
        text = "".join(char for _, _, char in units)
        results = self.model.align(audio=str(audio), text=text, language=QWEN_LANGUAGES[language])
        if len(results) != 1:
            raise ValueError("Qwen3 returned unexpected batch size")
        result = {"id": row.get("id", row.get("key")), "audio": paths[0], "duration_ms": seconds * 1000,
                  "words": restore_qwen_spans(text, results[0].items), "language": language,
                  "backend": "qwen3-forced-alignment-strict-timestamps", "alignment_model": self.model_name}
        words = alignment_words(result, seconds * 1000)
        if "".join(word["lexical"] for word in words) != lexical_chars(text):
            raise ValueError("Qwen3 restored timing text does not match transcript")
        return result


class WhisperXAligner:
    def __init__(self, device="cuda", model_name=None, model_dir=None, max_seconds=60.0):
        self.device, self.model_name, self.model_dir = device, model_name, model_dir
        self.max_seconds = max_seconds
        self.language = self.model = self.metadata = None

    def align(self, row, audio_root=None):
        import soundfile as sf
        import whisperx

        _, units, _ = annotation_units(row)
        languages = {language.split("-", 1)[0] for language, _, _ in units}
        if len(languages) != 1:
            raise ValueError("Automatic alignment is monolingual only; mixed speech needs real external timestamps")
        language = next(iter(languages))
        paths = row.get("audios")
        if not isinstance(paths, list) or len(paths) != 1 or not isinstance(paths[0], str):
            raise ValueError("Exactly one audio path is required")
        audio = Path(paths[0]).expanduser()
        if not audio.is_absolute():
            audio = Path(audio_root or Path.cwd()) / audio
        audio = audio.resolve(strict=True)
        info = sf.info(str(audio))
        seconds = info.frames / info.samplerate
        if not 0 < seconds <= self.max_seconds:
            raise ValueError("Audio exceeds forced-aligner length bound; provide externally timed subspans, no text-duration guessing")
        if self.language != language:
            # Keep one acoustic model loaded, so twelve languages do not retain
            # twelve models in GPU memory. Group source shards by language.
            self.model = self.metadata = None
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.model, self.metadata = whisperx.load_align_model(
                language_code=language, device=self.device,
                model_name=self.model_name, model_dir=self.model_dir)
            self.language = language
        text = "".join(char for _, _, char in units)
        dictionary = self.metadata["dictionary"]
        unsupported = sorted({char for char in text if lexical_chars(char) and char.lower() not in dictionary})
        if unsupported:
            raise ValueError("Acoustic model lacks lexical characters (including possible digits); "
                             "external timings required, wildcard/interpolated timing disabled: " + repr(unsupported[:20]))
        audio_values = whisperx.load_audio(str(audio))
        aligned = whisperx.align([{"start": 0.0, "end": seconds, "text": text}],
                                 self.model, self.metadata, audio_values, self.device,
                                 return_char_alignments=True)
        chars = []
        for segment in aligned.get("segments", []):
            for char in segment.get("chars", []) or []:
                value = char.get("char")
                if isinstance(value, str) and lexical_chars(value):
                    # Missing char timing is rejected by alignment_words below;
                    # never use WhisperX's interpolated word timing fallback.
                    chars.append({"text": value, "start": char.get("start"), "end": char.get("end")})
        result = {"id": row.get("id", row.get("key")), "audio": paths[0],
                  "duration_ms": seconds * 1000, "chars": chars,
                  "backend": "whisperx-forced-character-alignment",
                  "language": language, "alignment_model": self.model_name or "whisperx-default-" + language}
        words = alignment_words(result, seconds * 1000)
        if "".join(word["lexical"] for word in words) != lexical_chars(text):
            raise ValueError("Forced aligner omitted or duplicated lexical characters")
        return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="Annotated ASR JSONL")
    p.add_argument("--output", required=True)
    p.add_argument("--audio-root")
    p.add_argument("--device", default="cuda")
    p.add_argument("--backend", choices=["whisperx", "qwen3"], default="qwen3")
    p.add_argument("--qwen-model", default="Qwen/Qwen3-ForcedAligner-0.6B")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--align-model", help="Custom compatible Wav2Vec2 CTC model; required for Thai")
    p.add_argument("--model-dir")
    p.add_argument("--max-audio-seconds", type=float, default=60.0)
    p.add_argument("--resume", action="store_true")
    return p


def run(args):
    if Path(args.input).resolve() == Path(args.output).resolve():
        raise ValueError("Input and output must differ")
    if args.max_audio_seconds <= 0:
        raise ValueError("max-audio-seconds must be positive")
    aligner = (WhisperXAligner(args.device, args.align_model, args.model_dir, args.max_audio_seconds)
               if args.backend == "whisperx" else
               Qwen3Aligner(args.device, args.qwen_model, args.dtype, args.max_audio_seconds))
    config = {"pipeline": "forced-alignment-v2", "backend": args.backend, "device": args.device,
              "qwen_model": args.qwen_model, "dtype": args.dtype,
              "align_model": args.align_model, "model_dir": args.model_dir,
              "max_audio_seconds": args.max_audio_seconds, "audio_root": args.audio_root}
    journal = Journal(args.output, config, args.resume)
    counts = {"written": 0, "failed": 0, "resumed": 0}
    complete = False
    try:
        with open(str(args.output) + ".rejected.jsonl", "a", encoding="utf-8") as rejected:
            for line, row, error in source_rows(args.input):
                try:
                    identifier = row_id(row, args.input, line)
                except ValueError as exc:
                    identifier, error = row_id({}, args.input, line), str(exc)
                digest = fingerprint(row)
                if journal.previous(digest, identifier) is not None:
                    counts["resumed"] += 1
                    continue
                try:
                    if error:
                        raise ValueError(error)
                    result = aligner.align(row, args.audio_root or Path(args.input).resolve().parent)
                    result["id"], result["status"] = identifier, "ok"
                except Exception as exc:
                    result = {"id": identifier, "audios": row.get("audios"), "output": "",
                              "status": "error", "error": str(exc)}
                    counts["failed"] += 1
                    write_row(rejected, {"line": line, "id": identifier, "error": str(exc)})
                    print(f"[ERROR] id={identifier}: {exc}", file=sys.stderr, flush=True)
                result["input_hash"] = digest
                journal.append(result)
                counts["written"] += 1
        complete = True
    finally:
        journal.close(check_excess=complete)
    print(json.dumps(counts), flush=True)
    return counts


def main(argv=None):
    try:
        run(parser().parse_args(argv))
    except Exception as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 2
    return 0
