"""Batched local HF teacher collection with per-query durable resume."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import time

from .audio import inspect_audio
from .io import Journal, fingerprint, row_id, source_rows
from .prefix_source import parse_source, plan_queries

VERSION = "local-language-hf-prefix-v1"


def model_signature(path):
    path = Path(path).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise ValueError("--model must be a local HF directory")
    files = []
    for p in sorted(path.iterdir()):
        if p.is_file() and (p.suffix in (".json", ".safetensors", ".bin", ".model") or
                            p.name in ("merges.txt", "vocab.txt", "chat_template.jinja")):
            s = p.stat()
            item = {"name": p.name, "size": s.st_size, "mtime_ns": s.st_mtime_ns}
            if s.st_size < 16 * 1024 * 1024:
                item["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
            files.append(item)
    if not (path / "config.json").is_file():
        raise ValueError("HF directory has no config.json")
    return {"path": str(path), "files": files,
            "weight_identity": "path-size-mtime; small config/tokenizer files additionally SHA256"}


class QueryCache:
    """Successful queries committed after every batch, including unfinished rows."""
    def __init__(self, path, config_hash, resume=False):
        path = Path(path)
        if path.exists() and not resume:
            raise ValueError("Query cache exists; use --resume or a new output path")
        self.db = sqlite3.connect(str(path))
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        prior = self.db.execute("SELECT value FROM metadata WHERE key='config_hash'").fetchone()
        if prior and prior[0] != config_hash:
            self.db.close()
            raise ValueError("Query cache configuration differs; choose a new output path")
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('config_hash', ?)", (config_hash,))
        self.db.execute("CREATE TABLE IF NOT EXISTS queries (key TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        self.db.commit()

    def get(self, key):
        value = self.db.execute("SELECT payload FROM queries WHERE key=?", (key,)).fetchone()
        return json.loads(value[0]) if value else None

    def put_many(self, values):
        with self.db:
            self.db.executemany("INSERT OR REPLACE INTO queries VALUES (?, ?)",
                                [(key, json.dumps(value, ensure_ascii=False, allow_nan=False))
                                 for key, value in values if not value.get("error")])

    def close(self):
        self.db.close()


def cropped_waveform(pcm, source_rate, start, end):
    """Crop in source coordinates BEFORE resampling: never read future samples."""
    import numpy as np
    if not 0 <= start < end <= len(pcm):
        raise ValueError("Teacher crop outside source samples")
    value = np.asarray(pcm[start:end], dtype=np.float32)
    if source_rate != 16000:
        from scipy.signal import resample_poly
        divisor = math.gcd(source_rate, 16000)
        value = resample_poly(value, 16000 // divisor, source_rate // divisor)
    return np.ascontiguousarray(value, dtype=np.float32)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--model", required=True, type=Path, help="Local offline ASR HF checkpoint; not MCore")
    p.add_argument("--audio-root", type=Path, help="Root for relative paths; absolute paths are used unchanged")
    p.add_argument("--chunk-seconds", type=float, choices=(.5, .8, 1.), default=.5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--gpu-ids", default=None, help="CUDA_VISIBLE_DEVICES, set before importing torch")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-context-tokens", type=int, default=32768)
    p.add_argument("--max-audio-seconds", type=float, default=300.)
    p.add_argument("--number-format", choices=("normalized", "verbatim"), default="normalized")
    p.add_argument("--resume", action="store_true")
    return p


def run(args, teacher_factory=None):
    import numpy as np
    import soundfile as sf
    if args.batch_size < 1 or not math.isfinite(args.max_audio_seconds) or args.max_audio_seconds <= 0:
        raise ValueError("batch-size and max-audio-seconds must be positive")
    source_path, output = args.input.resolve(strict=True), args.output.resolve()
    cache_path = Path(str(output) + ".queries.sqlite3")
    report_path = Path(str(output) + ".summary.json")
    meta_path = Path(str(output) + ".run.json")
    if source_path in (output, cache_path, report_path, meta_path):
        raise ValueError("Input must differ from output and sidecars")
    if not args.resume and any(p.exists() for p in (output, cache_path, report_path, meta_path)):
        raise ValueError("Output/sidecar already exists; --resume or choose a new output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    signature = model_signature(args.model)
    config = {"version": VERSION, "model": signature, "input": str(source_path),
              "contract_code_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                       for name in ("prefix_collection.py", "prefix_source.py", "prefix_hf.py")},
              "audio_root": str((args.audio_root or source_path.parent).resolve()),
              "chunk_seconds": args.chunk_seconds, "number_format": args.number_format,
              "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens,
              "max_context_tokens": args.max_context_tokens, "device_map": args.device_map,
              "dtype": args.dtype, "attn_implementation": args.attn_implementation,
              "gpu_ids": args.gpu_ids, "max_audio_seconds": args.max_audio_seconds}
    config_hash = fingerprint(config)
    if meta_path.exists():
        if json.loads(meta_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Resume configuration differs from run metadata")
    else:
        meta_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    journal = Journal(output, config, args.resume)
    try:
        cache = QueryCache(cache_path, config_hash, args.resume)
    except Exception:
        journal.close(check_excess=False)
        raise
    counts = {"written": 0, "ok": 0, "failed": 0, "resumed": 0, "query_cache_hits": 0,
              "queries_inferred": 0, "real_hf_inference_executed": False,
              "resumed_ok": 0, "resumed_failed": 0}
    teacher = None
    completed = False
    started = time.monotonic()
    print(json.dumps({"output": str(output), "cache": str(cache_path), "model": signature["path"],
                      "chunk_seconds": args.chunk_seconds, "batch_size": args.batch_size}), flush=True)
    try:
        for line, record, parse_error in source_rows(source_path):
            info, parsed, audio, error = None, None, None, parse_error
            try:
                identifier = row_id(record, source_path, line)
                identifier.encode("utf-8")
                json.dumps(record, ensure_ascii=False).encode("utf-8")
            except ValueError as exc:
                identifier, error = row_id({}, source_path, line), str(exc)
            try:
                if error:
                    raise ValueError(error)
                audios = record.get("audios")
                if not isinstance(audios, list) or len(audios) != 1 or not isinstance(audios[0], str):
                    raise ValueError("Exactly one local audio path is required")
                audio = Path(audios[0]).expanduser()
                if not audio.is_absolute():
                    audio = (args.audio_root or source_path.parent) / audio
                audio = audio.resolve(strict=True)
                info = inspect_audio(audio, args.max_audio_seconds)
                parsed = parse_source(record, info["duration_ms"])
                queries = plan_queries(parsed, info["frames"], info["sample_rate"], args.chunk_seconds)
            except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
                error = str(exc)
            try:
                input_hash = fingerprint({"record": record, "audio": str(audio) if audio else None, "audio_info": info})
            except UnicodeEncodeError:
                input_hash = hashlib.sha256(json.dumps(record, ensure_ascii=True, sort_keys=True).encode()).hexdigest()
                identifier, error = row_id({}, source_path, line), "Input contains invalid Unicode surrogate characters"
            previous = journal.previous(input_hash, identifier)
            if previous is not None:
                counts["resumed"] += 1
                counts["resumed_ok" if previous.get("status") == "ok" else "resumed_failed"] += 1
                continue
            result = {"schema_version": 1, "kind": "local_prefix_trace", "id": identifier,
                      "source_line": line, "input_hash": input_hash, "audio": str(audio) if audio else None,
                      "audio_info": info, "chunk_seconds": args.chunk_seconds,
                      "number_format": args.number_format, "teacher": {"model": signature["path"],
                      "config_hash": config_hash, "reference_in_prompt": False}}
            if not error:
                try:
                    pcm, rate = sf.read(str(audio), dtype="float32", always_2d=True)
                    if rate != info["sample_rate"] or len(pcm) != info["frames"] or not np.isfinite(pcm).all():
                        raise ValueError("Audio changed or contains invalid samples")
                    pcm = pcm.mean(axis=1, dtype=np.float32)
                    waveform_hash = hashlib.sha256(pcm.tobytes()).hexdigest()
                    result["waveform_sha256"] = waveform_hash
                    pending, responses = [], {}
                    for q in queries:
                        key = fingerprint({"input": input_hash, "pcm": waveform_hash, "query": q})
                        prior = cache.get(key)
                        if prior is None:
                            pending.append((q, key))
                        else:
                            responses[q["query_id"]] = prior
                            counts["query_cache_hits"] += 1
                except (ValueError, TypeError, OSError, RuntimeError) as exc:
                    error = str(exc)
            if not error:
                # Lazy loading is outside the per-record error path: an invalid
                # checkpoint/unsupported runtime is a job failure, not bad data.
                if pending and teacher is None:
                    if args.gpu_ids is not None:
                        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
                    if teacher_factory is None:
                        from .prefix_hf import HFPrefixTeacher
                        teacher_factory = HFPrefixTeacher
                    teacher = teacher_factory(signature["path"], max_new_tokens=args.max_new_tokens,
                        max_context_tokens=args.max_context_tokens, device_map=args.device_map,
                        dtype=args.dtype, attn_implementation=args.attn_implementation,
                        number_format=args.number_format)
                # Similar durations share a batch. Batch rows are independent
                # prefixes, with no cross-sample attention or reference context.
                pending.sort(key=lambda item: item[0]["end_sample"] - item[0]["start_sample"])
                for offset in range(0, len(pending), args.batch_size):
                    batch = pending[offset:offset + args.batch_size]
                    try:
                        waves = [cropped_waveform(pcm, rate, q["start_sample"], q["end_sample"]) for q, _ in batch]
                        languages = [parsed["language_runs"][q["run_index"]]["language"] for q, _ in batch]
                        predictions = teacher.transcribe_batch(waves, languages)
                        if len(predictions) != len(batch):
                            raise ValueError("Teacher batch result count mismatch")
                    except (ValueError, TypeError) as exc:
                        predictions = [{"text": "", "token_ids": [], "error": str(exc), "prompt_tokens": None}
                                       for _ in batch]
                    cached = []
                    for (q, key), prediction in zip(batch, predictions):
                        if not isinstance(prediction, dict) or not isinstance(prediction.get("text"), str):
                            prediction = {"text": "", "token_ids": [], "error": "Malformed teacher result"}
                        responses[q["query_id"]] = prediction
                        cached.append((key, prediction))
                    # SQLite/file errors propagate; pretending to save is unsafe.
                    cache.put_many(cached)
                    counts["queries_inferred"] += len(batch)
                    counts["real_hf_inference_executed"] = teacher_factory.__module__ == "omnicommit.prefix_hf"
                    print(f"[prefix] line={line} inferred={min(offset + len(batch), len(pending))}/{len(pending)} "
                          f"cached={len(queries) - len(pending)}", flush=True)
                result["source"] = parsed
                result["queries"] = [{**q, **responses[q["query_id"]]} for q in queries]
                failures = [q for q in result["queries"] if q.get("error")]
                if failures:
                    error = f"{len(failures)} teacher queries failed: {failures[0]['error']}"
            result.update(status="error" if error else "ok", error=error)
            if error:
                result["output"] = ""
                counts["failed"] += 1
                print(f"[reject] line={line}: {error}", file=sys.stderr, flush=True)
            else:
                counts["ok"] += 1
            journal.append(result)
            counts["written"] += 1
        completed = True
    finally:
        cache.close()
        journal.close(check_excess=completed)
    counts["elapsed_seconds"] = time.monotonic() - started
    report_path.write_text(json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(counts, ensure_ascii=False), flush=True)
    return counts


def main(argv=None):
    try:
        counts = run(parser().parse_args(argv))
        return 0 if counts["ok"] or counts["resumed_ok"] else 1
    except Exception as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2
