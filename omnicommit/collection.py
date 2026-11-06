"""Collect independent cumulative-audio hypotheses, not forced-prefix decoding."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

from .audio import inspect_audio, prefix_endpoints, prefix_wav
from .io import Journal, fingerprint, row_id, source_rows, write_row
from .protocol import parse_tagged_text
from .teacher import AudioTeacher, ReplayTeacher, prompt_text


def resolve_audio(value, source_file, audio_root=None, path_maps=()):
    audios = value.get("audios")
    if not isinstance(audios, list) or len(audios) != 1 or not isinstance(audios[0], str):
        raise ValueError("Source requires exactly one audio path in audios")
    name = audios[0]
    for pair in path_maps:
        old, sep, new = pair.partition("=")
        if not sep or not old:
            raise ValueError("--path-map requires OLD=NEW")
        if name == old or name.startswith(old.rstrip("/\\") + "/") or name.startswith(old.rstrip("/\\") + "\\"):
            name = new.rstrip("/\\") + name[len(old.rstrip("/\\")):]
            break
    path = Path(name)
    if not path.is_absolute():
        path = Path(audio_root or Path(source_file).resolve().parent) / path
    return str(path.resolve())


def source_reference(value, strict=False):
    messages = value.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Source requires messages")
    answers = [m.get("content") for m in messages if isinstance(m, dict) and m.get("role") == "assistant"]
    if len(answers) != 1 or not isinstance(answers[0], str) or not answers[0].strip():
        raise ValueError("Source requires exactly one full assistant transcript")
    try:
        return answers[0], parse_tagged_text(answers[0]), None
    except ValueError as error:
        if strict:
            raise
        # Reference labels are an offline audit input, never teacher evidence.
        # Preserve ambiguity instead of guessing which empty speaker tag to drop.
        return answers[0], None, str(error)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="Original audios/messages JSONL")
    p.add_argument("--output", required=True, help="Append-only trace JSONL")
    p.add_argument("--config", help="JSON with api_base, model, generation, timeout")
    p.add_argument("--api-base")
    p.add_argument("--model")
    p.add_argument("--api-key-env", default="OMNICOMMIT_API_KEY")
    p.add_argument("--chunk-seconds", type=float, default=0.5)
    p.add_argument("--number-format", choices=("verbatim", "normalized"), default="verbatim")
    p.add_argument("--prompt", default=str(Path(__file__).resolve().parents[1] / "Data-Scripts/prompts/teacher_zh.txt"))
    p.add_argument("--audio-root")
    p.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW")
    p.add_argument("--max-audio-seconds", type=float, default=300.0)
    p.add_argument("--replay-responses", help="Ordered offline responses; makes zero HTTP requests")
    p.add_argument("--strict-reference", action="store_true",
                   help="Reject malformed reference tags; default preserves them for offline audit")
    p.add_argument("--resume", action="store_true")
    return p


def run(args):
    if Path(args.input).resolve() == Path(args.output).resolve():
        raise ValueError("Input and output must differ")
    for key in ("chunk_seconds", "max_audio_seconds"):
        number = getattr(args, key)
        if not math.isfinite(number) or number <= 0:
            raise ValueError(key + " must be positive and finite")
    config = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
    if not isinstance(config, dict):
        raise ValueError("Config must be an object")
    if set(config) - {"api_base", "model", "generation", "timeout"}:
        raise ValueError("Unknown config keys; secrets must use --api-key-env")
    base = args.api_base or config.get("api_base")
    model = args.model or config.get("model")
    prompt = prompt_text(args.prompt, args.number_format)
    generation = config.get("generation", {})
    timeout = config.get("timeout", 300)
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    if not args.replay_responses and (not base or not model):
        raise ValueError("Real inference requires explicit api_base and model via --config or CLI")
    replay_hash = None
    if args.replay_responses:
        import hashlib
        digest = hashlib.sha256()
        with open(args.replay_responses, "rb") as replay_file:
            for block in iter(lambda: replay_file.read(1024 * 1024), b""):
                digest.update(block)
        replay_hash = digest.hexdigest()
    settings = {"version": 1, "api_base": base, "model": model, "generation": generation,
                "timeout": timeout, "prompt_sha256": fingerprint(prompt),
                "number_format": args.number_format, "chunk_seconds": args.chunk_seconds,
                "max_audio_seconds": args.max_audio_seconds, "path_maps": args.path_map,
                "audio_root": args.audio_root, "replay_sha256": replay_hash,
                "strict_reference": args.strict_reference}
    teacher = (ReplayTeacher(args.replay_responses) if args.replay_responses else
               AudioTeacher(base, model, generation, prompt, timeout, os.getenv(args.api_key_env)))
    journal = Journal(args.output, settings, args.resume)
    counts = {"written": 0, "failed": 0, "resumed": 0}
    completed = False
    try:
        with open(str(args.output) + ".rejected.jsonl", "a", encoding="utf-8", newline="\n") as rejected:
            for line, value, parse_error in source_rows(args.input):
                prep_error, audio, info = parse_error, None, None
                try:
                    identifier = row_id(value, args.input, line)
                except ValueError as error:
                    identifier = row_id({}, args.input, line)
                    prep_error = str(error)
                try:
                    if prep_error:
                        raise ValueError(prep_error)
                    reference, reference_segments, reference_parse_error = source_reference(value, args.strict_reference)
                    audio = resolve_audio(value, args.input, args.audio_root, args.path_map)
                    info = inspect_audio(audio, args.max_audio_seconds)
                except Exception as error:
                    prep_error = str(error)
                input_hash = fingerprint({"source": value, "audio": audio, "audio_info": info})
                previous = journal.previous(input_hash, identifier)
                if previous is not None:
                    counts["resumed"] += 1
                    continue
                row = {"schema_version": 1, "id": identifier, "input_hash": input_hash,
                       "source_line": line, "audio": audio, "status": "ok"}
                stage = "input"
                try:
                    if prep_error:
                        raise ValueError(prep_error)
                    row.update(duration_ms=info["duration_ms"], audio_info=info,
                               reference=reference, reference_segments=reference_segments,
                               reference_parse_error=reference_parse_error,
                               teacher=settings, prefixes=[])
                    ends = prefix_endpoints(info, args.chunk_seconds)
                    stage = "full_audio"
                    print(f"[teacher] id={identifier} full_audio", flush=True)
                    if args.replay_responses:
                        teacher.select(identifier)
                        full = teacher.full()
                    else:
                        full = teacher.transcribe(prefix_wav(audio, info["duration_ms"]))
                    row["full"] = {"segments": full}
                    for end in ends:
                        stage = f"prefix_{end:.6f}ms"
                        print(f"[teacher] id={identifier} end_ms={end:.3f}", flush=True)
                        if end == info["duration_ms"]:
                            segments = full
                        elif args.replay_responses:
                            segments = teacher.prefix(end)
                        else:
                            segments = teacher.transcribe(prefix_wav(audio, end))
                        row["prefixes"].append({"end_ms": end, "segments": segments})
                    # A file changed during collection invalidates the whole trajectory.
                    if inspect_audio(audio, args.max_audio_seconds) != info:
                        raise ValueError("Audio changed during collection")
                except Exception as error:
                    row.update(status="error", output="", error={"stage": stage, "message": str(error)})
                    counts["failed"] += 1
                    write_row(rejected, {"id": identifier, "source_line": line, "audio": audio,
                                         "error": row["error"]})
                    print(f"[ERROR] id={identifier} stage={stage}: {error}", file=sys.stderr, flush=True)
                journal.append(row)
                counts["written"] += 1
        completed = True
    finally:
        teacher.close()
        journal.close(check_excess=completed)
    print(json.dumps(counts, ensure_ascii=False), flush=True)
    return counts


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        run(args)
    except Exception as error:
        print(f"[FATAL] {error}", file=sys.stderr)
        return 2
    return 0
