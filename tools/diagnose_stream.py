#!/usr/bin/env python3
"""Run an exported OmniCommit checkpoint on a media file and log each streaming turn."""
import argparse
from contextlib import closing, contextmanager
from dataclasses import asdict
from itertools import chain
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omnicommit.diagnostic_media import iter_media_chunks
from omnicommit.diagnostic_runtime import add_diagnostic_arguments, resolve_runtime
from omnicommit.streaming import SAMPLE_RATE


@contextmanager
def open_trace(path):
    """JSONL flushes each event. A JSON array also closes on handled inference errors."""
    as_array = path.suffix.lower() == ".json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        first = True
        if as_array:
            stream.write("[\n")

        def write(row):
            nonlocal first
            line = json.dumps(row, ensure_ascii=False, allow_nan=False)
            if as_array and not first:
                stream.write(",\n")
            stream.write(line if as_array else line + "\n")
            stream.flush()
            first = False

        try:
            yield write
        finally:
            if as_array:
                stream.write("\n]\n")


class ObservedBackend:
    """Observe replies before the session parser, including replies the protocol rejects."""
    def __init__(self, backend):
        self.backend = backend
        self.allowed_tokens = getattr(backend, "allowed_tokens", None)
        self.reset()

    def reset(self):
        self.called = False
        self.raw_text = None

    def generate(self, messages, audios):
        self.called = True
        generated = self.backend.generate(messages, audios)
        reply = generated.get("text") if isinstance(generated, dict) else generated
        if isinstance(reply, str):
            self.raw_text = reply
        return generated


def context_state(session):
    head = session.history[0].start if session.history else session.heard_seconds
    watermark = session.verified_commit_seconds
    if session.options.context_policy == "rolling":
        watermark = max(watermark, session.inferred_commit_seconds)
    return {"history_start_seconds": head, "commit_watermark_seconds": watermark,
            "pending_audio_seconds": session.heard_seconds - watermark}


def run_file(args, write):
    session = observed = None
    attempted_end = last_direct = None
    consecutive_waits = 0
    stage = "configuration"
    try:
        protocol, options, session_factory, backend_factory = resolve_runtime(args)
        stage = "read_media"
        with closing(iter_media_chunks(args.input, options.chunk_seconds, protocol)) as chunks:
            first_chunk = next(chunks, None)
            if first_chunk is None:
                raise ValueError("Media contains no audio samples")
            stage = "load_model"
            backend = backend_factory()
            observed = ObservedBackend(backend)
            session = session_factory(observed, options)
            write({"type": "ready", "protocol": protocol, "backend": type(backend).__name__,
                   "requested_dtype": args.dtype,
                   "enable_prefix_caching": getattr(backend, "enable_prefix_caching", None),
                   "mm_processor_cache_gb": getattr(backend, "mm_processor_cache_gb", None),
                   "model": str(args.model) if args.model else None,
                   "source_kind": "file", "input": str(args.input.resolve()),
                   "chunk_ms": round(options.chunk_seconds * 1000), **asdict(options),
                   "max_context_tokens": args.max_context_tokens, "max_new_tokens": args.max_new_tokens,
                   "automatic_reference": "disabled"})
            # EOF is a separate, text-only turn, as it is in the web file player.
            for pcm, final_chunk in chain((first_chunk,), chunks, ((None, False),)):
                eos = pcm is None
                stage = "end_of_audio" if eos else "audio"
                attempted_end = session.heard_seconds if eos else session.heard_seconds + len(pcm) / SAMPLE_RATE
                previous_head = session.history[0].start if session.history else session.heard_seconds
                observed.reset()
                result = session.step(pcm, end_of_audio=eos, final_chunk=final_chunk)
                if result["action"] == "DIRECT":
                    last_direct = result["end_seconds"]
                    consecutive_waits = 0
                elif not eos:
                    consecutive_waits += 1
                state = context_state(session)
                row = {"type": "result", **result, **state, "raw_model_text": observed.raw_text,
                       "consecutive_audio_waits": consecutive_waits, "last_direct_seconds": last_direct}
                if state["history_start_seconds"] > previous_head:
                    row["trimmed_interval_seconds"] = [previous_head, state["history_start_seconds"]]
                write(row)
                stage = "read_media"
                attempted_end = None
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        processing_turn = stage in ("audio", "end_of_audio")
        error = {"type": "error", "stage": stage, "error_type": type(exc).__name__,
                 "error": str(exc) or type(exc).__name__, "heard_seconds": session.heard_seconds if session else 0,
                 "backend_called": bool(processing_turn and observed and observed.called),
                 "raw_model_text": observed.raw_text if processing_turn and observed else None,
                 "consecutive_audio_waits": consecutive_waits, "last_direct_seconds": last_direct}
        if session is not None:
            error.update(context_state(session))
        if attempted_end is not None:
            error.update(attempted_start_seconds=session.heard_seconds, attempted_end_seconds=attempted_end)
        write(error)
        print(f"{stage}: {error['error']}", file=sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    finally:
        if session is not None:
            session.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Local audio or video file with an audio track")
    parser.add_argument("--output", type=Path, help="New .jsonl (one record per line) or .json (array) file")
    add_diagnostic_arguments(parser)
    args = parser.parse_args(argv)
    if not args.input.is_file():
        parser.error(f"Input file does not exist: {args.input}")
    output = args.output or args.input.with_suffix(f".{round(args.chunk_seconds * 1000)}ms.jsonl")
    if output.suffix.lower() not in (".json", ".jsonl"):
        parser.error("Use .jsonl or .json for the output file")
    try:
        with open_trace(output) as write:
            print(f"Streaming trace: {output.resolve()}", file=sys.stderr)
            return run_file(args, write)
    except FileExistsError:
        parser.error(f"Output already exists: {output}. Choose a new output path.")
    except OSError as exc:
        parser.exit(1, f"Cannot write trace {output}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
