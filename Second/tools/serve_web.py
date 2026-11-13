#!/usr/bin/env python3
"""Serve a Second checkpoint or replay a Second transcript over WebSocket."""
import argparse
from functools import partial
from pathlib import Path
import sys

SECOND = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SECOND), str(SECOND.parent)]

from omnicommit.stream_cli import add_runtime_arguments, runtime_options
from omnicommit.web_service import add_web_arguments, serve_web
from second_asr.gap_backend import WhisperGapBackend, local_whisper_directory
from second_asr.service_backend import build_backend
from second_asr.service_session import SecondStreamOptions
from second_asr.service_web import create_app


def milliseconds(value):
    ms = int(value)
    if not 1 <= ms <= 8000:
        raise argparse.ArgumentTypeError("chunk-ms must be between 1 and 8000")
    return ms / 1000


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser, include_chunk=False)
    add_web_arguments(parser)
    parser.add_argument("--gap-asr-model", type=Path, metavar="LOCAL_CT2_DIR",
                        help="Recover missed speech in the background with a local multilingual "
                             "faster-whisper model on CPU (INT8)")
    chunk = parser.add_mutually_exclusive_group()
    chunk.add_argument("--chunk-ms", dest="chunk_seconds", type=milliseconds, metavar="MS",
                       help="Integer audio step in milliseconds, e.g. 500, 700, 800 or 1000")
    chunk.add_argument("--chunk-seconds", type=float, help="Equivalent step in seconds")
    parser.set_defaults(chunk_seconds=.5, number_format="normalized", port=18890,
                        backend="vllm", context_policy="rolling")
    args = parser.parse_args(argv)
    if args.backend != "replay" and args.model is None:
        parser.error("--model is required for real inference")
    try:
        options = runtime_options(args, SecondStreamOptions)
        app_factory = create_app
        if args.gap_asr_model is not None:
            gap_path = local_whisper_directory(args.gap_asr_model)
            app_factory = partial(create_app, gap_backend_factory=partial(WhisperGapBackend, gap_path))
    except ValueError as exc:
        parser.error(str(exc))
    serve_web(parser, args, lambda: build_backend(args), options, app_factory)


if __name__ == "__main__":
    main()
