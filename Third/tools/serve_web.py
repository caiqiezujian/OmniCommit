#!/usr/bin/env python3
"""Start the Third streaming ASR Web service."""
import argparse
from pathlib import Path
import sys

THIRD = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(THIRD), str(THIRD.parent)]

from omnicommit.stream_cli import add_runtime_arguments, runtime_options
from omnicommit.web_service import add_web_arguments, serve_web
from third_asr.service_backend import build_backend
from third_asr.service_session import ThirdStreamOptions
from third_asr.service_web import create_app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser, include_chunk=False, include_references=False)
    add_web_arguments(parser, include_references=False)
    parser.add_argument("--chunk-ms", type=int, default=800, help="Audio step used during SFT")
    parser.set_defaults(backend="vllm", port=18891, context_policy="rolling", number_format="normalized")
    args = parser.parse_args(argv)
    args.chunk_seconds = args.chunk_ms / 1000
    if args.model is None and args.backend != "replay":
        parser.error("--model is required for HF and vLLM inference")
    try:
        options = runtime_options(args, ThirdStreamOptions)
    except ValueError as exc:
        parser.error(str(exc))
    serve_web(parser, args, lambda: build_backend(args), options, create_app)


if __name__ == "__main__":
    main()
