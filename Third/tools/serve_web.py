#!/usr/bin/env python3
"""Start the Third streaming ASR Web service."""
import argparse
from pathlib import Path
import sys

THIRD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THIRD))

from third_asr.runtime.stream_cli import add_runtime_arguments, runtime_options
from third_asr.service_backend import build_backend, detect_model_protocol
from third_asr.service_session import ThirdStreamOptions
from third_asr.service_web import add_web_arguments, create_app, serve_web


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    add_web_arguments(parser)
    parser.add_argument("--chunk-ms", type=int, default=800, help="Audio step used during SFT")
    parser.add_argument("--protocol", choices=["auto", "explicit", "plain"], default="auto",
                        help="Auto-detect the trained action format from the merged checkpoint")
    parser.add_argument("--recovery", dest="enable_recovery", action=argparse.BooleanOptionalAction,
                        default=True, help="Retry sustained voiced WAITs on explicit checkpoints")
    parser.add_argument("--recovery-voiced-seconds", type=float, default=2.4)
    parser.add_argument("--recovery-overlap-seconds", type=float, default=.8)
    parser.add_argument("--recovery-window-seconds", type=float, default=6.4)
    parser.set_defaults(backend="vllm", port=18891, context_policy="rolling", number_format="normalized")
    args = parser.parse_args(argv)
    args.chunk_seconds = args.chunk_ms / 1000
    if args.model is None and args.backend != "replay":
        parser.error("--model is required for HF and vLLM inference")
    try:
        if args.backend != "replay":
            trained = detect_model_protocol(args.model)
            if args.protocol not in ("auto", trained):
                parser.error(f"Checkpoint uses {trained!r}, incompatible with --protocol {args.protocol}")
            args.protocol = trained
        elif args.protocol == "auto":
            args.protocol = "explicit"
        args.enable_recovery = args.protocol == "explicit" and args.enable_recovery
        options = runtime_options(args, ThirdStreamOptions)
        if args.enable_recovery:
            from third_asr.recovery import check_recovery_dependencies
            check_recovery_dependencies()
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        parser.error(str(exc))
    serve_web(parser, args, lambda: build_backend(args), options, create_app)


if __name__ == "__main__":
    main()
