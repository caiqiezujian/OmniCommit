#!/usr/bin/env python3
"""Start the Fourth streaming ASR Web service."""
import argparse
from functools import partial
import math
from pathlib import Path
import sys

FOURTH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FOURTH))

from fourth_asr.runtime.stream_cli import add_runtime_arguments, runtime_options
from fourth_asr.service_backend import build_backend, detect_model_protocol
from fourth_asr.service_session import FourthStreamOptions
from fourth_asr.service_web import add_web_arguments, create_app, serve_web


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    add_web_arguments(parser)
    parser.add_argument("--chunk-ms", type=int, default=1000, help="Audio step used during SFT")
    parser.add_argument("--window-seconds", type=float, help="Default live audio window, including the new chunk (60 s)")
    parser.add_argument("--discard-seconds", type=float, help="Oldest live audio to discard on overflow (50 s)")
    parser.add_argument("--max-window-seconds", type=float,
                        help="Largest selectable window; vLLM reserves audio slots for at least 120 s by default")
    parser.add_argument("--protocol", choices=["auto", "speaker", "explicit", "plain"], default="auto",
                        help="Auto-detect the trained action format from the merged checkpoint")
    parser.add_argument("--recovery", dest="enable_recovery", action=argparse.BooleanOptionalAction,
                        default=True, help="Retry sustained voiced WAITs on explicit checkpoints")
    parser.add_argument("--recovery-voiced-seconds", type=float, default=2.4)
    parser.add_argument("--recovery-overlap-seconds", type=float, default=.8)
    parser.add_argument("--recovery-window-seconds", type=float, default=6.4)
    parser.set_defaults(backend="vllm", port=18891, number_format="normalized")
    args = parser.parse_args(argv)
    args.chunk_seconds = args.chunk_ms / 1000
    if args.model is None and args.backend != "replay":
        parser.error("--model is required for HF and vLLM inference")
    try:
        if args.context_policy != "window" and any(value is not None for value in
                (args.window_seconds, args.discard_seconds, args.max_window_seconds)):
            parser.error("Window settings require --context-policy window")
        if args.window_seconds is None:
            args.window_seconds = FourthStreamOptions.window_seconds
        if args.discard_seconds is None:
            args.discard_seconds = FourthStreamOptions.discard_seconds
        if args.context_policy == "window":
            if args.max_window_seconds is None and args.backend == "vllm":
                args.max_window_seconds = max(120.0, args.window_seconds)
            if args.max_window_seconds is not None and (not math.isfinite(args.max_window_seconds)
                    or args.max_window_seconds < args.window_seconds):
                parser.error("--max-window-seconds must be finite and at least --window-seconds")
        if args.backend != "replay":
            trained = detect_model_protocol(args.model)
            if args.protocol not in ("auto", trained):
                parser.error(f"Checkpoint uses {trained!r}, incompatible with --protocol {args.protocol}")
            args.protocol = trained
        elif args.protocol == "auto":
            args.protocol = "speaker"
        args.enable_recovery = args.protocol in ("speaker", "explicit") and args.enable_recovery
        options = runtime_options(args, FourthStreamOptions)
        if args.enable_recovery:
            from fourth_asr.recovery import check_recovery_dependencies
            check_recovery_dependencies()
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        parser.error(str(exc))
    serve_web(parser, args, lambda: build_backend(args), options,
              partial(create_app, max_window_seconds=args.max_window_seconds))


if __name__ == "__main__":
    main()
