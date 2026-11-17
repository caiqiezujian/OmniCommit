#!/usr/bin/env python3
"""Serve incremental Third ASR to HTTP clients that identify meetings by ID."""
import argparse
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from third_asr.runtime.stream_cli import add_runtime_arguments, runtime_options
from third_asr.service_backend import build_backend, detect_model_protocol
from third_asr.service_session import ThirdStreamOptions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser, include_legacy_policies=False)
    parser.set_defaults(backend="vllm", number_format="normalized",
                        recovery_window_seconds=ThirdStreamOptions.recovery_window_seconds)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18892)
    parser.add_argument("--chunk-ms", type=int, default=1000, help="Model audio step in milliseconds (default: 1000)")
    parser.add_argument("--window-seconds", type=float, default=ThirdStreamOptions.window_seconds)
    parser.add_argument("--discard-seconds", type=float, default=ThirdStreamOptions.discard_seconds)
    parser.add_argument("--protocol", choices=("auto", "plain", "explicit"), default="auto")
    parser.add_argument("--recovery", action=argparse.BooleanOptionalAction, default=False,
                        help="Run the optional second decode on voiced WAITs of explicit checkpoints")
    parser.add_argument("--max-meetings", type=int, default=16,
                        help="Number of meeting histories kept concurrently")
    parser.add_argument("--max-in-flight", type=int, default=8,
                        help="Maximum number of simultaneous meeting operations accepted")
    parser.add_argument("--batch-size", dest="max_batch_size", type=int, default=4,
                        help="Maximum number of different meetings in a single vLLM generate call")
    parser.add_argument("--batch-wait-ms", type=float, default=2.0,
                        help="Maximum time to wait for a batch to fill after its first job arrives")
    # Accept existing deployment commands; POST now waits for the transcription.
    parser.add_argument("--response-timeout-ms", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--upload-timeout-seconds", type=float, default=10,
                        help="Release the meeting slot if the client stalls while uploading one packet")
    parser.add_argument("--idle-timeout-seconds", type=float, default=600)
    parser.add_argument("--max-packet-seconds", type=float, default=8.0)
    args = parser.parse_args(argv)
    if args.response_timeout_ms is not None:
        print("--response-timeout-ms is obsolete: successful POST requests wait for src.", file=sys.stderr)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in 1..65535")
    if args.backend != "replay" and args.model is None:
        parser.error("--model is required for real inference")
    if min(args.max_meetings, args.max_in_flight, args.max_batch_size) < 1:
        parser.error("meeting, in-flight and batch limits must be positive")
    for name in ("upload_timeout_seconds", "idle_timeout_seconds", "max_packet_seconds"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error("--" + name.replace("_", "-") + " must be positive and finite")
    if not math.isfinite(args.batch_wait_ms) or args.batch_wait_ms < 0:
        parser.error("--batch-wait-ms must be nonnegative and finite")
    try:
        args.chunk_seconds = args.chunk_ms / 1000
        args.max_window_seconds = args.window_seconds
        if args.backend != "replay":
            trained = detect_model_protocol(args.model)
            if args.protocol not in ("auto", trained):
                parser.error(f"Checkpoint uses {trained!r}, incompatible with --protocol {args.protocol}")
            args.protocol = trained
        elif args.protocol == "auto":
            args.protocol = "explicit"
        args.enable_recovery = args.protocol == "explicit" and args.recovery
        options = runtime_options(args, ThirdStreamOptions)
        if args.enable_recovery:
            from third_asr.recovery import check_recovery_dependencies
            check_recovery_dependencies()
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))

    from third_asr.batch_backend import BatchBackend
    from third_asr.http_service import create_app
    import uvicorn

    def load_model():
        return BatchBackend(lambda: build_backend(args), max_batch_size=args.max_batch_size,
                            batch_wait_ms=args.batch_wait_ms)

    app = create_app(load_model, options, max_meetings=args.max_meetings, max_in_flight=args.max_in_flight,
                     idle_timeout_seconds=args.idle_timeout_seconds,
                     max_packet_seconds=args.max_packet_seconds, upload_timeout_seconds=args.upload_timeout_seconds)
    # Meeting histories and one copy of the GPU engine live in this process.
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
