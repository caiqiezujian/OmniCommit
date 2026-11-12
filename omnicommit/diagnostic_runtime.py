"""Select the same meeting backend and streaming protocol for local file inference."""
from __future__ import annotations

import argparse
from functools import partial
import json
import os
from pathlib import Path

from .runtime_protocol import END_OF_AUDIO, MAX_CHUNK_MS
from .stream_cli import add_runtime_arguments, runtime_options
from .streaming import DIRECT, WAIT, StreamOptions, StreamSession
from .web_service import build_web_backend


MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def add_diagnostic_arguments(parser):
    add_runtime_arguments(parser, include_chunk=False)
    parser.add_argument("--protocol", choices=("auto", "first", "second"), default="auto",
                        help="Default: infer the ASR protocol from the merged checkpoint manifest")
    parser.add_argument("--chunk-ms", type=_chunk_seconds, dest="chunk_seconds", metavar="MS",
                        help=f"New audio per turn in integer milliseconds (1–{MAX_CHUNK_MS}; default: 500)")
    # Mirror start_server.sh and Second/tools/serve_web.py.
    parser.set_defaults(chunk_seconds=.5, backend="vllm", context_policy="rolling",
                        number_format="normalized")


def _chunk_seconds(value):
    try:
        ms = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--chunk-ms needs an integer in milliseconds") from exc
    if not 1 <= ms <= MAX_CHUNK_MS:
        raise argparse.ArgumentTypeError(f"--chunk-ms must be between 1 and {MAX_CHUNK_MS}")
    return ms / 1000


def _checkpoint_protocol(path):
    manifest_path = path / "omnicommit_tokens.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read merged checkpoint manifest {manifest_path}: {exc}") from exc
    if isinstance(manifest, dict) and type(manifest.get("schema_version")) is int:
        protocol = manifest.get("protocol")
        if manifest["schema_version"] == 1:
            if protocol == {"wait": WAIT, "direct": DIRECT, "end_of_audio": END_OF_AUDIO}:
                return "first"
            # First-generation services also accept manifests with the action
            # inventory and tokenizer IDs but no separate protocol block.
            tokens = manifest.get("tokens")
            if "protocol" not in manifest and isinstance(tokens, list) and WAIT in tokens and DIRECT in tokens:
                return "first"
        if manifest["schema_version"] == 2 and protocol == {
                "name": "omnicommit_second", "wait": WAIT, "end_of_audio": END_OF_AUDIO}:
            return "second"
    raise ValueError(f"Unrecognized ASR protocol in {manifest_path}; use a merged OmniCommit checkpoint")


def _resolve_checkpoint(args):
    selected = args.model or os.environ.get("OMNICOMMIT_MODEL")
    if selected:
        candidates = [Path(selected).expanduser().resolve()]
    else:
        defaults = (MODELS_DIR / "omnicommit-merged", MODELS_DIR / "second-merged")
        candidates = [path for path in defaults if (path / "omnicommit_tokens.json").is_file()]
    if not candidates:
        raise ValueError(f"No merged model found in {MODELS_DIR}; pass --model PATH or set OMNICOMMIT_MODEL")
    matches = []
    for path in candidates:
        protocol = _checkpoint_protocol(path)
        if args.protocol in ("auto", protocol):
            matches.append((path, protocol))
        elif selected:
            raise ValueError(f"Checkpoint {path} uses protocol {protocol}, conflicting with --protocol {args.protocol}")
    if not matches:
        raise ValueError(f"No {args.protocol} checkpoint found in {MODELS_DIR}; pass --model PATH")
    if len(matches) > 1:
        paths = ", ".join(str(path) for path, _ in matches)
        raise ValueError(f"Multiple merged checkpoints match ({paths}); select --model PATH or --protocol")
    return matches[0]


def resolve_runtime(args):
    """Resolve protocol/options now and return an unevaluated backend constructor."""
    if args.backend == "replay":
        args.model = None
        protocol = "first" if args.protocol == "auto" else args.protocol
    else:
        args.model, protocol = _resolve_checkpoint(args)

    if protocol == "second":
        from Second.second_asr.service_backend import build_backend
        from Second.second_asr.service_session import SecondStreamOptions, SecondStreamSession
        return protocol, runtime_options(args, SecondStreamOptions), SecondStreamSession, partial(build_backend, args)
    return protocol, runtime_options(args, StreamOptions), StreamSession, partial(build_web_backend, args)
