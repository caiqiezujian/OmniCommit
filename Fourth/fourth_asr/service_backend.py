"""Load a merged Fourth checkpoint with its training-time tokenizer."""
import json
from pathlib import Path

from .protocol import DIRECT, EOS, SPK_CHANGE, WAIT, protocol_for_tokens


def _metadata(name):
    if name == "speaker":
        return {"schema_version": 5, "protocol": {
            "name": "omnicommit_fourth_speaker", "wait": WAIT, "direct": DIRECT,
            "speaker_change": SPK_CHANGE, "end_of_audio": EOS}}
    if name == "explicit":
        return {"schema_version": 4, "protocol": {
            "name": "omnicommit_third_explicit", "wait": WAIT, "direct": DIRECT, "end_of_audio": EOS}}
    return {"schema_version": 3, "protocol": {
        "name": "omnicommit_third", "wait": WAIT, "end_of_audio": EOS}}


def read_export_protocol(tokens_file):
    tokens = Path(tokens_file).read_text(encoding="utf-8-sig").splitlines()
    profile = protocol_for_tokens(tokens)
    return tokens, _metadata(profile.name)


def _manifest_mode(manifest):
    if isinstance(manifest, dict) and isinstance(manifest.get("tokens"), list):
        profile = protocol_for_tokens(manifest["tokens"])
        if all(manifest.get(k) == v for k, v in _metadata(profile.name).items()):
            return profile.name
    raise ValueError("Expected a Fourth export from Fourth/SFT-Scripts/merge_lora.sh")


def detect_model_protocol(model_dir):
    """Read protocol metadata before allocating GPU memory; the backend verifies token IDs on load."""
    path = Path(model_dir) / "omnicommit_tokens.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read Fourth export manifest {path}: {exc}") from exc
    return _manifest_mode(manifest)


def validate_manifest(tokenizer, manifest):
    from .runtime.inference_backend import validate_manifest_token_ids
    _manifest_mode(manifest)
    return validate_manifest_token_ids(tokenizer, manifest)


def build_backend(args):
    from .runtime.stream_cli import build_backend as load_backend
    options = {}
    if getattr(args, "protocol", None) in ("speaker", "explicit") and getattr(args, "enable_recovery", False):
        options["max_audio_item_seconds"] = args.recovery_window_seconds
    return load_backend(args, manifest_validator=validate_manifest, **options)
