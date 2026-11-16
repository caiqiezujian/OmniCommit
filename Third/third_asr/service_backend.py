"""Load a merged Third checkpoint with its training-time tokenizer."""
from .protocol import EOS, WAIT


def _metadata():
    return {"schema_version": 3, "protocol": {
        "name": "omnicommit_third", "wait": WAIT, "end_of_audio": EOS}}


def read_export_protocol(tokens_file):
    tokens = tokens_file.read_text(encoding="utf-8-sig").splitlines()
    if tokens != [WAIT]:
        raise ValueError("Third tokens.txt must contain exactly <|WAIT|>")
    return tokens, _metadata()


def validate_manifest(tokenizer, manifest):
    from omnicommit.inference_backend import validate_manifest_token_ids
    if (not isinstance(manifest, dict) or manifest.get("tokens") != [WAIT]
            or any(manifest.get(k) != v for k, v in _metadata().items())):
        raise ValueError("Expected a Third export from Third/SFT-Scripts/merge_lora.sh")
    return validate_manifest_token_ids(tokenizer, manifest)


def build_backend(args):
    from omnicommit.stream_cli import build_backend as load_backend
    return load_backend(args, manifest_validator=validate_manifest)
