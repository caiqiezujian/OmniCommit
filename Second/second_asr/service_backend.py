"""Select HF/vLLM with the tokenizer and reply contract from Second training."""
from __future__ import annotations

import re

from .protocol import EOS, LANGUAGE, SPEAKER, WAIT, sort_tokens


def _metadata():
    return {"schema_version": 2, "protocol": {"name": "omnicommit_second", "wait": WAIT, "end_of_audio": EOS}}


def _check_tokens(tokens):
    if (not isinstance(tokens, list) or not tokens or tokens[0] != WAIT
            or not all(isinstance(token, str) for token in tokens) or len(set(tokens)) != len(tokens)):
        raise ValueError("Expected the complete Second tokens.txt, beginning with WAIT and containing unique tokens")
    sort_tokens(tokens)  # Reuse the training vocabulary grammar, keeping the actual training order.
    if not (any(re.fullmatch(rf"<{LANGUAGE}>", token) for token in tokens)
            and any(re.fullmatch(rf"<{SPEAKER}>", token) for token in tokens)):
        raise ValueError("Second model needs at least one trained language and one speaker token")


def read_export_protocol(tokens_file):
    tokens = tokens_file.read_text(encoding="utf-8-sig").splitlines()
    _check_tokens(tokens)
    return tokens, _metadata()


def validate_manifest(tokenizer, manifest):
    from omnicommit.inference_backend import validate_manifest_token_ids
    if not isinstance(manifest, dict) or any(manifest.get(k) != v for k, v in _metadata().items()):
        raise ValueError("Expected a Second model exported using Second/SFT-Scripts/merge_lora.sh")
    _check_tokens(manifest.get("tokens"))
    return validate_manifest_token_ids(tokenizer, manifest)


def build_backend(args):
    from omnicommit.web_service import build_web_backend
    return build_web_backend(args, manifest_validator=validate_manifest)
