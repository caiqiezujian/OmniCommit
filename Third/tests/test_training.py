"""The Third training contract must reject old headers and supervise every text turn."""
import json
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace
import wave

import numpy as np
import pytest

THIRD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THIRD))
from third_asr.protocol import DIRECT, EOS, WAIT, get_protocol
from third_asr.service_backend import build_backend, detect_model_protocol, read_export_protocol, validate_manifest
from third_asr.sft_checks import read_tokens, validate_row
from third_asr.chunk_refs import encode_chunk_ref


def row_for(path, replies, mode="plain"):
    messages = [{"role": "system", "content": get_protocol(mode).system_prompt()}]
    refs = []
    for i, reply in enumerate(replies):
        refs.append(encode_chunk_ref(str(path), i * .8, (i + 1) * .8, .8))
        messages.extend([{"role": "user", "content": "<audio>"},
                         {"role": "assistant", "content": reply}])
    messages.extend([{"role": "user", "content": EOS}, {"role": "assistant", "content": WAIT}])
    return {"audios": refs, "messages": messages}


@pytest.mark.parametrize("reply", ["<en><spk2>Wait.", "<|DIRECT|>Hello.", "Hi<|WAIT|>", "   "])
def test_attributed_and_mixed_outputs_fail(tmp_path, reply):
    with pytest.raises(ValueError, match="audio turn 1"):
        validate_row(row_for(tmp_path / "sample.wav", [reply]))


def test_third_check_cli_decodes_multilingual_turns(tmp_path):
    audio = tmp_path / "sample.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", ""))
        wav.writeframes(bytes(3 * 12800 * 2))
    data = tmp_path / "training.jsonl"
    data.write_text(json.dumps(row_for(audio, [WAIT, "你好。Hi,", " again."])), encoding="utf8")
    tokens = Path(str(data) + ".tokens.txt")
    tokens.write_text(WAIT + "\n", encoding="utf8")
    result = subprocess.run([sys.executable, str(THIRD / "SFT-Scripts/check_sft.py"), str(data), "--decode"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["checked_rows"] == 1
    tokens.write_text(WAIT + "\n<en>\n", encoding="utf8")
    with pytest.raises(ValueError, match="exactly"):
        read_tokens(tokens)
    with pytest.raises(ValueError, match="no transcribed speech"):
        validate_row(row_for(audio, [WAIT]))


@pytest.mark.parametrize("mode", ["plain", "explicit"])
def test_training_sidecar_matches_every_turn_and_its_loss(mode, tmp_path):
    profile = get_protocol(mode)
    prefix = DIRECT if mode == "explicit" else ""
    row = row_for(tmp_path / "a.wav", [WAIT, prefix + "你好。Let's", prefix + " meet."], mode)
    sidecar = tmp_path / "train.jsonl.tokens.txt"
    sidecar.write_text("\n".join(profile.tokens) + "\n", encoding="utf-8")
    inventory = read_tokens(sidecar)
    assert validate_row(row, inventory) == inventory
    other = get_protocol("explicit" if mode == "plain" else "plain")
    with pytest.raises(ValueError, match="system instruction"):
        validate_row(row, set(other.tokens))
    if mode == "explicit":
        row["messages"][4]["content"] = "missing DIRECT"
        with pytest.raises(ValueError, match="audio turn 2"):
            validate_row(row, inventory)
        row["messages"][4]["content"] = DIRECT + "你好。Let's"

    # The encoded preflight must count every action in assistant labels, even
    # when DIRECT also occurs in the (masked) system prompt.
    protocol_ids = {token: i + 7 for i, token in enumerate(profile.tokens)}
    body_labels = [protocol_ids[DIRECT]] if mode == "explicit" else []
    labels = [-100, protocol_ids[WAIT], *body_labels, 100, *body_labels, 200, protocol_ids[WAIT]]
    encoded = {"labels": labels, "input_features": np.zeros((3, 1)), "feature_attention_mask": np.zeros((3, 1))}
    check = runpy.run_path(str(THIRD / "SFT-Scripts/audit_encoded_lengths.py"))["check_encoded_protocol"]
    check(row, encoded, protocol_ids)
    if mode == "explicit":
        labels[2] = -100
        with pytest.raises(ValueError, match="Supervised protocol tokens"):
            check(row, encoded, protocol_ids)


@pytest.mark.parametrize("mode,version", [("plain", 3), ("explicit", 4)])
def test_export_manifest_selects_serving_protocol(mode, version, tmp_path):
    profile = get_protocol(mode)
    sidecar = tmp_path / "tokens.txt"
    sidecar.write_text("\n".join(profile.tokens) + "\n", encoding="utf-8")
    tokens, metadata = read_export_protocol(sidecar)
    assert tokens == list(profile.tokens) and metadata["schema_version"] == version
    ids = {token: 8 + i for i, token in enumerate(tokens)}

    class Tokenizer:
        all_special_tokens = tokens

        def __len__(self):
            return 8 + len(tokens)

        def encode(self, token, **kwargs):
            return [ids[token]]

    manifest = {**metadata, "tokens": tokens, "token_ids": ids, "tokenizer_length": len(Tokenizer())}
    (tmp_path / "omnicommit_tokens.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert detect_model_protocol(tmp_path) == mode
    assert validate_manifest(Tokenizer(), manifest) == set(tokens)
    if mode == "explicit":
        corrupt = {**manifest, "schema_version": 3}
        with pytest.raises(ValueError, match="Expected a Third"):
            validate_manifest(Tokenizer(), corrupt)
        with pytest.raises(ValueError, match="mismatch"):
            validate_manifest(Tokenizer(), {**manifest, "token_ids": {WAIT: ids[DIRECT], DIRECT: ids[WAIT]}})
        sidecar.write_text("\n".join(tokens[::-1]), encoding="utf-8")
        with pytest.raises(ValueError):
            read_export_protocol(sidecar)


@pytest.mark.parametrize("mode,enabled,reserved", [("plain", True, None), ("explicit", False, None),
                                                   ("explicit", True, 5.0)])
def test_only_explicit_recovery_reserves_long_audio_items(mode, enabled, reserved, monkeypatch):
    monkeypatch.setattr("third_asr.runtime.stream_cli.build_backend", lambda args, **options: options)
    args = SimpleNamespace(protocol=mode, enable_recovery=enabled, recovery_window_seconds=5.0)
    options = build_backend(args)
    assert options["manifest_validator"] is validate_manifest
    assert options.get("max_audio_item_seconds") == reserved
