"""Exercise the native Swift 4 launch command without loading model weights."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

SCRIPT = Path(__file__).with_name("train_sft_swift4.sh")
BASH = "D:/Git/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")


def launch(tmp_path, args=(), *, hf_config=True, custom_tokens=False):
    if not BASH or not Path(BASH).is_file():
        pytest.skip("Bash is unavailable")
    model = tmp_path / "HF model"
    model.mkdir()
    if hf_config:
        (model / "config.json").write_text("{}")
    dataset = tmp_path / "streaming train.jsonl"
    dataset.write_text("{}\n")
    tokens = tmp_path / "explicit tokens.txt" if custom_tokens else Path(str(dataset) + ".tokens.txt")
    tokens.write_text("<|WAIT|>\n<|DIRECT|>\n")
    output = tmp_path / "packed output"
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    stub = b'''#!/usr/bin/env bash
printf 'CALL=%s\n' "${0##*/}"
printf 'ENV=%s,%s,%s,%s,%s\n' "$CUDA_VISIBLE_DEVICES" "$NPROC_PER_NODE" "$ENABLE_AUDIO_OUTPUT" "${OMNICOMMIT_SFT_ATTENTION_COMPAT-unset}" "${OMNICOMMIT_PACKED_NO_FLASH-unset}"
printf 'ARG=%s\n' "$@"
'''
    for name in ("python3", "megatron"):
        command = mock_bin / name
        command.write_bytes(stub)
        command.chmod(0o755)
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.update(OMNICOMMIT_SFT_ATTENTION_COMPAT="1", OMNICOMMIT_PACKED_NO_FLASH="1")
    args = ["--tokens-file", tokens.as_posix(), *args] if custom_tokens else list(args)
    add_path = 'mock_bin="$1"; shift; '
    if os.name == "nt":
        add_path += 'mock_bin="$(cygpath -u "$mock_bin")"; '
    result = subprocess.run(
        [BASH, "-c", add_path + 'export PATH="$mock_bin:$PATH"; exec bash "$@"', "test",
         mock_bin.as_posix(), SCRIPT.as_posix(), model.as_posix(), dataset.as_posix(), output.as_posix(), *args],
        env=env, capture_output=True, text=True, timeout=30,
    )
    return result, model.as_posix(), dataset.as_posix(), tokens.as_posix(), output.as_posix()


def call_args(stdout, name):
    block = stdout.split(f"CALL={name}\n", 1)[1].split("CALL=", 1)[0]
    return [line[4:] for line in block.splitlines() if line.startswith("ARG=")]


def test_native_packed_defaults_and_virtual_template(tmp_path):
    run, model, dataset, tokens, output = launch(tmp_path)
    assert run.returncode == 0, run.stderr
    assert run.stdout.index("CALL=python3") < run.stdout.index("CALL=megatron")
    assert run.stdout.count("ENV=0,1,2,3,4,5,6,7,8,0,unset,unset") == 2
    check = call_args(run.stdout, "python3")
    assert check[0].endswith("/check_sft.py") and "--framework" in check
    assert dataset in check and check[check.index("--tokens-file") + 1] == tokens
    train = call_args(run.stdout, "megatron")
    assert train[0] == "sft"
    expected = {
        "model": model, "dataset": dataset, "output_dir": output, "new_special_tokens": tokens,
        "template": "omnicommit_qwen3_omni", "tuner_type": "lora", "lora_rank": "8", "lora_alpha": "32",
        "packing": "true", "attention_backend": "flash", "max_length": "8192", "micro_batch_size": "4",
        "global_batch_size": "32", "num_train_epochs": "1", "save_steps": "1000", "split_dataset_ratio": "0.01",
    }
    for key, value in expected.items():
        assert train[train.index("--" + key) + 1] == value
    assert train[train.index("--external_plugins") + 1].endswith("/omnicommit_sft_plugin.py")
    i = train.index("--modules_to_save") + 1
    assert train[i:i + 2] == ["word_embeddings", "output_layer"]


@pytest.mark.parametrize("step_args", [("--train_iters", "2"), ("--train_iters=2",)])
def test_native_overrides_for_smoke_steps(tmp_path, step_args):
    run, _, _, tokens, _ = launch(tmp_path, ["--gpu-ids", "4,5,6,7", *step_args], custom_tokens=True)
    assert run.returncode == 0, run.stderr
    assert "ENV=4,5,6,7,4,0,unset,unset" in run.stdout
    check = call_args(run.stdout, "python3")
    train = call_args(run.stdout, "megatron")
    assert check[check.index("--tokens-file") + 1] == tokens
    assert train[train.index("--new_special_tokens") + 1] == tokens
    assert tuple(train[-len(step_args):]) == step_args and "--num_train_epochs" not in train


def test_native_rejects_model_without_hf_config(tmp_path):
    run, *_ = launch(tmp_path, hf_config=False)
    assert run.returncode != 0 and "config.json" in run.stderr
    assert "CALL=" not in run.stdout
