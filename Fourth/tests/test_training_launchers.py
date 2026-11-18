"""Launch Fourth with its own checker and template from an isolated deployment."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


FOURTH = Path(__file__).resolve().parents[1]
GPU_IDS = "0,1,2,3,4,5,6,7"


def find_bash():
    if os.name == "nt":
        git = shutil.which("git")
        if git:
            bash = Path(git).resolve().parents[1] / "bin" / "bash.exe"
            if bash.is_file():
                return str(bash)
    else:
        bash = shutil.which("bash")
        if bash:
            return bash
    pytest.skip("Bash or Git Bash is required for launcher tests")


@pytest.mark.parametrize("launcher,epochs,save", [
    ("train_sft.sh", "--max_epochs", "--save_interval"),
    ("train_sft_swift4.sh", "--num_train_epochs", "--save_steps"),
])
@pytest.mark.parametrize("vocab_update", [None, "full"])
def test_third_launch_from_isolated_deployment(tmp_path, launcher, epochs, save, vocab_update):
    bash = find_bash()
    work = tmp_path / "Fourth"
    scripts = work / "SFT-Scripts"
    scripts.mkdir(parents=True)
    shutil.copy(FOURTH / "SFT-Scripts" / launcher, scripts / launcher)

    dataset = "train.1000ms.jsonl"
    tokens = dataset + ".tokens.txt"
    (work / "model").mkdir()
    (work / "model" / "config.json").write_text("{}", encoding="utf-8")
    (work / dataset).write_text("{}\n", encoding="utf-8")
    (work / tokens).write_text("<|WAIT|>\n<|DIRECT|>\n", encoding="utf-8")
    mock_bin = work / "mock-bin"
    mock_bin.mkdir()
    for name in ("python3", "megatron"):
        mock = mock_bin / name
        mock.write_bytes(b'''#!/usr/bin/env bash
if [[ "$1" == *check_sft.py ]]; then
    printf '%s\n' "$@" > checker.argv
    printf '%s\n' "${OMNICOMMIT_ACTION_ROWS-unset}" > checker.mode
else
    printf '%s\n' "$@" > trainer.argv
    printf '%s\n' "$CUDA_VISIBLE_DEVICES" > trainer.gpus
    printf '%s\n' "${OMNICOMMIT_ACTION_ROWS-unset}" "${OMNICOMMIT_ACTION_TOKENS_FILE-unset}" > trainer.mode
fi
''')
        mock.chmod(0o755)

    options = {"--attention_backend": "flash", "--packing": "false", "--padding_free": "false",
               "--micro_batch_size": "4", "--global_batch_size": "64", "--max_length": "4096",
               epochs: "1", "--split_dataset_ratio": "0", save: "1000"}
    run = subprocess.run(
        [bash, "-c", 'export PATH="$PWD/mock-bin:$PATH"; exec bash "$@"', "test",
         f"SFT-Scripts/{launcher}", "model", dataset, "outputs", "--gpu-ids", GPU_IDS,
         *(["--vocab-update", vocab_update] if vocab_update else []),
         *(arg for pair in options.items() for arg in pair)],
        cwd=work, capture_output=True, text=True, timeout=30,
        env={**os.environ, "OMNICOMMIT_ACTION_ROWS": "stale", "OMNICOMMIT_ACTION_TOKENS_FILE": "stale"},
    )
    assert run.returncode == 0, run.stdout + run.stderr
    check = (work / "checker.argv").read_text().splitlines()
    assert check[0].endswith("/Fourth/SFT-Scripts/check_sft.py")
    assert check[1] == dataset and "--framework" in check
    assert check[check.index("--tokens-file") + 1] == tokens
    train = (work / "trainer.argv").read_text().splitlines()
    # CLI overrides follow defaults; check the final value passed to SWIFT.
    actual = {key: value for key, value in zip(train, train[1:]) if key.startswith("--")}
    for key, value in options.items():
        assert actual[key] == value
    assert actual["--dataset"] == dataset
    assert actual["--new_special_tokens"] == tokens
    assert actual["--template"] == "omnicommit_third_qwen3_omni"
    assert actual["--external_plugins"].endswith("/Fourth/SFT-Scripts/omnicommit_fourth_sft_plugin.py")
    assert (work / "trainer.gpus").read_text().strip() == GPU_IDS
    assert (work / "checker.mode").read_text().strip() == "unset"
    assert (work / "trainer.mode").read_text().splitlines() == (
        ["unset", "unset"] if vocab_update == "full" else ["1", tokens])
    assert "--vocab-update" not in train
