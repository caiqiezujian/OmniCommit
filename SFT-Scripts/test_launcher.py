"""Exercise shell argument/env construction with a fake Python; no GPU work."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

SCRIPT = Path(__file__).with_name("train_sft.sh")
BASH = "D:/Git/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")


def run_launcher(tmp_path, extra):
    if not BASH or not Path(BASH).is_file():
        pytest.skip("Bash is unavailable")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    python = fake_bin / "python3"
    python.write_bytes(('#!/usr/bin/env bash\nprintf "CALL\\n"\nprintf "ARG:%s\\n" "$@"\n'
                      'printf "CUDA:%s\\n" "$CUDA_VISIBLE_DEVICES"\n'
                      'printf "PACKED_NO_FLASH:%s\\n" "${OMNICOMMIT_PACKED_NO_FLASH-unset}"\n'
                      'if [[ "$1" == *check_no_flash_packing.py ]]; then exit "${NO_FLASH_PROBE_STATUS:-0}"; fi\n'
                      'if [[ "$1" == *check_packed_attention.py ]]; then exit "${PACKED_PROBE_STATUS:-0}"; fi\n'
                      'printf "ENV:%s,%s,%s,%s,%s\\n" "${OMNICOMMIT_SFT_ATTENTION_COMPAT-unset}" '
                      '"${NVTE_FLASH_ATTN-unset}" "${NVTE_FUSED_ATTN-unset}" '
                      '"${NVTE_UNFUSED_ATTN-unset}" "${NVTE_MASKED_SOFTMAX_FUSION-unset}"\n').encode("utf-8"))
    python.chmod(0o755)
    model = tmp_path / "model with spaces"
    model.mkdir()
    dataset = tmp_path / "train.jsonl"
    dataset.write_text("{}\n")
    Path(str(dataset) + ".tokens.txt").write_text("<|WAIT|>\n<|DIRECT|>\n")
    env = dict(os.environ)
    for key in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN", "NVTE_MASKED_SOFTMAX_FUSION",
                "OMNICOMMIT_SFT_ATTENTION_COMPAT", "CUDA_VISIBLE_DEVICES"):
        env.pop(key, None)
    prefix = 'mock_bin="$1"; shift; '
    if os.name == "nt":
        prefix += 'mock_bin="$(cygpath -u "$mock_bin")"; '
    return subprocess.run([BASH, "-c", prefix + 'export PATH="$mock_bin:$PATH"; exec bash "$@"',
                           "test", str(fake_bin), str(SCRIPT), str(model), str(dataset), str(tmp_path / "output"),
                           *extra], env=env, capture_output=True, text=True, timeout=30)


def test_default_launcher_uses_eight_gpus_and_8192_length(tmp_path):
    run = run_launcher(tmp_path, [])
    assert run.returncode == 0, run.stderr
    assert "ARG:--attention_backend\nARG:flash" in run.stdout
    assert "ARG:--packing\nARG:true" in run.stdout
    assert "CUDA:0,1,2,3,4,5,6,7" in run.stdout
    for flag, value in (("nproc_per_node", "8"), ("micro_batch_size", "2"),
                        ("global_batch_size", "16"), ("max_length", "8192")):
        assert f"ARG:--{flag}\nARG:{value}" in run.stdout
    assert "ARG:--padding_free" not in run.stdout
    assert "ENV:unset,unset,unset,unset,unset" in run.stdout


def test_packed_sdpa_checks_kernel_before_dataset_and_torchrun(tmp_path):
    run = run_launcher(tmp_path, ["--packed-sdpa", "--global_batch_size", "32"])
    assert run.returncode == 0, run.stderr
    assert run.stdout.index("check_packed_attention.py") < run.stdout.index("check_sft.py") < run.stdout.index("torch.distributed.run")
    for flag, value in (("attn_impl", "sdpa"), ("attention_backend", "flash"), ("packing", "true"),
                        ("padding_free", "true"), ("global_batch_size", "32")):
        assert f"ARG:--{flag}\nARG:{value}" in run.stdout
    assert "ENV:1,1,0,0,unset" in run.stdout


def test_failed_packed_probe_does_not_start_dataset_processing(tmp_path, monkeypatch):
    monkeypatch.setenv("PACKED_PROBE_STATUS", "17")
    run = run_launcher(tmp_path, ["--packed-sdpa"])
    assert run.returncode == 17
    assert "check_sft.py" not in run.stdout and "torch.distributed.run" not in run.stdout


def test_no_flash_packing_disables_flash_and_enables_adapter_only_for_training(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNICOMMIT_PACKED_NO_FLASH", "from-parent")
    run = run_launcher(tmp_path, ["--packed-no-flash", "--global_batch_size", "32"])
    assert run.returncode == 0, run.stderr
    assert "check_packed_attention.py" not in run.stdout
    assert run.stdout.index("check_no_flash_packing.py") < run.stdout.index("check_sft.py") < run.stdout.index("torch.distributed.run")
    assert run.stdout.count("PACKED_NO_FLASH:unset") == 2
    assert run.stdout.count("PACKED_NO_FLASH:1") == 1
    assert "ENV:1,0,0,1,0" in run.stdout
    for flag, value in (("attention_backend", "unfused"), ("attn_impl", "sdpa"), ("packing", "true"),
                        ("padding_free", "true"), ("micro_batch_size", "1"), ("global_batch_size", "32")):
        assert f"ARG:--{flag}\nARG:{value}" in run.stdout


def test_failed_no_flash_probe_stops_before_preprocessing(tmp_path, monkeypatch):
    monkeypatch.setenv("NO_FLASH_PROBE_STATUS", "18")
    run = run_launcher(tmp_path, ["--packed-no-flash"])
    assert run.returncode == 18
    assert "check_sft.py" not in run.stdout and "torch.distributed.run" not in run.stdout


@pytest.mark.parametrize("extra", [["--packed-sdpa", "--compat-old-driver"],
                                   ["--packed-no-flash", "--packed-sdpa"],
                                   ["--compat-old-driver", "--packed-no-flash"],
                                   ["--packed-no-flash", "--attention_backend=flash"],
                                   ["--packed-no-flash", "--packing", "false"],
                                   ["--compat-old-driver", "--packed-sdpa"],
                                   ["--packed-sdpa", "--attention_backend=fused"],
                                   ["--packed-sdpa", "--packing", "false"]])
def test_packed_preset_rejects_conflicting_layout(tmp_path, extra):
    run = run_launcher(tmp_path, extra)
    assert run.returncode == 2
    assert "CALL" not in run.stdout


def test_four_gpu_selection_uses_four_processes(tmp_path):
    run = run_launcher(tmp_path, ["--gpu-ids", "1,2,3,4"])
    assert run.returncode == 0, run.stderr
    assert "CUDA:1,2,3,4" in run.stdout
    assert "ARG:--nproc_per_node\nARG:4" in run.stdout
    assert "ARG:--micro_batch_size\nARG:4" in run.stdout


def test_compat_launcher_routes_both_attention_branches_and_preserves_global_batch(tmp_path):
    run = run_launcher(tmp_path, ["--compat-old-driver"])
    assert run.returncode == 0, run.stderr
    for flag, value in (("attn_impl", "sdpa"), ("attention_backend", "unfused"), ("packing", "false"),
                        ("padding_free", "false"), ("micro_batch_size", "1"), ("global_batch_size", "16")):
        assert f"ARG:--{flag}\nARG:{value}" in run.stdout
    assert "ENV:1,0,0,1,0" in run.stdout
    assert "ARG:--merge_lora\nARG:false" in run.stdout


@pytest.mark.parametrize("override", [["--packing", "true"], ["--padding_free=true"],
                                      ["--attention_backend", "auto"], ["--attn_impl", "flash_attn"]])
def test_compat_rejects_conflicting_attention_and_packing_overrides(tmp_path, override):
    run = run_launcher(tmp_path, ["--compat-old-driver", *override])
    assert run.returncode == 2
    assert "controls" in run.stderr and "CALL" not in run.stdout
