"""Exercise the actual Bash launcher with a Python stub in place of the GPU service."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

FOURTH = Path(__file__).resolve().parents[1]


@pytest.fixture
def launch(tmp_path):
    if os.name == "nt":
        git = shutil.which("git")
        bash = Path(git).resolve().parents[1] / "bin" / "bash.exe" if git else None
    else:
        bash = shutil.which("bash")
    if not bash or not Path(bash).is_file():
        pytest.skip("Requires Bash or Git Bash")

    recipe = tmp_path / "Fourth"
    tools = recipe / "tools"
    tools.mkdir(parents=True)
    shutil.copy(FOURTH / "start_server.sh", recipe)
    (tools / "serve_web.py").write_text(
        "import json, os, sys\nprint(json.dumps([os.getenv('CUDA_VISIBLE_DEVICES'), sys.argv[1:]]))\n",
        encoding="utf-8",
    )

    def run(args=(), config_path=None):
        if config_path is not None:
            config = (FOURTH / "web_config.sh").read_text(encoding="utf-8")
            config = config.replace("/data/yb/models/silero_vad.onnx", config_path)
            (recipe / "web_config.sh").write_bytes(config.encode("utf-8"))
        return subprocess.run(
            [str(bash), "Fourth/start_server.sh", *args], cwd=tmp_path, capture_output=True,
            text=True, timeout=30, env={**os.environ, "PYTHON": Path(sys.executable).as_posix(),
            "CUDA_VISIBLE_DEVICES": "2", "MSYS2_ARG_CONV_EXCL": "*"},
        )

    return recipe, run


@pytest.mark.parametrize("gate", [True, False])
def test_no_args_read_config_and_only_pass_silero_when_enabled(launch, gate):
    recipe, run = launch
    model = "weights with spaces/silero.onnx"
    if gate:
        (recipe / model).parent.mkdir()
        (recipe / model).touch()
    proc = run(config_path=model if gate else "")
    assert proc.returncode == 0, proc.stderr
    gpu, args = json.loads(proc.stdout)
    assert gpu == "6"
    assert args[args.index("--model") + 1].endswith("sft-checkpoint-2721")
    assert args[args.index("--port") + 1] == "18895"
    assert args[args.index("--https-host") + 1] == "10.185.1.71"
    if gate:
        assert args[args.index("--silero-vad-model") + 1] == model
        assert float(args[args.index("--silero-quiet-threshold") + 1]) == 0.10
    else:
        assert "--silero-vad-model" not in args and "--silero-quiet-threshold" not in args


def test_requested_missing_vad_reports_editable_setting(launch):
    _, run = launch
    proc = run(config_path="missing.onnx")
    assert proc.returncode == 2 and "missing.onnx" in proc.stderr
    assert "Fourth/web_config.sh" in proc.stderr


def test_explicit_launch_bypasses_config_and_preserves_gpu(launch):
    _, run = launch
    proc = run(["my model", "--port", "8001"])
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == ["2", ["--model", "my model", "--host", "0.0.0.0",
                                            "--port", "8001"]]
