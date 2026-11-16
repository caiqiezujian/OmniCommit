"""The Third training contract must reject old headers and supervise every text turn."""
import json
from pathlib import Path
import runpy
import subprocess
import sys
import wave

import pytest

THIRD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THIRD))
from third_asr.protocol import EOS, WAIT, system_prompt
from third_asr.sft_checks import SHARED_SFT, read_tokens, validate_row
from omnicommit_audio import encode_chunk_ref


def row_for(path, replies):
    messages = [{"role": "system", "content": system_prompt()}]
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


def test_encoded_audit_uses_third_template_and_checks(monkeypatch):
    audit_main = runpy.run_path(str(THIRD / "SFT-Scripts/audit_encoded_lengths.py"))["main"]

    def shared_main(argv, *, checks, template_name, plugin_name):
        assert argv == ["arg"]
        assert checks.validate_row is validate_row
        assert template_name == "omnicommit_third_qwen3_omni"
        assert plugin_name == "omnicommit_third_sft_plugin"
        return 0

    def load_shared(script):
        assert Path(script) == SHARED_SFT / "audit_encoded_lengths.py"
        return {"main": shared_main}

    monkeypatch.setattr(runpy, "run_path", load_shared)
    assert audit_main(["arg"]) == 0
