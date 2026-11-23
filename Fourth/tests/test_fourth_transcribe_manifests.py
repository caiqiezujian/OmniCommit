"""Fourth manifest evaluation through its real chunk reader and independent sessions."""
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from types import SimpleNamespace
import wave

import pytest


ENTRY = Path(__file__).resolve().parents[1] / "tools" / "transcribe_manifests.py"
spec = spec_from_file_location("fourth_manifest_runner", ENTRY)
runner = module_from_spec(spec)
spec.loader.exec_module(runner)

from fourth_asr.protocol import DIRECT, EOS, SPK_CHANGE, WAIT, get_protocol


FIELD = "OmniCommit_test_fourth"


def make_wav(directory, name, seconds):
    path = directory / "splited_audios" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", ""))
        audio.writeframes(b"\x10\x00" * int(seconds * 16000))


def write_manifest(root, lang, rows, extension):
    path = root / lang / f"splited_manifest.{extension}"
    path.parent.mkdir(parents=True, exist_ok=True)
    if extension == "json":
        contents = json.dumps(rows, ensure_ascii=False)
    else:
        contents = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(contents, encoding="utf-8")
    return path


def read_rows(path):
    contents = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return json.loads(contents)
    return [json.loads(line) for line in contents.splitlines() if line.strip()]


class Replay:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.histories = []
        self.prompts = []

    def generate_batch(self, calls):
        results = []
        for messages, audios, prefix in calls:
            assert prefix == ""
            assert messages[0]["content"] == get_protocol("speaker").system_prompt("normalized")
            assert "SECRET_REFERENCE" not in str(messages)
            assert sum(m["content"].count("<audio>") for m in messages) == len(audios)
            self.histories.append([m["content"] for m in messages if m["role"] == "assistant"])
            self.prompts.append(messages[-1]["content"])
            reply = next(self.replies)
            results.append(reply if isinstance(reply, dict) else {"text": reply})
        return results


def run(root, make_model, languages=("de",), **kwargs):
    return runner.run(root, FIELD, languages, make_model, 4, 1000, 60, 50, **kwargs)


def no_load():
    pytest.fail("all requested predictions already exist")


@pytest.mark.parametrize("ext", ["json", "jsonl"])
def test_plain_predictions_preserve_raw_model_history_and_resume(tmp_path, ext):
    source = {"audio": r"C:\old\de\splited_audios\meeting.wav", "text": "SECRET_REFERENCE",
              "entities": [["Haus", "", ""]], "third_prediction": "existing"}
    already = [{"audio": "old.wav", FIELD: "saved"}, {"audio": "silent.wav", FIELD: ""}]
    path = write_manifest(tmp_path, "de", [source, *already], ext)
    make_wav(path.parent, "meeting.wav", 2.1)
    replies = [DIRECT + "Hallo!", WAIT, DIRECT + SPK_CHANGE + " Guten\nTag", DIRECT + SPK_CHANGE + "."]
    model = Replay(replies)
    assert run(tmp_path, lambda: model) == 0
    assert read_rows(path) == [{**source, FIELD: "Hallo! Guten\nTag."}, *already]
    assert model.prompts == ["<audio>", "<audio>", "<audio>", EOS]
    assert model.histories == [replies[:i] for i in range(len(replies))]
    if ext == "jsonl":
        assert len(path.read_text(encoding="utf-8").splitlines()) == 3
    assert run(tmp_path, no_load) == 0


def test_missing_wav_does_not_block_other_languages_or_selected_retry(tmp_path):
    first = write_manifest(tmp_path, "de", [{"audio": "missing.wav"}], "jsonl")
    second = write_manifest(tmp_path, "en", [{"audio": "ok.wav"}], "json")
    make_wav(second.parent, "ok.wav", 1)
    model = Replay([DIRECT + "Okay.", WAIT])
    assert run(tmp_path, lambda: model, ("de", "en")) == 1
    assert read_rows(first) == [{"audio": "missing.wav"}]
    assert read_rows(second) == [{"audio": "ok.wav", FIELD: "Okay."}]
    make_wav(first.parent, "missing.wav", 1)
    retry = tmp_path / "retry.tsv"
    retry.write_text("de\t0\tmissing.wav\n", encoding="utf-8")
    assert run(tmp_path, lambda: Replay([WAIT, WAIT]), ("de", "en"),
               retry_targets=runner.read_retry_list(retry)) == 0
    assert read_rows(first) == [{"audio": "missing.wav", FIELD: ""}]
    assert run(tmp_path, no_load, ("de", "en")) == 0


def test_ambiguous_format_requires_explicit_manifest_name(tmp_path):
    rows = [{"audio": "a.wav"}]
    array = write_manifest(tmp_path, "de", rows, "json")
    lines = write_manifest(tmp_path, "de", rows, "jsonl")
    original = [path.read_bytes() for path in (array, lines)]
    assert run(tmp_path, no_load) > 0
    assert [path.read_bytes() for path in (array, lines)] == original
    make_wav(lines.parent, "a.wav", 1)
    assert run(tmp_path, lambda: Replay([DIRECT + "hi", WAIT]), manifest_name=lines.name) == 0
    assert array.read_bytes() == original[0]
    assert read_rows(lines) == [{**rows[0], FIELD: "hi"}]


def test_capped_vllm_output_is_scored_after_stripping_only_protocol_markers(tmp_path):
    raw = DIRECT + SPK_CHANGE + "Очень! " * 45
    backend = runner.ManifestVLLMBackend.__new__(runner.ManifestVLLMBackend)
    backend.max_new_tokens, backend.max_context_tokens = len(raw), 4352
    backend.tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: "".join(map(chr, ids)))
    completion = SimpleNamespace(finish_reason="length", token_ids=list(map(ord, raw)))
    capped = backend._decode_reply(SimpleNamespace(outputs=[completion], prompt_token_ids=[1, 2]), 2)
    assert (capped["text"], capped["finish_reason"]) == (raw, "length")
    path = write_manifest(tmp_path, "ru", [{"audio": "test.wav"}], "json")
    make_wav(path.parent, "test.wav", 1)
    model = Replay([capped, WAIT])
    assert run(tmp_path, lambda: model, ("ru",)) == 0
    assert model.histories[-1] == [raw]
    assert read_rows(path)[0][FIELD] == "Очень! " * 45


def test_bad_jsonl_line_preserved_and_valid_lines_can_finish(tmp_path):
    rows = [{"audio": "a.wav"}]
    path = write_manifest(tmp_path, "de", rows, "jsonl")
    contents = path.read_text(encoding="utf-8")
    path.write_text("\n{broken json}\n" + contents, encoding="utf-8")
    make_wav(path.parent, "a.wav", 1)
    assert run(tmp_path, lambda: Replay([DIRECT + "fertig", WAIT])) > 0
    lines = path.read_text(encoding="utf-8").splitlines()
    assert "{broken json}" in lines
    assert json.loads(lines[-1]) == {**rows[0], FIELD: "fertig"}
