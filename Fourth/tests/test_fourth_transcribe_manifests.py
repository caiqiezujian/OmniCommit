"""Fourth manifest evaluation through its real chunk reader and independent sessions."""
from importlib.util import module_from_spec, spec_from_file_location
from collections import Counter
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import wave

import pytest


ENTRY = Path(__file__).resolve().parents[1] / "tools" / "transcribe_manifests.py"
spec = spec_from_file_location("fourth_manifest_runner", ENTRY)
runner = module_from_spec(spec)
spec.loader.exec_module(runner)

from fourth_asr.protocol import DIRECT, EOS, SPK_CHANGE, WAIT, get_protocol


FIELD = "OmniCommit_test_fourth"


def make_wav(directory, name, seconds, sample=16):
    path = directory / "splited_audios" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", ""))
        audio.writeframes(sample.to_bytes(2, "little", signed=True) * int(seconds * 16000))


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


def failure_rows(root):
    report, = root.glob(f"failed-{FIELD}-*.jsonl")
    return read_rows(report)


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
    already = [{"audio": "old.wav", FIELD: "saved"}]
    path = write_manifest(tmp_path, "de", [source, *already], ext)
    make_wav(path.parent, "meeting.wav", 2.1)
    replies = [DIRECT + "Hallo!", WAIT, DIRECT + SPK_CHANGE + " Guten\nTag", DIRECT + SPK_CHANGE + "."]
    model = Replay(replies)
    assert run(tmp_path, lambda: model, resume=True) == 0
    assert read_rows(path) == [{**source, FIELD: "Hallo! Guten\nTag."}, *already]
    assert model.prompts == ["<audio>", "<audio>", "<audio>", EOS]
    assert model.histories == [replies[:i] for i in range(len(replies))]
    if ext == "jsonl":
        assert len(path.read_text(encoding="utf-8").splitlines()) == 1 + len(already)
    assert run(tmp_path, no_load, resume=True) == 0


@pytest.mark.parametrize("ext,previous", [("json", ""), ("jsonl", "earlier prediction")])
def test_default_overwrites_existing_field_only_after_decode(tmp_path, ext, previous):
    source = {"audio": "meeting.wav", "text": "SECRET_REFERENCE", FIELD: previous}
    missing = {"audio": "missing.wav", FIELD: "last successful prediction"}
    path = write_manifest(tmp_path, "de", [source, missing], ext)
    make_wav(path.parent, "meeting.wav", 1)
    assert run(tmp_path, lambda: Replay([DIRECT + "New.", WAIT])) == 1
    assert read_rows(path) == [{**source, FIELD: "New."}, {**missing, FIELD: ""}]
    failure, = failure_rows(tmp_path)
    assert (failure["language"], failure["index"], failure["audio"]) == ("de", 1, "missing.wav")
    assert len(failure["attempts"]) == 3
    assert all(t["stage"] == "open_wav" and t["error_type"] == "FileNotFoundError"
               and t["completed_turns"] == 0 for t in failure["attempts"])


def test_missing_wav_does_not_block_other_languages_or_selected_retry(tmp_path):
    first = write_manifest(tmp_path, "de", [{"audio": "missing.wav"}], "jsonl")
    second = write_manifest(tmp_path, "en", [{"audio": "ok.wav"}], "json")
    make_wav(second.parent, "ok.wav", 1)
    model = Replay([DIRECT + "Okay.", WAIT])
    assert run(tmp_path, lambda: model, ("de", "en")) == 1
    assert read_rows(first) == [{"audio": "missing.wav", FIELD: ""}]
    assert read_rows(second) == [{"audio": "ok.wav", FIELD: "Okay."}]
    make_wav(first.parent, "missing.wav", 1)
    retry = tmp_path / "retry.tsv"
    retry.write_text("de\t0\tmissing.wav\n", encoding="utf-8")
    assert run(tmp_path, lambda: Replay([WAIT] * 6), ("de", "en"),
               retry_targets=runner.read_retry_list(retry)) == 1
    assert read_rows(first) == [{"audio": "missing.wav", FIELD: ""}]
    assert run(tmp_path, lambda: Replay([DIRECT + "Updated.", WAIT]), ("de", "en"),
               retry_targets=runner.read_retry_list(retry)) == 0
    assert read_rows(first) == [{"audio": "missing.wav", FIELD: "Updated."}]
    assert read_rows(second) == [{"audio": "ok.wav", FIELD: "Okay."}]
    assert run(tmp_path, no_load, ("de", "en"), resume=True) == 0


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
    failure, = failure_rows(tmp_path)
    assert failure["index"] == 1 and failure["kind"] == "invalid_record" and failure["attempts"] == []


def test_only_unsuccessful_wavs_replay_from_zero_and_last_pass_is_isolated(tmp_path):
    # The first PCM sample identifies each fixture to the fake engine without ever sending a file name.
    class Model:
        def __init__(self):
            self.starts = Counter()
            self.groups = []

        def generate_batch(self, calls):
            replies = []
            for messages, audios, _ in calls:
                clip = round(float(audios[0][0]) * 32768)
                at_eos = messages[-1]["content"] == EOS
                if not at_eos:
                    assert not any(m["role"] == "assistant" for m in messages)
                    assert len(audios) == 1
                    self.starts[clip] += 1
                attempt = self.starts[clip]
                self.groups.append((clip, attempt, len(calls)))
                if clip == 3 and attempt == 2:
                    replies.append(RuntimeError("test engine error"))
                    continue
                text = WAIT
                if clip <= 3 and attempt == clip and not at_eos:
                    text = DIRECT + f"ok-{clip}"
                if clip == 5 and at_eos:
                    text = DIRECT + SPK_CHANGE + " "
                replies.append({"text": text, "request_batch_size": len(calls), "finish_reason": "stop"})
            return replies

    rows = [{"audio": f"{i}.wav", FIELD: "stale", "text": "reference"} for i in range(1, 6)]
    path = write_manifest(tmp_path, "de", rows, "jsonl")
    for i in range(1, 6):
        make_wav(path.parent, f"{i}.wav", 1, sample=i)
    model = Model()
    assert run(tmp_path, lambda: model) == 2
    assert model.starts == {1: 1, 2: 2, 3: 3, 4: 3, 5: 3}
    assert all(size <= (4, 2, 1)[attempt - 1] for _, attempt, size in model.groups)
    assert [r[FIELD] for r in read_rows(path)] == ["ok-1", "ok-2", "ok-3", "", ""]
    failures = {row["index"]: row for row in failure_rows(tmp_path)}
    assert set(failures) == {3, 4}
    for failure in failures.values():
        assert [a["concurrent_wavs"] for a in failure["attempts"]] == [4, 2, 1]
        assert failure["attempts"][-1]["largest_request_batch"] == 1
    assert all(a["status"] == "all_wait" and a["wait_replies"] == 2 for a in failures[3]["attempts"])
    trace = failures[4]["attempts"][-1]
    assert trace["status"] == "empty_transcript" and trace["empty_direct_replies"] == 1
    assert trace["last_replies"][-1]["raw"] == DIRECT + SPK_CHANGE + " "
    assert trace["last_replies"][-1]["event"] == "end_of_audio" and trace["duration_seconds"] == 1
    assert trace["peak_abs"] > 0


@pytest.mark.parametrize("previous", ["", "   ", WAIT, DIRECT + SPK_CHANGE])
def test_resume_retries_old_empty_fields_and_accepts_text_at_eos(tmp_path, previous):
    path = write_manifest(tmp_path, "en", [{"audio": "tail.wav", FIELD: previous}], "json")
    make_wav(path.parent, "tail.wav", 1)
    assert run(tmp_path, lambda: Replay([WAIT, DIRECT + "Yes"]), ("en",), resume=True) == 0
    assert read_rows(path)[0][FIELD] == "Yes" and failure_rows(tmp_path) == []


def test_incomplete_decode_is_not_published_and_other_languages_share_one_report(tmp_path):
    class FailsAtEos:
        def generate_batch(self, calls):
            if calls[0][0][-1]["content"] == EOS:
                raise RuntimeError("engine request failed")
            return [{"text": DIRECT + "partial"} for _ in calls]

    for lang in ("ar", "en"):
        path = write_manifest(tmp_path, lang, [{"audio": "same.wav", FIELD: "stale"}], "json")
        make_wav(path.parent, "same.wav", 1)
    assert run(tmp_path, FailsAtEos, ("ar", "en")) == 2
    failures = failure_rows(tmp_path)
    assert [r["language"] for r in failures] == ["ar", "en"]
    for failure in failures:
        assert read_rows(Path(failure["manifest"]))[0][FIELD] == ""
        assert len(failure["attempts"]) == 3
        for trace in failure["attempts"]:
            assert trace["status"] == "error" and trace["stage"] == "decode" and trace["event"] == "end_of_audio"
            assert trace["turn"] == 2 and trace["emitted_chars"] == len("partial")
            assert "engine request failed" in trace["error"]


def test_batch_wide_generate_error_can_recover_on_single_wav(tmp_path, monkeypatch, capsys):
    path = write_manifest(tmp_path, "de", [{"audio": "a.wav"}, {"audio": "b.wav"}], "json")
    for name in ("a.wav", "b.wav"):
        make_wav(path.parent, name, 1)
    native = runner.ManifestVLLMBackend.__new__(runner.ManifestVLLMBackend)
    native.max_batch_size, native.stop_ids, native.sampling = 2, [], object()
    native.tokenizer = SimpleNamespace(decode=lambda ids, **kw: "".join(map(chr, ids)))
    native._prepare_request = lambda messages, audios, prefix: (messages[-1]["content"], 1)
    group_sizes = []

    def generate(requests, sampling, **kwargs):
        group_sizes.append(len(requests))
        if len(requests) > 1:
            raise RuntimeError("fixture: grouped requests fail")
        text = WAIT if requests[0] == EOS else DIRECT + "Hallo"
        reply = SimpleNamespace(token_ids=list(map(ord, text)), finish_reason="stop", stop_reason=9)
        return [SimpleNamespace(outputs=[reply], prompt_token_ids=[1])]

    native.llm = SimpleNamespace(generate=generate)
    # Synchronize the two workers before decoding so this exercises an actual grouped submission.
    barrier = threading.Barrier(2)
    original_attempt = runner.transcribe_attempt
    original_batch = runner.BatchBackend

    def attempt(row, directory, backend, options, chunk_ms, number, workers):
        if workers == 2:
            barrier.wait(timeout=5)
        return original_attempt(row, directory, backend, options, chunk_ms, number, workers)

    monkeypatch.setattr(runner, "transcribe_attempt", attempt)
    monkeypatch.setattr(runner, "BatchBackend",
                        lambda factory, max_batch_size: original_batch(factory, max_batch_size=max_batch_size,
                                                                        batch_wait_ms=250))
    assert runner.run(tmp_path, FIELD, ["de"], lambda: native, 2, 1000, 60, 50) == 0
    assert group_sizes == [2, 1, 1, 1, 1]
    assert [row[FIELD] for row in read_rows(path)] == ["Hallo", "Hallo"]
    assert failure_rows(tmp_path) == []
    output = capsys.readouterr().out
    assert "vLLM.generate(batch=2)" in output and output.count("[恢复]") == 2
