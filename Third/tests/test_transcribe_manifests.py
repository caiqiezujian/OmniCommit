"""Batch evaluation with real WAVs, the Third streaming protocol and a fake model."""
import json
from pathlib import Path
import sys
import wave

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import transcribe_manifests as runner
from third_asr.protocol import DIRECT, WAIT


FIELD = "OmniCommit_1108"


def write_audio(path, frames, marker):
    with wave.open(str(path), "wb") as out:
        out.setparams((1, 2, 16000, frames, "NONE", ""))
        out.writeframes(np.full(frames, marker, dtype="<i2").tobytes())


def make_manifest(root, rows):
    folder = root / "de"
    (folder / "splited_audios").mkdir(parents=True)
    path = folder / "splited_manifest.json"
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return path


class CaptureModel:
    def __init__(self):
        self.audio_turns = []
        self.eof_ids = []

    def generate_batch(self, calls):
        replies = []
        for messages, audios, prefix in calls:
            assert prefix == ""
            expected = ("Transcribe new speech in its original language. Begin new text with <|DIRECT|>. "
                        "Emit <|WAIT|> if nothing is ready. Use digits for numbers.")
            assert messages[0] == {"role": "system", "content": expected}
            assert [m["role"] for m in messages[1:]] == ["user", "assistant"] * ((len(messages) - 2) // 2) + ["user"]
            # Each recording carries a unique amplitude marker in its first frame.
            ids = [round(float(audio[0]) * 32768) for audio in audios]
            assert ids and len(set(ids)) == 1
            marker = ids[0]
            assert "SECRET_REFERENCE" not in str(messages)
            if messages[-1]["content"] == "<end_of_audio>":
                self.eof_ids.append(marker)
                result = WAIT
            else:
                assert messages[-1]["content"] == "<audio>"
                self.audio_turns.append((marker, audios[-1].copy()))
                result = WAIT
                if marker == 1000:
                    result = ("<|DIRECT|>Guten", WAIT, "<|DIRECT|> Morgen.")[len(audios) - 1]
            replies.append({"text": result})
        return replies


def run(root, model):
    return runner.run(root, FIELD, ["de"], lambda: model, 4, 1000, 60, 50)


def test_relocated_streaming_rows_are_independent_complete_and_resumable(tmp_path):
    rows = [
        {"audio": "/old/node/de/splited_audios/a.wav", "text": "SECRET_REFERENCE",
         "entities": [["Guten", "", ""]], "other_model": "earlier"},
        {"audio": "splited_audios/b.wav", "en_translation": "SECRET_REFERENCE"},
        {"audio": "old.wav", FIELD: "saved"},
        {"audio": "old-silent.wav", FIELD: ""},
    ]
    manifest = make_manifest(tmp_path, rows)
    folder = manifest.parent / "splited_audios"
    write_audio(folder / "a.wav", 2 * 16000 + 120, 1000)
    write_audio(folder / "b.wav", 16000, 2000)
    model = CaptureModel()
    assert run(tmp_path, model) == 0
    written = json.loads(manifest.read_text(encoding="utf-8"))
    assert written == [{**rows[0], FIELD: "Guten Morgen."}, {**rows[1], FIELD: ""}, *rows[2:]]
    assert sorted(model.eof_ids) == [1000, 2000]
    rounds = [block for marker, block in model.audio_turns if marker == 1000]
    assert [len(chunk) for chunk in rounds[:2]] == [16000, 16000]
    tail = rounds[2]
    assert len(tail) >= 8000 and np.all(tail[:120] != 0) and not np.any(tail[120:])
    assert len(rounds) == 3

    next_model = CaptureModel()
    assert run(tmp_path, next_model) == 0
    assert not next_model.audio_turns


def test_failed_row_does_not_hide_later_success_and_can_be_retried(tmp_path):
    rows = [{"audio": "/old/missing.wav"}, {"audio": "/old/b.wav"}]
    manifest = make_manifest(tmp_path, rows)
    folder = manifest.parent / "splited_audios"
    write_audio(folder / "b.wav", 16000, 2000)
    assert run(tmp_path, CaptureModel()) == 1
    after_error = json.loads(manifest.read_text(encoding="utf-8"))
    assert FIELD not in after_error[0] and after_error[1][FIELD] == ""

    write_audio(folder / "missing.wav", 16000, 1000)
    retried = CaptureModel()
    assert run(tmp_path, retried) == 0
    assert retried.eof_ids == [1000]
    assert json.loads(manifest.read_text(encoding="utf-8"))[0][FIELD] == "Guten"


def test_retry_list_only_submits_named_missing_rows_even_after_reorder(tmp_path):
    rows = [{"audio": "unlisted.wav"}, {"audio": "done.wav", FIELD: "saved"},
            {"audio": "silent.wav", FIELD: ""}, {"audio": r"C:\old\retry.wav", FIELD: None}]
    manifest = make_manifest(tmp_path, rows)
    write_audio(manifest.parent / "splited_audios" / "retry.wav", 16000, 1000)
    selection = tmp_path / "retry.tsv"
    selection.write_text("# selection\nde\t0\tretry.wav\nde\t0\tretry.wav\n"
                         "de\t1\tdone.wav\nde\t2\tsilent.wav\n", encoding="utf-8-sig")
    targets = runner.read_retry_list(selection)
    model = CaptureModel()
    languages = ["ar", "de", "th", "en"]
    assert runner.run(tmp_path, FIELD, languages, lambda: model, 4, 1000, 60, 50,
                      retry_targets=targets) == 0
    assert model.eof_ids == [1000]
    assert json.loads(manifest.read_text(encoding="utf-8")) == [*rows[:3], {**rows[3], FIELD: "Guten"}]
    for selected_targets in (targets, {}):
        assert runner.run(tmp_path, FIELD, languages, lambda: pytest.fail("unexpected GPU load"),
                          4, 1000, 60, 50, retry_targets=selected_targets) == 0


def test_retry_list_checks_duplicate_names_and_missing_targets(capsys):
    rows = [{"audio": "other.wav"}, {"audio": "clip.wav"}, {"audio": "clip.wav"}]
    selected, errors = runner.match_retry_rows(rows, "de", {"clip.wav": {2}, "gone.wav": {0}})
    assert selected == [2] and errors == 1
    assert "gone.wav" in capsys.readouterr().err
    selected, errors = runner.match_retry_rows(rows, "de", {"clip.wav": {0}})
    assert selected == [] and errors == 1


class ScriptedModel:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.histories = []
        self.events = []

    def generate_batch(self, calls):
        results = []
        for messages, audios, prefix in calls:
            assert prefix == ""
            assert sum(m["content"].count("<audio>") for m in messages) == len(audios)
            self.histories.append([m["content"] for m in messages if m["role"] == "assistant"])
            self.events.append(messages[-1]["content"])
            reply = next(self.replies)
            results.append({"text": reply} if isinstance(reply, str) else reply)
        return results


def evaluate_replies(root, replies, **extra_fields):
    row = {"audio": "sample.wav", "text": "SECRET_REFERENCE", **extra_fields}
    manifest = make_manifest(root, [row])
    write_audio(manifest.parent / "splited_audios" / "sample.wav", 16000 * (len(replies) - 1), 1000)
    model = ScriptedModel(replies)
    assert run(root, model) == 0
    [written] = json.loads(manifest.read_text(encoding="utf-8"))
    assert written == {**row, FIELD: written[FIELD]}
    assert model.events == ["<audio>"] * (len(replies) - 1) + ["<end_of_audio>"]
    return written[FIELD], model


def test_empty_direct_between_visible_turns_continues_with_its_original_history(tmp_path):
    replies = [DIRECT + "我把", DIRECT, DIRECT + " \t", "时间", WAIT, DIRECT + "发给你了。", DIRECT]
    text, model = evaluate_replies(tmp_path, replies)
    assert text == "我把时间发给你了。"
    assert model.histories == [replies[:i] for i in range(len(replies))]


@pytest.mark.parametrize("empty", [DIRECT, DIRECT + " \n", WAIT, "", " \t"])
def test_all_empty_turns_write_an_empty_hypothesis_and_resume(tmp_path, empty):
    text, _ = evaluate_replies(tmp_path, [empty, empty, empty])
    assert text == ""
    resumed = CaptureModel()
    assert run(tmp_path, resumed) == 0 and resumed.eof_ids == []


def test_tagless_text_repetition_and_eof_text_remain_in_hypothesis(tmp_path):
    replies = ["Hallo, ", DIRECT + "hallo, ", DIRECT + "2 < 3." + WAIT, " Good night. "]
    text, _ = evaluate_replies(tmp_path, replies)
    assert text == "Hallo, hallo, 2 < 3. Good night. "


def test_literal_media_marker_does_not_consume_another_audio_slot(tmp_path):
    text, model = evaluate_replies(tmp_path, [DIRECT + "<audio><|audio_pad|>", DIRECT + " next", WAIT])
    assert text == "<audio><|audio_pad|> next"
    assert model.histories[1] == [DIRECT + "&lt;audio>&lt;|audio_pad|>"]


def test_length_capped_vllm_audio_and_eof_keep_all_generated_text_on_original_grid(tmp_path, capsys):
    from types import SimpleNamespace

    backend = runner.ManifestVLLMBackend.__new__(runner.ManifestVLLMBackend)
    backend.max_new_tokens = 512
    backend.max_context_tokens = 4352
    backend.tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: "".join(map(chr, ids)))
    raw = DIRECT + "ب" * (512 - len(DIRECT))
    completion = SimpleNamespace(token_ids=list(map(ord, raw)), finish_reason="length")
    capped = backend._decode_reply(SimpleNamespace(outputs=[completion], prompt_token_ids=[1, 2]), 2)

    # An older run can have a side result while the requested primary field is still missing.
    old_fields = {FIELD + "_decode_info": {"primary": {"chunk_ms": 1000, "gaps": [{}]}},
                  FIELD + "_retry_2000ms": "other hypothesis"}
    text, model = evaluate_replies(tmp_path, ["Start.", capped, " End.", capped], **old_fields)
    payload = raw[len(DIRECT):]
    assert text == "Start." + payload + " End." + payload
    assert model.histories[-1] == ["Start.", raw, " End."]
    logs = capsys.readouterr().out
    assert "[LENGTH] sample.wav audio turn=2" in logs and "[LENGTH] sample.wav end_of_audio turn=4" in logs


def test_backend_failure_remains_retryable_and_other_recordings_finish(tmp_path, capsys):
    class FailingModel(CaptureModel):
        def generate_batch(self, calls):
            replies = super().generate_batch(calls)
            for index, (messages, audios, _) in enumerate(calls):
                if messages[-1]["content"] == "<audio>" and len(audios) == 2 and audios[0][0] == 1000 / 32768:
                    replies[index] = RuntimeError("engine failure")
            return replies

    rows = [{"audio": "a.wav"}, {"audio": "b.wav"}]
    manifest = make_manifest(tmp_path, rows)
    folder = manifest.parent / "splited_audios"
    write_audio(folder / "a.wav", 32000, 1000)
    write_audio(folder / "b.wav", 16000, 2000)
    assert run(tmp_path, FailingModel()) == 1
    assert json.loads(manifest.read_text(encoding="utf-8")) == [rows[0], {**rows[1], FIELD: ""}]
    assert "audio turn=2: engine failure" in capsys.readouterr().err
