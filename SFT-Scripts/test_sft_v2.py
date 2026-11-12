"""EOS, causal reference conditioning and safe windowing integration tests."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_sft import prepare_row
from prepare_windows import split_row, window_boundaries
from check_sft import check_row
from audit_encoded_lengths import audit_encoded, check_encoded_protocol, check_packed_batch


def example(tmp_path):
    path = tmp_path / "meeting.wav"
    sf.write(path, np.zeros(48000, dtype=np.float32), 16000)
    turns = []
    for i in range(6):
        segment = {"language": "zh" if i < 4 else "en", "speaker": "spk1", "text": f"word{i} "}
        turns.append({"event": "audio", "start_ms": i * 500, "end_ms": (i + 1) * 500,
                      "action": "DIRECT", "segments": [segment], "safe_cut": True})
    turns[-1].update(action="WAIT", segments=[], safe_cut=False)
    turns.append({"event": "end_of_audio", "start_ms": 3000, "end_ms": 3000,
                  "action": "DIRECT", "segments": [{"language": "en", "speaker": "spk1", "text": "hundred million."}]})
    row = {"schema_version": 2, "id": "meeting-a", "audio": str(path), "duration_ms": 3000,
           "turns": turns}
    return row


def reference(row, **overrides):
    result = {"audio": row["audio"], "start_ms": 0, "end_ms": 500, "speaker": "spk1",
              "language": "zh", "text": "word0 ", "verified": True, "clean": True, "overlap": False}
    result.update(overrides)
    return result


def test_eos_releases_pending_without_extra_audio(tmp_path):
    row = example(tmp_path)
    prepared, tokens = prepare_row(row)
    assert len(prepared["audios"]) == 6
    assert len(prepared["messages"]) == 15
    assert prepared["messages"][-2] == {"role": "user", "content": "<end_of_audio>"}
    assert prepared["messages"][-1]["content"] == "<|DIRECT|><en><spk1>hundred million."
    metrics = check_row(prepared, token_inventory=tokens)
    assert metrics["eos_turns"] == 1
    assert metrics["audio_turns"] == 6


def test_minimal_audios_messages_pass_train_preflight_without_losing_validation(tmp_path):
    row = example(tmp_path)
    prepared, tokens = prepare_row(row)
    compact = {"audios": prepared["audios"], "messages": prepared["messages"]}
    checked = check_row(compact, decode=True, token_inventory=tokens)
    assert checked["audio_turns"] == 6 and checked["eos_turns"] == 1
    damaged = copy.deepcopy(compact)
    damaged["messages"][-2]["content"] = "<audio>"
    with pytest.raises(ValueError, match="end_of_audio"):
        check_row(damaged, decode=False, token_inventory=tokens)
    damaged = copy.deepcopy(compact)
    damaged["audios"] = damaged["audios"][1:]
    with pytest.raises(ValueError, match="Expected system"):
        check_row(damaged, decode=False, token_inventory=tokens)


def test_full_meeting_ids_match_stream_registry_without_silent_remap(tmp_path):
    row = example(tmp_path)
    row["turns"][0]["segments"][0]["speaker"] = "spk3"
    with pytest.raises(ValueError, match="first occurrence"):
        prepare_row(row)
    row = example(tmp_path)
    row["turns"][1]["segments"][0]["speaker"] = "spk3"
    with pytest.raises(ValueError, match="spk2"):
        prepare_row(row)


def test_no_metadata_newline_inserted_and_actual_newlines_preserved(tmp_path):
    row = example(tmp_path)
    row["turns"][0]["segments"] = [
        {"language": "zh", "speaker": "spk1", "text": "你好\n\n世界\n"},
        {"language": "en", "speaker": "spk1", "text": "Hello."}]
    prepared, tokens = prepare_row(row)
    assert prepared["messages"][2]["content"] == "<|DIRECT|><zh><spk1>你好\n\n世界\n<en><spk1>Hello."
    check_row(prepared, token_inventory=tokens)


@pytest.mark.parametrize("mutation", [
    lambda row: row["turns"].pop(),
    lambda row: row["turns"][-1].update(start_ms=2500),
    lambda row: row["turns"][2].update(event="end_of_audio"),
    lambda row: row["turns"][0].update(event="unknown"),
])
def test_bad_eos_rejected(tmp_path, mutation):
    row = example(tmp_path)
    mutation(row)
    with pytest.raises(ValueError):
        prepare_row(row)


def test_causal_reference_windows_mask_refs_as_user_input(tmp_path):
    row = example(tmp_path)
    prepared, tokens = prepare_row(row)
    windows = split_row(prepared, [reference(row)], max_seconds=1)
    assert len(windows) == 3
    assert windows[1]["window"]["start_ms"] == 1000
    assert windows[1]["messages"][1]["content"].count("<audio>") == 2
    assert "Speaker reference" in windows[1]["messages"][1]["content"]
    assert len(windows[1]["audios"]) == 3  # one reference + two current chunks
    assert len(windows[1]["messages"]) == 5  # no fake reference assistant target
    assert windows[1]["messages"][2]["content"].endswith("word2 ")
    assert "<end_of_audio>" not in [m["content"] for m in windows[1]["messages"]]
    assert windows[-1]["messages"][-2]["content"] == "<end_of_audio>"
    for window in windows:
        check_row(window, token_inventory=tokens)


@pytest.mark.parametrize("updates", [{"end_ms": 2000}, {"verified": False}, {"clean": False}, {"overlap": True}, {"end_ms": 100}])
def test_future_unsafe_short_refs_cannot_enable_window(tmp_path, updates):
    row = example(tmp_path)
    prepared, _ = prepare_row(row)
    with pytest.raises(ValueError, match="causal verified references"):
        split_row(prepared, [reference(row, **updates)], max_seconds=1)


def test_cannot_split_partial_entity_even_after_direct(tmp_path):
    prepared, _ = prepare_row(example(tmp_path))
    prepared["turn_provenance"][0]["safe_cut"] = False
    prepared["turn_provenance"][1]["safe_cut"] = False
    with pytest.raises(ValueError, match="No safe boundary"):
        window_boundaries(prepared, max_seconds=1)


def test_window_validator_rejects_tampered_future_reference(tmp_path):
    row = example(tmp_path)
    prepared, _ = prepare_row(row)
    window = split_row(prepared, [reference(row)], max_seconds=1)[1]
    window["speaker_references"][0]["end_ms"] = 1700
    with pytest.raises(ValueError, match="interval differs"):
        check_row(window, decode=False)


def test_actual_encoded_length_audit_rejects_truncation_and_no_labels():
    assert audit_encoded({"input_ids": [1, 2], "labels": [-100, 2]}, 2)["supervised_tokens"] == 1
    with pytest.raises(ValueError, match="exceeds"):
        audit_encoded({"input_ids": [1, 2], "labels": [-100, 2]}, 1)
    with pytest.raises(ValueError, match="No supervised"):
        audit_encoded({"input_ids": [1], "labels": [-100]}, 2)


def test_packing_audit_detects_masked_wait_or_reordered_tags():
    torch = pytest.importorskip("torch")
    row = {"audios": ["one", "two"], "messages": [
        {"role": "assistant", "content": "<|WAIT|>"},
        {"role": "assistant", "content": "<|DIRECT|><zh><spk1>开始。"}]}
    ids = {"<|WAIT|>": 41, "<|DIRECT|>": 42, "<zh>": 43, "<spk1>": 44}
    encoded = {"labels": [-100, 41, 42, 43, 44, 88], "input_features": torch.ones(2, 1, 4),
               "feature_attention_mask": torch.ones(2, 4)}
    check_encoded_protocol(row, encoded, ids)
    for labels in ([-100, -100, 42, 43, 44, 88], [-100, 41, 42, 44, 43, 88]):
        with pytest.raises(ValueError, match="actions/language/speaker"):
            check_encoded_protocol(row, dict(encoded, labels=labels), ids)


@pytest.mark.parametrize("damaged", [None, "text_position_ids", "cu_seq_lens_k", "input_features", "labels"])
def test_packing_audit_compares_originals_to_collated_batch(damaged):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    source = [{"input_ids": [1, 6], "labels": [-100, 6], "input_features": torch.ones(1, 1, 2),
               "feature_attention_mask": torch.ones(1, 2)},
              {"input_ids": [1, 9, 7], "labels": [-100, 9, 7], "input_features": torch.full((1, 1, 1), 2.),
               "feature_attention_mask": torch.ones(1, 1)}]
    batch = {"input_ids": torch.tensor([[1, 6, 1, 9, 7]]), "labels": torch.tensor([[-100, 6, -100, 9, 7]]),
             "text_position_ids": torch.tensor([[0, 1, 0, 1, 2]]), "position_ids": torch.zeros(3, 1, 5),
             "cu_seq_lens_k": torch.tensor([0, 2, 5]), "input_features": torch.tensor([[[1., 1.]], [[2., 0.]]]),
             "feature_attention_mask": torch.tensor([[1., 1.], [1., 0.]])}
    template = SimpleNamespace(data_collator=lambda packs: batch)
    if damaged:
        batch[damaged].flatten()[1] = 99
        with pytest.raises(ValueError):
            check_packed_batch(template, source, max_length=8)
    else:
        assert check_packed_batch(template, source, max_length=8)["records_per_pack"] == [2]
        assert check_packed_batch(template, source, max_length=4)["records_per_pack"] == [1, 1]


def test_total_reference_budget_is_shared_across_speakers(tmp_path):
    row = example(tmp_path)
    row["turns"][1]["segments"][0]["speaker"] = "spk2"
    prepared, _ = prepare_row(row)
    candidates = [reference(row), reference(row, speaker="spk2", start_ms=500,
                                           end_ms=1000, text="word1 ")]
    # Each clip fits the per-speaker cap; together they exceed the total cap.
    with pytest.raises(ValueError, match="[Tt]otal.*reference|reference.*[Bb]udget"):
        split_row(prepared, candidates, max_seconds=1.5, max_total_reference_seconds=.75)
    windows = split_row(prepared, candidates, max_seconds=1.5, max_total_reference_seconds=1.)
    assert len(windows[-1]["speaker_references"]) == 2
    assert sum((ref["end_ms"] - ref["start_ms"]) / 1000 for ref in windows[-1]["speaker_references"]) == 1.
