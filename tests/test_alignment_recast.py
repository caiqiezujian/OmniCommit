import argparse
import copy
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import soundfile as sf

from omnicommit import alignment, recast
from omnicommit.forced_alignment import WhisperXAligner, Qwen3Aligner, restore_qwen_spans, strict_qwen_timestamps


def source(tmp_path, pieces, seconds=2.0, raw=None):
    audio = tmp_path / "meeting.wav"
    sf.write(str(audio), np.zeros(round(seconds * 16000), dtype=np.float32), 16000)
    text = raw or "".join(text for text, action in pieces)
    return {"id": "meeting-1", "audios": [str(audio)],
            "messages": [{"role": "assistant", "content": text}],
            "output": json.dumps({"segments": [{"source_text": text, "action": action} for text, action in pieces],
                                  "flush": pieces[-1][1] == "WAIT", "error": None}, ensure_ascii=False)}


def timed(*values):
    return {"words": [{"text": text, "start_ms": start, "end_ms": end} for text, start, end in values]}


def emitted(row):
    return "".join(segment["text"] for turn in row["turns"] for segment in turn["segments"])


def test_collision_preserves_wait_then_eos_flush(tmp_path):
    row = source(tmp_path, [("<zh><spk1>请联系", "DIRECT"), ("张伟", "WAIT"), ("确认。", "DIRECT")], .9)
    aligned = alignment.align_annotation(row, timed(("请联系", 20, 200), ("张伟", 210, 350), ("确认", 360, 490)))
    result = recast.recast_row(aligned)
    assert [t["action"] for t in result["turns"]] == ["WAIT", "DIRECT", "WAIT"]
    assert result["turns"][0]["provenance"]["hard_wait_source_indices"] == [1]
    assert emitted(result) == "请联系张伟确认。"
    assert result["turns"][-1]["event"] == "end_of_audio"
    assert result["turns"][-1]["start_ms"] == result["turns"][-1]["end_ms"] == 900


def test_final_collision_flush_false_still_eos_direct(tmp_path):
    row = source(tmp_path, [("<en><spk1>Pay ", "DIRECT"), ("100 million", "WAIT"), (" now.", "DIRECT")], .5)
    aligned = alignment.align_annotation(row, timed(("Pay", 10, 80), ("100", 90, 200), ("million", 210, 350), ("now", 360, 490)))
    result = recast.recast_row(aligned)
    assert aligned["annotation_flush"] is False
    assert [t["action"] for t in result["turns"]] == ["WAIT", "DIRECT"]
    assert result["provenance"]["eos_flush_source_count"] == 3
    assert emitted(result) == "Pay 100 million now."


def test_atomic_incomplete_number_no_partial_commit(tmp_path):
    row = source(tmp_path, [("<en><spk1>Pay ", "DIRECT"), ("100 million", "WAIT"), (" now.", "DIRECT")], 2)
    aligned = alignment.align_annotation(row, timed(("Pay", 10, 100), ("100", 200, 400), ("million", 600, 1100), ("now", 1200, 1400)))
    result = recast.recast_row(aligned)
    assert [t["action"] for t in result["turns"]] == ["DIRECT", "WAIT", "WAIT", "DIRECT", "WAIT"]
    assert result["turns"][0]["segments"][0]["text"] == "Pay "
    assert result["turns"][0]["safe_cut"] is False
    assert result["turns"][3]["safe_cut"] is True
    assert emitted(result) == "Pay 100 million now."


def test_switch_events_consumed_once_and_identity_preserved(tmp_path):
    row = source(tmp_path, [("<zh><spk1>你好。", "DIRECT"), ("<en><spk1>Hello.", "WAIT"),
                            ("<en><spk2>Yes.", "WAIT"), ("<zh><spk1>继续。", "WAIT")], 2)
    aligned = alignment.align_annotation(row, timed(("你好", 20, 200), ("Hello", 510, 700),
                                                   ("Yes", 1010, 1200), ("继续", 1510, 1700)))
    result = recast.recast_row(aligned)
    assert [t["action"] for t in result["turns"]] == ["DIRECT", "WAIT", "WAIT", "WAIT", "DIRECT"]
    assert [(s["language"], s["speaker"]) for s in result["turns"][-1]["segments"]] == [
        ("en", "spk1"), ("en", "spk2"), ("zh", "spk1")]
    assert emitted(result) == "你好。Hello.Yes.继续。"


def test_many_consecutive_direct_and_one_second_mode(tmp_path):
    row = source(tmp_path, [("<en><spk1>We ", "DIRECT"), ("can ", "DIRECT"), ("start ", "DIRECT"), ("now.", "DIRECT")], 2)
    acoustic = timed(("We", 10, 300), ("can", 510, 800), ("start", 1010, 1300), ("now", 1510, 1800))
    aligned = alignment.align_annotation(row, acoustic)
    half = recast.recast_row(aligned)
    assert [t["action"] for t in half["turns"]] == ["DIRECT"] * 4 + ["WAIT"]
    whole = recast.recast_row(aligned, 1)
    assert [t["end_ms"] for t in whole["turns"]] == [1000, 2000, 2000]
    assert [t["action"] for t in whole["turns"]] == ["DIRECT", "DIRECT", "WAIT"]
    assert emitted(half) == emitted(whole) == "We can start now."


def test_every_direct_header_does_not_create_switches_or_change_offsets(tmp_path):
    row = source(tmp_path, [("<en><spk1>We ", "DIRECT"), ("<en><spk1>can ", "DIRECT"),
                            ("<en><spk1>start ", "DIRECT"), ("<en><spk1>now.", "DIRECT")],
                 2, raw="<en><spk1>We can start now.")
    aligned = alignment.align_annotation(row, timed(("We", 10, 300), ("can", 510, 800),
                                                   ("start", 1010, 1300), ("now", 1510, 1800)))
    assert [(s["char_start"], s["char_end"]) for s in aligned["segments"]] == [(0, 3), (3, 7), (7, 13), (13, 17)]
    assert not any(s["switched"] for s in aligned["segments"])
    assert [t["action"] for t in recast.recast_row(aligned)["turns"]] == ["DIRECT"] * 4 + ["WAIT"]
    assert emitted(recast.recast_row(aligned)) == "We can start now."


def test_prefixed_direct_releases_previous_wait_without_repeating_source(tmp_path):
    row = source(tmp_path, [("<zh><spk1>请联系", "DIRECT"), ("张伟", "WAIT"),
                            ("<zh><spk1>确认方案。", "DIRECT")],
                 1.5, raw="<zh><spk1>请联系张伟确认方案。")
    aligned = alignment.align_annotation(row, timed(("请联系", 20, 300), ("张伟", 520, 800),
                                                   ("确认方案", 1010, 1300)))
    output = recast.recast_row(aligned)
    assert [t["action"] for t in output["turns"]] == ["DIRECT", "WAIT", "DIRECT", "WAIT"]
    assert output["turns"][2]["segments"] == [{"language": "zh", "speaker": "spk1", "text": "张伟确认方案。"}]
    assert emitted(output) == "请联系张伟确认方案。"


def test_direct_headers_preserve_real_language_and_speaker_switch_waits(tmp_path):
    row = source(tmp_path, [("<zh><spk1>开始。", "DIRECT"), ("<en><spk1>Please ", "WAIT"),
                            ("<en><spk1>continue.", "DIRECT"), ("<en><spk2>I ", "WAIT"),
                            ("<en><spk2>agree.", "DIRECT")], 2.5,
                 raw="<zh><spk1>开始。<en><spk1>Please continue.<en><spk2>I agree.")
    aligned = alignment.align_annotation(row, timed(("开始", 10, 300), ("Please", 510, 800),
                                                   ("continue", 1010, 1300), ("I", 1510, 1800),
                                                   ("agree", 2010, 2300)))
    assert [s["switched"] for s in aligned["segments"]] == [False, True, False, True, False]
    assert [t["action"] for t in recast.recast_row(aligned)["turns"]] == ["DIRECT", "WAIT", "DIRECT", "WAIT", "DIRECT", "WAIT"]


@pytest.mark.parametrize("header", ["<zh><spk2>", "<en><spk1>"])
def test_repeated_direct_header_cannot_change_true_attribution(tmp_path, header):
    row = source(tmp_path, [("<zh><spk1>我们", "DIRECT"), (header + "继续。", "DIRECT")],
                 1, raw="<zh><spk1>我们继续。")
    with pytest.raises(ValueError, match="attribution"):
        alignment.align_annotation(row, timed(("我们", 10, 300), ("继续", 600, 800)))


@pytest.mark.parametrize("pieces", [
    [("<en><spk1>We<en", "DIRECT"), ("><spk1> go.", "DIRECT")],
    [("<en><spk1>We<en>", "DIRECT"), ("<spk1> go.", "DIRECT")],
])
def test_concatenation_cannot_hide_tag_cut_between_source_units(tmp_path, pieces):
    row = source(tmp_path, pieces, 1, raw="<en><spk1>We go.")
    with pytest.raises(ValueError, match="tag|attribution"):
        alignment.align_annotation(row, timed(("We", 10, 300), ("go", 600, 800)))


def test_full_direct_header_keeps_leading_whitespace_verbatim(tmp_path):
    row = source(tmp_path, [("<en><spk1> \n We ", "DIRECT"), ("<en><spk1>continue.\n", "DIRECT")],
                 1, raw=" \n <en><spk1>We continue.\n")
    aligned = alignment.align_annotation(row, timed(("We", 10, 300), ("continue", 600, 800)))
    assert emitted(recast.recast_row(aligned)) == " \n We continue.\n"


def test_real_word_boundaries_conservative_no_proportional_timing(tmp_path):
    row = source(tmp_path, [("<zh><spk1>我们", "DIRECT"), ("出发。", "DIRECT")], 1)
    aligned = alignment.align_annotation(row, timed(("我们出发", 10, 800)))
    assert [(s["start_ms"], s["end_ms"]) for s in aligned["segments"]] == [(10, 800), (10, 800)]
    result = recast.recast_row(aligned)
    assert result["turns"][0]["action"] == "WAIT"
    assert result["turns"][1]["action"] == "DIRECT"


def test_repeated_words_follow_order_and_original_punctuation(tmp_path):
    row = source(tmp_path, [("<en><spk1>Go, ", "DIRECT"), ("go!", "DIRECT")], 1)
    aligned = alignment.align_annotation(row, timed(("go", 10, 200), ("go", 600, 800)))
    assert [s["end_ms"] for s in aligned["segments"]] == [200, 800]
    assert emitted(recast.recast_row(aligned)) == "Go, go!"


@pytest.mark.parametrize("text,lang", [("ภาษาไทย", "th"), ("こんにちは", "ja"), ("안녕하세요", "ko"),
                                      ("Привет", "ru"), ("فجأةً", "ar"), ("Olá", "pt"), ("你好", "zh")])
def test_multilingual_external_alignment(tmp_path, text, lang):
    row = source(tmp_path, [(f"<{lang}><spk1>" + text, "DIRECT")], .7)
    aligned = alignment.align_annotation(row, timed((text, 10, 600)))
    assert emitted(recast.recast_row(aligned)) == text


@pytest.mark.parametrize("bad", [
    timed(("wrong", 10, 100)), timed(("Hello", -1, 200)), timed(("Hello", 1, 1001)),
    timed(("Hello", 200, 200)), timed(("Hello", float("nan"), 200)),
    {"words": [{"text": "Hello"}]}, {"status": "error", "error": "bad"},
])
def test_invalid_alignment_rejected(tmp_path, bad):
    row = source(tmp_path, [("<en><spk1>Hello.", "DIRECT")], 1)
    with pytest.raises(ValueError):
        alignment.align_annotation(row, bad)


@pytest.mark.parametrize("text,wrong", [("1.00", "100"), ("100%", "100"), ("-100", "100")])
def test_numeric_punctuation_mismatch_not_silently_ignored(tmp_path, text, wrong):
    row = source(tmp_path, [("<en><spk1>" + text, "WAIT")], 1)
    with pytest.raises(ValueError, match="differs"):
        alignment.align_annotation(row, timed((wrong, 10, 700)))


def test_duplicate_headers_and_11122211_normalization(tmp_path):
    raw = "<zh><spk1>甲<zh><spk1>乙<zh><spk1>丙<zh><spk2>丁<zh><spk2>戊<zh><spk2>己<zh><spk1>庚<zh><spk1>辛"
    normalized = alignment.normalize_source(raw)
    assert normalized == "<zh><spk1>甲乙丙<zh><spk2>丁戊己<zh><spk1>庚辛"
    row = source(tmp_path, [("<zh><spk1>甲乙丙", "DIRECT"), ("<zh><spk2>丁戊己", "WAIT"),
                            ("<zh><spk1>庚辛", "WAIT")], 1, raw)
    aligned = alignment.align_annotation(row, timed(("甲乙丙", 10, 200), ("丁戊己", 300, 500), ("庚辛", 600, 800)))
    assert emitted(recast.recast_row(aligned)) == "甲乙丙丁戊己庚辛"


def test_switch_cannot_be_direct(tmp_path):
    row = source(tmp_path, [("<en><spk1>Yes.", "DIRECT"), ("<en><spk2>No.", "DIRECT")], 1)
    with pytest.raises(ValueError, match="switch"):
        alignment.align_annotation(row, timed(("Yes", 10, 200), ("No", 300, 800)))


def test_broken_row_continues_and_resume(tmp_path):
    row = source(tmp_path, [("<en><spk1>Hello.", "DIRECT")], 1)
    source_file, time_file, output = [tmp_path / name for name in ("annotations.jsonl", "times.jsonl", "aligned.jsonl")]
    source_file.write_bytes(b"not-json\n\xff\n" + json.dumps(row).encode() + b"\n")
    time_file.write_text(json.dumps(dict(timed(("Hello", 20, 800)), id="meeting-1")) + "\n", encoding="utf-8")
    args = alignment.parser().parse_args([str(source_file), "--timestamps", str(time_file), "--output", str(output)])
    counts = alignment.run(args)
    assert counts == {"written": 3, "failed": 2, "resumed": 0}
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [r["status"] for r in rows] == ["error", "error", "ok"]
    assert rows[0]["output"] == rows[1]["output"] == ""
    args.resume = True
    assert alignment.run(args)["resumed"] == 3
    trajectory_file = tmp_path / "trajectory.jsonl"
    counts = recast.run(recast.parser().parse_args([str(output), "--output", str(trajectory_file)]))
    assert counts["written"] == 3 and counts["failed"] == 2


def test_duplicate_timestamp_key_rejected(tmp_path):
    path = tmp_path / "timestamps.jsonl"
    record = dict(timed(("yes", 1, 10)), id="same")
    path.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        alignment.find_alignment({"id": "same"}, alignment.load_alignment_index(path))


def test_unicode_surrogate_row_does_not_abort(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_bytes(b'{"x":"\\ud800"}\n{}\n')
    rows = list(alignment.source_rows(path))
    assert rows[0][2] is not None and rows[1][2] is None


def test_short_audio_eos_releases_atomic_word(tmp_path):
    row = source(tmp_path, [("<zh><spk1>张伟", "WAIT")], .2)
    aligned = alignment.align_annotation(row, timed(("张伟", 10, 190)))
    result = recast.recast_row(aligned)
    assert [turn["action"] for turn in result["turns"]] == ["WAIT", "DIRECT"]
    assert result["turns"][0]["end_ms"] == 200
    assert emitted(result) == "张伟"


def test_seconds_adapter(tmp_path):
    row = source(tmp_path, [("<en><spk1>Hello.", "DIRECT")], 1)
    aligned = alignment.align_annotation(row, {"word_segments": [{"word": "Hello", "start": .1, "end": .6}]})
    assert aligned["segments"][0]["start_ms"] == 100
    assert aligned["segments"][0]["end_ms"] == 600


def test_forced_aligner_uses_char_timing_and_refuses_interpolation(tmp_path, monkeypatch):
    row = source(tmp_path, [("<en><spk1>Hi.", "DIRECT")], 1)
    fake = types.ModuleType("whisperx")
    fake.load_align_model = lambda **kwargs: (object(), {"dictionary": {"h": 1, "i": 2}})
    fake.load_audio = lambda path: np.zeros(16000)
    seen = {}
    def align(*args, **kwargs):
        seen.update(kwargs)
        return {"segments": [{"chars": [{"char": "H", "start": .1, "end": .3},
                                           {"char": "i", "start": .3, "end": .5}, {"char": "."}]}]}
    fake.align = align
    monkeypatch.setitem(sys.modules, "whisperx", fake)
    # No real torch/GPU/model is needed for the adapter contract.
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)))
    aligner = WhisperXAligner("cpu")
    result = aligner.align(row)
    assert seen["return_char_alignments"] is True
    assert len(result["chars"]) == 2
    fake.align = lambda *args, **kwargs: {"segments": [{"chars": [{"char": "H"}, {"char": "i"}]}]}
    with pytest.raises(ValueError, match="finite"):
        aligner.align(row)


def test_forced_aligner_rejects_unknown_digits_and_mix(tmp_path, monkeypatch):
    fake = types.ModuleType("whisperx")
    fake.load_align_model = lambda **kwargs: (object(), {"dictionary": {"a": 1}})
    monkeypatch.setitem(sys.modules, "whisperx", fake)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)))
    row = source(tmp_path, [("<en><spk1>100", "WAIT")], 1)
    with pytest.raises(ValueError, match="lacks lexical"):
        WhisperXAligner("cpu").align(row)
    row = source(tmp_path, [("<en><spk1>a", "DIRECT"), ("<zh><spk1>好", "WAIT")], 1)
    with pytest.raises(ValueError, match="monolingual"):
        WhisperXAligner("cpu").align(row)


def test_qwen_numeric_punctuation_restored_losslessly():
    text = "Pay 1,000.50 euros."
    items = [types.SimpleNamespace(text=text, start_time=start, end_time=end)
             for text, start, end in [("Pay", 0, .2), ("100050", .3, .8), ("euros", .9, 1.2)]]
    words = restore_qwen_spans(text, items)
    assert "".join(w["text"] for w in words) == text
    assert words[1]["text"] == " 1,000.50"
    assert words[1]["end"] == .8
    items[1].text = "1000500"
    with pytest.raises(ValueError, match="differ"):
        restore_qwen_spans(text, items)


def test_qwen_no_interpolated_timestamp_repairs():
    assert strict_qwen_timestamps(np.array([10., 20., 30., 40.])) == [10, 20, 30, 40]
    with pytest.raises(ValueError, match="monotonic"):
        strict_qwen_timestamps(np.array([10., 20., 5., 40.]))
    with pytest.raises(ValueError, match="invalid"):
        strict_qwen_timestamps(np.array([10., float("nan")]))


def test_qwen_adapter_real_api_shape_with_mocked_model(tmp_path, monkeypatch):
    row = source(tmp_path, [("<en><spk1>100 million", "WAIT")], 1)
    seen = {}
    class FakeModel:
        aligner_processor = types.SimpleNamespace()
        def align(self, **kwargs):
            seen.update(kwargs)
            return [types.SimpleNamespace(items=[types.SimpleNamespace(text="100", start_time=.1, end_time=.3),
                                                   types.SimpleNamespace(text="million", start_time=.4, end_time=.8)])]
    model = FakeModel()
    fake = types.ModuleType("qwen_asr")
    fake.Qwen3ForcedAligner = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: model)
    monkeypatch.setitem(sys.modules, "qwen_asr", fake)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(float32=object()))
    result = Qwen3Aligner("cpu", dtype="float32").align(row)
    assert seen["language"] == "English"
    assert seen["text"] == "100 million"
    assert model.aligner_processor.fix_timestamp is strict_qwen_timestamps
    assert result["words"][1]["text"] == " million"
    aligned = alignment.align_annotation(row, result)
    assert aligned["segments"][0]["end_ms"] == 800


def test_annotation_to_sft_eos_audio_decode_integration(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "SFT-Scripts"))
    import prepare_sft
    import check_sft
    row = source(tmp_path, [("<zh><spk1>请联系\n", "DIRECT"), ("张伟", "WAIT"),
                            ("<en><spk2>Thanks.", "WAIT")], .5)
    aligned = alignment.align_annotation(row, timed(("请联系", 10, 100), ("张伟", 110, 250), ("Thanks", 260, 490)))
    trajectory = recast.recast_row(aligned)
    prepared, tokens = prepare_sft.prepare_row(trajectory)
    assert len(prepared["audios"]) == 1  # EOS never creates an extra audio.
    assert prepared["messages"][-2] == {"role": "user", "content": "<end_of_audio>"}
    assert prepared["messages"][-1]["content"] == "<|DIRECT|><zh><spk1>请联系\n张伟<en><spk2>Thanks."
    checked = check_sft.check_row(prepared, decode=True, token_inventory=tokens)
    assert checked["audio_turns"] == checked["eos_turns"] == checked["direct_turns"] == 1
    assert checked["sample_counts"] == [8000]


def test_long_meeting_silence_window_preserved_with_evidence(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "SFT-Scripts"))
    import prepare_sft
    import prepare_windows
    import check_sft
    row = source(tmp_path, [("<en><spk1>Hello. ", "DIRECT"), ("Bye.", "DIRECT")], 60)
    aligned = alignment.align_annotation(row, timed(("Hello", 10, 200), ("Bye", 58900, 59000)))
    prepared, _ = prepare_sft.prepare_row(recast.recast_row(aligned))
    refs = [{"audio": row["audios"][0], "start_ms": 0, "end_ms": 500,
             "speaker": "spk1", "language": "en", "text": "Hello.",
             "verified": True, "clean": True, "overlap": False}]
    windows = prepare_windows.split_row(prepared, refs, max_seconds=20)
    assert len(windows) == 3
    silence = windows[1]
    assert silence["window"]["validated_silence"] is True
    assert silence["window"]["safe_start"] is silence["window"]["safe_end"] is True
    assert check_sft.check_row(silence, decode=False)["direct_turns"] == 0
    assert check_sft.check_row(windows[-1], decode=False)["direct_turns"] == 1
    damaged = copy.deepcopy(silence)
    damaged["source_speech_spans"].append({"start_ms": 21000, "end_ms": 22000})
    with pytest.raises(ValueError, match="overlaps"):
        check_sft.check_row(damaged, decode=False)
    for field in ("validated_silence", "parent_has_transcription", "safe_end"):
        damaged = copy.deepcopy(silence)
        damaged["window"].pop(field)
        with pytest.raises(ValueError, match="evidence"):
            check_sft.check_row(damaged, decode=False)
    damaged = copy.deepcopy(silence)
    damaged.pop("source_speech_spans")
    with pytest.raises(ValueError, match="evidence"):
        check_sft.check_row(damaged, decode=False)


def test_speech_evidence_must_cover_all_emitted_content(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "SFT-Scripts"))
    import prepare_sft
    row = source(tmp_path, [("<en><spk1>Hello. ", "DIRECT"), ("Bye.", "DIRECT")], 1)
    aligned = alignment.align_annotation(row, timed(("Hello", 10, 200), ("Bye", 600, 900)))
    trajectory = recast.recast_row(aligned)
    trajectory["source_segments"].pop()
    with pytest.raises(ValueError, match="cover"):
        prepare_sft.prepare_row(trajectory)


def test_window_last_committed_attribution_ignores_reference_language(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "SFT-Scripts"))
    import prepare_sft
    import prepare_windows
    from omnicommit.runtime_protocol import last_committed_text
    row = source(tmp_path, [("<en><spk1>Hi. ", "DIRECT"), ("<zh><spk1>你好。", "WAIT"),
                            ("继续。", "DIRECT"), ("<zh><spk2>是。", "WAIT"), ("好的。", "DIRECT")], 6)
    acoustic = timed(("Hi", 10, 300), ("你好", 2100, 2300), ("继续", 2600, 2800),
                     ("是", 4100, 4300), ("好的", 4600, 4800))
    prepared, _ = prepare_sft.prepare_row(recast.recast_row(alignment.align_annotation(row, acoustic)))
    refs = [{"audio": row["audios"][0], "start_ms": 0, "end_ms": 500,
             "speaker": "spk1", "language": "en", "text": "Hi.",
             "verified": True, "clean": True, "overlap": False}]
    windows = prepare_windows.split_row(prepared, refs, max_seconds=2)
    assert len(windows) == 3
    assert "Last committed live attribution:" not in windows[0]["messages"][0]["content"]
    assert last_committed_text("en", "spk1") in windows[1]["messages"][0]["content"]
    assert last_committed_text("zh", "spk1") in windows[2]["messages"][0]["content"]
    assert windows[2]["speaker_references"][0]["language"] == "en"
    assert windows[1]["messages"][2]["content"] == "<|WAIT|>"
    assert windows[2]["messages"][2]["content"] == "<|WAIT|>"
    with pytest.raises(ValueError, match="total reference"):
        prepare_windows.split_row(prepared, refs, max_seconds=2, max_total_reference_seconds=.4)
