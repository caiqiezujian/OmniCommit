import json
from pathlib import Path
import sys
import wave

import pytest

FOURTH = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(FOURTH), str(FOURTH / "Data-Scripts")]
from filter_final_sft import TRAIN_FILE, main, punctuation_evidence
from fourth_asr.chunk_refs import encode_chunk_ref
from fourth_asr.protocol import DIRECT, EOS, SPK_CHANGE, WAIT, get_protocol
from fourth_asr.sft_checks import main as check_training


def recording(*replies, mode="speaker", open_end=False, silence=False, path="/missing/on/purpose/meeting.wav"):
    profile = get_protocol(mode)
    path = str(path)
    refs, events = [], []
    msgs = [{"role": "system", "content": profile.system_prompt("normalized")}]
    for turn, body in enumerate(replies, 1):
        refs.append(encode_chunk_ref(path, turn - 1, turn, 1))
        reply = WAIT if body == WAIT else profile.prefix + body
        msgs.extend([{"role": "user", "content": "<audio>"}, {"role": "assistant", "content": reply}])
        if SPK_CHANGE in reply:
            events.append({"turn": turn, "kind": "speaker_change"})
    if silence:
        events.append({"turn": 1, "kind": "silence_only"})
    events.append({"turn": len(refs), "kind": "annotation_end" if open_end else "audio_end"})
    meta = dict(source_audio=path, merged_units=0, chunk_ms=1000, events=events)
    if open_end:
        meta.update(supervised_frames=len(refs)*16000, source_frames=len(refs)*16000+8000)
    else:
        msgs.extend([{"role": "user", "content": EOS}, {"role": "assistant", "content": WAIT}])
    return {"audios": refs, "messages": msgs, "recast": meta}


@pytest.mark.parametrize("bodies,expected,peak", [
    (["我们把。", "时间改了。如果明天下雨。", "我们就改线上。"], {"suspect_period": 2}, 2),
    (["我们把。", "日期改了。", WAIT, "We need to.", " check the figures."], {"suspect_period": 2}, 1),
    (["我们把。", SPK_CHANGE + "门关上。"], {}, 0),
    (["我知道。", WAIT, "知道大家的想法。"], {}, 0),
    (["我知道。", "知道大家的想法。"], {"repeated_boundary_text": 1}, 0),
    (["我知道。" + SPK_CHANGE + "知道大家的想法。"], {}, 0),
    (["我。觉得。这个。方案。"], {"four_short_sentences": 1}, 0),
    (["好。", "对。", "嗯。", "行。"], {}, 0),
    (["好的。好的。我们开始吧。"], {}, 0),
    (["We use 3.14 in the U.S. Ask Dr. Smith about the.com domain."], {}, 0),
    (["我知道。这件事我来处理。"], {}, 0),
    (["We need the.", "latest figures."], {"suspect_period": 1}, 1),
])
def test_text_boundaries(bodies, expected, peak):
    maximum, counts, _ = punctuation_evidence(recording(*bodies), get_protocol("speaker"))
    assert counts == expected
    assert maximum == peak


def test_distinct_turn_location_and_older_protocol_without_speaker_metadata():
    row = recording("我们把。", "门关上。")
    _, _, samples = punctuation_evidence(row, get_protocol("speaker"))
    assert (samples["suspect_period"][0]["turn"], samples["suspect_period"][0]["next_text_turn"]) == (1, 2)
    row = recording("我们把。", "门关上。", mode="plain")
    assert punctuation_evidence(row, get_protocol("plain"))[0] == 0


def encoded(row):
    return json.dumps(row, ensure_ascii=False).encode("utf-8")


def read_lines(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


@pytest.mark.parametrize("policy,kept,review", [
    ("conservative", [0, 2, 3, 4], [3, 4]),
    ("strict", [0, 4], []),
])
def test_filter_then_train_check(tmp_path, policy, kept, review):
    rows = [
        recording("我们继续", WAIT, "开会。", open_end=True),
        recording("如果明天下雨。我们就改线上。我们把。日期改了。"),
        recording("我知道。", "知道大家的想法。"),
        recording("我。觉得。这个。方案。"),
        recording(WAIT, WAIT, silence=True),
    ]
    payloads = [b"  " + encoded(row) + b"\r\n" for row in rows]
    bad = recording("坏协议")
    bad["messages"][2]["content"] = DIRECT
    source = tmp_path / "train.jsonl"
    original = b"\xef\xbb\xbf" + b"".join(payloads) + b"{bad json}\n" + encoded(bad)
    source.write_bytes(original)
    out = tmp_path / policy
    assert main([str(source), "--output-dir", str(out), "--policy", policy]) == 0
    assert source.read_bytes() == original
    assert (out / TRAIN_FILE).read_bytes() == b"".join(payloads[index] for index in kept)
    assert check_training([str(out / TRAIN_FILE)]) == 0
    assert (out / (TRAIN_FILE + ".tokens.txt")).read_bytes() == (out / "token.txt").read_bytes()
    candidates = read_lines(out / "review.jsonl")
    assert [r["line"] for r in candidates] == review
    rejects = read_lines(out / "rejected.jsonl")
    assert rejects[0]["record"] == rows[1]
    assert rejects[0]["reason"] == "suspect_periods"
    assert rejects[-2]["raw_line"] == "{bad json}"
    assert "audio turn 1" in rejects[-1]["error"]
    report = json.loads((out / "summary.json").read_bytes())
    assert (report["rows"], report["kept"], report["kept_for_review"]) == (7, len(kept), len(review))
    assert report["validated_before"]["rows"] == 5
    assert report["kept_after"]["silence_only_rows"] == 1
    assert report["kept_after"]["wait_turns"] == 3  # excludes the synthetic EOF replies
    assert report["rejected_by_reason"]["invalid_record"] == 2
    if policy == "strict":
        assert report["rejected_by_reason"]["quarantined_punctuation_candidate"] == 2
    with pytest.raises(SystemExit):
        main([str(source), "--output-dir", str(out)])


def test_sidecar_selects_protocol_and_all_rejections_are_inspectable(tmp_path):
    source = tmp_path / "train.jsonl"
    source.write_bytes(encoded(recording("自然结束。", mode="plain")))
    Path(str(source) + ".tokens.txt").write_text("\n".join(get_protocol("speaker").tokens), encoding="utf-8")
    out = tmp_path / "filtered"
    assert main([str(source), "--output-dir", str(out)]) == 1
    assert (out / TRAIN_FILE).read_bytes() == b""
    reject, = read_lines(out / "rejected.jsonl")
    assert (reject["reason"], reject["actual_protocol"]) == ("protocol_mismatch", "plain")


@pytest.mark.parametrize("policy", ["conservative", "strict"])
def test_all_wait_recordings_survive_filter_and_training_decode(tmp_path, policy):
    def silent_wav(name, seconds):
        path = tmp_path / name
        with wave.open(str(path), "wb") as stream:
            stream.setparams((1, 2, 16000, 0, "NONE", ""))
            stream.writeframes(bytes(int(seconds * 16000) * 2))
        return path

    one = silent_wav("one.wav", 1)
    two = silent_wav("two.wav", 2)
    excerpt = silent_wav("excerpt.wav", 2.5)
    rows = [
        recording(WAIT, silence=True, path=one),
        recording(WAIT, WAIT, path=two),
        recording(WAIT, WAIT, path=two),
        recording(WAIT, WAIT, open_end=True, path=excerpt),
        recording(WAIT, WAIT, open_end=True, silence=True, path=excerpt),
    ]
    del rows[2]["recast"]
    source = tmp_path / "original.jsonl"
    source.write_bytes(b"".join(encoded(row) + b"\n" for row in rows))
    out = tmp_path / "out"
    assert main([str(source), "--output-dir", str(out), "--policy", policy]) == 0
    assert (out / TRAIN_FILE).read_bytes() == source.read_bytes()
    assert not (out / "rejected.jsonl").read_bytes()
    assert not (out / "review.jsonl").read_bytes()
    report = json.loads((out / "summary.json").read_bytes())
    assert report["validated_before"] == report["kept_after"]
    assert report["kept_after"]["silence_only_rows"] == len(rows)
    assert report["kept_after"]["wait_turns"] == 9  # live audio turns only
    assert report["invalid_all_wait_rows"] == report["all_wait_system_prompts_adapted"] == 0
    assert check_training([str(out / TRAIN_FILE), "--decode"]) == 0


@pytest.mark.parametrize("sidecar", [False, True])
def test_legacy_silence_uses_speech_protocol_and_keeps_all_targets(tmp_path, sidecar):
    source = tmp_path / "mixed.jsonl"
    target = get_protocol("speaker")
    if sidecar:
        Path(str(source) + ".tokens.txt").write_text("\n".join(target.tokens), encoding="utf-8")
    silent = recording(WAIT, WAIT, mode="plain")
    silent["messages"][0]["content"] = get_protocol("plain").system_prompt("verbatim")
    del silent["recast"]
    filtered_speech = recording("我。觉得。这个。方案。", mode="explicit")
    retained_speech = recording("明天继续。")
    other_silence = recording(WAIT, mode="explicit", silence=True)
    rows = [silent, filtered_speech, retained_speech, other_silence]
    original = b"".join(encoded(row) + b"\n" for row in rows)
    source.write_bytes(original)
    out = tmp_path / "out"
    assert main([str(source), "--output-dir", str(out), "--policy", "strict"]) == 0
    assert source.read_bytes() == original
    first, speech, last = read_lines(out / TRAIN_FILE)
    for kept, old, numbers in ((first, silent, "verbatim"), (last, other_silence, "normalized")):
        assert kept["messages"][0]["content"] == target.system_prompt(numbers)
        assert kept["messages"][1:] == old["messages"][1:]
        assert kept["audios"] == old["audios"]
        assert kept.get("recast") == old.get("recast")
    assert speech == retained_speech
    assert [r["line"] for r in read_lines(out / "rejected.jsonl")] == [2]
    report = json.loads((out / "summary.json").read_bytes())
    assert report["all_wait_system_prompts_adapted"] == 2
    assert report["validated_before"]["silence_only_rows"] == report["kept_after"]["silence_only_rows"] == 2
    assert (out / (TRAIN_FILE + ".tokens.txt")).read_text("utf-8").splitlines() == list(target.tokens)
    assert check_training([str(out / TRAIN_FILE)]) == 0


def test_all_silent_input_with_multiple_protocols_audits_corrupt_rows(tmp_path):
    source = tmp_path / "silence.jsonl"
    broken = recording(WAIT)
    broken["audios"][0] = "invalid virtual audio reference"
    rows = [broken, recording(WAIT, mode="explicit"), recording(WAIT, WAIT, mode="plain"), recording(WAIT)]
    source.write_bytes(b"".join(encoded(row) + b"\n" for row in rows))
    out = tmp_path / "out"
    assert main([str(source), "--output-dir", str(out), "--policy", "strict"]) == 0
    report = json.loads((out / "summary.json").read_bytes())
    assert (report["protocol"], report["invalid_all_wait_rows"], report["all_wait_system_prompts_adapted"]) == ("explicit", 1, 2)
    assert report["validated_before"]["silence_only_rows"] == report["kept_after"]["silence_only_rows"] == 3
    reject, = read_lines(out / "rejected.jsonl")
    assert (reject["line"], reject["reason"], reject["all_wait_targets"]) == (1, "invalid_record", True)
    assert check_training([str(out / TRAIN_FILE)]) == 0


def test_malformed_bytes_do_not_stop_later_records_or_need_audio_files(tmp_path):
    data = encoded(recording("正常的最后一行。"))
    src = tmp_path / "records.jsonl"
    src.write_bytes(b"\xff \n \n" + data)
    out = tmp_path / "filtered"
    assert main([str(src), "--output-dir", str(out)]) == 0
    assert (out / TRAIN_FILE).read_bytes() == data + b"\n"
    assert [r["line"] for r in read_lines(out / "rejected.jsonl")] == [1, 2]
