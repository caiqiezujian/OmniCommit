"""Check training rows that stop on the last annotated sample, including mixed JSONL files."""
import json
from pathlib import Path
import sys
import wave

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fourth_asr.chunk_refs import encode_chunk_ref
from fourth_asr.protocol import DIRECT, EOS, SPK_CHANGE, WAIT, get_protocol
from fourth_asr.sft_checks import main, validate_row


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "meeting.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, 22050, 0, "NONE", ""))
        stream.writeframes(bytes(2 * 33075))  # physical duration 1.5s
    return path


def row_for(path, open_end):
    profile = get_protocol("speaker")
    end = 1.2 if open_end else 1.5
    messages = [{"role": "system", "content": profile.system_prompt()}]
    refs = []
    for start, stop, text in [(0, 1, "确认。"), (1, end, SPK_CHANGE + "Yes")]:
        refs.append(encode_chunk_ref(str(path), start, stop, max(0.5, stop-start)))
        messages.extend([{"role": "user", "content": "<audio>"}, {"role": "assistant", "content": DIRECT + text}])
    events = [{"turn": 2, "kind": "speaker_change"}]
    meta = {"source_audio": str(path), "merged_units": 0, "chunk_ms": 1000, "events": events}
    if open_end:
        meta.update(source_frames=33075, supervised_frames=26460)
        events.append({"turn": 2, "kind": "annotation_end"})
    else:
        messages.extend([{"role": "user", "content": EOS}, {"role": "assistant", "content": WAIT}])
        events.append({"turn": 2, "kind": "audio_end"})
    return {"messages": messages, "audios": refs, "recast": meta}


def test_open_and_closed_in_same_training_file(source, capsys):
    output = source.with_suffix(".jsonl")
    closed, open_row = row_for(source, False), row_for(source, True)
    output.write_text("\n".join(json.dumps(row) for row in [open_row, closed]) + "\n", encoding="utf-8")
    Path(str(output) + ".tokens.txt").write_text("\n".join(get_protocol("speaker").tokens), encoding="utf-8")
    assert main([str(output), "--decode"]) == 0
    assert json.loads(capsys.readouterr().out)["checked_rows"] == 2
    assert validate_row(open_row) == {DIRECT, SPK_CHANGE}
    assert validate_row(closed) == {DIRECT, SPK_CHANGE, WAIT}


def test_preflight_uses_source_header_and_unpadded_last_ref_for_missing_pair(source):
    row = row_for(source, True)
    for key in ("source_frames", "supervised_frames"):
        del row["recast"][key]
    dataset = source.with_suffix(".jsonl")
    original = json.dumps(row).encode() + b"\n"
    dataset.write_bytes(original)
    Path(str(dataset) + ".tokens.txt").write_text("\n".join(get_protocol("speaker").tokens), encoding="utf-8")
    assert main([str(dataset), "--rows", "2"]) == 0
    assert main([str(dataset), "--decode"]) == 0
    assert dataset.read_bytes() == original
    # The offline text filter has no access to WAVs and keeps its strict metadata check.
    with pytest.raises(ValueError, match="integer frame counts"):
        validate_row(row)
    row["recast"]["source_frames"] = 33075
    with pytest.raises(ValueError, match="integer frame counts"):
        validate_row(row, allow_header_lookup=True)


@pytest.mark.parametrize("end,match", [(1.6, "integer frame counts"), (1.20001, "does not end at")])
def test_missing_pair_cannot_hide_invalid_crop(source, end, match):
    row = row_for(source, True)
    del row["recast"]["source_frames"], row["recast"]["supervised_frames"]
    row["audios"][-1] = encode_chunk_ref(str(source), 1, end, 1)
    with pytest.raises(ValueError, match=match):
        validate_row(row, allow_header_lookup=True)


@pytest.mark.parametrize("field,value", [
    ("supervised_frames", 0), ("supervised_frames", 33076),
    ("supervised_frames", True), ("source_frames", None), ("source_frames", 33075.0),
])
def test_invalid_annotation_end_metadata(source, field, value):
    row = row_for(source, True)
    row["recast"][field] = value
    with pytest.raises(ValueError, match="integer frame counts"):
        validate_row(row)


@pytest.mark.parametrize("field,match", [
    ("supervised_frames", "supervised_frames"), ("source_frames", "original audio header"),
])
def test_decode_checks_real_source_frame_counts(source, field, match):
    row = row_for(source, True)
    row["recast"][field] += 1
    with pytest.raises(ValueError, match=match):
        validate_row(row, decode=True)


def test_open_end_rejects_ambiguous_end_or_unsupervised_last_turn(source):
    row = row_for(source, True)
    events = row["recast"]["events"]
    events.reverse()
    with pytest.raises(ValueError, match="occur last"):
        validate_row(row)
    events.reverse()
    events.insert(1, {"kind": "audio_end", "turn": 2})
    with pytest.raises(ValueError, match="sole end event"):
        validate_row(row)
    events.pop(1)
    row["messages"].extend([{"role": "user", "content": EOS}, {"role": "assistant", "content": WAIT}])
    with pytest.raises(ValueError, match="EOS pair only for closed"):
        validate_row(row)
    del row["messages"][-2:]
    row["messages"][-1]["content"] = WAIT
    with pytest.raises(ValueError, match="DIRECT containing transcript text"):
        validate_row(row)


def test_closed_rows_still_require_complete_audio_and_eos(source):
    row = row_for(source, False)
    row["audios"][-1] = encode_chunk_ref(str(source), 1, 1.2, 0.5)
    with pytest.raises(ValueError, match="before or beyond"):
        validate_row(row, decode=True)
    del row["recast"]
    del row["messages"][-2:]
    with pytest.raises(ValueError, match="EOS pair only for closed"):
        validate_row(row)
