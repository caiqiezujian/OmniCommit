"""Run the completed merger's output through the one-second Fourth training entry points."""
import json
from pathlib import Path
import subprocess
import sys
import wave


FOURTH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FOURTH / "Data-Scripts"))
from merge_timestamped_segments import merge_record
from fourth_asr.chunk_refs import decode_chunk_ref
from fourth_asr.protocol import DIRECT, EOS, SPK_CHANGE, WAIT


def merged_example(directory, name, duration_ms, parts):
    path = directory / (name + ".wav")
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", ""))
        wav.writeframes(bytes(duration_ms * 16 * 2))
    snippets = [f"<{language}><{speaker}>{text}" for _, _, language, speaker, text in parts]
    record = {
        "audios": [path.name],
        "messages": [{"role": "assistant", "content": "".join(snippets)}],
        "output": {"segments": [{"source_text": text, "action": "WAIT"} for text in snippets]},
        "timestamps": [{"start_ms": start, "end_ms": end} for start, end, *_ in parts],
    }
    merged, _ = merge_record(record)
    return merged


def test_merged_jsonl_to_decoded_training_data(tmp_path):
    records = [
        # A finishes, B answers briefly, then B continues in English across the 1 s edge.
        merged_example(tmp_path, "switch", 1900, [
            (100, 550, "zh", "spk1", "准备好了吗？"),
            (570, 780, "zh", "spk2", "好。"),
            (830, 1250, "en", "spk2", " All set."),
        ]),
        merged_example(tmp_path, "zero-gap", 850, [(80, 850, "en", "spk1", "Based in the U.S.")]),
        merged_example(tmp_path, "below-one-second", 1849, [(80, 850, "zh", "spk1", "可以吗？")]),
        merged_example(tmp_path, "exactly-one-second", 1850, [(80, 850, "zh", "spk1", "可以。")]),
        # The detached period is in a later window. Its timestamp determines the
        # tail policy, but its removal must not leave a silent training window.
        merged_example(tmp_path, "detached-period", 1920, [
            (35, 800, "zh", "spk1", "我是你"), (1200, 1300, "zh", "spk1", "。"),
        ]),
        # Step 1 attached the period. Its child timestamp still identifies it.
        merged_example(tmp_path, "attached-period", 1680, [
            (35, 970, "zh", "spk1", "我是你"), (1000, 1120, "zh", "spk1", "。"),
        ]),
    ]
    assert len(records[-2]["timestamps"]) == 2
    assert len(records[-1]["timestamps"]) == 1
    source = tmp_path / "merged.jsonl"
    original = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records).encode()
    source.write_bytes(original)
    output = tmp_path / "built"
    run = subprocess.run([
        sys.executable, str(FOURTH / "Data-Scripts" / "build_sft.py"), str(source),
        "--output-dir", str(output), "--chunk-ms", "1000",
        "--release-on", "merged-unit", "--tail-policy", "short-open",
    ], capture_output=True)
    assert run.returncode == 0, run.stderr.decode(errors="replace")
    assert source.read_bytes() == original
    assert (output / "rejected.jsonl").read_bytes() == b""
    training_file = output / "train.1000ms.jsonl"
    rows = [json.loads(line) for line in training_file.read_text(encoding="utf8").splitlines()]
    expected_replies = [
        [DIRECT + "准备好了吗？" + SPK_CHANGE + "好。", DIRECT + " All set"],
        [DIRECT + "Based in the U.S."],
        [DIRECT + "可以吗？"],
        [DIRECT + "可以。", WAIT, WAIT],
        [DIRECT + "我是你"],
        [DIRECT + "我是你"],
    ]
    assert len(rows) == len(expected_replies)
    for i, (row, expected) in enumerate(zip(rows, expected_replies)):
        assert [m["content"] for m in row["messages"] if m["role"] == "assistant"] == expected
        has_eos = any(m["role"] == "user" and m["content"] == EOS for m in row["messages"])
        assert has_eos == (i == 3)
        refs = [decode_chunk_ref(ref) for ref in row["audios"]]
        assert all(Path(ref["path"]) == tmp_path / records[i]["audios"][0] for ref in refs)
        assert refs[-1]["end"] == [1.25, .85, .85, 1.85, .8, .97][i]
    verify = subprocess.run([sys.executable, str(FOURTH / "SFT-Scripts" / "check_sft.py"),
                             str(training_file), "--decode"], capture_output=True)
    assert verify.returncode == 0, verify.stderr.decode(errors="replace")
    assert json.loads(verify.stdout)["checked_rows"] == len(records)


def test_known_partial_annotations_do_not_turn_a_long_unknown_suffix_into_wait(tmp_path):
    record = merged_example(tmp_path, "partial", 4680, [(80, 680, "zh", "spk1", "我知道。")])
    source = tmp_path / "partial.jsonl"
    source.write_text(json.dumps(record) + "\n", encoding="utf8")
    out = tmp_path / "prefix"
    build = subprocess.run([
        sys.executable, str(FOURTH / "Data-Scripts" / "build_sft.py"), str(source),
        "--output-dir", str(out), "--chunk-ms", "1000",
        "--release-on", "merged-unit", "--tail-policy", "annotated-prefix",
    ], capture_output=True)
    assert build.returncode == 0, build.stderr.decode(errors="replace")
    row = json.loads((out / "train.1000ms.jsonl").read_text(encoding="utf8"))
    assert [msg["content"] for msg in row["messages"][1:]] == ["<audio>", DIRECT + "我知道。"]
    ref, = row["audios"]
    assert decode_chunk_ref(ref)["end"] == .68
