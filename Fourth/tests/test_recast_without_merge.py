"""The standalone 1 s builder must release the original timestamped segments intact."""
import json
from pathlib import Path
import subprocess
import sys
import wave


FOURTH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FOURTH))
from fourth_asr.chunk_refs import decode_chunk_ref
from fourth_asr.protocol import DIRECT, EOS, SPK_CHANGE, WAIT


def test_original_end_times_and_actions_through_training_reader(tmp_path):
    audio = tmp_path / "meeting.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", ""))
        wav.writeframes(bytes(3500 * 16 * 2))

    # In the merge workflow a split decimal can move "3." into the second
    # turn. Here it must be committed at its original end, exactly 1000 ms.
    texts = ["<en><spk1>3.", "14", "<zh><spk1>收到了。", "<en><spk2> Yes.",
             "<en><spk2> See you.<zh><spk1>再见。"]
    times = [{"start_ms": start, "end_ms": end} for start, end in
             [(930, 1000), (1000, 1010), (1120, 1580), (1600, 1720), (1850, 3250)]]
    # A first child ends in turn 2. Even with validated subsegment times,
    # a mixed source segment is submitted only when its parent has ended.
    times[-1]["spans"] = [
        {"start_ms": 1850, "end_ms": 1980, "language": "en", "speaker": "spk2", "text": " See you."},
        {"start_ms": 2600, "end_ms": 3250, "language": "zh", "speaker": "spk1", "text": "再见。"},
    ]
    record = {"audios": [audio.name], "messages": [{"role": "assistant", "content": "".join(texts)}],
              "output": json.dumps({"segments": [{"source_text": text, "action": "WAIT"} for text in texts]}),
              "timestamps": times}
    source = tmp_path / "source.jsonl"
    original = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
    source.write_bytes(original)
    out = tmp_path / "training"
    subprocess.run([sys.executable, str(FOURTH / "Data-Scripts/recast_1s.py"), str(source),
                    "--output-dir", str(out), "--tail-policy", "keep"], check=True)
    train = out / "train.1000ms.jsonl"
    row = json.loads(train.read_bytes())
    assert source.read_bytes() == original
    assert (out / "rejected.jsonl").read_bytes() == b""
    assert row["recast"]["merged_units"] == 0
    assert [m["content"] for m in row["messages"] if m["role"] == "assistant"] == [
        DIRECT + "3.", DIRECT + "14收到了。" + SPK_CHANGE + " Yes.", WAIT,
        DIRECT + " See you." + SPK_CHANGE + "再见。", WAIT,
    ]
    assert row["messages"][-2]["content"] == EOS
    refs = [decode_chunk_ref(ref) for ref in row["audios"]]
    assert [(ref["start"], ref["end"]) for ref in refs] == [(0, 1), (1, 2), (2, 3), (3, 3.5)]
    assert all(Path(ref["path"]) == audio for ref in refs)
    assert Path(str(train) + ".tokens.txt").read_bytes() == (out / "token.txt").read_bytes()
    checked = subprocess.run([sys.executable, str(FOURTH / "SFT-Scripts/check_sft.py"), str(train), "--decode"],
                             capture_output=True, check=True)
    assert json.loads(checked.stdout)["checked_rows"] == 1
