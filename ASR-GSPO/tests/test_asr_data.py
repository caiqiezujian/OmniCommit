import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asr_gspo.data import audio_chunks, prepare_record
from asr_gspo.protocol import initial_messages
from prepare_data import main
from third_asr.chunk_refs import decode_chunk_ref


class TestASRPreparation(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.path = self.root / "录音.wav"
        with wave.open(str(self.path), "wb") as wav:
            wav.setparams((1, 2, 11025, 0, "NONE", "not compressed"))
            wav.writeframes(bytes(2 * 21180))

    def source(self, text):
        return {"audios": [self.path.name], "messages": [
            {"role": "user", "content": "<audio>**Entity Reference:** [SECRET]"},
            {"role": "assistant", "content": text}]}

    def task(self, record):
        formatted = prepare_record(record, self.root)
        return json.loads(formatted["asr_task"])

    def test_switches_are_onsets_and_repeated_headers_are_not_boundaries(self):
        original = "<ar><spk1>مرحبا،<ar><spk1> أنا<en><spk1> Alex.<en><spk2> Hi!"
        row = self.source(original.replace("<en><spk1>", "[400ms]<en><spk1>"))
        row["messages"][-1]["content_original"] = original
        row["timestamps"] = [
            {"language": "en", "speaker": "spk1", "time_ms": 400},
            {"language": "en", "speaker": "spk2", "time_ms": 980}]
        prepared = prepare_record(row, self.root)
        self.assertEqual(prepared["messages"], initial_messages())
        self.assertNotIn("SECRET", json.dumps(prepared, ensure_ascii=False))
        self.assertEqual(len(prepared["audios"]), 1)
        self.assertEqual(decode_chunk_ref(prepared["audios"][0])["end"], 1.0)
        task = json.loads(prepared["asr_task"])
        self.assertEqual(task["spans"], [{"lang": "ar", "text": "مرحبا، أنا"}, {"lang": "en", "text": " Alex. Hi!"}])
        self.assertEqual(task["bucket"], "switch")
        self.assertEqual(task["alignment"], [])
        self.assertEqual([x["char_start"] for x in task["switches"]], [len("مرحبا، أنا"), len("مرحبا، أنا Alex.")])
        self.assertEqual(task["reference_text"], "مرحبا، أنا Alex. Hi!")
        row["messages"][-1]["content_original"] += "?"
        with self.assertRaisesRegex(ValueError, "content_original"):
            self.task(row)

    def test_ctc_unicode_offsets_partial_invalid_stamps_and_sparse_words(self):
        row = self.source("<zh><spk1>甲，甲！<en><spk2> Go 😀")
        row["chars"] = [{"char": "甲", "start": 0.1, "end": 0.2},
                        {"char": "甲", "start_ms": None, "end_ms": 400},
                        {"char": "😀", "char_start": 8, "char_end": 9, "start_ms": 980}]
        anchors = self.task(row)["alignment"]
        self.assertEqual([(a["char_start"], a["char_end"]) for a in anchors], [(0, 1), (2, 3), (8, 9)])
        self.assertEqual((anchors[1]["start_ms"], anchors[1]["end_ms"]), (None, 400))
        self.assertEqual((anchors[2]["start_ms"], anchors[2]["end_ms"]), (980, None))
        row["chars"][-1]["start_ms"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.task(row)
        row["chars"] = [{"char": "Go", "start_ms": 820, "end_ms": 910}]
        with self.assertRaisesRegex(ValueError, "disagrees"):
            self.task(row)

    def test_native_third_semantic_alignment_and_synthesis_sources(self):
        row = self.source("<zh><spk1>预算<en><spk2> 30 dollars.")
        row["output"] = json.dumps({"segments": [
            {"source_text": "<zh><spk1>预算", "action": "WAIT"},
            {"source_text": "<en><spk2> 30 dollars.", "action": "DIRECT"}], "error": None, "flush": False})
        row["timestamps"] = [{"index": 1, "start_ms": 0, "end_ms": 440},
                             {"index": 2, "start_ms": 650, "end_ms": 1500}]
        row["synthesis"] = {"clips": [{"audio": "donor-a.wav"}, {"audio": "donor-b.wav"}]}
        task = self.task(row)
        self.assertEqual(task["source_ids"], sorted(str(self.root / x) for x in ("donor-a.wav", "donor-b.wav", self.path.name)))
        self.assertEqual(task["reference_text"], "预算 30 dollars.")
        self.assertEqual(task["alignment"][1], dict(text=" 30 dollars.", start_ms=650, end_ms=1500, char_start=2, char_end=14))
        self.assertNotIn("WAIT", json.dumps(task))
        row["timestamps"][1]["end_ms"] = 5000
        with self.assertRaisesRegex(ValueError, "outside"):
            self.task(row)

    def test_existing_aligned_asr_and_audio_identity_validation(self):
        record = {"kind": "aligned_asr", "schema_version": 2, "status": "ok", "audio": self.path.name,
                  "transcript": "<en><spk1>Go<zh><spk1>吧", "duration_ms": 21180 / 11.025,
                  "segments": [{"source_text": "<en><spk1>Go", "start_ms": 225, "end_ms": 710},
                               {"source_text": "<zh><spk1>吧", "start_ms": 750, "end_ms": 900}]}
        task = self.task(record)
        self.assertEqual((task["reference_text"], task["bucket"]), ("Go吧", "switch"))
        self.assertEqual([a["end_ms"] for a in task["alignment"]], [710, 900])
        record["duration_ms"] = 999
        with self.assertRaisesRegex(ValueError, "duration_ms"):
            self.task(record)
        raw = self.source("Good.")
        raw["alignment"] = {"audio": "other.wav", "words": [{"word": "Good", "start": 0, "end": 0.3}]}
        with self.assertRaisesRegex(ValueError, "audio path"):
            self.task(raw)

    def test_frame_boundaries_tail_and_silence(self):
        chunks = audio_chunks(4400, 11025, 125)
        self.assertEqual([round(c["end_seconds"] * 11025) for c in chunks], [1379, 2757, 4135, 4400])
        self.assertEqual(chunks[-1]["input_seconds"], 0.5)
        self.assertEqual(chunks[-1]["end_seconds"], 4400 / 11025)
        with self.assertRaisesRegex(ValueError, "5 turns including EOF"):
            audio_chunks(4400, 11025, 125, max_turns=4)
        silent = self.task(self.source(""))
        self.assertEqual((silent["reference_text"], silent["spans"], silent["bucket"]), ("", [], "stability"))
        with self.assertRaisesRegex(ValueError, "only attribution"):
            self.task(self.source("<en><spk1>"))

    def test_duplicate_ids_and_donor_recording_groups(self):
        record = self.source("<en><spk1> Good.")
        record.update(key="k1", source_ids=["user-group"], meeting_id="m", recording_id="r")
        self.assertEqual(set(self.task(record)["source_ids"]),
                         {str(self.path), "user-group", "meeting:m", "recording:r"})
        src, out, rejects = [self.root / x for x in ("in", "out", "rejects")]
        src.write_text((json.dumps(record) + "\n") * 2, encoding="utf-8")
        main([str(src), "--output", str(out), "--rejects", str(rejects)])
        self.assertEqual(len(out.read_text(encoding="utf-8").splitlines()), 1)
        self.assertIn("Duplicate task ID", json.loads(rejects.read_text(encoding="utf-8"))["error"])

    def test_skipped_bad_lines_and_atomic_previous_output(self):
        src, dst, rej = [self.root / (x + ".jsonl") for x in ("input", "out", "rejects")]
        valid = json.dumps(self.source("<zh><spk1>你好"), ensure_ascii=False)
        src.write_bytes(valid.encode() + b"\n{bad\n")
        dst.write_text("old\n", encoding="utf8")
        with self.assertRaisesRegex(SystemExit, ":2:"):
            main([str(src), "--output", str(dst)])
        self.assertEqual(dst.read_text(encoding="utf8"), "old\n")
        main([str(src), "--output", str(dst), "--rejects", str(rej)])
        self.assertEqual(json.loads(json.loads(dst.read_text(encoding="utf8"))["asr_task"])["reference_text"], "你好")
        self.assertEqual(json.loads(rej.read_text(encoding="utf8"))["line"], 2)
        src.write_bytes(b"{}\n")
        retained = dst.read_bytes()
        with self.assertRaisesRegex(SystemExit, "No valid"):
            main([str(src), "--output", str(dst), "--rejects", str(rej)])
        self.assertEqual(dst.read_bytes(), retained)
        self.assertEqual(json.loads(rej.read_text(encoding="utf8"))["line"], 1)
        self.assertFalse(list(self.root.glob("*.partial")))


if __name__ == "__main__":
    unittest.main()
