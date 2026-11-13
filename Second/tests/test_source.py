import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from second_asr.source import Unit, parse_record


def record():
    return {
        "audios": ["/audio/file.wav"],
        "messages": [{"role": "assistant", "content": "<zh><spk1>请联系<zh><spk1>张伟，<en><spk2> OK."}],
        "output": {
            "segments": [
                {"source_text": "<zh><spk1>请联系", "action": "DIRECT"},
                {"source_text": "张伟，", "action": "WAIT"},
                {"source_text": "<en><spk2> OK.", "action": "DIRECT"},
            ],
            "flush": False, "error": None,
        },
        "timestamps": [
            {"index": 1, "start_ms": 0, "end_ms": 410},
            {"index": 2, "start_ms": 350, "end_ms": 810, "text_plain": "张伟，"},
            {"index": 3, "start_ms": 900, "end_ms": 1210, "text": " OK.",
             "spans": [{"language": "en", "speaker": "spk2", "start_ms": 910,
                        "end_ms": 1210, "text": "<en><spk2> OK.", "text_plain": " OK."}]},
        ],
        "tags": ["<|DIRECT|>", "<|WAIT|>", "<|DIRECT|>"],
    }


class SourceTest(unittest.TestCase):
    def test_legacy_labels_and_sparse_headers(self):
        row = record()
        row["output"] = json.dumps(row["output"])
        row["timestamps"].reverse()
        expected = [
            Unit(0, 410, (("zh", "spk1", "请联系"),)),
            Unit(350, 810, (("zh", "spk1", "张伟，"),)),
            Unit(900, 1210, (("en", "spk2", " OK."),)),
        ]
        self.assertEqual(parse_record(row, 1250), expected)
        for action in ("DIRECT", "WAIT", None):
            row["output"] = {"segments": [{"source_text": text} for text in ("请联系", "张伟，", " OK.")]}
            if action:
                for segment in row["output"]["segments"]:
                    segment["action"] = action
            self.assertEqual(parse_record(row, 1250), expected)

    def test_unit_crossing_both_switches(self):
        row = record()
        row["output"]["segments"] = [{"source_text": "请联系张伟，<en><spk2> OK.", "action": "DIRECT"}]
        row["timestamps"] = [{"start_ms": 0, "end_ms": 1210}]
        expected = [Unit(0, 1210, (("zh", "spk1", "请联系张伟，"), ("en", "spk2", " OK.")))]
        self.assertEqual(parse_record(row, 1210), expected)
        row["timestamps"][0]["spans"] = [
            {"start_ms": 0, "end_ms": 810, "language": "zh", "speaker": "spk1", "text": "请联系张伟，"},
            {"start_ms": 900, "end_ms": 1210, "language": "en", "speaker": "spk2", "text": " OK."},
        ]
        self.assertEqual(parse_record(row, 1210), expected)

    def test_marked_reference_and_lexical_literal(self):
        row = record()
        raw = "<zh><spk1>请联系张伟，<en>[900ms]<spk2> OK [120ms]."
        row["messages"][0].update(content=raw, content_original=raw.replace("[900ms]", ""))
        row["output"]["segments"][-1]["source_text"] = "<en> OK [120ms]."
        row["timestamps"][-1] = {"index": 3, "start_ms": 900, "end_ms": 1210}
        self.assertEqual(parse_record(row, 1300)[-1].spans, (("en", "spk2", " OK [120ms]."),))
        row["messages"][0]["content_original"] += "!"
        with self.assertRaises(ValueError):
            parse_record(row, 1300)

    def test_inverted_span_end_across_overlapping_parents(self):
        row = record()
        row["timestamps"][1]["spans"] = [
            {"start_ms": 350, "end_ms": 405, "language": "zh", "speaker": "spk1", "text_plain": "张伟，"}]
        self.assertEqual(parse_record(row, 1300)[1].end_ms, 810)
        with self.assertRaisesRegex(ValueError, "out of chronological order across segments"):
            parse_record(row, 1300, release_on="timed-span")

    def test_spacing_only_child_keeps_parent_intact(self):
        row = record()
        spans = row["timestamps"][-1]["spans"]
        spans[:] = [{"start_ms": 910, "end_ms": 995, "language": "en", "speaker": "spk2", "text_plain": " "},
                    {"start_ms": 1000, "end_ms": 1210, "language": "en", "speaker": "spk2", "text_plain": "OK."}]
        self.assertEqual(parse_record(row, 1250, release_on="timed-span"), parse_record(row, 1250))

    def test_invalid_reference_timing_or_attribution(self):
        cases = [
            lambda r: r["timestamps"][0].update(end_ms=0),
            lambda r: r["timestamps"][0].update(index=True),
            lambda r: r["timestamps"][1].update(index=3),
            lambda r: r["timestamps"][1].update(end_ms=float("nan")),
            lambda r: r["timestamps"][1].update(end_ms=405),
            lambda r: r["timestamps"][1].update(text_plain="张伟"),
            lambda r: r["timestamps"][0].update(text="<de><spk1>请联系"),
            lambda r: r["timestamps"][0].update(source_text="<zh><spk9>请联系"),
            lambda r: r["timestamps"][-1].update(end_ms=1251),
            lambda r: r["timestamps"][-1]["spans"][0].update(start_ms=899),
            lambda r: r["timestamps"][-1]["spans"][0].update(speaker="spk1"),
            lambda r: r["output"]["segments"][1].update(source_text="<en><spk2>张伟，"),
            lambda r: r["output"]["segments"][1].update(source_text="張偉，"),
            lambda r: r["output"]["segments"][-1].update(source_text="<en><spk2> OK"),
            lambda r: r["output"].update(error="invalid"),
            lambda r: r.update(timestamps=[{"time_ms": 900}]),
        ]
        for case in cases:
            row = record()
            case(row)
            with self.subTest(case=case), self.assertRaises(ValueError):
                parse_record(row, 1250)


if __name__ == "__main__":
    unittest.main()
