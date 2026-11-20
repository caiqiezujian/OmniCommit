import json
from pathlib import Path
import sys
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Data-Scripts"))
import profile_timestamped as profile


def record(audio, parts):
    texts = [f"<{lang}><{speaker}>{text}" for _, _, lang, speaker, text in parts]
    return {"audios": [audio], "messages": [{"role": "assistant", "content": "".join(texts)}],
            "output": json.dumps({"segments": [{"source_text": t} for t in texts]}),
            "timestamps": [{"index": i, "start_ms": a, "end_ms": b}
                           for i, (a, b, *_) in enumerate(parts, 1)]}


def test_profile_paths_and_denominators(tmp_path):
    with wave.open(str(tmp_path / "sample.wav"), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", ""))
        wav.writeframes(bytes(32000 * 4))
    parts = [(100, 700, "zh", "spk1", "到。"), (1850, 1980, "en", "spk2", "I"),
             (2200, 2350, "en", "spk2", " can"), (2400, 2950, "en", "spk2", " go.")]
    good, missing = record("sample.wav", parts), record("missing.wav", parts)
    path = tmp_path / "test.jsonl"
    path.write_text("invalid\n" + "\n".join(map(json.dumps, (good, missing))), encoding="utf-8")
    p = profile.scan(path, tmp_path, 100)
    profile.write_report(p, tmp_path / "stats", 100, path, tmp_path)
    report = json.loads((tmp_path / "stats/profile.json").read_text(encoding="utf-8"))
    assert report["rows"]["valid_source"] == report["rows"]["prepared"] == 2
    assert report["rows"]["audio_header_error"] == report["rows"]["invalid_source"] == 1
    assert report["rows"]["replayed"] == 1
    assert report["raw_all"]["ALL"]["n"] == 8
    assert report["raw_all"]["zh"]["space_groups"]["n"] == 0
    assert report["raw_all"]["en"]["one_space_group_pct"] == 100
    assert report["turn_counts"]["audio_turns"] == 4  # no EOF
    assert report["turn_counts"]["direct"] == 3
    assert report["short_source_outcomes"]["en"] == {"alone": 1, "with_other_text": 1}
    switch = report["switches"]["both->en"]
    assert switch["near_end_and_emitted_in_onset_turn"] == 1
    assert switch["onset_to_commit_ms"]["p50"] == 150
    assert report["records_per_language"] == {"zh": 2, "en": 2}
    assert report["between_direct_commits_ms"]["p50"] == 1000
    examples = [json.loads(line) for line in (tmp_path / "stats/examples.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(e["kind"] == "immediate_late_both" and e["line"] == 2 and e["turn"] == 2 for e in examples)


def test_mixed_without_child_times_and_unicode():
    from fourth_asr.timestamped_source import Unit
    u = Unit(100, 700, (("zh", "spk1", "嗯。"), ("th", "spk2", "ดี")))
    assert profile.timed_states([u]) == [(0, "zh", "spk1", 100), (2, "th", "spk2", None)]
    assert profile.char_count("  دي، e\u0301。한! ") == 4
    prepared = profile.Stage()
    prepared.units([u])
    assert set(prepared.report()) == {"ALL", "(mixed)"}
    assert prepared.report()["ALL"]["space_groups"]["n"] == 0


def test_exact_threshold_and_split_short_unit():
    from fourth_asr.timestamped_source import Unit
    parts = (("zh", "spk1", "上下"),)
    u = Unit(800, 1100, parts)
    turns = [(1, 1000, "", "上", "zh", []), (2, 2000, "", "下", "zh", [])]
    assert list(profile.short_unit_outcomes([u], turns)) == [("zh", "split_across_turns")]
    stage = profile.Stage()
    stage.units([u, Unit(1250, 1550.1, parts)])
    report = stage.report()["zh"]
    assert report["short_ms_pct"] == 50
    assert report["duration_ms"]["bins_upper_inclusive"]["300"] == 1
    assert report["next_same_lang_spk_nonoverlap_gap_ms"]["p50"] == 150
