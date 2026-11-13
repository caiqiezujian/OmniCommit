import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

SECOND = Path(__file__).resolve().parents[1]
SCRIPT = SECOND / "Data-Scripts/augment_codeswitch.py"


def run(script, *args):
    return subprocess.run([sys.executable, str(script), *map(str, args)], capture_output=True,
                          text=True, encoding="utf-8", errors="replace")


def material(tmp_path, mixed_rate=False, copies=1, invalid=True):
    rows = []
    mono = [("zh", ("你好。", "再见。")), ("it", ("Sì.", " Ciao.")), ("en", ("OK.", " Bye."))]
    for lang, words in mono * copies + [("native", ("你好。", " Sure."))]:
        rate = 22050 if mixed_rate and lang == "it" else 16000
        path = tmp_path / f"{lang}-{len(rows)}.wav"
        # Audible, distinguishable source material, stereo, nonzero cut-in.
        value = (len(rows) % 8 + 1)*0.1
        sf.write(path, np.tile([value, value/2], (rate, 1)), rate, subtype="PCM_16")
        langs = ["zh", "en"] if lang == "native" else [lang, lang]
        segs = [f"<{l}><spk1>{w}" for l, w in zip(langs, words)]
        rows.append(dict(audios=[path.name], messages=[{"role": "assistant", "content": "".join(segs)}],
                         output=json.dumps({"segments": [{"source_text": t, "action": "WAIT"} for t in segs]}),
                         timestamps=[dict(index=1, start_ms=25, end_ms=271.25),
                                     dict(index=2, start_ms=285, end_ms=763.5)]))
    source = tmp_path / "source.jsonl"
    source.write_text(("bad line\n" if invalid else "")+"\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return source


@pytest.mark.parametrize("no_gap,gap", [(1, 0), (0, 37)])
def test_audio_json_and_sft_roundtrip(tmp_path, no_gap, gap):
    source = material(tmp_path, mixed_rate=not no_gap, copies=4)
    folder = tmp_path / "audio"
    result = run(SCRIPT, source, "--audio-dir", folder, "--count", 21, "--no-gap-prob", no_gap,
                 "--gap-ms", 37, 37, "--short-max-ms", 500)
    assert result.returncode == 0, result.stdout + result.stderr
    output = folder / "codeswitch.jsonl"
    rows = [json.loads(l) for l in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(list(folder.rglob("*.wav"))) == 21
    scenarios = set()
    for row in rows:
        meta = row["synthesis"]
        scenarios.add(meta["scenario"])
        audio, rate = sf.read(row["audios"][0])
        assert rate == 16000 and audio.ndim == 1
        previous = 0
        for i, clip in enumerate(meta["clips"]):
            original, _ = sf.read(clip["audio"], always_2d=True)
            start, end = clip["start_frame"], clip["end_frame"]
            assert end-start == round((clip["source_end_frame"]-clip["source_start_frame"])*rate/clip["source_rate"])
            assert start-previous == (gap*16 if i else 0)
            assert np.all(audio[previous:start] == 0)
            assert np.abs(np.mean(audio[start:end])-np.mean(original)) < .001
            previous = end
        assert previous == len(audio)
        assert 0 < round(row["timestamps"][-1]["end_ms"]*16) <= len(audio)
        stamps = row["timestamps"]
        runs = [s for i, s in enumerate(stamps) if not i or stamps[i-1]["language"] != s["language"]]
        assert [s["language"] for s in runs] == meta["languages"]
        if "short" in meta["scenario"]:
            middle = [s for s in stamps if s["language"] == runs[1]["language"]]
            assert middle[-1]["end_ms"]-middle[0]["start_ms"] <= 500
        if "return" in meta["scenario"]:
            assert runs[0]["speaker"] == runs[2]["speaker"] != runs[1]["speaker"]
        assert runs[0]["speaker"] != runs[1]["speaker"]
    assert scenarios == {"pair", "triple", "short-triple", "return", "short-return"}
    recast = run(SECOND / "Data-Scripts/build_sft.py", output, "--output-dir", tmp_path/"sft",
                 "--chunk-ms", 500, 800)
    assert recast.returncode == 0, recast.stderr
    assert json.loads((tmp_path/"sft/summary.json").read_text())["kept"] == 21
    assert (tmp_path/"sft/rejected.jsonl").stat().st_size == 0


def test_deficits_bad_audio_and_small_pools(tmp_path):
    source = material(tmp_path)
    path = tmp_path / "audio"
    result = run(SCRIPT, source, "--audio-dir", path, "--scenarios", "pair", "--count", 6)
    assert result.returncode == 0, result.stderr
    output = path / "codeswitch.jsonl"
    routes = [json.loads(l)["synthesis"]["languages"] for l in output.read_text(encoding="utf-8").splitlines()]
    assert len(set(map(tuple, routes))) == 6
    assert routes[-1] == ["zh", "en"]  # the only pre-existing direction
    before = output.read_bytes()
    assert run(SCRIPT, source, "--audio-dir", path, "--count", 1).returncode != 0
    assert output.read_bytes() == before
    for wav in tmp_path.glob("*.wav"):
        wav.unlink()
    result = run(SCRIPT, source, "--audio-dir", tmp_path / "empty", "--count", 2)
    assert result.returncode != 0 and '"written": 0' in result.stdout
    assert not list((tmp_path/"empty").rglob("*.wav"))


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_quota_mode_combines_unordered_sets_and_source_roles(tmp_path):
    source = material(tmp_path, copies=6, invalid=False)
    original = source.read_bytes()
    original_wavs = {p.resolve(): p.read_bytes() for p in tmp_path.glob("*.wav")}
    dst = tmp_path / "output"
    run_args = (source, "--audio-dir", dst, "--languages", "zh", "en", "it",
                "--pair-target", 2, "--triple-target", 1)
    result = run(SCRIPT, *run_args)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = jsonl(dst / "codeswitch.jsonl")
    combos = Counter(frozenset(re.findall(r"<(zh|it|en)>", row["messages"][-1]["content"])) for row in rows)
    assert {c: n for c, n in combos.items() if len(c) == 2} == {
        frozenset(c): 2 for c in combinations(["zh", "en", "it"], 2)}
    assert combos[frozenset(["zh", "it", "en"])] == 1
    synthesized = [r for r in rows if "synthesis" in r]
    assert len(synthesized) == 6  # original bilingual counts toward exactly one unordered pair
    full, partial = set(), set()
    for row in synthesized:
        assert Path(row["audios"][0]).is_file()
        for clip in row["synthesis"]["clips"]:
            p = Path(clip["audio"]).resolve()
            if clip["source_usage"] == "full":
                full.add(p)
            else:
                assert clip["source_usage"] == "partial"
                partial.add(p)
    retained = {Path(r["audios"][0]).resolve() for r in rows if "synthesis" not in r}
    assert full and retained == (set(original_wavs) - full)
    assert partial <= retained
    assert {p: p.read_bytes() for p in original_wavs} == original_wavs
    assert original == source.read_bytes()


def test_quota_preserves_above_target_without_planning_new_wavs(tmp_path):
    source = material(tmp_path, invalid=False)
    original = source.read_bytes()
    output = tmp_path / "other" / "training.jsonl"
    result = run(SCRIPT, source, "--audio-dir", tmp_path/"out", "--output", output,
                 "--languages", "zh", "en", "--pair-target", 1, "--triple-target", 0)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(jsonl(output)) == len(jsonl(source))
    assert not list((tmp_path/"out").rglob("*.wav"))
    assert source.read_bytes() == original


def test_unreachable_combination_reports_deficit_and_retains_valid_rows(tmp_path):
    source = material(tmp_path, invalid=False)
    output = tmp_path/"new"/"combined.jsonl"
    result = run(SCRIPT, source, "--audio-dir", tmp_path/"wav", "--output", output,
                 "--languages", "en", "ja", "--pair-target", 2, "--triple-target", 0)
    assert result.returncode != 0
    assert "ja" in result.stdout
    assert len(jsonl(output)) == len(jsonl(source))


def test_quota_short_triples_retain_middle_donors_and_full_waveforms(tmp_path):
    source = material(tmp_path, copies=4, invalid=False)
    output = tmp_path/"synth"/"all.jsonl"
    result = run(SCRIPT, source, "--audio-dir", tmp_path/"wav", "--output", output,
                 "--scenarios", "short-triple", "--languages", "zh", "it", "en",
                 "--pair-target", 0, "--triple-target", 2, "--short-max-ms", 500,
                 "--no-gap-prob", 1)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = jsonl(output)
    retained = {Path(r["audios"][0]).resolve() for r in rows if "synthesis" not in r}
    synthesized = [r for r in rows if "synthesis" in r]
    assert len(synthesized) == 2
    for row in synthesized:
        clips = row["synthesis"]["clips"]
        assert [c["source_usage"] for c in clips] == ["full", "partial", "full"]
        for i in (0, 2):
            full = clips[i]
            assert Path(full["audio"]).resolve() not in retained
            start, end = full["source_start_frame"], full["source_end_frame"]
            assert (start, end) == (0, sf.info(full["audio"]).frames)
        insert = clips[1]
        assert Path(insert["audio"]).resolve() in retained
        assert 0 < insert["source_end_frame"] - insert["source_start_frame"] < insert["source_frames"]
    assert {p.name for p in output.parent.iterdir()} == {output.name}
