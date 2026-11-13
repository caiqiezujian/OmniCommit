"""Exercise the switch audit through its public CLI, including its listening artifacts."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from omnicommit.runtime_protocol import END_OF_AUDIO, system_prompt
from omnicommit.streaming import DIRECT, WAIT
from Second.second_asr.protocol import system_prompt as second_prompt

CLI = Path(__file__).resolve().parents[1] / "SFT-Scripts" / "audit_switches.py"
sys.path.insert(0, str(CLI.parent))
from omnicommit_audio import encode_chunk_ref


def jsonl(path, *rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return path


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def run(sft, output, *args):
    proc = subprocess.run([sys.executable, str(CLI), str(sft), "--output-dir", str(output), *map(str, args)],
                          cwd=sft.parent, env=dict(os.environ, PYTHONUTF8="1"), capture_output=True,
                          text=True, encoding="utf-8", timeout=30)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    return json.loads((output / "summary.json").read_text(encoding="utf-8"))


def training_row(audio, protocol):
    first = protocol == "first"
    tag = DIRECT if first else ""
    ja_de = ["<ja><spk1>次です。", None, "<de><spk2>Hallo", "<de><spk2> Welt." if first else " Welt."]
    messages = [{"role": "system", "content": system_prompt() if first else second_prompt()}]
    refs = []
    for i, reply in enumerate(ja_de):
        refs.append(encode_chunk_ref(audio, i / 2, (i + 1) / 2, .5))
        messages.extend([{"role": "user", "content": "<audio>"},
                         {"role": "assistant", "content": WAIT if reply is None else tag + reply}])
    messages.extend([{"role": "user", "content": END_OF_AUDIO}, {"role": "assistant", "content": WAIT}])
    return {"audios": refs, "messages": messages}


def observed(turn, start, end, *, lang=None, text="", protocol="first", eos=False):
    spans = [{"language": lang, "speaker": "spk3", "text": text}] if lang else []
    prefix = DIRECT if protocol == "first" else ""
    raw = prefix + f"<{lang}><spk3>{text}" if lang else WAIT
    return {"type": "result", "event": "end_of_audio" if eos else "audio", "turn": turn,
            "start_seconds": start, "end_seconds": end, "action": "DIRECT" if lang else "WAIT",
            "raw_model_text": raw, "segments": spans, "generation_seconds": .07}


@pytest.mark.parametrize("protocol", ["first", "second"])
def test_raw_timestamps_clipped_model_trace_export_and_invalid_row(tmp_path, protocol):
    source = tmp_path / "ja-de.wav"
    fs = 16000
    # Stereo PCM with a different deterministic signal in each channel and each frame.
    frame = np.arange(2 * fs, dtype=np.int32)
    pcm = np.column_stack((frame - fs, (frame * 91) % 32768 - fs)).astype(np.int16)
    sf.write(str(source), pcm, fs, subtype="PCM_16")
    sft = tmp_path / "train.jsonl"
    sft.write_text("{bad row\n" + json.dumps(training_row(source, protocol)) + "\n", encoding="utf-8")
    original = "<ja><spk1>次<ja><spk1>です。<de><spk2>Hallo Welt."
    reference = jsonl(tmp_path / "source.switches.jsonl", {
        "audios": [str(source)], "messages": [
            {"role": "system", "content": "Transcribe."}, {"role": "user", "content": "<audio>"},
            {"role": "assistant", "content_original": original, "content": original.replace("<de>", "[700ms]<de>")}],
        "timestamps": [{"time_ms": 700, "message_index": 2, "changed": ["language", "speaker"],
                        "language": "de", "previous_language": "ja", "speaker": "spk2", "previous_speaker": "spk1",
                        "text": "Hallo Welt."}]})
    # The model was started 500 ms into this source. Compare by its original audio clock.
    ready = {"type": "ready", "protocol": protocol, "input": str(tmp_path / "cold-cut.wav")}
    trace = jsonl(tmp_path / "cold.jsonl", ready, observed(1, 0, .5),
                  observed(2, .5, 1, lang="ja", text="Hallo", protocol=protocol),
                  observed(3, 1, 1.5, lang="de", text=" Welt.", protocol=protocol),
                  observed(4, 1.5, 1.5, eos=True))
    out = tmp_path / "out"
    summary = run(sft, out, "--source-jsonl", reference, "--trace", trace, source, ".5",
                  "--before-seconds", ".1", "--after-seconds", ".5", "--write-clips")
    assert summary["counts"]["rows_read"] == 2 and summary["counts"]["rows_parsed"] == 1
    assert summary["by_kind"] == {"both": 1} and summary["by_language_pair"] == {"ja->de": 1}
    near = summary["gold_audio_turn_profiles"]["at_or_after_switch"]
    assert near["audio_turns"] == 2 and near["wait_fraction"] == .5
    assert near["characters_per_direct"] == 5
    assert summary["issues"] == {"invalid_sft": 1}
    assert read_rows(out / "issues.jsonl")[0]["sft_line"] == 1
    case, = read_rows(out / "cases.jsonl")
    switch, = read_rows(out / "switches.jsonl")
    assert case["case_id"] == switch["case_id"] and case["protocol"] == protocol and case["sft_line"] == 2
    assert "turns" not in switch
    assert case["gold_attribution_matches"] and case["reference_origin"] == "source"
    assert (case["anchor_seconds"], case["gold_commit_seconds"], case["gold_delay_seconds"]) == (.7, 1.5, .8)
    obs = case["models"][str(trace)]
    assert obs["terminal"]["status"] == "eos"
    assert obs["direct_turns_by_language"] == {"ja": 1, "de": 1}
    assert obs["observed_wait_turns"] == 1 and obs["longest_observed_wait_seconds"] == pytest.approx(.3)
    assert obs["first_target_tag_delay_seconds"] == pytest.approx(1.3)
    assert (obs["first_target_tag_gold_turn"], obs["first_target_tag_turn"]) == (4, 3)
    wait, wrong, right, eos = case["turns"]
    assert wait["gold"] == wait["models"][str(trace)]["raw_model_text"] == WAIT
    assert wrong["gold_segments"][0]["language"] == "de" and wrong["models"][str(trace)]["segments"][0]["language"] == "ja"
    assert right["gold_segments"][0]["text"] == " Welt." and eos["event"] == "end_of_audio"
    review = (out / "review.md").read_text(encoding="utf-8")
    assert WAIT in review and "<ja><spk3>Hallo" in review and case["clip"] in review
    excerpt, clip_fs = sf.read(str(out / case["clip"]), dtype="float32", always_2d=True)
    assert clip_fs == fs
    np.testing.assert_array_equal(excerpt, pcm[9600:] / 32768.0)  # .6 to 2.0 s, without source resampling or padding


@pytest.mark.parametrize("sample_size", [0, 1])
def test_sft_only_sampling_and_same_basename_unjoined_trace(tmp_path, sample_size):
    sft = jsonl(tmp_path / "train.jsonl", training_row(tmp_path / "a" / "meeting.wav", "second"))
    other = jsonl(tmp_path / "other.jsonl", {"type": "ready", "input": str(tmp_path / "b" / "meeting.wav")})
    out = tmp_path / "out"
    summary = run(sft, out, "--sample-size", sample_size, "--trace", other)
    assert summary["row_reference_origins"] == {"sft_only": 1}
    assert summary["traces"][str(other)]["joined_sft_rows"] == 0
    assert summary["issues"] == {"trace_source_not_in_selected_sft": 1}
    assert summary["sampled"] == sample_size and summary["counts"]["switches_selected"] == 1
    commit = summary["sft_commit_profiles"]
    assert commit["switch_commit"]["mean_preceding_wait_turns"] == 1
    assert commit["continuing_attribution"]["mean_preceding_wait_turns"] == 0
    assert not summary["gold_audio_turn_profiles"]
    switch, = read_rows(out / "switches.jsonl")
    assert switch["event_basis"] == "sft" and switch["reference_origin"] == "sft_only" and switch["kind"] == "both"
    assert switch["gold_turn"] == 3 and switch["gold_commit_seconds"] == 1.5
    assert switch["anchor_seconds"] is None and switch["gold_delay_seconds"] is None and not switch["models"]
    assert len(read_rows(out / "cases.jsonl")) == sample_size
