"""Regression tests for switch attribution and timing evidence in actual SFT protocols."""
import pytest

from omnicommit.runtime_protocol import (
    END_OF_AUDIO, LIVE_AUDIO_PREFIX, last_committed_text, speaker_reference_text, system_prompt,
)
from omnicommit.switch_audit import enrich_with_alignment, find_switches, parse_sft_row
from omnicommit_audio import encode_chunk_ref
from Second.second_asr.protocol import system_prompt as second_prompt

AUDIO = "/meeting/test.wav"
D = "<|DIRECT|>"
W = "<|WAIT|>"


def sft(*answers, start=0., eos=W, system=None, end=None):
    messages = [{"role": "system", "content": system or system_prompt()}]
    audios = []
    for i, answer in enumerate(answers):
        begin = start + .5 * i
        finish = end if end is not None and i == len(answers) - 1 else begin + .5
        audios.append(encode_chunk_ref(AUDIO, begin, finish, .5))
        messages.extend([{"role": "user", "content": "<audio>"},
                         {"role": "assistant", "content": answer}])
    if eos is not None:
        messages.extend([{"role": "user", "content": END_OF_AUDIO},
                         {"role": "assistant", "content": eos}])
    return {"audios": audios, "messages": messages}


def source(text, *times):
    return {"audios": [AUDIO], "messages": [{"role": "assistant", "content": text}],
            "timestamps": [{"time_ms": t} for t in times]}


def test_sparse_turns_inherit_state_and_several_switches_share_commit():
    row = sft("<ja><spk1>こんにちは", W, "。<de><spk1>Hallo<de><spk2> Welt",
              end=1.3, system=second_prompt())
    parsed = parse_sft_row(row)
    assert parsed["protocol"] == "second" and parsed["end_seconds"] == 1.3
    assert parsed["turns"][-1]["event"] == "end_of_audio"
    assert parsed["turns"][-1]["end_seconds"] == 1.3
    assert parsed["turns"][2]["segments"][0]["language"] == "ja"
    ref = source("<ja><spk1>こんに<ja><spk1>ちは。<de><spk1>Hallo<de><spk2> Welt", 721, 1034)
    a, b = find_switches(parsed, ref)
    assert [(c["kind"], c["gold_turn"], c["gold_segment_index"], c["gold_text"]) for c in (a, b)] == [
        ("language", 3, 1, "Hallo"), ("speaker", 3, 2, " Welt")]
    assert a["onset_seconds"] == .721 and a["next_onset_seconds"] == 1.034
    assert a["first_commit_characters"] == 5 and a["first_unit_end_seconds"] is None
    assert all(c["gold_transition_matches"] for c in (a, b))
    assert find_switches(parsed)[0]["onset_seconds"] is None


def test_wrong_or_delayed_gold_tag_and_spurious_transition_retained_separately():
    row = sft(D + "<ja><spk1>日", D + "<ja><spk1>Hello", D + "<de><spk2> gut")
    ref = source("<ja><spk1>日<de><spk2>Hello gut", 700)
    expected, extra = find_switches(parse_sft_row(row), ref)
    assert expected["kind"] == "both" and expected["gold_language"] == "ja"
    assert expected["first_commit_characters"] == 0 and expected["gold_text"] == "Hello"
    assert not expected["gold_attribution_matches"] and not expected["gold_transition_matches"]
    assert expected["gold_expected_tag_turn"] == 3 and expected["gold_expected_tag_seconds"] == 1.5
    assert extra["anchor_source"] == "sft_change_without_reference_switch"
    assert extra["gold_attribution_matches"] and not extra["gold_transition_matches"]


def test_multiple_same_state_spans_use_whole_new_attribution_including_eos():
    ref = source("<ja><spk1>X<de><spk2>guten tag<ja><spk2>X", 510, 815)
    row = sft(D + "<ja><spk1>X", W, eos=D + "<de><spk2>guten<de><spk2> tag<ja><spk2>X")
    first, following = find_switches(parse_sft_row(row), ref)
    assert first["first_commit_characters"] == 9 and first["gold_text"] == "guten tag"
    assert following["first_commit_characters"] == 1 and following["gold_turn"] == 3


def test_semantic_unit_and_inner_span_evidence():
    reference = source("<ja><spk1>甲<de><spk1>Ja<en><spk2> OK")
    reference["output"] = {"segments": [{"source_text": "<ja><spk1>甲<de><spk1>Ja"}, {"source_text": " OK"}]}
    reference["timestamps"] = [{"index": 2, "start_ms": 900, "end_ms": 1130},
                               {"index": 1, "start_ms": 0, "end_ms": 880}]
    parsed = parse_sft_row(sft(D + "<ja><spk1>甲", W, D + "<de><spk1>Ja<en><spk2> OK"))
    a, b = find_switches(parsed, reference)
    assert a["onset_seconds"] is None and a["anchor_source"] == "untimed_internal_change"
    assert b["onset_seconds"] == .9 and b["first_unit_end_seconds"] == 1.13
    reference["timestamps"][1]["spans"] = [
        {"language": "ja", "speaker": "spk1", "text": "甲", "start_ms": 0, "end_ms": 180},
        {"language": "de", "speaker": "spk1", "text": "Ja", "start_ms": 510, "end_ms": 880}]
    a, b = find_switches(parsed, reference)
    assert a["onset_seconds"] == .51 and a["first_unit_end_seconds"] == .88
    assert a["anchor_source"] == "aligned_attribution_span"


def test_native_aligned_asr_preserves_first_unit_end_and_wrong_sft_tags():
    parsed = parse_sft_row(sft(D + "<ja><spk1>甲", D + "<zh><spk1>A"))
    ref = {"kind": "aligned_asr", "schema_version": 2, "status": "ok", "audio": AUDIO,
           "transcript": "<ja><spk1>甲<en><spk1>A", "duration_ms": 1000,
           "segments": [{"start_ms": begin, "end_ms": end,
                         "attributed": [{"language": lang, "speaker": "spk1", "text": text}]}
                        for begin, end, lang, text in ((0, 180, "ja", "甲"), (520, 750, "en", "A"))]}
    case, = find_switches(parsed, ref)
    assert case["onset_seconds"] == .52 and case["first_unit_end_seconds"] == .75
    assert case["gold_language"] == "zh" and not case["gold_attribution_matches"]


def test_cropped_unique_text_join_uses_live_audio_clock_skipping_speaker_prompt():
    row = sft(D + "<de><spk2>Hallo", start=2.5, system=system_prompt()+last_committed_text("ja", "spk1"))
    row["audios"].insert(0, encode_chunk_ref(AUDIO, 0, .5, .5))
    row["messages"][1]["content"] = speaker_reference_text("spk1", "ja", "甲") + LIVE_AUDIO_PREFIX + "<audio>"
    ref = source("<ja><spk1>甲<de><spk2>Hallo", 2600)
    parsed = parse_sft_row(row)
    case, = find_switches(parsed, ref)
    assert parsed["start_seconds"] == 2.5 and parsed["end_seconds"] == 3
    assert case["char_offset"] == 0 and case["reference_char_offset"] == 1
    assert case["onset_seconds"] == 2.6 and case["gold_transition_matches"] and case["onset_in_window"]
    with pytest.raises(ValueError, match="unique exact"):
        find_switches(parsed, source("<ja><spk1>Hallo<de><spk2>Hallo", 2600))
    with pytest.raises(ValueError, match="different audio"):
        find_switches(parsed, dict(ref, audios=["other.wav"]))


def test_validation_does_not_assume_a_timestamp_or_unobserved_attribution():
    parsed = parse_sft_row(sft(D+"<ja><spk1>甲", D+"<de><spk1>A"))
    with pytest.raises(ValueError, match="Every real"):
        find_switches(parsed, source("<ja><spk1>甲<de><spk1>A"))
    with pytest.raises(ValueError, match="text differs"):
        find_switches(parsed, source("<ja><spk1>甲<de><spk1>B", 600))
    with pytest.raises(ValueError, match="requires language"):
        parse_sft_row(sft("inherited?", system=second_prompt()))
    with pytest.raises(ValueError, match="Ambiguous"):
        parse_sft_row(sft(W, system="Test."))
    with pytest.raises(ValueError, match="first-protocol"):
        parse_sft_row(sft("<en><spk1>reply missing DIRECT"))
    with pytest.raises(ValueError, match="All-WAIT window"):
        find_switches(parse_sft_row(sft(W, start=.5, eos=None)), source("<ja><spk1>日<de><spk2>ja", 600))


def test_alignment_enriches_first_evidence_only_for_exact_reference_change():
    parsed = parse_sft_row(sft(D+"<ja><spk1>甲", D+"<de><spk1>JA!", D+"<en><spk2>Hi"))
    source_text = "<ja><spk1>甲<de><spk1>JA!<en><spk2>Hi"
    cases = find_switches(parsed, source(source_text, 515, 1050))
    units = ["<ja><spk1>甲", "<de><spk1>JA", "!", "<en><spk2>Hi"]
    timed = source(source_text)
    timed["output"] = {"segments": [{"source_text": t} for t in units]}
    timed["timestamps"] = [{"start_ms": s, "end_ms": f} for s, f in
                           [(100, 300), (520, 650), (700, 930), (1050, 1320)]]
    enriched, warnings = enrich_with_alignment(cases, parsed, timed)
    assert len(warnings) == 1 and "0.515s" in warnings[0]
    assert enriched[0]["onset_seconds"] == .515 and enriched[0]["trajectory_onset_seconds"] == .52
    assert enriched[0]["first_unit_end_seconds"] == .65 and enriched[0]["evidence_text"] == "JA"
    assert enriched[0]["evidence_source"] == "trajectory_aligned_semantic_unit"
    assert cases[0]["first_unit_end_seconds"] is None
    timed["messages"][0]["content"] = source_text.replace("<de>", "<zh>")
    timed["output"]["segments"][1]["source_text"] = "<zh><spk1>JA"
    unchanged, warnings = enrich_with_alignment(cases, parsed, timed)
    assert all(e["first_unit_end_seconds"] is None for e in unchanged)
    assert any("unmatched" in w for w in warnings)


def test_secondary_onset_fills_untimed_change_inside_a_mixed_unit_only():
    parsed = parse_sft_row(sft(D+"<ja><spk1>甲", D+"<de><spk2>Ja"))
    ref = source("<ja><spk1>甲<de><spk2>Ja")
    ref["output"] = {"segments": [{"source_text": "<ja><spk1>甲<de><spk2>Ja"}]}
    ref["timestamps"] = [{"start_ms": 100, "end_ms": 900}]
    untimed = find_switches(parsed, ref)
    ref["timestamps"][0]["spans"] = [
        dict(language="ja", speaker="spk1", text="甲", start_ms=100, end_ms=400),
        dict(language="de", speaker="spk2", text="Ja", start_ms=520, end_ms=870)]
    enriched, warnings = enrich_with_alignment(untimed, parsed, ref)
    assert not warnings
    assert enriched[0]["onset_seconds"] == .52 and enriched[0]["onset_in_window"]
    assert enriched[0]["anchor_source"] == "trajectory_aligned_attribution_span"
