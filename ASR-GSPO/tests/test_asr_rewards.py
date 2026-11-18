import pytest

from asr_gspo.reward import score_trajectory


WAIT = "<|WAIT|>"
DIRECT = "<|DIRECT|>"


def task(spans, seconds=5, anchors=()):
    text = "".join(s for _, s in spans)
    alignment = []
    for word, ms_start, ms_end in anchors:
        i = text.index(word)
        alignment.append(dict(text=word, char_start=i, char_end=i + len(word),
                              start_ms=ms_start, end_ms=ms_end))
    return dict(reference_text=text, spans=[{"lang": lang, "text": s} for lang, s in spans],
                duration_seconds=seconds, chunks=[{"end_seconds": t} for t in range(1, seconds + 1)],
                alignment=alignment)


def turns(t, emissions):
    history = []
    for n, c in enumerate(t["chunks"], 1):
        history.append(dict(kind="audio", response=DIRECT + emissions[n] if n in emissions else WAIT,
                            audio_end_seconds=c["end_seconds"], finish_reason="stop"))
    return history + [dict(kind="eof", response=WAIT, audio_end_seconds=t["duration_seconds"], finish_reason="stop")]


def test_fast_late_deleted_and_silent_cases():
    t = task([("en", "Hello, "), ("de", "Alles klar. "), ("fr", "Bonjour.")],
             anchors=[("Hello", 100, 800), ("Alles", 1100, 1700), ("klar", 1800, 2450),
                      ("Bonjour", 4200, 4800)])
    fast = turns(t, {1: "Hello, ", 2: "Alles ", 3: "klar. ", 5: "Bonjour."})
    late = turns(t, {1: "Hello, ", 5: "Alles klar. Bonjour."})
    lost = turns(t, {1: "Hello, ", 5: "Bonjour."})
    a, b, c = [score_trajectory(t, tr) for tr in (fast, late, lost)]
    assert a["reward"] > b["reward"] > c["reward"]
    assert a["delay_p95_seconds"] == 0 and b["delay_p95_seconds"] == 3
    assert c["short_switch_deletions"] == 1 and c["deletions"] == 2
    assert c["timing_cost"] > b["timing_cost"]
    assert score_trajectory(t, fast, "quality")["reward"] == score_trajectory(t, late, "quality")["reward"]
    silent = task([], seconds=6)
    assert score_trajectory(silent, turns(silent, {}))["reward"] == 1
    assert score_trajectory(silent, turns(silent, {2: "."}))["reward"] < 0
    assert score_trajectory(t, turns(t, {}))["reward"] == -2.5
    assert score_trajectory(t, turns(t, {2: "unrelated hallucination"}))["reward"] < -2.5


def test_protocol_and_provenance():
    t = task([("en", "OK")], seconds=1)
    tr = turns(t, {1: "OK"})
    for reply in (DIRECT, "OK", WAIT + " hi", DIRECT + "<en>OK"):
        assert score_trajectory(t, [dict(tr[0], response=reply), tr[1]])["reward"] == -3
    assert score_trajectory(t, [tr[0], dict(tr[1], response=DIRECT + " end")])["valid"] is False
    assert score_trajectory(t, tr[:1])["valid"] is False
    assert score_trajectory(t, [dict(tr[0], finish_reason="length"), tr[1]])["valid"] is False
    with pytest.raises(ValueError, match="schedule"):
        score_trajectory(t, [dict(tr[0], audio_end_seconds=2), tr[1]])


def test_early_guess_and_coarse_endpoints():
    t = task([("en", "the new budget")], anchors=[("the new budget", 2100, 3650)])
    early = score_trajectory(t, turns(t, {1: "the new budget"}))
    valid = score_trajectory(t, turns(t, {3: "the new ", 4: "budget"}))
    assert early["timed_units"] == 1  # no synthetic times inside phrase
    assert early["early_units"] == 1 and valid["early_units"] == 0
    assert early["reward"] < valid["reward"]
    t["alignment"] = []
    assert score_trajectory(t, turns(t, {5: "the new budget"}))["delay_p95_seconds"] is None


def test_tokenization_spans_and_subword_commit():
    t = task([("en", "Hello "), ("en", "world "), ("zh", "预算３００元")],
             anchors=[("Hello", 80, 880)])
    result = score_trajectory(t, turns(t, {1: "He", 2: "LLO, world ", 5: "预算300元。"}))
    assert result["errors"] == 0 and result["delay_p95_seconds"] == 1
    assert result["reference_units"] == 6
    assert result["switch_segments"] == 1 and result["short_switch_segments"] == 0
    t["spans"][0]["text"] = "bad"
    with pytest.raises(ValueError, match="reference spans"):
        score_trajectory(t, turns(t, {}))


def test_switch_error_counts_head_and_boundary_insertions():
    t = task([("zh", "开始。"), ("de", "Guten Morgen aus Berlin.")])
    good = score_trajectory(t, turns(t, {2: "开始 Guten Morgen aus Berlin."}))
    tail = score_trajectory(t, turns(t, {2: "开始 Guten Morgen aus Bonn."}))
    head = score_trajectory(t, turns(t, {2: "开始 Hello Morgen aus Berlin."}))
    insert = score_trajectory(t, turns(t, {2: "开始 Hello Guten Morgen aus Berlin."}))
    assert good["switch_error_rate"] == tail["switch_error_rate"] == 0
    assert head["reward"] < tail["reward"] < good["reward"]
    assert insert["insertions"] == 1 and insert["switch_error_rate"] > 0


def test_partial_ctc_endpoints_and_switch_onset_only():
    t = task([("zh", "开始。"), ("en", "Good morning.")],
             anchors=[("Good", 2200, 2900), ("morning", 3000, 3900)])
    t["switches"] = [{"char_start": 3, "time_ms": 2100, "lang": "en", "speaker": "spk1"}]
    t["alignment"][0]["start_ms"] = None
    t["alignment"][1]["end_ms"] = None
    early = score_trajectory(t, turns(t, {1: "开始。Good morning."}))
    timely = score_trajectory(t, turns(t, {1: "开始。", 3: "Good ", 4: "morning."}))
    assert early["early_units"] == 2 and early["delay_samples"] == 0
    assert timely["early_units"] == 0 and timely["delay_samples"] == 1
    assert early["reward"] < timely["reward"]
    t["alignment"] = []
    early = score_trajectory(t, turns(t, {1: "开始。Good morning."}))
    assert early["early_units"] == 2 and early["timed_units"] == 0
    assert early["mean_extra_wait_seconds"] is None


def test_unseparated_latin_language_switch_preserves_new_language_head():
    t = task([("en", "Hello"), ("de", "Guten Morgen")])
    accurate = score_trajectory(t, turns(t, {2: "HELLO ", 4: "Guten Morgen"}))
    substituted = score_trajectory(t, turns(t, {2: "Hello", 4: "XXXen Morgen"}))
    dropped = score_trajectory(t, turns(t, {2: "Hello", 4: " Morgen"}))
    assert accurate["unit_mode"] == "character_cross_boundary"
    assert accurate["reference_units"] == 16
    assert accurate["errors"] == accurate["switch_error_rate"] == 0
    assert substituted["substitutions"] == 3 and substituted["switch_error_rate"] == 1
    assert dropped["deletions"] == 5 and dropped["switch_error_rate"] == 1


def test_repeated_reference_words_and_ambiguous_edit_paths():
    t = task([("en", "OK "), ("de", "OK")])
    t["alignment"] = [
        dict(text="OK", char_start=0, char_end=2, start_ms=100, end_ms=950),
        dict(text="OK", char_start=3, char_end=5, start_ms=2400, end_ms=2950),
    ]
    early = score_trajectory(t, turns(t, {1: "OK"}))
    later = score_trajectory(t, turns(t, {3: "OK"}))
    assert early["early_units"] == later["early_units"] == 0
    assert early["deletions"] == later["deletions"] == 1
    assert early["short_switch_deletions"] == 1 and later["short_switch_deletions"] == 0
    assert early["mean_extra_wait_seconds"] == later["mean_extra_wait_seconds"] == 0
    assert (score_trajectory(t, turns(t, {1: "OK"}), "quality")["reward"]
            == score_trajectory(t, turns(t, {3: "OK"}), "quality")["reward"])
    transposed = task([("en", "budget first")])
    result = score_trajectory(transposed, turns(transposed, {2: "first budget"}))
    assert result["errors"] == 2 and result["matched_units"] == 1
