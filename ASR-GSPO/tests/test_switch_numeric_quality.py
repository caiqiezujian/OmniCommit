"""Local CER must preserve numeric meaning exactly as the global CER does."""
import pytest

from asr_gspo.character_quality import _aligned_raw_text, local_switch_cer
from asr_gspo.reward import _edits, _units, score_trajectory
from .test_asr_rewards import task, turns


@pytest.mark.parametrize("value", [
    "Price 3.50 dollars.", "Price 50%.", "Budget -50 euros.",
    "+50%", "−50%", "＋５０％", "2026-11-21", "12:30", "1/2", "1,000.50",
])
def test_exact_numeric_switch_is_not_penalized(value):
    t = task([("zh", "好。"), ("en", value)])
    result = score_trajectory(t, turns(t, {2: t["reference_text"]}))
    assert result["cer"] == 0
    assert result["switch_cer"] == 0
    assert result["language_penalty"] == 0
    assert result["reward"] == 1


@pytest.mark.parametrize("reference,candidate", [
    ("3.50", "350"), ("50%", "50"), ("-50", "50"), ("+50", "50"),
    ("12:30", "1230"), ("1/2", "12"), ("2026-11-21", "20261121"),
])
def test_numeric_corruption_is_visible_locally(reference, candidate):
    t = task([("zh", "好。"), ("en", reference)])
    good = score_trajectory(t, turns(t, {2: t["reference_text"]}))
    wrong = score_trajectory(t, turns(t, {2: "好。" + candidate}))
    assert wrong["cer"] > 0 and wrong["switch_cer"] > 0
    assert wrong["reward"] < good["reward"]


def test_raw_candidate_does_not_copy_reference_numeric_marks():
    ref, hyp = _units("-50%"), _units("+50")
    edits = _edits(ref, hyp, [1.] * len(hyp), {}, {}, 2.)
    metrics = local_switch_cer("-50%", ref, hyp, [(0, 4)], edits, hypothesis="+50")
    assert metrics["switch_cer"] == .5


def test_disjoint_aligned_runs_do_not_absorb_other_span_words():
    raw = "3.50 ignored -20%"
    hyp = _units(raw)
    assert [u.value for u in hyp] == ["3", "50", "ignored", "20"]
    assert _aligned_raw_text(raw, hyp, [0, 1, 3]) == "3.50 -20%"


def test_shared_latin_text_is_not_claimed_to_identify_language():
    # The observed word is genuinely shared: lexical scoring cannot determine
    # its spoken language. It must not invent a script/language mismatch.
    t = task([("en", "Ready. "), ("de", "OK.")])
    score = score_trajectory(t, turns(t, {2: t["reference_text"]}))
    assert score["switch_cer"] == 0
    t = task([("en", "Ready. "), ("fr", "Merci beaucoup.")])
    translated = score_trajectory(t, turns(t, {2: "Ready. Thank you."}))
    assert translated["switch_cer"] > 0
    assert translated["script_mismatch_runs"] == 0
