import pytest

from asr_gspo.reward import _changed_spans, _edits, _units
from asr_gspo.switch_quality import disjoint_script_metrics, switch_quality_metrics


def score(spans, heard):
    text = "".join(s for _, s in spans)
    changes = _changed_spans([{"lang": lang, "text": s} for lang, s in spans], text)
    ref, hyp = _units(text), _units(heard)
    ops = _edits(ref, hyp, [1] * len(hyp), {}, {}, 1)
    return switch_quality_metrics(ref, hyp, changes, ops)


def test_translation_or_omission_of_short_switch_remains_visible_in_long_context():
    before = "这是很长的会议记录，" * 10
    after = "然后继续讨论。"
    spans = [("zh", before), ("en", "budget "), ("zh", after)]
    for missing in ("预算", ""):
        result = score(spans, before + missing + after)
        assert result["switch_content_fully_missed"] == 1
        assert result["switch_content_spans"] == 2
        assert result["switch_content_error"] == pytest.approx(0.75)
    assert score(spans, before + "budget" + after)["switch_content_error"] == 0


@pytest.mark.parametrize("mistyped", ["Morgne", "Morgem", "Morgn", "Morgeen"])
def test_same_script_translation_is_worse_than_spelling_slip(mistyped):
    spans = [("en", "Next, "), ("de", "Morgen.")]
    typo = score(spans, "Next, " + mistyped)
    translated = score(spans, "Next, tomorrow")
    assert typo["switch_content_fully_missed"] == 0
    assert 0 < typo["switch_content_error"] < translated["switch_content_error"] == 1.0


def test_same_script_long_switch_rewards_preserving_head_and_tail():
    spans = [("en", "Next, "), ("de", "Guten Morgen an alle hier.")]
    near = score(spans, "Next, Guten Morgne an alle hier.")
    lost_head = score(spans, "Next, Good morning an alle hier.")
    lost_tail = score(spans, "Next, Guten Morgen an everyone here.")
    assert 0 < near["switch_content_error"] < lost_tail["switch_content_error"]
    assert lost_tail["switch_content_error"] < lost_head["switch_content_error"] < 1


def test_correct_borrowing_and_neutral_characters_do_not_signal_a_wrong_language():
    spans = [("zh", "请启动"), ("fr", "Windows １１！")]
    assert score(spans, "请启动 windows 11.")["switch_content_error"] == 0.0
    only_digits = score([("en", "Budget: "), ("fr", "300 €")], "Budget: 100.")
    assert only_digits["switch_content_spans"] == only_digits["switch_content_error"] == 0


def test_same_language_and_unknown_language_have_no_switch_penalty():
    for next_lang in ("en", "und"):
        result = score([("en", "Maybe "), (next_lang, "tomorrow.")], "Maybe.")
        assert result["switch_content_spans"] == result["switch_content_error"] == 0


def scripts(reference, hypothesis):
    ref, hyp = _units(reference), _units(hypothesis)
    ops = _edits(ref, hyp, [1] * len(hyp), {}, {}, 1)
    return disjoint_script_metrics(ref, hyp, ops)


def test_single_foreign_script_replacement_in_otherwise_correct_monolingual_clip():
    before = "我们先来看这一次讨论的" * 4
    after = "有没有问题。"
    stats = scripts(before + "预算" + after, before + "budget" + after)
    assert stats == {"script_mismatch_runs": 1, "script_mismatch_ref_units": 2,
                     "script_mismatch_hyp_units": 1}
    assert scripts("My budget is fine.", "My 预算 is fine.")["script_mismatch_runs"] == 1


def test_distinct_mismatches_and_real_cyrillic_translation():
    stats = scripts("我的预算已经确认。", "我的budget已经confirmed。")
    assert stats["script_mismatch_runs"] == 2
    assert scripts("Just спасибо now.", "Just thank you now.")["script_mismatch_runs"] == 1


@pytest.mark.parametrize("reference, hypothesis", [
    ("我们用Windows。", "我们用WINDOWS!"),   # Borrowing retained with its actual spelling
    ("我们用Windows。", "我们用Windos。"),     # Same-alphabet spelling error
    ("预算300元。", "预算three hundred元。"),  # Numerals supply no alphabet evidence
    ("价格不会变。", "价格 budget 不会变。"), # Unsupported insertion is already an ASR edit
    ("价格不会变。", "价格变。"),             # No hypothesis content in the gap
    ("今日は東京です。", "今日はとうきょうです。"),  # Shared Japanese writing system
    ("遇见かな。", "遇见カナ。"),
    ("银行在銀行。", "银行在은행。"),          # Hanja and Hangul
    ("Account РАУРАL.", "Account PAYPAL."), # Mixed-script company name / confusables
    ("КОРТ", "KOPT"),                       # Cyrillic and Latin visual lookalikes
    ("ΡΑΡΑ", "papa"),                       # Greek and Latin visual lookalikes
    ("ΝΝ", "NN"),                           # Original capital shapes are lost on case-folding
    ("A", "甲"),                            # Too little evidence
])
def test_ambiguous_scripts_shared_words_and_unpaired_errors(reference, hypothesis):
    assert scripts(reference, hypothesis)["script_mismatch_runs"] == 0
