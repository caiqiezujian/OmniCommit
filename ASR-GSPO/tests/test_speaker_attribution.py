"""Behavioral tests for local turn attribution; no persistent identity claim."""
import pytest

from asr_gspo.reward import _edits, _units
from asr_gspo.speaker_quality import speaker_metrics


def score(reference, changes, hypothesis=None, markers=None, *, scorable=True, characters=False):
    hypothesis = reference if hypothesis is None else hypothesis
    ref, hyp = _units(reference, characters), _units(hypothesis, characters)
    operations = _edits(ref, hyp, [1.] * len(hyp), {}, {}, 3.)
    return speaker_metrics(dict(speaker_scorable=scorable, speaker_changes=changes),
                           ref, hyp, operations, changes if markers is None else markers,
                           return_details=True)


def test_exact_boundary_and_arbitrary_audio_round_splits_have_no_attribution_cost():
    result, correct = score("好的。我知道这个问题。", [3])
    assert result["speaker_error"] == result["speaker_attribution_error"] == 0
    assert result["speaker_correct_changes"] == 1
    assert correct == set(range(9))


def test_larger_shift_loses_more_correct_characters_on_either_side():
    text = "现在我们确认方案。明天开始具体实施。"
    boundary = text.index("明")
    one, one_correct = score(text, [boundary], markers=[boundary + 1])
    three, three_correct = score(text, [boundary], markers=[boundary + 3])
    early, _ = score(text, [boundary], markers=[boundary - 3])
    assert one["speaker_error"] == three["speaker_error"] == 1
    assert one["speaker_misattributed_characters"] == 1
    assert three["speaker_misattributed_characters"] == 3
    assert 0 < one["speaker_attribution_error"] < three["speaker_attribution_error"] < 1
    assert one_correct > three_correct
    assert early["speaker_misattributed_characters"] == 2  # punctuation has no mass


def test_missing_change_does_not_corrupt_the_next_correct_change():
    text = "甲乙丙丁戊己庚辛壬癸"
    result, correct = score(text, [3, 6], markers=[6])
    assert result["speaker_unpaired_changes"] == 1
    assert result["speaker_correct_changes"] == 1
    assert result["speaker_misattributed_characters"] == 3
    assert correct == {0, 1, 2, 6, 7, 8, 9}


def test_missing_short_turn_cannot_hide_behind_long_correct_turns():
    text = "甲" * 50 + "对" + "乙" * 50
    result, correct = score(text, [50, 51], markers=[51])
    assert result["speaker_misattributed_characters"] == 1
    assert result["speaker_attribution_error"] == .75
    assert 50 not in correct and len(correct) == 100


def test_deleting_new_speaker_head_does_not_make_attribution_perfect():
    result, correct = score("好的我知道这个问题", [2], "好的这个问题", [2])
    assert result["speaker_deleted_turn_characters"] == 3
    assert result["speaker_attribution_error"] > 0
    assert result["speaker_ambiguous_changes"] == 1
    assert {0, 1, 5, 6, 7, 8} == correct


def test_all_deleted_text_has_no_latency_credit_or_attribution_success():
    result, correct = score("好的我知道", [2], "", [])
    assert result["speaker_attribution_error"] == 1
    assert result["speaker_deleted_turn_characters"] == 3
    assert not correct


@pytest.mark.parametrize("markers", ([0, 3], [3, 3], [3, 9]))
def test_empty_and_duplicate_turns_are_boundary_errors_without_global_identity_flip(markers):
    result, correct = score("甲乙丙丁戊己庚辛壬", [3], markers=markers)
    assert result["speaker_extra_changes"] == 1
    assert result["speaker_error"] > 0
    assert result["speaker_misattributed_characters"] == 0
    assert correct == set(range(9))


def test_spurious_change_inside_real_turn_is_local_and_recovery_is_preserved():
    result, correct = score("甲乙丙丁戊己庚辛壬", [6], markers=[3, 6])
    assert result["speaker_extra_changes"] == 1
    assert result["speaker_misattributed_characters"] == 3
    assert correct == {0, 1, 2, 6, 7, 8}


def test_latin_boundary_motion_uses_characters_not_number_of_words():
    text = "Okay. I understand this problem."
    one, _ = score(text, [6], markers=[8])
    longer, _ = score(text, [6], markers=[19])
    assert one["speaker_misattributed_characters"] == 1
    assert longer["speaker_misattributed_characters"] == 11
    assert longer["speaker_attribution_error"] > one["speaker_attribution_error"]


def test_marker_inside_word_is_not_repaired_to_make_it_correct():
    result, _ = score("Okay. Understand this.", [6], markers=[10])
    assert result["speaker_correct_changes"] == 0
    assert result["speaker_unpaired_changes"] == 1
    assert result["speaker_attribution_error"] == 1


def test_intra_word_spurious_marker_without_real_change_is_not_latency_success():
    result, correct = score("Okay understand this.", [], markers=[8])
    assert result["speaker_extra_changes"] == 1
    assert result["speaker_attribution_error"] > 0
    assert correct == {0}


def test_substitution_at_first_word_does_not_fabricate_a_shift():
    result, correct = score("Okay. I understand.", [6], "Okay. We understand.", [6])
    assert result["speaker_attribution_error"] == 0
    assert result["speaker_misattributed_characters"] == 0
    assert correct == {0, 2}


def test_two_shifted_boundaries_pair_monotonically_and_do_not_double_count():
    result, correct = score("甲乙丙丁戊己庚辛壬癸", [3, 6], markers=[4, 7])
    assert result["speaker_shifted_changes"] == 2
    assert result["speaker_misattributed_characters"] == 2
    assert correct == {0, 1, 2, 4, 5, 7, 8, 9}


def test_unlabeled_record_does_not_invent_a_speaker_objective():
    result, correct = score("甲乙丙丁", [], markers=[2], scorable=False)
    assert not result["speaker_scorable"]
    assert result["speaker_attribution_error"] == result["speaker_error"] == 0
    assert correct == {0, 1, 2, 3}


def test_same_speaker_language_change_needs_no_speaker_marker():
    result, correct = score("好的 Bonjour", [], markers=[])
    assert result["speaker_attribution_error"] == result["speaker_error"] == 0
    assert correct == {0, 1, 2}


def test_repeated_words_follow_existing_lexical_alignment_not_marker_count():
    result, correct = score("好好好好好好", [2, 4], markers=[4])
    assert result["speaker_misattributed_characters"] == 2
    assert correct == {0, 1, 4, 5}
