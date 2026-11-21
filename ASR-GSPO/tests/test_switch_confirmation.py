"""Behavioral contracts: confirm a switch without rewarding indefinite silence."""
import pytest

from asr_gspo.protocol import SPK_CHANGE
from asr_gspo.reward import score_trajectory
from .test_asr_rewards import task, turns


def test_new_speaker_can_confirm_one_round_without_losing_quality():
    t = task([("zh", "这样吧。我知道这个问题。")], seconds=5,
             anchors=[("这样吧", 0, 700), ("我", 850, 1000), ("知道", 1050, 1700)])
    t.update(speaker_scorable=True, speaker_changes=[4])
    fast = score_trajectory(t, turns(t, {1: "这样吧。" + SPK_CHANGE + "我", 2: "知道这个问题。"}))
    confirmed = score_trajectory(t, turns(t, {1: "这样吧。", 2: SPK_CHANGE + "我知道这个问题。"}))
    stalled = score_trajectory(t, turns(t, {1: "这样吧。", 5: SPK_CHANGE + "我知道这个问题。"}))
    assert fast["reward"] == confirmed["reward"] == 1.
    assert confirmed["switch_grace_used_units"] == 1
    assert confirmed["switch_extra_wait_ms_histogram"] == {1000: 1, 0: 1}
    assert stalled["reward"] < confirmed["reward"]
    assert stalled["timing_cost"] > 0
    assert score_trajectory(t, turns(t, {1: "这样吧。", 2: SPK_CHANGE + "我知道这个问题。"}),
                            switch_grace_turns=0)["reward"] < confirmed["reward"]


def test_language_confirmation_neither_forces_speaker_marker_nor_period():
    t = task([("zh", "请看 "), ("en", "updated version"), ("zh", "，明天发。")],
             anchors=[("updated", 800, 1000), ("version", 1100, 1600)])
    t.update(speaker_scorable=True, speaker_changes=[])
    r = score_trajectory(t, turns(t, {1: "请看 ", 2: "updated version", 3: "，明天发。"}))
    assert r["reward"] == 1 and r["speaker_error"] == 0
    assert r["extra_internal_periods"] == 0
    wrong = score_trajectory(t, turns(t, {1: "请看 更新版本，明天发。"}))
    assert wrong["reward"] < r["reward"]


def test_one_shared_deadline_not_an_extra_round_for_each_word():
    t = task([("zh", "好。"), ("en", "one two three four")], seconds=6,
             anchors=[("one", 800, 1000), ("two", 1200, 2000), ("three", 2200, 3000),
                      ("four", 3200, 4000)])
    r = score_trajectory(t, turns(t, {1: "好。", 2: "one ", 3: "two ", 4: "three ", 5: "four"}))
    assert r["switch_grace_used_units"] == 1
    assert r["excess_wait_ms_histogram"] == {0: 1, 1000: 3}
    assert r["reward"] < 1


def test_grace_does_not_relax_normal_speech_or_erase_raw_delays():
    t = task([("en", "a normal sentence")], anchors=[("normal", 100, 800)])
    r = score_trajectory(t, turns(t, {2: t["reference_text"]}))
    assert r["switch_grace_used_units"] == 0 and r["timing_cost"] > 0
    assert r["extra_wait_ms_histogram"] == r["excess_wait_ms_histogram"] == {1000: 1}


def test_no_end_timestamps_means_no_fabricated_grace_or_latency():
    t = task([("en", "Yes. "), ("fr", "Bonjour.")])
    t["switches"] = [{"char_start": 5, "time_ms": 900}]
    r = score_trajectory(t, turns(t, {5: t["reference_text"]}))
    assert r["timed_units"] == 0 and r["latency_penalty"] == 0
    assert r["delay_p95_seconds"] is None and r["switch_grace_used_units"] == 0


def test_whole_short_turn_shares_confirmation_deadline():
    t = task([("zh", "好。我知道这事")], anchors=[(ch, 600 + n * 50, 650 + n * 50)
                                               for n, ch in enumerate("我知道这事")])
    t.update(speaker_scorable=True, speaker_changes=[2])
    r = score_trajectory(t, turns(t, {1: "好。", 2: SPK_CHANGE + "我知道这事"}))
    assert r["reward"] == 1 and r["switch_grace_used_units"] == 5


def test_coarse_first_unit_has_one_real_completion_not_interpolated_word_times():
    t = task([("en", "OK. "), ("zh", "我知道这个问题。")],
             anchors=[("我知道这个问题", 1100, 1800)])
    t.update(speaker_scorable=True, speaker_changes=[4])
    r = score_trajectory(t, turns(t, {1: "OK. ", 3: SPK_CHANGE + "我知道这个问题。"}))
    assert r["reward"] == 1 and r["timed_units"] == 1
    assert r["switch_coarse_completion_boundaries"] == 1
    assert r["extra_wait_ms_histogram"] == {1000: 1}
    # A sparse late word's end cannot masquerade as the missing first-unit end.
    t["alignment"][0].update(text="问题", char_start=9, char_end=11)
    r = score_trajectory(t, turns(t, {1: "OK. ", 3: SPK_CHANGE + "我知道这个问题。"}))
    assert r["switch_coarse_completion_boundaries"] == 0 and r["reward"] < 1


def test_partial_final_chunk_and_two_short_turns_do_not_disappear():
    t = task([("en", "OK. "), ("zh", "对。"), ("en", "Sure.")], seconds=3,
             anchors=[("OK", 0, 500), ("对", 800, 1100), ("Sure", 1200, 2300)])
    t.update(duration_seconds=2.3, end_policy="annotation_end", speaker_scorable=True,
             speaker_changes=[4, 6], chunks=[{"end_seconds": x} for x in (1., 2., 2.3)])
    tr = turns(t, {1: "OK. ", 3: SPK_CHANGE + "对。" + SPK_CHANGE + "Sure."})[:-1]
    r = score_trajectory(t, tr)
    assert r["reward"] == 1 and r["speaker_correct_changes"] == 2
    assert r["switch_grace_used_units"] == 1


def test_correct_spelling_on_wrong_side_is_not_fast_correct_commit():
    t = task([("zh", "好。我知道这个问题")], anchors=[("我", 700, 1000), ("知", 1000, 1400),
                                                   ("道", 1400, 1800)])
    t.update(speaker_scorable=True, speaker_changes=[2])
    one = score_trajectory(t, turns(t, {2: "好。我" + SPK_CHANGE + "知道这个问题"}))
    three = score_trajectory(t, turns(t, {2: "好。我知道" + SPK_CHANGE + "这个问题"}))
    correct = score_trajectory(t, turns(t, {1: "好。", 2: SPK_CHANGE + "我知道这个问题"}))
    assert correct["reward"] > one["reward"] > three["reward"]
    assert correct["cer"] == one["cer"] == three["cer"] == 0
    assert one["timing_wrong_side_units"] == 1 and three["timing_wrong_side_units"] == 3
    assert one["attributed_timed_matches"] == 2 and three["attributed_timed_matches"] == 0
    assert three["switch_extra_wait_ms_histogram"] == {}


def test_cli_configuration_and_boundary_cohorts_are_explicit():
    import train_gspo
    import evaluate
    args = train_gspo.parser().parse_args(["model", "tasks", "output", "--switch-grace-turns", "0"])
    env = train_gspo.build_environment(args)
    assert env["ASR_GSPO_SWITCH_GRACE_TURNS"] == "0"
    assert "--switch-grace-turns" not in train_gspo.build_command(args)  # plugin, not an upstream flag
    t = task([("zh", "甲乙")])
    t.update(bucket="general", speaker_scorable=True, speaker_changes=[1])
    assert evaluate.evaluation_groups(t) == ("all", "general", "boundary/speaker_only")
    t["spans"] = [{"lang": "zh", "text": "甲"}, {"lang": "en", "text": "乙"}]
    assert "boundary/both" in evaluate.evaluation_groups(t)


@pytest.mark.parametrize("value", [-1, 5, 1.5, True])
def test_invalid_allowance_rejected(value):
    t = task([])
    with pytest.raises(ValueError, match="switch_grace_turns"):
        score_trajectory(t, turns(t, {}), switch_grace_turns=value)
