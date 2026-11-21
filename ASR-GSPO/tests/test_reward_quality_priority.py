"""A transcription error must not buy a better reward by hiding punctuation."""
import pytest

from asr_gspo.reward import score_trajectory
from .test_asr_rewards import task, turns


@pytest.mark.parametrize("damaged", ["我知倒。你有问题", "我知。你有问题"])
def test_corrupting_sentence_boundary_anchor_cannot_improve_reward(damaged):
    t = task([("zh", "我知道你有问题")])
    exact = score_trajectory(t, turns(t, {2: t["reference_text"]}))
    fragmented = score_trajectory(t, turns(t, {2: "我知道。你有问题"}))
    hidden = score_trajectory(t, turns(t, {2: damaged}))
    assert fragmented["cer"] == 0 and fragmented["punctuation_penalty"] > 0
    # The ambiguous alignment intentionally withholds the punctuation judgment.
    assert hidden["cer"] > 0 and hidden["punctuation_penalty"] == 0
    assert exact["reward"] > fragmented["reward"] > hidden["reward"]


def test_punctuation_and_waiting_together_cannot_outweigh_one_character():
    text = "背景" * 100 + "我知道你有问题"
    t = task([("zh", text)], seconds=40, anchors=[("我知道你有问题", 100, 500)])
    fragmented = text[:-4] + "。" + text[-4:]
    late = score_trajectory(t, turns(t, {40: fragmented}))
    damaged = text[:-5] + "倒。" + text[-4:]
    fast_error = score_trajectory(t, turns(t, {1: damaged}))
    assert late["punctuation_penalty"] > 0 and late["latency_penalty"] > 0
    assert fast_error["character_errors"] == 1
    assert late["punctuation_penalty"] + late["latency_penalty"] < 1 / late["reference_characters"]
    assert late["reward"] > fast_error["reward"]


def test_reference_supported_short_sentences_are_preserved():
    t = task([("zh", "好。可以。你说。")])
    score = score_trajectory(t, turns(t, {1: "好。", 2: "可以。", 3: "你说。"}))
    assert score["cer"] == 0 and score["punctuation_penalty"] == 0
    assert score["reward"] == 1


@pytest.mark.parametrize("inserted", ["<|WAIT|>", "<zh>"])
def test_malformed_output_cannot_hide_an_unsupported_sentence_stop(inserted):
    t = task([("zh", "我知道你有问题")])
    malformed = score_trajectory(t, turns(t, {2: "我知道。" + inserted + "你有问题"}))
    fragmented = score_trajectory(t, turns(t, {2: "我知道。你有问题"}))
    assert not malformed["valid"] and malformed["reward"] < fragmented["reward"]
