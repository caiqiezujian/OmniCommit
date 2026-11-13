"""Exercise the real per-session tracker and reconciliation on synthetic PCM."""
from pathlib import Path
import sys

import numpy as np
import pytest

SECOND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SECOND))
from second_asr.gap_recovery import GapTracker, RATE, reconcile


def voice(seconds=.8):
    t = np.arange(round(seconds * RATE)) / RATE
    return (.13 * np.sin(2 * np.pi * 241 * t)).astype("float32")


def turn(n, text=None, lang="de"):
    return {"turn": n, "start_seconds": (n - 1) * .8, "end_seconds": n * .8,
            "action": "WAIT" if text is None else "DIRECT",
            "segments": [] if text is None else [{"text": text, "speaker": "spk2", "language": lang}]}


def request():
    tracker = GapTracker()
    assert tracker.observe(voice(), turn(1, "皆さん。", "ja")) is None
    assert tracker.observe(voice(), turn(2)) is None
    result = tracker.observe(voice(), turn(3))
    assert result is not None
    return result


def test_trigger_once_with_lead_and_absolute_bounds():
    req = request()
    assert req.reason == "voiced_wait"
    assert (req.first_turn, req.last_turn) == (1, 3)
    assert req.end_seconds == pytest.approx(2.4)
    assert req.lead_end_seconds == pytest.approx(.8)
    assert req.wait_start_seconds == pytest.approx(.8)
    assert req.wait_end_seconds == pytest.approx(2.4)
    assert req.audio.shape == (38400,)
    assert not req.audio.flags.writeable


def test_short_voiced_tail_on_eos_and_silence():
    tracker = GapTracker()
    silence = np.full(12800, .1, dtype="float32")  # DC is not speech
    for n in range(1, 5):
        assert tracker.observe(silence, turn(n)) is None
    assert tracker.finish() is None
    assert tracker.observe(voice(.2), turn(5)) is None
    assert tracker.finish().reason == "end_of_audio"
    assert tracker.finish() is None
    block = np.concatenate([voice(.24), np.zeros(8960, dtype="float32")])
    assert GapTracker().observe(block, turn(1)).reason == "pause"


def test_short_insert_between_languages_is_not_masked_by_next_direct():
    tracker = GapTracker()
    assert tracker.observe(voice(), turn(1, "はい", "ja")) is None
    assert tracker.observe(voice(), turn(2)) is None
    req = tracker.observe(voice(), turn(3, "Salut", "fr"))
    assert req.reason == "main_resumed"
    assert (req.wait_start_seconds, req.wait_end_seconds) == pytest.approx((.8, 1.6))
    assert req.end_seconds == pytest.approx(2.4)
    assert tracker.finish() is None


def test_long_wait_gives_bounded_windows_not_one_request_each_chunk():
    tracker = GapTracker()
    requests = [r for i in range(1, 51) if (r := tracker.observe(voice(), turn(i)))]
    assert len(requests) == 25
    for previous, following in zip(requests, requests[1:]):
        assert following.eligible_after_seconds == previous.end_seconds
        assert following.end_seconds > previous.end_seconds
        assert following.start_seconds < previous.end_seconds
        assert len(following.audio) == round(2.1 * RATE)


def span(words, lang="de"):
    timed = [dict(text=t, start_seconds=a, end_seconds=b) for t, a, b in words]
    return {"text": "".join(w["text"] for w in timed), "words": timed,
            "start_seconds": timed[0]["start_seconds"], "end_seconds": timed[-1]["end_seconds"],
            "language": lang}


def test_reconcile_removes_only_known_lead_and_partial_catchup():
    hyp = span([("皆さん。", .05, .50), (" Guten", .7, .9),
                (" Morgen", 1., 1.3), (" zusammen.", 1.4, 2.)])
    fixed = reconcile(request(), [hyp], [turn(1, "皆さん。", "ja"), turn(4, "GUTEN!")])
    assert len(fixed) == 1
    assert fixed[0]["text"] == " Morgen zusammen."
    assert (fixed[0]["start_seconds"], fixed[0]["end_seconds"]) == (1., 2.)
    assert [w["text"] for w in fixed[0]["words"]] == [" Morgen", " zusammen."]
    assert (fixed[0]["sort_after_turn"], fixed[0]["sort_after_segment"]) == (4, 0)
    assert "speaker" not in fixed[0]
    assert reconcile(request(), [hyp], [turn(4, "Guten Morgen zusammen.")]) == []
    assert reconcile(request(), [hyp], [turn(4, "Ja. Guten Morgen zusammen. Hallo!")]) == []


def test_reconcile_keeps_repetition_after_lead():
    tracker = GapTracker(wait_seconds=.8)
    tracker.observe(voice(), turn(1, "Hi!", "en"))
    req = tracker.observe(voice(), turn(2))
    hyp = span([("Hi!", .31, .55), (" Hi!", 1., 1.23)], "en")
    kept = reconcile(req, [hyp], [turn(1, "Hi!", "en")])
    assert kept[0]["text"] == " Hi!"
    assert kept[0]["start_seconds"] == 1.


def test_right_context_from_next_language_is_not_reemitted():
    tracker = GapTracker()
    tracker.observe(voice(), turn(1, "はい。", "ja"))
    tracker.observe(voice(), turn(2))
    req = tracker.observe(voice(), turn(3, "Bonjour!", "fr"))
    hyp = span([("はい", .2, .45), (" Danke.", .82, 1.26), (" Bonjour.", 1.75, 2.2)])
    retained = reconcile(req, [hyp], [turn(1, "はい。", "ja"), turn(3, "Bonjour!", "fr")])
    assert retained[0]["text"] == " Danke."
    assert retained[0]["end_seconds"] == 1.26
    assert (retained[0]["sort_before_turn"], retained[0]["sort_before_segment"]) == (3, 0)


def test_delayed_catchup_after_waits_has_bounded_word_and_time_evidence():
    req = request()
    hyp = span([("Guten", .81, 1.03), (" Morgen", 1.1, 1.45), (" zusammen.", 1.6, 1.9)])
    results = [turn(4), turn(5), turn(6, "GUTEN Morgen")]
    kept = reconcile(req, [hyp], results)
    assert (kept[0]["text"], kept[0]["sort_after_turn"]) == (" zusammen.", 6)
    assert reconcile(req, [hyp], [*results[:2], turn(6, "Guten Morgen zusammen.")]) == []
    assert reconcile(req, [hyp], [*results[:2], turn(6, "Guten")])[0]["text"] == hyp["text"]
    assert reconcile(req, [hyp], [turn(4, "OK"), turn(5, hyp["text"])])[0]["text"] == hyp["text"]
    assert reconcile(req, [hyp], [turn(9, hyp["text"])])[0]["text"] == hyp["text"]


def test_silent_wait_after_direct_never_requests_or_repeats_at_end():
    tracker = GapTracker()
    assert tracker.observe(voice(), turn(1, "はい。", "ja")) is None
    for n in range(2, 35):
        assert tracker.observe(np.zeros(12800), turn(n)) is None
    assert tracker.finish() is None
    assert tracker.observe(voice(), turn(35, "はい。", "ja")) is None
    assert tracker.finish() is None


@pytest.mark.parametrize("lead", [False, True])
def test_resumed_main_checks_voiced_wait_regardless_of_prior_language(lead):
    tracker = GapTracker()
    n = 1
    if lead:
        tracker.observe(voice(), turn(n, "Eins."))
        n += 1
    assert tracker.observe(voice(), turn(n)) is None
    current = turn(n+1, "Drei.")
    req = tracker.observe(voice(), current)
    assert req.reason == "main_resumed"
    assert (req.wait_start_seconds, req.wait_end_seconds) == pytest.approx(((n-1)*.8, n*.8))
    base = req.wait_start_seconds
    recovered = [("Zwei.", base+.05, base+.5), (" Drei.", base+.85, base+1.)]
    assert reconcile(req, [span(recovered)], [current])[0]["text"] == "Zwei."
    assert reconcile(req, [span(recovered[1:])], [current]) == []
    assert tracker.finish() is None


def test_unmatched_lead_and_resumed_main_rephrasing_cannot_invent_gap_text():
    tracker = GapTracker()
    tracker.observe(voice(), turn(1, "今日は。", "ja"))
    tracker.observe(voice(), turn(2))
    current = turn(3, "Bonjour.", "fr")
    req = tracker.observe(voice(), current)
    different = [("Today.", .05, .5), (" Guten", .7, .95), (" Morgen.", 1.2, 1.5),
                 (" Hello.", 1.7, 2.)]
    kept = reconcile(req, [span(different)], [current])
    assert [s["text"] for s in kept] == [" Guten Morgen."]


def test_silence_flushes_preceding_voiced_wait_once():
    tracker = GapTracker()
    tracker.observe(voice(), turn(1, "会議", "ja"))
    tracker.observe(voice(), turn(2))
    req = tracker.observe(np.zeros(12800), turn(3))
    assert req.reason == "pause" and req.wait_start_seconds == pytest.approx(.8)
    for n in range(4, 8):
        assert tracker.observe(np.zeros(12800), turn(n)) is None
    assert tracker.finish() is None


def test_do_not_call_on_every_direct_or_when_tail_lacks_speech():
    tracker = GapTracker()
    for i in range(1, 12):
        assert tracker.observe(voice(), turn(i, "Hallo")) is None
    ultimately_silent = np.concatenate([voice(.22), np.zeros(int(.58 * RATE))])
    assert tracker.observe(ultimately_silent, turn(12, "Hallo")) is None
    assert tracker.finish() is None
