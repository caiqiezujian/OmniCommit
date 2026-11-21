import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asr_gspo.reward import _edits, _units
from asr_gspo.sentence_boundaries import sentence_boundary_errors


def score(ref, *emissions, characters=False):
    text = "".join(emissions)
    source, candidate = _units(ref, characters=characters), _units(text, characters=characters)
    path = _edits(source, candidate, [1] * len(candidate), {}, {}, 1)
    return sentence_boundary_errors(ref, text, source, candidate, path)


class SentenceBoundaryTests(unittest.TestCase):
    def test_unwarranted_stop_at_stream_commit(self):
        s = score("我知道你有点问题。", "我知道。", "你有点问题。")
        self.assertEqual(s["extra_internal_periods"], 1)
        self.assertEqual(s["missing_sentence_stops"], 0)
        self.assertTrue(s["punctuation_scorable"])
        self.assertEqual(score("我知道你有点问题。", "我知道", "你有点问题。")["extra_internal_periods"], 0)

    def test_repeated_echo_does_not_hide_added_stop(self):
        for text in ["我知道。知道你有点问题。", "我知道知道。你有点问题。"]:
            with self.subTest(text=text):
                self.assertEqual(score("我知道你有点问题。", text)["extra_internal_periods"], 1)

    def test_legal_stops_kept_even_if_the_sentence_is_short(self):
        s = score("好。你呢？我来！", "好。", "你呢。", "我来。")
        self.assertEqual(s["reference_sentence_stops"], 3)
        self.assertEqual(s["extra_internal_periods"], 0)
        self.assertEqual(s["missing_sentence_stops"], 0)
        self.assertEqual(score("好。你呢？我来！", "好你呢我来")["missing_sentence_stops"], 3)

    def test_dot_inside_reference_comma_is_extra(self):
        s = score("如果方便，明天见。", "如果方便。", "明天见。")
        self.assertEqual(s["extra_internal_periods"], 1)

    def test_missing_terminal_punctuation_does_not_require_final_period(self):
        for ref in ["有时间联系我", "有时间，联系我", "I paid 3.50 to Dr. Lee", ""]:
            with self.subTest(ref=ref):
                s = score(ref, ref + ".")
                self.assertEqual(s["reference_sentence_stops"], 0)
                self.assertEqual(s["extra_internal_periods"], 0)

    def test_unpunctuated_reference_still_scores_internal_fragmentation(self):
        s = score("有时间联系我", "有时间。联系我。")
        self.assertEqual(s["extra_internal_periods"], 1)
        self.assertEqual(s["unsupported_short_breaks"], 1)

    def test_question_or_exclamation_cannot_evade_false_boundary_penalty(self):
        for mark in ["。", "!", "？", "؟"]:
            s = score("我知道这个问题。", "我知道" + mark + "这个问题。")
            self.assertEqual(s["extra_internal_stops"], 1)
            self.assertEqual(s["unsupported_short_breaks"], 1)
        self.assertEqual(score("好！你说。", "好！你说。")["extra_internal_stops"], 0)

    def test_technical_dots_and_acronyms(self):
        ref = "Dr Lee paid US $3.50 on example.com today."
        hyp = "Dr. Lee paid U.S. $3.50 on example.com today."
        s = score(ref, hyp, characters=True)
        self.assertEqual(s["extra_internal_periods"], 0)
        self.assertEqual(s["missing_sentence_stops"], 0)
        self.assertEqual(s["reference_sentence_stops"], 1)

    def test_no_ascii_space_at_chunk_join_does_not_hide_period(self):
        s = score("I know you can help.", "I know.", "you can help.")
        self.assertEqual(s["extra_internal_periods"], 1)

    def test_ordinals_and_initials(self):
        for ref, hyp in [
            ("Am 3 März kommt sie.", "Am 3. März kommt sie."),
            ("Он встретил А Пушкина.", "Он встретил А. Пушкина."),
            ("تحدث د أحمد.", "تحدث د. أحمد."),
        ]:
            with self.subTest(ref=ref):
                self.assertEqual(score(ref, hyp)["extra_internal_periods"], 0)

    def test_multilingual_questions_and_internal_stops(self):
        for ref, hyp in [
            ("你问了吗？我问了。", "你问。了吗。我问了！"),
            ("هل وصل؟ نعم وصل.", "هل. وصل. نعم وصل."),
            ("Mañana nos vemos. ¿Sí?", "Mañana. nos vemos. ¿Sí?"),
            ("はい。明日ですね。", "はい。明日。ですね。"),
        ]:
            with self.subTest(ref=ref):
                s = score(ref, hyp)
                self.assertEqual(s["extra_internal_periods"], 1)
                self.assertEqual(s["missing_sentence_stops"], 0)

    def test_no_punctuation_penalty_at_uncertain_text(self):
        s = score("We will meet tomorrow.", "We may. leave tomorrow.")
        self.assertEqual(s["extra_internal_periods"], 0)
        self.assertEqual(s["missing_sentence_stops"], 0)
        self.assertEqual(score("We meet today. Yes.", "We meet")["missing_sentence_stops"], 0)

    def test_ellipsis_and_optional_newline_are_not_required_stops(self):
        s = score("I thought... well\nperhaps tomorrow.", "I thought. well. perhaps tomorrow.")
        self.assertEqual(s["extra_internal_periods"], 0)
        self.assertEqual(s["reference_sentence_stops"], 1)


if __name__ == "__main__":
    unittest.main()
