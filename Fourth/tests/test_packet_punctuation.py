"""HTTP punctuation rules can be tested without an ASR model or a Web server."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fourth_asr.packet_punctuation import edit_packet_period, replace_terminal_period
from fourth_asr.protocol import SPK_CHANGE


@pytest.mark.parametrize("before,after", [
    ("已收到。", "已收到，"),
    ("已收到.", "已收到，"),
    ("请打开 API。", "请打开 API，"),
    ("「確認します。」 ", "「確認します、」 "),
    ("今日は東京。", "今日は東京、"),
    ("東京。", "東京，"),  # Kanji-only Japanese is indistinguishable from Chinese.
    ("알겠습니다.", "알겠습니다,"),
    ("We can.", "We can,"),
    ("«Merci.»", "«Merci,»"),
    ("Danke.", "Danke,"),
    ("Gracias.", "Gracias,"),
    ("Grazie.", "Grazie,"),
    ("Obrigado.", "Obrigado,"),
    ("Спасибо.", "Спасибо,"),
    ("เข้าใจแล้ว.", "เข้าใจแล้ว,"),
    ("حسنًا.", "حسنًا،"),
    ("تمام۔", "تمام،"),
    ("他们来了。我知道了。", "他们来了。我知道了，"),
    ("你好。We can.", "你好。We can,"),
    ("大家好。" + SPK_CHANGE + "分かりました。", "大家好。" + SPK_CHANGE + "分かりました、"),
    ("好的！”", "好的！”"),
    ("好的……", "好的……"),
    ("«Vale...»", "«Vale...»"),
    ("Is it ready?", "Is it ready?"),
    ("Ask Dr.", "Ask Dr."),
    ("The U.S.", "The U.S."),
    ("и т. д.", "и т. д."),
    ("Value 3.14", "Value 3.14"),
    ("Number 3.", "Number 3."),
    ("see https://example.org/foo.", "see https://example.org/foo."),
    ("try example.org.", "try example.org."),
    ("mail me@example.org.", "mail me@example.org."),
    ("See example.org. We can.", "See example.org. We can,"),
])
def test_terminal_period(before, after):
    assert replace_terminal_period(before) == after


@pytest.mark.parametrize("previous,marked,expected", [
    ("你好", "。", "，"),
    ("わかりました", "。』 ", "、』 "),
    ("ありがとう", SPK_CHANGE + "你好。", SPK_CHANGE + "你好，"),
    ("Ask Dr", ".", "."),
    ("Ask ", "Dr.", "Dr."),
    ("Finished", ".", ","),
    ("Visit https://example.com", ".", "."),
    ("你好。", "。", "。"),
    ("", "。", "。"),
    ("你知道", SPK_CHANGE + "。", SPK_CHANGE + "。"),
    ("", "还未提交", "还未提交"),
])
def test_streamed_tail(previous, marked, expected):
    assert replace_terminal_period(marked, previous) == expected


@pytest.mark.parametrize("previous,marked,ms,expected", [
    ("", "我知道。", 100, "我知道"),
    ("", "对。", 100, "对"),
    ("", "是的。", 100, "是的"),
    ("", "Got it.", 100, "Got it"),
    ("", "Wir haben also Zeit.", 400, "Wir haben also Zeit."),
    ("", "请打开 API。", 100, "请打开 API"),
    ("", "“我知道。” ", 100, "“我知道” "),
    ("", "你好。I know.", 449, "你好。I know,"),
    ("", "I agree." + SPK_CHANGE + "我知道。", 80, "I agree." + SPK_CHANGE + "我知道"),
    ("All done", ".", 80, ""),
    ("我知道", "。", 80, ""),
    ("请确认", "方案。", 80, "方案"),
    ("We discussed the ", "schedule.", 80, "schedule"),
    ("Das ist gut.", "I know.", 80, "I know"),
    ("", "Ask Dr.", 80, "Ask Dr."),
    ("", "I paid 3.", 80, "I paid 3."),
    ("", "The U.S.", 80, "The U.S."),
    ("", "Thanks...", 80, "Thanks..."),
    ("", "Visit https://example.com.", 80, "Visit https://example.com."),
    ("", "I know.", 450, "I know."),
    ("", "Bonjour à tous.", 100, "Bonjour à tous,"),
    ("Thanks.", "Guten Morgen.", 360, "Guten Morgen."),
    ("", "Ich bin am Bahnhof.", 120, "Ich bin am Bahnhof,"),
    ("", "Je suis sure.", 360, "Je suis sure."),
    ("I know", SPK_CHANGE + "Paris.", 360, SPK_CHANGE + "Paris."),
    ("", "東京。", 400, "東京。"),
    ("", "今日は東京。", 100, "今日は東京、"),
    ("", "알겠습니다.", 400, "알겠습니다."),
    ("", "Спасибо.", 100, "Спасибо,"),
    ("", "تمام۔", 100, "تمام،"),
])
def test_eligible_language_and_period_at_two_thresholds(previous, marked, ms, expected):
    edit = edit_packet_period(marked, previous, ms, 450, 150)
    assert (edit.text if edit else marked) == expected
