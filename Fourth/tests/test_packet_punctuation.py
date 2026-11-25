"""HTTP punctuation rules can be tested without an ASR model or a Web server."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fourth_asr.packet_punctuation import replace_terminal_period
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
