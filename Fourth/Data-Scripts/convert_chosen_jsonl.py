#!/usr/bin/env python3
"""把顶层 chosen 按原有换行转换成 OmniCommit 的全 DIRECT 标注。"""

import json
import re
import sys
import unicodedata
from pathlib import Path
from tempfile import TemporaryDirectory


INPUT_JSONL = Path("/data/yb/Code/OmniCommit/datasets/chosen.jsonl")
OUTPUT_JSONL = Path("/data/yb/Code/OmniCommit/datasets/chosen.direct.jsonl")

SYSTEM_PROMPT = (
    "You are a high-precision speech transcriber. Transcribe the audio into text. Strictly follow these rules:\n"
    "1. Listen to the provided speech and produce a transcription in text, and convert all numerical expressions to standard digit format.\n"
    '2. Use the provided "Entity Reference" for the exact spelling of names or terms. If an entity is NOT in the audio, do NOT force its use.\n'
    "3. Split the text based on semantics and use a newline character as the separator.\n"
    "4. Insert speaker labels <spk1>, <spk2>, ... at each speaker change, and insert language labels <zh>, <en>, ... when the language changes.\n"
)
LANGUAGES = {
    "zh": "zh", "chinese": "zh", "mandarin": "zh", "中文": "zh", "汉语": "zh",
    "en": "en", "english": "en", "英文": "en", "英语": "en",
}
TAG = re.compile(r"<([^<>\s]+)>")
SPEAKER = re.compile(r"spk[1-9][0-9]*")
CONTRACTION = re.compile(r"^(?:['’](?:s|t|re|ve|ll|d|m)|n['’]t)\b", re.IGNORECASE)


def json_text(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def latin_letter_or_digit(char):
    return char.isdecimal() or (char.isalpha() and "LATIN" in unicodedata.name(char, ""))


def newline_separator(left, right):
    """只判断原换行处的空格，返回的空格属于右侧块的正文。"""
    if not left or CONTRACTION.match(right):
        return ""
    left = unicodedata.normalize("NFC", left)
    last, first = left[-1], right[0]
    if len(left) >= 2 and left[-2].isdecimal() and last in ".," and first.isdecimal():
        return ""  # 3.\n14 或 1,\n000
    ends_word = latin_letter_or_digit(last) or last in ".,;:!?)]}”’"
    if last in "\"'":
        ends_word = len(left) >= 2 and (latin_letter_or_digit(left[-2]) or left[-2] in ".!?")
    starts_word = latin_letter_or_digit(first) or first in "([{“‘$£€"
    if first in "\"'":
        starts_word = len(right) >= 2 and latin_letter_or_digit(right[1])
    return " " if ends_word and starts_word else ""


def chosen_segments(chosen, language):
    segments = []
    speaker = "spk1"
    tail = ""
    for line in chosen.splitlines():
        parts = []
        last_state = None
        for index, value in enumerate(TAG.split(line.strip())):
            if index % 2:  # 捕获组是标签，此外都是原文。
                label = value.casefold()
                if SPEAKER.fullmatch(label):
                    speaker = label
                elif label in ("zh", "en"):
                    language = label
                else:
                    raise ValueError(f"chosen 中有未知标签 <{value}>")
                continue
            if not value.strip():
                continue
            if language is None:
                raise ValueError("正文缺少有效语种，请检查 language 或行首的 <zh>/<en>")
            if not parts:
                value = value.lstrip()
                value = newline_separator(tail, value) + value
            state = (language, speaker)
            if state != last_state:
                parts.append(f"<{language}><{speaker}>")
                last_state = state
            parts.append(value)
            tail = (tail + value)[-8:]
        if parts:
            segments.append({"source_text": "".join(parts), "action": "DIRECT"})
    if not segments:
        raise ValueError("chosen 中没有正文")
    return segments


def convert_record(row):
    if not isinstance(row, dict):
        raise ValueError("记录必须是 JSON 对象")
    audio, chosen = row.get("audio"), row.get("chosen")
    if not isinstance(audio, str) or not audio.strip():
        raise ValueError("缺少 audio 路径")
    if not isinstance(chosen, str) or not chosen.strip():
        raise ValueError("缺少顶层 chosen 正文")
    raw_language = str(row.get("language", "")).strip().casefold().replace("_", "-")
    language = LANGUAGES.get(raw_language.split("-", 1)[0])
    segments = chosen_segments(chosen, language)
    return {
        "audios": [audio],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "<audio>**Entity Reference:** []"},
            {"role": "assistant", "content": "".join(s["source_text"] for s in segments)},
        ],
        "output": json_text({"segments": segments, "flush": False, "error": None}),
    }


def convert_file(source, target):
    if source.resolve() == target.resolve():
        raise ValueError("输入和输出路径必须不同")
    target.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = blank = 0
    with source.open("rb") as reader, TemporaryDirectory(dir=target.parent) as work:
        temporary = Path(work) / target.name
        with temporary.open("wb") as writer:
            for number, raw in enumerate(reader, 1):
                if not raw.strip():
                    blank += 1
                    continue
                try:
                    result = convert_record(json.loads(raw))
                    encoded = (json_text(result) + "\n").encode("utf-8")
                except ValueError as exc:
                    skipped += 1
                    if skipped <= 20:
                        print(f"[跳过] 第 {number} 行：{exc}", file=sys.stderr)
                else:
                    writer.write(encoded)
                    written += 1
                if number % 10000 == 0:
                    print(f"[进度] 已读 {number} 行，成功 {written}，异常 {skipped}", flush=True)
        temporary.replace(target)
    print(f"[完成] 成功 {written}，异常跳过 {skipped}，空行 {blank}，输出 {target}")


if __name__ == "__main__":
    convert_file(INPUT_JSONL, OUTPUT_JSONL)
