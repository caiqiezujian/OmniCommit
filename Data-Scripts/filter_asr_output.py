#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""过滤 OmniCommit 蒸馏结果，只保留 output 符合当前格式的 JSONL 行。"""

import json
import re
from collections import Counter
from pathlib import Path


# 只修改这两个路径；输出必须是一个不存在的新文件。
INPUT_JSONL = "/data/your_input.jsonl"
OUTPUT_JSONL = "/data/your_filtered.jsonl"

TAG = re.compile(r"<([^<>]+)>")
LANGUAGE = re.compile(r"[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*\Z")
SPEAKER = re.compile(r"spk[1-9][0-9]*\Z")
DIRECT_PREFIX = re.compile(r"^<([a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*)><(spk[1-9][0-9]*)>")


def normalize_transcript(source):
    """与蒸馏脚本一致：保留正文，压缩重复的语种/说话人标签。"""
    if not isinstance(source, str) or not source.strip():
        raise ValueError("原始转录为空")
    source.encode("utf-8")
    state = {"language": None, "speaker": None}
    pending = {}
    previous_pair = None
    result = []
    position = 0

    for match in list(TAG.finditer(source)) + [None]:
        end = match.start() if match is not None else len(source)
        body = source[position:end]
        if body.strip():
            if state["language"] is None or state["speaker"] is None:
                raise ValueError("文字之前缺少语种或说话人标签")
            pair = (state["language"], state["speaker"])
            if pair != previous_pair:
                result.append(f"<{pair[0]}><{pair[1]}>")
                previous_pair = pair
            result.append(body)
            pending.clear()
        elif body:
            result.append(body)

        if match is None:
            break
        value = match.group(1)
        if SPEAKER.fullmatch(value):
            kind = "speaker"
        elif LANGUAGE.fullmatch(value):
            kind = "language"
        else:
            raise ValueError("存在未知标签")
        if kind in pending and pending[kind] != value:
            raise ValueError("没有文字的连续标签互相冲突")
        pending[kind] = value
        state[kind] = value
        position = match.end()

    if pending or previous_pair is None:
        raise ValueError("标签悬空或没有正文")
    normalized = "".join(result)
    if TAG.sub("", normalized) != TAG.sub("", source):
        raise ValueError("标签处理改变了正文")
    return normalized


def comparison_text(text):
    """容忍完整前缀放在最开头的空白之前，不改变任何正文字符。"""
    normalized = normalize_transcript(text)
    first = re.match(r"(\s*)(<[^<>]+><spk[1-9][0-9]*>)", normalized)
    if first:
        return first.group(2) + first.group(1) + normalized[first.end():]
    return normalized


def check_chunk_switch(chunk, state):
    """按已经读入的状态识别实际切换；重复相同前缀不算切换。"""
    switched = False
    pending = {}
    position = 0
    for match in list(TAG.finditer(chunk)) + [None]:
        end = match.start() if match is not None else len(chunk)
        body = chunk[position:end]
        if body.strip():
            if state["language"] is None or state["speaker"] is None:
                raise ValueError("片段缺少归属标签")
            pending.clear()
        if match is None:
            break
        value = match.group(1)
        if SPEAKER.fullmatch(value):
            kind = "speaker"
        elif LANGUAGE.fullmatch(value):
            kind = "language"
        else:
            raise ValueError("片段存在未知标签")
        if kind in pending and pending[kind] != value:
            raise ValueError("片段中连续标签互相冲突")
        if state[kind] is not None and state[kind] != value:
            switched = True
        pending[kind] = value
        state[kind] = value
        position = match.end()
    if pending:
        raise ValueError("片段末尾存在没有文字的标签")
    return switched


def check_record(record):
    """不修复、不改写：任一格式或确定性协议检查失败即过滤。"""
    if not isinstance(record, dict):
        raise ValueError("输入行不是 JSON 对象")

    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("messages 不是数组")
    assistants = [m for m in messages if isinstance(m, dict) and m.get("role") == "assistant"]
    if len(assistants) != 1:
        raise ValueError("原始 assistant 转录不是唯一一条")
    source = normalize_transcript(assistants[0].get("content"))

    raw_output = record.get("output")
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise ValueError("output 为空或不是 JSON 字符串")
    # 蒸馏脚本最终写入的是纯 JSON 字符串，保留行原样供后续 json.loads 使用。
    output = json.loads(raw_output)
    if not isinstance(output, dict) or set(output) != {"segments", "flush", "error"}:
        raise ValueError("output 顶层字段不正确")
    if output["error"] is not None:
        raise ValueError("output 标记了错误")
    if type(output["flush"]) is not bool:
        raise ValueError("flush 不是布尔值")
    segments = output["segments"]
    if not isinstance(segments, list) or not segments:
        raise ValueError("segments 为空或不是数组")

    chunks = []
    for index, segment in enumerate(segments, 1):
        if not isinstance(segment, dict) or set(segment) != {"source_text", "action"}:
            raise ValueError(f"第 {index} 片字段不正确")
        chunk, action = segment["source_text"], segment["action"]
        if not isinstance(chunk, str) or not chunk.strip():
            raise ValueError(f"第 {index} 片文字为空")
        plain = TAG.sub("", chunk)
        if "<" in plain or ">" in plain:
            raise ValueError(f"第 {index} 片从标签中间切开")
        if not any(character.isalnum() for character in plain):
            raise ValueError(f"第 {index} 片没有正文")
        if action not in ("WAIT", "DIRECT") or not isinstance(action, str):
            raise ValueError(f"第 {index} 片动作不是 WAIT 或 DIRECT")
        if action == "DIRECT" and not DIRECT_PREFIX.match(chunk):
            raise ValueError(f"第 {index} 片 DIRECT 缺少完整语种和说话人前缀")
        chunks.append(chunk)

    if comparison_text("".join(chunks)) != comparison_text(source):
        raise ValueError("切分后的正文或归属与原始转录不一致")
    if output["flush"] != (segments[-1]["action"] == "WAIT"):
        raise ValueError("flush 与最后一个动作不一致")

    state = {"language": None, "speaker": None}
    for index, segment in enumerate(segments, 1):
        switched = check_chunk_switch(segment["source_text"], state)
        if switched and segment["action"] != "WAIT":
            raise ValueError(f"第 {index} 片发生语种或说话人切换，但不是 WAIT")


def main():
    source = Path(INPUT_JSONL).expanduser().resolve()
    destination = Path(OUTPUT_JSONL).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"输入文件不存在：{source}")
    if source == destination:
        raise ValueError("输入文件和输出文件不能是同一个文件")

    destination.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    examples = []

    # 二进制逐行读取：保留通过记录的原始字节，不改写其他字段和 output。
    with source.open("rb") as input_file, destination.open("xb") as output_file:
        for line_number, raw_line in enumerate(input_file, 1):
            if not raw_line.strip():
                counts["blank"] += 1
                continue
            counts["total"] += 1
            if counts["total"] % 10000 == 0:
                print(f"正在处理第 {counts['total']} 条，已保留 {counts['kept']} 条，已过滤 {counts['filtered']} 条", flush=True)
            try:
                text = raw_line.decode("utf-8-sig" if line_number == 1 else "utf-8")
                record = json.loads(text)
                check_record(record)
            except Exception as exc:
                counts["filtered"] += 1
                if len(examples) < 10:
                    examples.append((line_number, str(exc)))
                continue

            output_file.write(raw_line)
            counts["kept"] += 1

    print(f"处理完成：共 {counts['total']} 条，保留 {counts['kept']} 条，过滤 {counts['filtered']} 条，空行 {counts['blank']} 行。")
    print(f"输出文件：{destination}")
    if examples:
        print("前 10 条被过滤原因：")
        for line_number, reason in examples:
            print(f"  第 {line_number} 行：{reason}")


if __name__ == "__main__":
    main()
