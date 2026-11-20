"""Measure fragmentation in a timed corpus and replay the existing Fourth policy."""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
import unicodedata
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fourth_asr.protocol import SPK_CHANGE, WAIT, get_protocol
from fourth_asr.recast import build_sft_row
from fourth_asr.semantic_preparation import STANDALONE, prepare_units
from fourth_asr.timestamped_source import parse_record


# 服务器配置。全量扫描 JSONL；回放只读取选中 WAV 的文件头。
INPUT = Path("/data/yb/Code/OmniCommit/datasets/timestamped.jsonl")
OUTPUT_DIR = Path("/data/yb/Code/OmniCommit/datasets/fourth_fragment_profile")
AUDIO_ROOT = None  # 相对音频路径默认从输入 JSONL 所在目录解析
CHUNK_MS = 1000
SHORT_MS = 300
REPLAY_PERCENT = 5  # 100 表示每条都执行当前 Fourth 的重铸
EXAMPLES_PER_GROUP = 5
PROGRESS_EVERY = 10000

DURATION_BINS = (100, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000, 10000)
CHAR_BINS = (0, 1, 2, 3, 5, 8, 12, 20, 40, 80)
GROUP_BINS = (0, 1, 2, 3, 5, 8, 12, 20)
SPACE_LANGS = set("ar de en es fr it ko pt ru".split())
POLICY = get_protocol("speaker")
MIXED = "(mixed)"


def percent(n, total):
    return round(100 * n / total, 2) if total else None


class Distribution:
    """Integer ceiling histograms. Memory is bounded even for extreme outliers."""
    def __init__(self, bounds, cap):
        self.bounds, self.cap = bounds, cap
        self.hist = Counter()

    def add(self, value):
        self.hist[min(math.ceil(value), self.cap + 1)] += 1

    def describe(self):
        total = sum(self.hist.values())
        cuts = (10, 25, 50, 75, 90, 95)
        quantiles = dict.fromkeys((f"p{p}" for p in cuts))
        buckets = Counter()
        cumulative = 0
        for value, count in sorted(self.hist.items()):
            key = next((str(b) for b in self.bounds if value <= b), f">{self.bounds[-1]}")
            buckets[key] += count
            cumulative += count
            for p in cuts:
                if quantiles[f"p{p}"] is None and 100 * cumulative >= total * p:
                    quantiles[f"p{p}"] = value if value <= self.cap else f">{self.cap}"
        keys = [str(b) for b in self.bounds] + [f">{self.bounds[-1]}"]
        return {"n": total, **quantiles, "bins_upper_inclusive": {k: buckets[k] for k in keys}}


def char_count(text):
    return sum(c.isalnum() for c in unicodedata.normalize("NFC", text))


def space_group_count(lang, text):
    if lang.split("-")[0] in SPACE_LANGS:
        return sum(any(c.isalnum() for c in item) for item in text.split())
    return None


def standalone(text):
    body = text.strip().casefold()
    return body.rstrip(".!?。！？،,，؛;؟…").rstrip() in STANDALONE


def language_of(languages):
    langs = set(languages)
    return next(iter(langs)) if len(langs) == 1 else MIXED


class TextStats:
    def __init__(self):
        self.chars = Distribution(CHAR_BINS, 4096)
        self.space_groups = Distribution(GROUP_BINS, 4096)
        self.duration = Distribution(DURATION_BINS, 60000)
        self.next_gap = Distribution(DURATION_BINS, 60000)
        self.counts = Counter()

    def add(self, n, groups, in_lexicon, duration, following_same_gap):
        counts = self.counts
        counts["n"] += 1
        self.chars.add(n)
        counts["no_letters_or_digits"] += n == 0
        counts["one_or_two_chars"] += 1 <= n <= 2
        if 1 <= n <= 2:
            counts["one_or_two_chars_standalone_lexicon"] += in_lexicon
        if groups is not None:
            self.space_groups.add(groups)
            counts["one_space_group"] += groups == 1
        if duration is not None:
            self.duration.add(duration)
            if following_same_gap is not None:
                if following_same_gap < 0:
                    counts["overlaps_next_same_lang_spk"] += 1
                else:
                    self.next_gap.add(following_same_gap)
            if duration <= SHORT_MS:
                counts["short_ms"] += 1
                counts["short_ms_and_one_or_two_chars"] += 1 <= n <= 2
                counts["short_ms_standalone_lexicon"] += in_lexicon
                if following_same_gap is not None and 0 <= following_same_gap <= SHORT_MS:
                    counts["short_ms_next_same_lang_spk_near"] += 1

    def report(self):
        c = self.counts
        groups = self.space_groups.describe()
        duration = self.duration.describe()
        return {"n": c["n"], "counts": dict(c), "chars": self.chars.describe(),
                "duration_ms": duration, "space_groups": groups,
                "next_same_lang_spk_nonoverlap_gap_ms": self.next_gap.describe(),
                "one_or_two_chars_pct": percent(c["one_or_two_chars"], c["n"]),
                "one_space_group_pct": percent(c["one_space_group"], groups["n"]),
                "short_ms_pct": percent(c["short_ms"], duration["n"])}


class Stage:
    def __init__(self):
        self.stats = defaultdict(TextStats)

    def add(self, lang, text, duration=None, following_same_gap=None):
        n = char_count(text)
        groups = space_group_count(lang, text)
        in_lexicon = standalone(text)
        for key in ("ALL", lang):
            self.stats[key].add(n, groups, in_lexicon, duration, following_same_gap)
        return n, groups

    def units(self, units):
        for i, u in enumerate(units):
            lang = language_of(s[0] for s in u.spans)
            gap = None
            if i + 1 < len(units) and len(u.spans) == len(units[i+1].spans) == 1:
                nxt = units[i+1]
                if u.spans[0][:2] == nxt.spans[0][:2]:
                    gap = nxt.start_ms - u.end_ms
            self.add(lang, u.text, u.end_ms - u.start_ms, gap)

    def report(self):
        return {lang: group.report() for lang, group in sorted(self.stats.items())}


class Examples:
    def __init__(self):
        self.groups = defaultdict(list)
        self.seen = Counter()
        self.rng = random.Random(20261120)

    def offer(self, kind, lang, example):
        key = (kind, lang)
        self.seen[key] += 1
        count = self.seen[key]
        pos = count - 1 if count <= EXAMPLES_PER_GROUP else self.rng.randrange(count)
        if pos < EXAMPLES_PER_GROUP:
            items = self.groups[key]
            if pos == len(items):
                items.append(example)
            else:
                items[pos] = example


def timed_states(units):
    """Text offsets and known onsets; never interpolate inside a mixed parent."""
    runs = []
    offset = 0
    previous = None
    for unit in units:
        for part in unit.children or (unit,):
            for index, (lang, speaker, text) in enumerate(part.spans):
                if text.strip() and (lang, speaker) != previous:
                    start = part.start_ms if index == 0 else None
                    runs.append((offset, lang, speaker, start))
                    previous = lang, speaker
                offset += len(text)
    return runs


def replay(raw_units, prepared, path, info):
    audit = {}
    row = build_sft_row(prepared, path, info.frames, info.samplerate, CHUNK_MS,
                        "normalized", protocol=POLICY, audit=audit)
    runs = timed_states(raw_units)
    reference = "".join(u.text for u in raw_units)
    cursor = state_index = 0
    turns = []
    for i, message in enumerate(row["messages"][2:-2:2], 1):
        end_frame = min(info.frames, (i * CHUNK_MS * info.samplerate + 999) // 1000)
        end_ms = end_frame * 1000 / info.samplerate
        reply = message["content"]
        text = POLICY.parse_reply(reply).replace(SPK_CHANGE, "")
        stop = cursor + len(text)
        if not reference.startswith(text, cursor):
            raise ValueError("Fourth replay changed source text")
        langs = set()
        switches = []
        if text and state_index and runs[state_index-1][0] < cursor:
            if state_index == len(runs) or runs[state_index][0] > cursor:
                langs.add(runs[state_index-1][1])
        while state_index < len(runs) and runs[state_index][0] < stop:
            offset, lang, speaker, onset = runs[state_index]
            langs.add(lang)
            if state_index:
                _, old_lang, old_speaker, _ = runs[state_index-1]
                kind = ("both" if old_lang != lang and old_speaker != speaker else
                        "speaker" if old_speaker != speaker else "language")
                switches.append((kind, lang, onset, offset > cursor))
            state_index += 1
        if reply.count(SPK_CHANGE) != sum(s[0] != "language" for s in switches):
            raise ValueError("Fourth replay changed speaker markers")
        turns.append((i, end_ms, reply, text, language_of(langs), switches))
        cursor = stop
    if cursor != len(reference):
        raise ValueError("Fourth replay omitted source text")
    return turns, audit


def short_unit_outcomes(raw, turns):
    """Map original segment character intervals onto disjoint committed replies."""
    ends = []
    length = 0
    for _, _, _, text, _, _ in turns:
        if text:
            length += len(text)
            ends.append(length)
    start = 0
    for unit in raw:
        end = start + len(unit.text)
        if unit.end_ms - unit.start_ms <= SHORT_MS:
            first, last = bisect_right(ends, start), bisect_left(ends, end)
            if first != last:
                outcome = "split_across_turns"
            elif start == (ends[first-1] if first else 0) and end == ends[first]:
                outcome = "alone"
            else:
                outcome = "with_other_text"
            yield language_of(s[0] for s in unit.spans), outcome
        start = end


class Profile:
    def __init__(self):
        self.source, self.paired_source, self.prepared, self.direct = (Stage() for _ in range(4))
        self.rows = Counter()
        self.lang_rows = Counter()
        self.turns = Counter()
        self.between_commits = Distribution(DURATION_BINS, 60000)
        self.first_commit = Distribution(DURATION_BINS, 60000)
        self.events = defaultdict(Counter)
        self.short_outcomes = defaultdict(Counter)
        self.latency = defaultdict(lambda: Distribution(DURATION_BINS, 60000))
        self.examples = Examples()
        self.errors = defaultdict(list)

    def error(self, stage, line, exc):
        self.rows[stage] += 1
        if len(self.errors[stage]) < 20:
            self.errors[stage].append({"line": line, "reason": str(exc)[:400]})

    def record_turns(self, line, path, turns, audit, info):
        t = self.turns
        previous_direct = False
        last_commit = None
        for i, end_ms, reply, text, lang, switches in turns:
            t["audio_turns"] += 1
            if reply == WAIT:
                t["wait"] += 1
                previous_direct = False
                continue
            n_chars, n_groups = self.direct.add(lang, text)
            if last_commit is None:
                self.first_commit.add(end_ms)
            else:
                self.between_commits.add(end_ms-last_commit)
            last_commit = end_ms
            t["direct"] += 1
            t["direct_immediately_after_direct"] += previous_direct
            previous_direct = True
            sample = {"line": line, "audio": str(path), "turn": i, "audio_end_ms": end_ms,
                      "previous": turns[i-2][2][:160] if i > 1 else None, "reply": reply[:160],
                      "next": turns[i][2][:160] if i < len(turns) else None}
            if 1 <= n_chars <= 2:
                self.examples.offer("short_direct", lang, sample)
            elif n_groups == 1:
                self.examples.offer("one_space_group_direct", lang, sample)
            for kind, target, onset, has_previous_text in switches:
                key = f"{kind}->{target}"
                c = self.events[key]
                c["n"] += 1
                c["previous_text_in_same_reply"] += has_previous_text
                if onset is None:
                    c["unknown_onset"] += 1
                    continue
                delay = end_ms - onset
                self.latency[key].add(delay)
                # Grid interval [start, end), matching Fourth switch semantics.
                onset_frame = math.ceil(onset * info.samplerate / 1000)
                onset_turn = min(len(turns), onset_frame * 1000 // (CHUNK_MS * info.samplerate) + 1)
                in_first = i == onset_turn
                c["in_onset_turn"] += in_first
                if turns[onset_turn-1][1] - onset <= SHORT_MS:
                    c["starts_near_grid_end"] += 1
                    c["near_end_and_emitted_in_onset_turn"] += in_first
                    c["of_those_on_final_audio_turn"] += in_first and i == len(turns)
                    if in_first:
                        self.examples.offer("immediate_late_" + kind, target, dict(sample, onset_ms=onset))
        for name, indices in audit.items():
            t[name] += len(indices)

    def prepared_examples(self, line, path, groups):
        for i, unit in enumerate(groups):
            if unit.end_ms - unit.start_ms > SHORT_MS:
                continue
            lang = language_of(s[0] for s in unit.spans)
            self.examples.offer("short_after_preparation", lang, {
                "line": line, "audio": str(path), "start_ms": unit.start_ms, "end_ms": unit.end_ms,
                "text": unit.text[:160], "speakers": sorted({s[1] for s in unit.spans}),
                "previous": groups[i-1].text[-160:] if i else None,
                "next": groups[i+1].text[:160] if i+1 < len(groups) else None,
            })


def annotated_upper_bound(record):
    times = record.get("timestamps") if isinstance(record, dict) else None
    if not isinstance(times, list) or not times:
        raise ValueError("missing per-segment timestamps")
    if any(not isinstance(t, dict) or type(t.get("end_ms")) not in (int, float) for t in times):
        raise ValueError("requires per-segment start_ms/end_ms; switch time_ms alone is insufficient")
    return max(t["end_ms"] for t in times)


def selected(audio, pct):
    h = hashlib.blake2b(audio.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") % 100 < pct


ROW_ERRORS = (ValueError, TypeError, KeyError, IndexError, OverflowError, RecursionError, OSError)


def scan(input_path, audio_root, pct):
    p = Profile()
    with input_path.open("rb") as src:
        for line, blob in enumerate(src, 1):
            p.rows["lines"] += 1
            if line % PROGRESS_EVERY == 0:
                print(f"[{line:,}] valid={p.rows['valid_source']:,} replayed={p.rows['replayed']:,}", flush=True)
            try:
                record = json.loads(blob.decode("utf-8-sig" if line == 1 else "utf-8"))
                upper = annotated_upper_bound(record)
                raw = parse_record(record, upper)
            except ROW_ERRORS as exc:
                p.error("invalid_source", line, exc)
                continue
            p.rows["valid_source"] += 1
            p.source.units(raw)
            langs = {lang for u in raw for lang, _, _ in u.spans}
            p.lang_rows.update(langs)
            p.rows["multilingual"] += len(langs) > 1
            p.rows["multispeaker"] += len({s for u in raw for _, s, _ in u.spans}) > 1
            audio = record["audios"][0]
            if not selected(audio, pct):
                continue
            p.rows["selected"] += 1
            path = audio_root / audio
            try:
                groups, _ = prepare_units(record, upper, chunk_ms=CHUNK_MS, short_ms=SHORT_MS)
            except ROW_ERRORS as exc:
                p.error("preparation_error", line, exc)
                continue
            p.paired_source.units(raw)
            p.prepared.units(groups)
            p.prepared_examples(line, path, groups)
            p.rows["prepared"] += 1
            try:
                info = sf.info(str(path))
            except (RuntimeError, OSError, ValueError) as exc:
                p.error("audio_header_error", line, exc)
                continue
            try:
                if not info.samplerate or upper > info.frames * 1000 / info.samplerate:
                    raise ValueError("timestamps extend past real audio or invalid sample rate")
                turns, audit = replay(raw, groups, path, info)
            except ROW_ERRORS as exc:
                p.error("replay_error", line, exc)
                continue
            p.rows["replayed"] += 1
            p.record_turns(line, path, turns, audit, info)
            for lang, outcome in short_unit_outcomes(raw, turns):
                p.short_outcomes[lang][outcome] += 1
                p.short_outcomes["ALL"][outcome] += 1
    return p


def write_report(p, output, pct, input_path, audio_root):
    t = p.turns
    report = {"input": str(input_path), "audio_root": str(audio_root), "chunk_ms": CHUNK_MS,
              "short_ms": SHORT_MS, "replay_percent": pct, "rows": p.rows, "records_per_language": p.lang_rows,
              "raw_all": p.source.report(), "raw_paired": p.paired_source.report(),
              "prepared_paired": p.prepared.report(), "direct_turns": p.direct.report(),
              "turn_counts": t, "wait_pct": percent(t["wait"], t["audio_turns"]),
              "first_commit_from_audio_start_ms": p.first_commit.describe(),
              "between_direct_commits_ms": p.between_commits.describe(),
              "short_source_outcomes": dict(p.short_outcomes),
              "direct_after_direct_pct_of_direct": percent(t["direct_immediately_after_direct"], t["direct"]),
              "switches": {k: {**v, "onset_to_commit_ms": p.latency[k].describe()}
                           for k, v in sorted(p.events.items())},
              "first_errors_by_stage": dict(p.errors)}
    output.mkdir(parents=True, exist_ok=True)
    (output / "profile.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "examples.jsonl").open("w", encoding="utf-8") as stream:
        for (kind, lang), items in sorted(p.examples.groups.items()):
            for item in items:
                stream.write(json.dumps({"kind": kind, "language": lang, **item}, ensure_ascii=False) + "\n")
    lines = [f"输入 {input_path}", f"音频步长 {CHUNK_MS} ms，短片段候选阈值 {SHORT_MS} ms，按源路径固定抽取 {pct}%。",
             f"数据行 {dict(p.rows)}",
             "原始统计校验文本与标注；真实音频时长仅在回放时从 WAV 文件头核对。",
             "字数按 NFC 后 Unicode 字母及数字计，排除标点、空白和组合附标；跨语比较还应看时长。",
             "空格组按含字母/数字的空白分隔片段计；中日泰不计算。混语段归 (mixed)，不重复分配时长。",
             "≤2字比例只计1或2个正文字母/数字，分母包括所有该组段。百分比均以本列对应样本计算。",
             "时长分位数向上取整至毫秒。profile.json 的分桶上界含等号，各桶互斥。",
             "raw_paired 和 prepared_paired 对应同一批成功准备的记录。DIRECT 和切换只统计真实回放成功记录，排除 EOF。"]
    lines.append("各语种出现的有效录音数（多语可重复计入）：" + str(dict(p.lang_rows.most_common())))
    lines.append("回放运行训练数据构造规则；提交延迟指新归属标注起点到首次提交对应文字的音频轮末。")
    for title, key in [("原始切分（全量）", "raw_all"), ("合并前（抽样）", "raw_paired"),
                       ("合并后（相同抽样）", "prepared_paired"), ("实际 DIRECT 正文（回放）", "direct_turns")]:
        lines.extend(["", title, "语言\t段数/轮数\t时长P50(ms)\t≤阈值%\t字数P50\t1–2字%\t单空格组%"])
        for lang, s in sorted(report[key].items(), key=lambda kv: -kv[1]["one_or_two_chars_pct"]):
            values = [lang, s["n"], s["duration_ms"]["p50"], s["short_ms_pct"], s["chars"]["p50"],
                      s["one_or_two_chars_pct"], s["one_space_group_pct"]]
            lines.append("\t".join("—" if v is None else str(v) for v in values))
    lines.extend(["", f"原始 ≤{SHORT_MS} ms 片段在训练目标里的去向（同一批成功回放的录音）",
                  "语种\t短段数\t独占一轮\t独占比例%\t与其他文字同轮\t跨多轮"])
    for lang, counts in sorted(p.short_outcomes.items()):
        n = sum(counts.values())
        lines.append(f"{lang}\t{n}\t{counts['alone']}\t{percent(counts['alone'],n)}"
                     f"\t{counts['with_other_text']}\t{counts['split_across_turns']}")
    lines.extend(["", f"训练目标音频轮次 {dict(t)}；WAIT% {report['wait_pct']}。",
                  f"连续两轮都有正文的 DIRECT 占比 {report['direct_after_direct_pct_of_direct']}%；"
                  f"同一录音相邻两次提交间隔 P50 {report['between_direct_commits_ms']['p50']} ms。",
                  "切换类型->新语种\t事件数\t起点未知\t块尾起声数\t其中当轮已提交\t其中末轮收尾\t起声到首次提交P50(ms)"])
    for k, s in report["switches"].items():
        lines.append(f"{k}\t{s['n']}\t{s.get('unknown_onset',0)}\t{s.get('starts_near_grid_end',0)}"
                     f"\t{s.get('near_end_and_emitted_in_onset_turn',0)}\t{s.get('of_those_on_final_audio_turn',0)}"
                     f"\t{s['onset_to_commit_ms']['p50']}")
    if not p.rows["replayed"]:
        lines.append("没有可回放记录。请查看抽样数量、音频根目录与 profile.json 内的错误示例。")
    summary = "\n".join(lines) + "\n"
    (output / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"结果目录 {output}")


def main():
    if not INPUT.is_file() or not 0 <= REPLAY_PERCENT <= 100:
        raise SystemExit("请检查文件顶部的 INPUT 和 REPLAY_PERCENT。")
    audio_root = AUDIO_ROOT or INPUT.parent
    profile = scan(INPUT, audio_root, REPLAY_PERCENT)
    write_report(profile, OUTPUT_DIR, REPLAY_PERCENT, INPUT, audio_root)
    return 0 if profile.rows["valid_source"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
