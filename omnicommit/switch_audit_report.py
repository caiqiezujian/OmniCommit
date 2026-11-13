"""Render the human listening review for a switch audit. No filesystem access."""
import json
import re
from urllib.parse import quote


def _fence(text, language="text"):
    # Transcripts can themselves contain Markdown fences, pipes and HTML-like tags.
    ticks = "`" * max(3, 1 + max((len(m[0]) for m in re.finditer(r"`+", text)), default=0))
    return f"{ticks}{language}\n{text}\n{ticks}"


def _literal(value):
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))


def _seconds(value):
    if value is None:
        return "未知"
    return f"{value:.6f}".rstrip("0").rstrip(".") + " s"


def _state(state):
    return f"<{state['language']}><{state['speaker']}>" if state else "未知"


def _attributions(segments):
    # Second can inherit its tags; expose the state even on untagged body-only turns.
    return " ".join(_state(segment) for segment in segments)


def _turns(case):
    turns = case["turns"]
    models = list(case.get("models", {}))
    for turn in turns:
        for name in turn.get("models", {}):
            if name not in models:
                models.append(name)
    lines = ["\t".join(["轮次", "事件", "原文件区间", "SFT 目标", *(_literal(name) for name in models)])]
    for turn in turns:
        span = f"{_seconds(turn['start_seconds'])} – {_seconds(turn['end_seconds'])}"
        cells = [str(turn["turn"]), turn["event"], span, _literal(turn["gold"])]
        details = []
        if turn.get("gold_segments"):
            details.append(f"  SFT 正文归属  {_attributions(turn['gold_segments'])}")
        for name in models:
            observed = turn.get("models", {}).get(name, {})
            status = observed.get("status")
            raw = observed.get("raw", observed.get("raw_model_text"))
            if raw is None:
                cells.append(f"[{status or '无结果'}]")
            elif status in (None, "ok", "matched", "result"):
                cells.append(_literal(raw))
            else:
                cells.append(f"[{status}] {_literal(raw)}")
            if observed.get("error"):
                details.append(f"  {_literal(name)} 错误  {_literal(observed['error'])}")
            if observed.get("segments"):
                details.append(f"  {_literal(name)} 正文归属  {_attributions(observed['segments'])}")
            timings = [f"{label} {_seconds(observed[key])}" for key, label in
                       (("generation_seconds", "生成"), ("service_seconds", "整轮"), ("queue_wait_seconds", "排队"))
                       if observed.get(key) is not None]
            if timings:
                details.append(f"  {_literal(name)}  " + " · ".join(timings))
        lines.append("\t".join(cells))
        if turn.get("provenance"):
            details.append(f"  构造记录  {_literal(turn['provenance'])}")
        lines.extend(details)
    return _fence("\n".join(lines))


def _models(case):
    lines = []
    for name, model in case["models"].items():
        lines.append(f"{_literal(name)}  {model.get('status', '未知')}")
        lines.append(f"  对齐的音频轮数  {model.get('exact_audio_turns', '未知')}"
                     f"    观测到的 WAIT 轮数  {model.get('observed_wait_turns', '未知')}")
        if model.get("longest_observed_wait_seconds") is not None:
            lines.append(f"  最长观测连续 WAIT  {_seconds(model['longest_observed_wait_seconds'])}")
        if model.get("direct_turns_by_language"):
            lines.append(f"  提交轮次按有效语种统计  {_literal(model['direct_turns_by_language'])}")
        if model.get("coverage_counts"):
            lines.append(f"  日志覆盖  {_literal(model['coverage_counts'])}")
        if (case["kind"] != "speaker" and case.get("anchor_seconds") is not None
                and case.get("onset_in_window") is not False):
            first_tag = model.get("first_target_tag_seconds")
            if first_tag is None:
                lines.append("  当前日志覆盖范围内未观测到新语种正文归属")
            else:
                lines.append(f"  首次提交带新语种归属的正文  {_seconds(first_tag)}"
                             f"    距参考切换点  {_seconds(model.get('first_target_tag_delay_seconds'))}")
        if model.get("terminal"):
            lines.append(f"  日志结束记录  {_literal(model['terminal'])}")
    return _fence("\n".join(lines))


def render_review(summary, cases):
    """Return a Markdown report whose audio times are on the original source clock."""
    cases = list(cases)
    out = ["# SFT 切换试听审查", f"本报告列出 {len(cases)} 个抽查案例。时间均以原录音起点为零。"]
    if summary.get("counts"):
        out += ["## 数据计数", _fence(json.dumps(summary["counts"], ensure_ascii=False, indent=2), "json")]
    if summary.get("sft_commit_profiles"):
        out += ["## 切换提交和普通提交", "按 SFT 输出标签统计每次提交的字符数，以及提交前连续 WAIT 的轮数。",
                _fence(json.dumps(summary["sft_commit_profiles"], ensure_ascii=False, indent=2), "json")]
    if summary.get("gold_audio_turn_profiles"):
        out += ["## 按原始切换时间对照音频轮次", _fence(json.dumps(summary["gold_audio_turn_profiles"],
                                                         ensure_ascii=False, indent=2), "json")]
    if summary.get("traces"):
        joins = [f"{_literal(name)}  已匹配的 SFT 行数 {trace['joined_sft_rows']}"
                 for name, trace in summary["traces"].items()]
        out += ["日志与原录音匹配情况", _fence("\n".join(joins))]
    if summary.get("issues"):
        out += ["读取或匹配异常按类别计数，逐条记录见 `issues.jsonl`。", _fence(_literal(summary["issues"]))]
    if not cases:
        out.append("当前选择中没有可展示的切换案例。请结合计数检查输入行、切换标注和筛选条件。")
    else:
        out += ["从试听窗口起点开始听。先记下第一次能听清新语种或听出新说话人的原文件时间，"
                "再与 SFT 目标和实际输出对照。标注的切换点与首个对齐单元的结束时间衡量的是不同事件，"
                "人能确认归属的时刻需要试听。",
                "逐轮记录中的引号保留原始回复的首尾空格，换行以 `\\n` 表示。方括号标示日志状态。"
                "只有实际生成的等待回复才代表模型返回了 WAIT。提交延迟以音频秒数计算。"
                "模型首次提交目标语种只核对有效标签，正文正确性按原音试听，新会话的说话人编号需单独核对。"]
    kinds = {"language": "语种切换", "speaker": "说话人切换", "both": "语种和说话人同时切换"}
    for index, case in enumerate(cases, 1):
        kind = kinds.get(case["kind"], case["kind"])
        out += [f"## 案例 {index} · {kind} · SFT 第 {case['sft_line']} 行",
                _fence(f"案例  {case['case_id']}\n原录音  {case['audio']}\n"
                       f"协议  {case['protocol']}\n归属  {_state(case['before'])} → {_state(case['after'])}\n"
                       f"试听窗口  {_seconds(case['window_start_seconds'])} – {_seconds(case['window_end_seconds'])}")]
        if case.get("clip"):
            clip = str(case["clip"]).replace("\\", "/")
            out.append(f"[播放本案例切换片段]({quote(clip, safe='/')})")
        if case.get("reference_text"):
            out += ["参考正文", _fence(case["reference_text"])]
        if case.get("evidence_text") and case["evidence_text"] != case.get("reference_text"):
            out += ["首个有结束时间的参考单元", _fence(case["evidence_text"])]
        facts = [
            f"时间依据  {case['time_basis']}",
            f"参考切换起点  {_seconds(case.get('anchor_seconds'))}",
            f"首个有结束时间的相关参考单元完成  {_seconds(case.get('evidence_seconds'))}",
            f"SFT 提交  {_seconds(case.get('gold_commit_seconds'))}（第 {case['gold_turn']} 轮）"
            if case.get("gold_turn") is not None else "SFT 提交  未观测到",
            f"SFT 提交时刻减参考切换起点  {_seconds(case.get('gold_delay_seconds'))}",
            f"SFT 提交前连续 WAIT 轮数（可含切换前的轮次）  {case['wait_turns_before_commit'] if case.get('wait_turns_before_commit') is not None else '未知'}",
            f"SFT 首次提交轮中归属正确的新文本字符数  {case['first_commit_characters'] if case.get('first_commit_characters') is not None else '未知'}",
            f"SFT 提交轮全部正文字符数  {case['turn_characters'] if case.get('turn_characters') is not None else '未知'}",
        ]
        if case.get("new_span_characters") is not None:
            facts.append(f"SFT 首次提交轮中新片段的正文字符数（含未核实归属的文字）  {case['new_span_characters']}")
        if case.get("event_basis") == "reference":
            expected_turn = case.get("gold_expected_tag_turn")
            result = (f"{_seconds(case['gold_expected_tag_seconds'])}（第 {expected_turn} 轮）"
                      if expected_turn is not None else "该参考片段内未观测到")
            facts.append(f"SFT 在新参考片段内首次正确标注正文归属  {result}")
        if case.get("gold_language") and case.get("gold_speaker"):
            facts.append(f"SFT 实际归属  <{case['gold_language']}><{case['gold_speaker']}>")
        if case.get("gold_attribution_matches") is not None:
            facts.append(f"SFT 与参考归属一致  {'是' if case['gold_attribution_matches'] else '否'}")
        out.append(_fence("\n".join(facts)))
        if case.get("models"):
            out += ["推理日志观测", _models(case)]
        if case.get("warnings"):
            out += ["核对提示", _fence("\n".join(str(item) for item in case["warnings"]))]
        out += ["逐轮对照", _turns(case),
                "试听记录（原文件秒）　可确认新归属 ____　首次正确新增正文 ____　首次正确归属且正文正确 ____　"
                "漏词、重复或录音中听不到的文字 ____"]
    return "\n\n".join(out) + "\n"
