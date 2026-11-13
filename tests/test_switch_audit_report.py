from omnicommit.switch_audit_report import render_review


def sample_case():
    return {
        "case_id": "line-12-switch-1", "sft_line": 12, "audio": "/audio/session.wav", "protocol": "first",
        "kind": "both", "before": {"language": "ja", "speaker": "spk1"},
        "after": {"language": "de", "speaker": "spk2"}, "time_basis": "switch timestamp",
        "anchor_seconds": 58.0, "gold_commit_seconds": 59.0, "gold_turn": 118, "gold_delay_seconds": 1.0,
        "wait_turns_before_commit": 0, "first_commit_characters": 0, "turn_characters": 6,
        "gold_language": "zh", "gold_speaker": "spk2", "gold_attribution_matches": False,
        "window_start_seconds": 57.5, "window_end_seconds": 59, "clip": "clips/日德 (A).wav",
        "turns": [
            {"turn": 117, "event": "audio", "start_seconds": 58, "end_seconds": 58.5, "gold": "<|WAIT|>",
             "models": {"vllm": {"status": "missing"}}},
            {"turn": 118, "event": "audio", "start_seconds": 58.5, "end_seconds": 59,
             "gold": "<|DIRECT|><zh><spk2>Hello.", "models": {
                 "vllm": {"status": "result", "raw_model_text": "<|DIRECT|><ja><spk1> `a | b`\n``` ",
                          "generation_seconds": .12, "queue_wait_seconds": 0}}}
        ],
        "models": {"vllm": {"status": "partial", "exact_audio_turns": 1, "observed_wait_turns": 0,
                            "first_target_tag_seconds": None, "terminal": {"error": "stopped"}}},
    }


def test_review_preserves_literals_in_fences_and_distinguishes_missing_and_wrong_labels():
    report = render_review({"counts": {"rows": 20}}, [sample_case()])
    assert "[播放本案例切换片段](clips/%E6%97%A5%E5%BE%B7%20%28A%29.wav)" in report
    assert '````text\n轮次' in report
    assert '"<|DIRECT|><ja><spk1> `a | b`\\n``` "' in report
    assert '"<|WAIT|>"\t[missing]' in report
    assert "生成 0.12 s · 排队 0 s" in report
    assert "SFT 与参考归属一致  否" in report
    assert "SFT 首次提交轮中归属正确的新文本字符数  0" in report
    assert "SFT 提交轮全部正文字符数  6" in report
    assert '"rows": 20' in report


def test_review_has_no_fabricated_audible_time_or_automatic_speaker_match():
    case = sample_case()
    case.update(kind="speaker", clip=None, anchor_seconds=None, gold_delay_seconds=None)
    report = render_review({}, [case])
    assert "参考切换起点  未知" in report and "SFT 提交时刻减参考切换起点  未知" in report
    assert "首次提交带新语种归属的正文" not in report and "[播放" not in report
    assert "当前选择中没有可展示的切换案例" in render_review({}, [])


def test_untagged_second_turn_keeps_original_and_displays_effective_attribution():
    case = sample_case()
    span = {"language": "de", "speaker": "spk2", "text": " noch."}
    turn = case["turns"][-1]
    turn.update(gold=" noch.", gold_segments=[span], models={"vllm": {
        "status": "result", "raw": " noch.", "segments": [span]}})
    report = render_review({}, [case])
    assert '" noch."\t" noch."' in report
    assert 'SFT 正文归属  <de><spk2>' in report and '"vllm" 正文归属  <de><spk2>' in report


def test_new_span_size_and_correct_tag_can_have_different_evidence_and_timing():
    case = sample_case()
    case.update(event_basis="reference", new_span_characters=6, gold_expected_tag_turn=120,
                gold_expected_tag_seconds=60)
    summary = {"gold_audio_turn_profiles": {"ordinary": {"wait": 2}}}
    report = render_review(summary, [case])
    assert '"ordinary"' in report
    assert "SFT 提交  59 s（第 118 轮）" in report
    assert "首次正确标注正文归属  60 s（第 120 轮）" in report
    case.update(event_basis="sft_only", first_commit_characters=None)
    report = render_review({}, [case])
    assert "新片段的正文字符数（含未核实归属的文字）  6" in report
    assert "首次正确标注正文归属" not in report
