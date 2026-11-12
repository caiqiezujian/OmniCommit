import importlib.util
import json
from pathlib import Path
import sys
import wave

import pytest

SECOND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SECOND))
sys.path.insert(0, str(SECOND / "SFT-Scripts"))
from omnicommit_audio import encode_chunk_ref
from second_asr.protocol import EOS, WAIT, parse_reply, render_reply, sort_tokens, system_prompt

spec = importlib.util.spec_from_file_location("second_sft_check", SECOND / "SFT-Scripts" / "check_sft.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_sparse_round_trip_across_multiple_changes_and_waits():
    turns = [
        [],
        [("zh", "spk4", "请联系")],
        [],
        [("zh", "spk4", "张伟，"), ("zh", "spk4", "预算是300元。"),
         ("en", "spk4", " OK?"), ("en", "spk17", " Sure. ")],
        [],
        [("zh", "spk4", "\n谢谢。")],
    ]
    expected = [WAIT, "<zh><spk4>请联系", WAIT,
                "张伟，预算是300元。<en><spk4> OK?<en><spk17> Sure. ",
                WAIT, "<zh><spk4>\n谢谢。"]
    render_state = parse_state = None
    for turn, target in zip(turns, expected):
        reply, render_state = render_reply(turn, render_state)
        recovered, parse_state = parse_reply(reply, parse_state)
        assert reply == target
        assert parse_state == render_state
        assert [(lang, speaker, ch) for lang, speaker, text in recovered for ch in text] == [
            (lang, speaker, ch) for lang, speaker, text in turn for ch in text]


@pytest.mark.parametrize("reply", [
    "", "hello", " <zh><spk1>说话", "<spk3>hello", "<ar>hello", "<en><spk0>hi",
    "<en><spk1> ", "<en><spk1>hello<|WAIT|>", "<|WAIT|>hello",
    "<|DIRECT|><zh><spk1>请联系", "<en><spk1>A<en><spk1>B",
    "<en><spk1>one<zh><spk1>", "<en><spk1>one > two", EOS,
])
def test_invalid_or_legacy_replies(reply):
    with pytest.raises(ValueError):
        parse_reply(reply)


def test_tokens_do_not_add_direct_or_eos():
    assert sort_tokens(["<spk12>", "<en>", "<spk2>", "<de>", "<en>"]) == [
        WAIT, "<de>", "<en>", "<spk2>", "<spk12>"]
    for forbidden in ["<|DIRECT|>", EOS, "<spk0>"]:
        with pytest.raises(ValueError):
            sort_tokens([forbidden])


def test_transition_with_separator_owned_by_previous_speaker():
    old = ("zh", "spk1")
    spans = [(*old, " "), ("en", "spk2", "OK")]
    reply, state = render_reply(spans, old)
    assert reply == " <en><spk2>OK"
    assert parse_reply(reply, old) == (spans, state)
    with pytest.raises(ValueError, match="whitespace"):
        render_reply([(*old, " ")], old)


def sample(path):
    messages = [{"role": "system", "content": system_prompt("normalized")}]
    for user, answer in [("<audio>", WAIT), ("<audio>", "<en><spk12> Sue, "), (EOS, WAIT)]:
        messages.extend([{"role": "user", "content": user}, {"role": "assistant", "content": answer}])
    return {"audios": [encode_chunk_ref(path, 0, 0.5, 0.5), encode_chunk_ref(path, 0.5, 0.65, 0.5)],
            "messages": messages}


def test_audit_fresh_state_and_token_coverage(tmp_path):
    row = sample(tmp_path / "audio.wav")
    assert audit.validate_row(row) == {WAIT, "<en>", "<spk12>"}
    with pytest.raises(ValueError, match="missing"):
        audit.validate_row(row, {WAIT, "<en>"})
    row["messages"][4]["content"] = " Sue, "
    with pytest.raises(ValueError, match="First text"):
        audit.validate_row(row)
    row["messages"][4]["content"] = WAIT
    with pytest.raises(ValueError, match="no transcribed speech"):
        audit.validate_row(row)


def test_audit_validates_audio_and_eos(tmp_path):
    pytest.importorskip("soundfile")
    path = tmp_path / "speech.wav"
    with wave.open(str(path), "wb") as w:
        w.setparams((1, 2, 16000, 0, "NONE", ""))
        w.writeframes(b"\x00\x00" * 10400)
    row = sample(path)
    audit.validate_row(row, decode=True)
    row["audios"][1] = encode_chunk_ref(path, 0.5, 0.6, 0.5)
    with pytest.raises(ValueError, match="recording end"):
        audit.validate_row(row, decode=True)
    row["audios"][1] = encode_chunk_ref(path, 0.55, 0.65, 0.5)
    with pytest.raises(ValueError, match="contiguously"):
        audit.validate_row(row)
    row["messages"][-1]["content"] = "unexpected"
    with pytest.raises(ValueError, match="end_of_audio"):
        audit.validate_row(row)


def test_cli_checks_manifest_and_counts_partial_scans(tmp_path, capsys):
    data = tmp_path / "train.jsonl"
    tokens = tmp_path / "train.jsonl.tokens.txt"
    good = json.dumps(sample(tmp_path / "audio.wav")) + "\n"
    data.write_text(good * 2, encoding="utf-8")
    tokens.write_text(f"{WAIT}\n<en>\n<spk12>\n", encoding="utf-8")
    assert audit.main([str(data), "--rows", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["full_dataset_checked"] is False
    assert audit.main([str(data)]) == 0
    assert json.loads(capsys.readouterr().out)["checked_rows"] == 2
    data.write_text(good + "{}\n", encoding="utf-8")
    assert audit.main([str(data)]) == 1
    assert "line 2" in capsys.readouterr().err
    tokens.write_text(f"{WAIT}\n<|DIRECT|>\n", encoding="utf-8")
    assert audit.main([str(data)]) == 1
    assert "Unexpected training token" in capsys.readouterr().err
