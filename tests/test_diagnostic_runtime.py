import argparse
import json

import numpy as np
import pytest

from omnicommit import diagnostic_runtime as runtime
from omnicommit.streaming import WAIT


def arguments(*argv):
    parser = argparse.ArgumentParser()
    runtime.add_diagnostic_arguments(parser)
    return parser.parse_args(list(argv))


def checkpoint(path, protocol):
    path.mkdir()
    if protocol == "first":
        metadata = {"schema_version": 1, "protocol": {"direct": "<|DIRECT|>", "wait": WAIT,
                                                     "end_of_audio": "<end_of_audio>"}}
    else:
        metadata = {"schema_version": 2, "protocol": {"name": "omnicommit_second",
                                                     "wait": WAIT, "end_of_audio": "<end_of_audio>"}}
    (path / "omnicommit_tokens.json").write_text(json.dumps(metadata), encoding="utf-8")
    return path


def test_model_selection_defaults_precedence_and_lazy_load(tmp_path, monkeypatch):
    monkeypatch.delenv("OMNICOMMIT_MODEL", raising=False)
    monkeypatch.setattr(runtime, "MODELS_DIR", tmp_path)
    with pytest.raises(ValueError, match="--model"):
        runtime.resolve_runtime(arguments())
    first = checkpoint(tmp_path / "omnicommit-merged", "first")
    args = arguments()
    protocol, options, session, make_backend = runtime.resolve_runtime(args)
    assert protocol == "first" and session.__name__ == "StreamSession"
    assert args.model == first and args.backend == "vllm"
    assert options.context_policy == "rolling" and options.number_format == "normalized"
    assert options.chunk_seconds == .5 and options.max_pending_seconds == 30

    second = checkpoint(tmp_path / "second-merged", "second")
    with pytest.raises(ValueError, match="Multiple merged"):
        runtime.resolve_runtime(arguments())
    args = arguments("--protocol", "second", "--chunk-ms", "200")
    protocol, options, session, make_backend = runtime.resolve_runtime(args)
    assert args.model == second and protocol == "second" and options.chunk_seconds == .2
    assert session.__name__ == "SecondStreamSession"
    monkeypatch.setenv("OMNICOMMIT_MODEL", str(first))
    assert runtime.resolve_runtime(arguments())[0] == "first"
    assert runtime.resolve_runtime(arguments("--model", str(second)))[0] == "second"
    with pytest.raises(ValueError, match="conflicting"):
        runtime.resolve_runtime(arguments("--protocol", "second"))


@pytest.mark.parametrize("bad", [None, {}, {"schema_version": 3}, {"schema_version": True},
                                 {"schema_version": 1, "protocol": {"name": "wrong"}}])
def test_explicit_checkpoint_requires_a_recognized_export_manifest(tmp_path, bad):
    if bad is not None:
        (tmp_path / "omnicommit_tokens.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="omnicommit_tokens.json"):
        runtime.resolve_runtime(arguments("--model", str(tmp_path)))


def test_first_manifest_without_protocol_block_like_web_loader(tmp_path):
    manifest = {"schema_version": 1, "tokens": ["<|WAIT|>", "<|DIRECT|>", "<en>", "<spk1>"]}
    (tmp_path / "omnicommit_tokens.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert runtime.resolve_runtime(arguments("--model", str(tmp_path)))[0] == "first"


@pytest.mark.parametrize("protocol", ["auto", "second"])
def test_replay_uses_real_session_protocol_without_a_model(tmp_path, protocol, monkeypatch):
    monkeypatch.setenv("OMNICOMMIT_MODEL", str(tmp_path / "unused"))
    text = "<en><spk1>hello"
    reply = "<|DIRECT|>" + text if protocol == "auto" else text
    replies = tmp_path / "replies.json"
    replies.write_text(json.dumps([reply, WAIT]), encoding="utf-8")
    args = arguments("--protocol", protocol, "--backend", "replay", "--replies", str(replies), "--chunk-ms", "333")
    name, options, factory, load = runtime.resolve_runtime(args)
    session = factory(load(), options)
    try:
        result = session.step(np.zeros(333 * 16, dtype=np.float32), final_chunk=True)
        assert result["text"] == reply and result["end_seconds"] == .333
        assert session.step(end_of_audio=True)["text"] == WAIT
        assert name == ("first" if protocol == "auto" else "second")
        assert args.model is None
    finally:
        session.close()


@pytest.mark.parametrize("value", ["0", "-1", "8001", ".5", "200.5"])
def test_invalid_chunk_ms(value):
    with pytest.raises(SystemExit):
        arguments("--chunk-ms", value)
