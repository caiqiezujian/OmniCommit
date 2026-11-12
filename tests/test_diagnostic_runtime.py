import argparse
import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

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


@pytest.mark.parametrize("protocol", ["first", "second"])
@pytest.mark.parametrize(("flags", "enabled", "mm_gb"), [
    ([], True, 1), (["--prefix-caching", "--mm-processor-cache-gb", "0"], True, 0),
    (["--no-prefix-caching"], False, 1),
])
def test_cli_prefix_cache_switch_reaches_native_vllm(tmp_path, monkeypatch, protocol, flags, enabled, mm_gb):
    from omnicommit.vllm_backend import AUDIO_PLACEHOLDER

    model = checkpoint(tmp_path / "merged", protocol)
    path = model / "omnicommit_tokens.json"
    tokens = [WAIT, "<ja>", "<de>", "<spk1>"]
    if protocol == "first":
        tokens.insert(1, "<|DIRECT|>")
    special = [*tokens, "<|im_end|>", "<|endoftext|>"]
    ids = {text: i for i, text in enumerate(special)}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(tokens=tokens, token_ids={text: ids[text] for text in tokens}, tokenizer_length=len(special))
    path.write_text(json.dumps(manifest), encoding="utf-8")
    tokenizer = MagicMock(all_special_tokens=special)
    tokenizer.__len__.return_value = len(special)
    tokenizer.encode.side_effect = lambda text, **kwargs: [ids.get(text, 0)]
    tokenizer.apply_chat_template.return_value = (
        f"<|im_start|>user\n{AUDIO_PLACEHOLDER}<|im_end|>\n<|im_start|>assistant\n")
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=Mock(from_pretrained=Mock(return_value=tokenizer))))

    # The effective engine settings can differ from the requested settings.
    # Force concrete values so this also checks the source of the trace metadata.
    config = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        model_config=SimpleNamespace(multimodal_config=SimpleNamespace(mm_processor_cache_gb=0.25)))
    engine = SimpleNamespace(vllm_config=config, model_config=config.model_config)
    native_llm = Mock(return_value=SimpleNamespace(llm_engine=engine))
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(
        __version__="0.11.1", LLM=native_llm, SamplingParams=dict))
    monkeypatch.setitem(sys.modules, "vllm.inputs", SimpleNamespace(TokensPrompt=dict))
    monkeypatch.setitem(sys.modules, "vllm.inputs.preprocess", SimpleNamespace(InputPreprocessor=Mock()))
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    args = arguments("--model", str(model), "--chunk-ms", "500", *flags)
    detected, options, _, loader = runtime.resolve_runtime(args)
    backend = loader()
    assert detected == protocol and options.chunk_seconds == .5
    assert native_llm.call_args.kwargs["enable_prefix_caching"] is enabled
    assert native_llm.call_args.kwargs["mm_processor_cache_gb"] == mm_gb
    assert backend.enable_prefix_caching is False
    assert backend.mm_processor_cache_gb == .25


def test_hf_constructor_does_not_receive_vllm_cache_settings(tmp_path, monkeypatch):
    model = checkpoint(tmp_path / "merged", "first")
    hf = Mock()
    monkeypatch.setattr("omnicommit.inference_backend.HFBackend", hf)
    _, _, _, loader = runtime.resolve_runtime(arguments("--backend", "hf", "--model", str(model)))
    assert loader() is hf.return_value
    assert "enable_prefix_caching" not in hf.call_args.kwargs
    assert "mm_processor_cache_gb" not in hf.call_args.kwargs
