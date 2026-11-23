"""Exercise Fourth's engine and exact-count cache version gates without loading CUDA."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fourth_asr.runtime.audio_token_counter import AudioTokenCounter
from fourth_asr.runtime.vllm_backend import VLLMBackend


@pytest.mark.parametrize("version", ["0.11.1", "0.11.1+cu129"])
def test_engine_accepts_the_web_release_and_cuda_build(tmp_path, monkeypatch, version):
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(__version__=version, LLM=object, SamplingParams=object))
    monkeypatch.setitem(sys.modules, "vllm.inputs", SimpleNamespace(TokensPrompt=dict))
    monkeypatch.setitem(sys.modules, "vllm.inputs.preprocess", SimpleNamespace(InputPreprocessor=object))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=object))
    with pytest.raises(FileNotFoundError, match="omnicommit_tokens.json"):
        VLLMBackend(tmp_path)


@pytest.mark.parametrize("version", ["0.19.1", "0.11.10", "0.11.1rc1+cu129", "0.11.1.dev1",
                                     "0.11.1.post1", "0.11.1+"])
def test_other_builds_fail_before_private_imports_and_keep_exact_counter(tmp_path, monkeypatch, version):
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(__version__=version))
    with pytest.raises(RuntimeError, match="requires vLLM 0.11.1"):
        VLLMBackend(tmp_path)
    assert AudioTokenCounter.for_preprocessor(None, vllm_version=version).reason == "unverified vLLM version"


def test_cuda_build_uses_same_checked_counter_as_public_release(monkeypatch):
    def processor_instance(module, name, **attributes):
        cls = type(name, (), {"__module__": module})
        obj = cls()
        vars(obj).update(attributes)
        return obj

    feature = processor_instance("transformers.models.whisper.feature_extraction_whisper",
                                 "WhisperFeatureExtractor", sampling_rate=16000, hop_length=160)
    hf = processor_instance("transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe",
                            "Qwen3OmniMoeProcessor", audio_token="<|audio_pad|>", feature_extractor=feature)
    mm = processor_instance("vllm.model_executor.models.qwen3_omni_moe_thinker",
                            "Qwen3OmniMoeThinkerMultiModalProcessor", info=SimpleNamespace(get_hf_processor=lambda: hf))
    proc = SimpleNamespace(_get_mm_processor=lambda: mm,
                           model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3_omni_moe")))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(__version__="4.57.1"))
    for version in ("0.11.1+cu129", "0.11.1"):
        assert AudioTokenCounter.for_preprocessor(proc, vllm_version=version).mode == "length_lru"
    feature.sampling_rate = 48000
    assert AudioTokenCounter.for_preprocessor(proc, vllm_version="0.11.1+cu129").mode == "exact"
