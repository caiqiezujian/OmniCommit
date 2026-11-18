"""Check the recorded primary, retry and recovery decodes of one stream turn."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fourth_asr.debug_backend import DebugBackend
from fourth_asr.protocol import DIRECT, WAIT
from fourth_asr.runtime.streaming import GenerationTruncated
from fourth_asr.service_session import FourthStreamOptions, FourthStreamSession


def test_retry_and_forced_decode_share_source_turn_and_preserve_backend_inputs():
    records = []

    def record(event, **fields):
        records.append(json.loads(json.dumps(dict(event=event, **fields))))

    class AllVoiced:
        def observe(self, audio):
            return 0, len(audio)

    class Engine:
        backend_name = "test"
        replies = iter([DIRECT + "One.", GenerationTruncated("test", 8, 12, 8, 20), WAIT])

        def generate(self, messages, audios):
            reply = next(self.replies)
            if isinstance(reply, Exception):
                raise reply
            return {"text": reply, "prompt_tokens": 12, "num_cached_tokens": 1}

        def generate_with_prefix(self, messages, audios, prefix):
            return prefix + " Two."

    options = FourthStreamOptions(protocol="explicit", chunk_seconds=.5, recovery_voiced_seconds=.5,
                                 recovery_overlap_seconds=0)
    engine = Engine()
    session = FourthStreamSession(engine, options, activity=AllVoiced())
    pcm = np.full(8000, .1, dtype=np.float32)
    session.step(pcm)
    session.backend = DebugBackend(engine, record)
    result = session.step(pcm)
    assert result["primary_text"] == WAIT and result["text"] == DIRECT + " Two."
    assert [r["event"] for r in records] == ["model_call", "model_error", "model_call", "model_reply",
                                              "model_call", "model_reply"]
    calls = records[::2]
    assert [r["kind"] for r in calls] == ["primary", "retry", "recovery"]
    assert len(set(r["call_id"] for r in calls)) == 3
    assert [r["call_id"] for r in calls] == [r["call_id"] for r in records[1::2]]
    assert [r["audio_samples"] for r in calls] == [[8000, 8000], [8000], [8000]]
    assert [r["prefix"] for r in calls] == ["", "", DIRECT]
    assert calls[0]["messages"][-2] == {"role": "assistant", "content": DIRECT + "One."}
    assert len(calls[1]["messages"]) == 2
    assert records[1]["finish_reason"] == "length" and records[1]["generated_tokens"] == 8
    assert records[3]["raw_text"] == WAIT and records[3]["num_cached_tokens"] == 1
    assert records[5]["raw_text"] == DIRECT + " Two."


def test_debug_writer_failure_does_not_swallow_model_result_or_replace_its_error():
    def broken(event, **fields):
        raise OSError("write failed")

    class Engine:
        reply = {"text": "malformed model reply"}

        def generate(self, messages, audios):
            return self.reply

        def generate_with_prefix(self, messages, audios, prefix):
            raise ValueError("backend error")

    backend = DebugBackend(Engine(), broken)
    assert backend.generate([], []) is Engine.reply
    with pytest.raises(ValueError, match="backend error"):
        backend.generate_with_prefix([], [], DIRECT)
