"""Actual waveform collection for auxiliary same-language speaker prefixes."""
import json

import numpy as np
import pytest
import soundfile as sf

from omnicommit.prefix_collection import LOCAL_QUERY_POLICY, parser, run
from omnicommit.prefix_source import plan_queries, plan_span_queries


def job(tmp_path, chunk=.5):
    audio = tmp_path / "meeting.wav"
    # Distinct sample ranges make accidental inclusion of an earlier speaker
    # observable at the backend boundary; no ASR quality is simulated.
    pcm = np.zeros(40000, dtype=np.float32)
    pcm[:19200] = .1
    pcm[19200:22400] = .2
    pcm[22400:] = .3
    sf.write(audio, pcm, 16000, subtype="FLOAT")
    record = {"id": "same-language", "audios": [str(audio)], "messages": [
        {"role": "user", "content": "<audio>**Entity Reference:** [DoNotLeak]"},
        {"role": "assistant", "content": "<en><spk1>AAA<spk2>B<spk1>CCC"}],
        "timestamps": [{"time_ms": 1200}, {"time_ms": 1400}]}
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    return parser().parse_args([str(source), "--model", str(model), "--output", str(tmp_path / "trace.jsonl"),
                               "--batch-size", "3", "--chunk-seconds", str(chunk)])


def teacher_type(*, fail_b=False, interrupt_on=None):
    calls = []

    class Teacher:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe_batch(self, waves, languages):
            assert languages == ["en"] * len(waves)
            calls.append([wave.copy() for wave in waves])
            if interrupt_on and len(calls) == interrupt_on:
                raise KeyboardInterrupt()
            results = []
            for wave in waves:
                is_b = np.allclose(wave, .2)
                results.append({"text": "B" if is_b else "raw-local-or-cumulative",
                                "token_ids": [], "prompt_tokens": 10,
                                "error": "Generation truncated" if fail_b and is_b else None})
            return results

    return Teacher, calls


@pytest.mark.parametrize("chunk", [.5, .8, 1.])
def test_collector_preserves_primary_queries_and_adds_independent_local_audio(tmp_path, chunk):
    args = job(tmp_path, chunk)
    teacher, calls = teacher_type()
    summary = run(args, teacher_factory=teacher)
    trace = json.loads(args.output.read_text(encoding="utf-8"))
    assert trace["status"] == "ok"
    assert trace["teacher"]["local_query_policy"] == LOCAL_QUERY_POLICY
    source = trace["source"]
    primary = plan_queries(source, 40000, 16000, chunk)
    local = plan_span_queries(source, 40000, 16000, chunk)
    assert len(trace["queries"]) == len(primary)
    assert len(trace["local_queries"]) == len(local)
    assert summary["queries_inferred"] == len(primary) + len(local)
    assert [q["query_id"] for q in trace["queries"]] == [q["query_id"] for q in primary]
    assert [q["query_id"] for q in trace["local_queries"]] == [q["query_id"] for q in local]
    b = next(q for q in trace["local_queries"] if q["span_index"] == 1)
    assert b["start_sample"] == 19200 and b["end_sample"] == 22400 and b["text"] == "B"
    waves = [wave for batch in calls for wave in batch]
    local_b, = [wave for wave in waves if np.allclose(wave, .2)]
    assert len(local_b) == 3200  # no A or C samples enter the B request
    assert any(np.allclose(wave, .3) for wave in waves)
    args.resume = True

    def forbidden(*args, **kwargs):
        raise AssertionError("completed rows must not reload a model")

    assert run(args, teacher_factory=forbidden)["resumed_ok"] == 1


def test_local_query_failure_remains_an_error_not_fabricated_wait(tmp_path):
    args = job(tmp_path)
    teacher, _ = teacher_type(fail_b=True)
    summary = run(args, teacher_factory=teacher)
    trace = json.loads(args.output.read_text(encoding="utf-8"))
    assert summary["failed"] == 1 and trace["status"] == "error" and trace["output"] == ""
    assert not any(q.get("error") for q in trace["queries"])
    failures = [q for q in trace["local_queries"] if q.get("error")]
    assert len(failures) == 1 and failures[0]["span_index"] == 1
    assert "Generation truncated" in trace["error"]


def test_auxiliary_queries_are_cached_even_if_current_row_is_interrupted(tmp_path):
    args = job(tmp_path)
    interrupted, first_calls = teacher_type(interrupt_on=2)
    with pytest.raises(KeyboardInterrupt):
        run(args, teacher_factory=interrupted)
    assert len(first_calls) == 2
    args.resume = True
    teacher, calls = teacher_type()
    summary = run(args, teacher_factory=teacher)
    assert summary["query_cache_hits"] == 3
    trace = json.loads(args.output.read_text(encoding="utf-8"))
    assert summary["queries_inferred"] == len(trace["queries"]) + len(trace["local_queries"]) - 3
    assert trace["local_queries"]
    # The first completed batch contains the shortest B query and is not rerun.
    assert not any(np.allclose(wave, .2) for batch in calls for wave in batch)
