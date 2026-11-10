"""Independent source-order/causality checks across variable chunk phases."""
from copy import deepcopy
import random

import pytest

from omnicommit.prefix_labels import build_trajectory
from omnicommit.prefix_source import parse_source, plan_queries, plan_span_queries


@pytest.mark.parametrize("chunk_seconds", [.5, .8, 1.])
def test_empty_teacher_fallback_is_ordered_closed_and_past_onset_gate(chunk_seconds):
    rng = random.Random(307)
    for _ in range(35):
        duration = rng.randrange(1600, 5100)
        switch_count = rng.randrange(1, 8)
        switches = sorted(rng.sample(range(1, duration), switch_count))
        # Repeated languages exercise same-language speaker changes, rapid
        # switches, returning identities/languages and multiple events per chunk.
        spans, previous = [], None
        for i in range(switch_count + 1):
            choices = [(lang, speaker) for lang in ("zh", "en", "fr")
                       for speaker in ("spk1", "spk2") if (lang, speaker) != previous]
            language, speaker = rng.choice(choices)
            spans.append({"language": language, "speaker": speaker, "text": f"part{i} "})
            previous = language, speaker
        content = "".join(f"<{s['language']}><{s['speaker']}>{s['text']}" for s in spans)
        source = parse_source({"audios": ["/data/test.wav"],
                               "messages": [{"role": "assistant", "content": content}],
                               "timestamps": [{"time_ms": t} for t in switches]}, duration)
        primary = plan_queries(source, duration, 1000, chunk_seconds)
        local = plan_span_queries(source, duration, 1000, chunk_seconds)
        for query in primary + local:
            query.update(text="", token_ids=[], error=None)
        trace = {"kind": "local_prefix_trace", "status": "ok", "id": "random-review",
                 "audio": "/data/test.wav", "source": source,
                 "audio_info": {"sample_rate": 1000, "frames": duration},
                 "chunk_seconds": chunk_seconds, "number_format": "normalized", "input_hash": "review",
                 "queries": primary, "local_queries": local}
        before = deepcopy(trace)
        result = build_trajectory(trace)
        assert trace == before  # Reference fallback never edits raw teacher evidence.
        assert "".join(s["text"] for turn in result["turns"] for s in turn["segments"]) == source["text"]
        events = result["provenance"]["reference_fallback_events"]
        assert sum(e["char_end"] - e["char_start"] for e in events) == len(source["text"])
        assert result["metrics"]["reference_fallback_characters"] == len(source["text"])
        previous_end = 0
        for event in events:
            assert event["char_start"] == previous_end
            previous_end = event["char_end"]
            assert event["emitted_at_ms"] >= event["span_end_ms"]
            if event["event"] == "audio" and event["span_index"]:
                turn = result["turns"][event["turn_index"]]
                # The segment's onset chunk must have ended already, even when
                # several new language/speaker spans fit in that same chunk.
                assert event["span_start_ms"] < turn["start_ms"]
        assert previous_end == len(source["text"])
