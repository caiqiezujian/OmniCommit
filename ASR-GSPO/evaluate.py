"""Evaluate full Third trajectories from a local merged model or saved candidate traces."""
import argparse
from collections import Counter, defaultdict
import io
import itertools
import json
import math
from pathlib import Path

from asr_gspo.reward import score_trajectory
from asr_gspo.protocol import EOS, parse_reply


def read_jsonl(path):
    with path.open(encoding="utf-8-sig") as stream:
        for n, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except ValueError as exc:
                    raise ValueError(f"{path} line {n}: {exc}") from exc


def model_turns(backend, row, task):
    import soundfile as sf
    from asr_gspo.audio import encode_chunk_ref, materialize_chunk_refs
    from third_asr.runtime.streaming import ContextBudgetError, GenerationTruncated

    messages = [dict(row["messages"][0])]
    audios, turns = [], []
    for ix in range(len(task["chunks"]) + 1):
        eof = ix == len(task["chunks"])
        if not eof:
            wav, = materialize_chunk_refs([encode_chunk_ref(task, ix)])
            pcm, rate = sf.read(io.BytesIO(wav), dtype="float32")
            if rate != 16000 or pcm.ndim != 1:
                raise ValueError("Expected mono 16 kHz chunk")
            audios.append(pcm)
        messages.append({"role": "user", "content": EOS if eof else "<audio>"})
        end = task["chunks"][-1 if eof else ix]["end_seconds"]
        try:
            response = backend.generate(messages, audios)["text"]
            reason = "stop"
        except GenerationTruncated:
            response, reason = "", "length"
        except ContextBudgetError:
            response, reason = "", "context_limit"
        turns.append({"kind": "eof" if eof else "audio", "audio_end_seconds": end,
                      "response": response, "finish_reason": reason})
        messages.append({"role": "assistant", "content": response})
        if reason != "stop":
            break
        try:
            parse_reply(response)
        except ValueError:
            break
    return turns


def eval_cases(args):
    if args.command == "replay":
        # Keep references for the selected trace only, even when scoring a few
        # rollouts against a training manifest with millions of audio records.
        def select():
            return itertools.islice(read_jsonl(args.traces), args.limit)
        def key(record):
            return record.get("id", record.get("task_id"))
        needed = {key(candidate) for candidate in select()}
        tasks = {}
        for row in read_jsonl(args.dataset):
            task = json.loads(row["asr_task"])
            if task["id"] not in needed:
                continue
            if task["id"] in tasks:
                raise ValueError(f"Duplicate evaluation ID: {task['id']}")
            tasks[task["id"]] = task
        for candidate in select():
            task_id = key(candidate)
            if task_id not in tasks:
                raise ValueError(f"Trace ID {task_id!r} is absent in the supplied manifest")
            yield tasks[task_id], candidate["turns"]
        return
    from third_asr.service_backend import build_backend, detect_model_protocol
    if detect_model_protocol(args.model) != "explicit":
        raise ValueError("Expected a merged explicit Third checkpoint")
    # vLLM profiles these audiovisual limits once when creating the engine.
    max_rounds, max_input = 0, 0.
    for row in itertools.islice(read_jsonl(args.dataset), args.limit):
        chunks = json.loads(row["asr_task"])["chunks"]
        max_rounds = max(max_rounds, len(chunks))
        max_input = max(max_input, max(c["input_seconds"] for c in chunks))
    if not max_rounds:
        raise ValueError("No evaluation tasks")
    args.chunk_seconds = max_input
    args.max_window_seconds = max_rounds * max_input
    backend = build_backend(args)
    for row in itertools.islice(read_jsonl(args.dataset), args.limit):
        task = json.loads(row["asr_task"])
        yield task, model_turns(backend, row, task)


def percentile(histogram, quantile):
    count = sum(histogram.values())
    if not count:
        return None
    remaining = math.ceil(count * quantile)
    for milliseconds, n in sorted(histogram.items()):
        remaining -= n
        if remaining <= 0:
            return milliseconds / 1000


def summarize(group, waits, switch_waits):
    def ratio(a, b):
        return group[a] / group[b] if group[b] else None
    return {"cases": group["cases"], "invalid": group["invalid"],
            "mean_reward": ratio("reward", "cases"),
            "valid_mixed_unit_error_rate": ratio("errors", "reference_units"),
            "valid_reference_units": group["reference_units"],
            "switches": group["switch_segments"],
            "mean_valid_switch_head_error": ratio("switch_errors_sum", "switch_segments"),
            "switch_content_spans": group["switch_content_spans"],
            "switch_content_fully_missed": group["switch_content_fully_missed"],
            "script_mismatch_runs": group["script_mismatch_runs"],
            "mean_switch_content_error_on_scored_trajectories": ratio("switch_content_error_sum", "switch_scored_trajectories"),
            "short_switch_deletions": group["short_switch_deletions"],
            "punctuation_scored_trajectories": group["punctuation_scorable"],
            "scored_internal_gaps": group["scored_internal_gaps"],
            "extra_internal_periods": group["extra_internal_periods"],
            "reference_sentence_stops": group["reference_sentence_stops"],
            "missing_sentence_stops": group["missing_sentence_stops"],
            "short_switch_segments": group["short_switch_segments"], "timed_matches": group["timed_matches"],
            "timed_units": group["timed_units"], "early_units": group["early_units"],
            "p95_extra_wait_seconds": percentile(waits, .95),
            "p95_switch_head_extra_wait_seconds": percentile(switch_waits, .95),
            "switch_head_delay_samples": sum(switch_waits.values()),
            "mean_valid_matched_extra_wait_seconds": ratio("sum_delay", "delay_weight")}


def evaluate(args):
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    groups = defaultdict(Counter)
    waits = defaultdict(Counter)
    switch_waits = defaultdict(Counter)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Keep completed trajectories if a long GPU run is interrupted.
    summary_file = args.output.with_suffix(args.output.suffix + ".summary.json")
    if summary_file.exists():
        raise FileExistsError(summary_file)
    with args.output.open("x", encoding="utf-8") as output:
        for task, turns in eval_cases(args):
            result = score_trajectory(task, turns, mode=args.reward_mode)
            record = {"id": task["id"], "bucket": task["bucket"], "mode": args.reward_mode,
                      "turns": turns, "score": result}
            output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            output.flush()
            for key in ("all", task["bucket"]):
                group = groups[key]
                group.update(cases=1, reward=result["reward"], invalid=int(not result["valid"]))
                if result["valid"]:
                    waits[key].update(result["extra_wait_ms_histogram"])
                    switch_waits[key].update(result["switch_extra_wait_ms_histogram"])
                    group.update({name: result[name] for name in (
                        "errors", "reference_units", "short_switch_deletions", "short_switch_segments",
                        "switch_segments", "timed_units", "timed_matches", "early_units",
                        "switch_content_spans", "switch_content_fully_missed", "script_mismatch_runs", "punctuation_scorable",
                        "scored_internal_gaps", "extra_internal_periods", "reference_sentence_stops",
                        "missing_sentence_stops")})
                    group["switch_errors_sum"] += result["switch_error_rate"] * result["switch_segments"]
                    if result["switch_content_spans"]:
                        group.update(switch_scored_trajectories=1, switch_content_error_sum=result["switch_content_error"])
                    if result["mean_extra_wait_seconds"] is not None:
                        group.update(sum_delay=result["mean_extra_wait_seconds"] * result["delay_samples"],
                                     delay_weight=result["delay_samples"])
    if not groups:
        raise ValueError("No candidates evaluated")
    report = {key: summarize(group, waits[key], switch_waits[key]) for key, group in groups.items()}
    summary_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("model", "replay"):
        parser = sub.add_parser(name)
        parser.add_argument("dataset", type=Path, help="Prepared evaluation JSONL")
        parser.add_argument("--output", type=Path, required=True, help="New file for traces and per-example scores")
        parser.add_argument("--reward-mode", choices=("quality", "quality_latency"), default="quality_latency")
        parser.add_argument("--limit", type=int, help="Use the same first N evaluation rows for all checkpoints")
        if name == "model":
            parser.add_argument("--model", type=Path, required=True)
            parser.add_argument("--backend", choices=("hf", "vllm"), default="hf")
            parser.add_argument("--max-context-tokens", type=int, default=8320)
            parser.add_argument("--max-new-tokens", type=int, default=128)
            parser.add_argument("--tensor-parallel-size", type=int, default=1)
            parser.add_argument("--gpu-memory-utilization", type=float, default=.85)
            parser.add_argument("--attention", choices=("sdpa", "flash_attention_2", "eager"), default="sdpa")
            parser.set_defaults(protocol="explicit", context_policy="window", enable_recovery=False,
                                device_map="auto", dtype="auto", max_reference_seconds=.5,
                                enable_prefix_caching=True, mm_processor_cache_gb=0, vllm_image_slots=1)
        else:
            parser.add_argument("traces", type=Path, help='JSONL records with id and list of turns')
    args = p.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
