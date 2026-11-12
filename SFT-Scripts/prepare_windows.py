#!/usr/bin/env python3
"""Bound SFT history at safe commit boundaries, with optional causal speaker references.

Input is prepared schema v2 JSONL. References are manually/externally verified,
not inferred from a DIRECT prediction. No future meeting audio is exposed.
"""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import sys

from prepare_sft import number, sort_tokens
from check_sft import check_row, read_tokens
from omnicommit_audio import encode_chunk_ref
from omnicommit.runtime_protocol import speaker_reference_text, speaker_registry_text, last_committed_text, LIVE_AUDIO_PREFIX


def window_boundaries(row, max_seconds):
    if row.get("schema_version") != 2 or row.get("window"):
        raise ValueError("Windowing requires a complete prepared schema_version=2 meeting")
    turns = row["turn_provenance"][:-1]
    result, first = [], 0
    while first < len(turns):
        start = turns[first]["start_ms"]
        within = [i + 1 for i in range(first, len(turns))
                  if turns[i]["end_ms"] - start <= max_seconds * 1000 + 1e-6]
        if within and within[-1] == len(turns):
            result.append((first, len(turns)))
            break
        safe = [end for end in within if turns[end - 1].get("safe_cut")]
        if not safe:
            raise ValueError(f"No safe boundary within {max_seconds}s after {start}ms; increase window budget or review alignment")
        last = max(safe)
        result.append((first, last))
        first = last
    return result


def choose_references(row, candidates, start_ms, previous_speakers, max_reference_seconds, max_speakers):
    if len(previous_speakers) > max_speakers:
        raise ValueError("Prior speaker count exceeds reference budget; do not silently forget known identities")
    selected = {}
    for item in candidates:
        if item.get("verified") is not True or item.get("overlap") is not False or item.get("clean") is not True:
            continue
        if Path(item.get("audio", "")).resolve() != Path(row["source_audio"]).resolve():
            raise ValueError("Reference audio must belong to the same original meeting recording")
        begin = number(item.get("start_ms"), "reference.start_ms")
        end = number(item.get("end_ms"), "reference.end_ms")
        if begin < 0 or end <= begin or end > row["duration_ms"] + 1e-6:
            raise ValueError("Reference interval lies outside the recording")
        if end > start_ms + 1e-6 or end - begin > max_reference_seconds * 1000 or end - begin < 500:
            continue
        speaker = item.get("speaker")
        # Validates identity/transcript without treating it as supervision.
        speaker_reference_text(speaker, item.get("language"), item.get("text"))
        if speaker not in previous_speakers:
            continue
        if speaker not in selected or end - begin > selected[speaker]["end_ms"] - selected[speaker]["start_ms"]:
            selected[speaker] = dict(item)
    missing = previous_speakers - set(selected)
    if missing:
        raise ValueError(f"Window needs causal verified references for prior speakers: {sorted(missing)}")
    return [selected[key] for key in sorted(selected, key=lambda s: int(s[3:]))]


def split_row(row, candidates=(), max_seconds=20., max_reference_seconds=5., max_speakers=8,
              max_total_reference_seconds=30.):
    max_total_reference_seconds = number(max_total_reference_seconds, "max_total_reference_seconds")
    if max_total_reference_seconds <= 0:
        raise ValueError("Total reference audio budget must be positive")
    check_row(row, decode=False)
    bounds = window_boundaries(row, max_seconds)
    previous_speakers, outputs = set(), []
    last_language = last_speaker = None
    for index, (first, last) in enumerate(bounds):
        start = row["turn_provenance"][first]["start_ms"]
        end = row["turn_provenance"][last - 1]["end_ms"]
        final = last == len(row["audios"])
        references = choose_references(row, candidates, start, previous_speakers,
                                       max_reference_seconds, max_speakers)
        reference_seconds = sum((reference["end_ms"] - reference["start_ms"]) / 1000 for reference in references)
        if reference_seconds > max_total_reference_seconds + 1e-6:
            raise ValueError("Causal speaker references exceed total reference audio budget")
        out = deepcopy(row)
        out["id"] = f"{row.get('id')}::window-{index:05d}"
        out["source_id"] = row.get("id")
        out["window"] = {"index": index, "start_ms": start, "end_ms": end,
                         "ends_meeting": final, "safe_start": first == 0 or bool(row["turn_provenance"][first - 1].get("safe_cut")),
                         "safe_end": final or bool(row["turn_provenance"][last - 1].get("safe_cut"))}
        out["messages"] = [deepcopy(row["messages"][0])] + deepcopy(row["messages"][1 + 2 * first:1 + 2 * last])
        out["messages"][0]["content"] += speaker_registry_text(previous_speakers)
        out["messages"][0]["content"] += last_committed_text(last_language, last_speaker)
        out["turn_provenance"] = deepcopy(row["turn_provenance"][first:last])
        if final:
            out["messages"] += deepcopy(row["messages"][-2:])
            out["turn_provenance"].append(deepcopy(row["turn_provenance"][-1]))
        refs, durations, audio_segments, prefix = [], [], [], []
        for ref in references:
            seconds = max(.5, (ref["end_ms"] - ref["start_ms"]) / 1000)
            refs.append(encode_chunk_ref(row["source_audio"], ref["start_ms"] / 1000, ref["end_ms"] / 1000, seconds))
            durations.append(seconds)
            audio_segments.append({"path": row["source_audio"], "start": ref["start_ms"] / 1000,
                                   "end": ref["end_ms"] / 1000, "input_seconds": seconds, "purpose": "speaker_reference"})
            prefix.append(speaker_reference_text(ref["speaker"], ref["language"], ref["text"]))
        out["speaker_references"] = references
        out["audios"] = refs + row["audios"][first:last]
        out["audio_chunk_durations"] = durations + row["audio_chunk_durations"][first:last]
        out["audio_segments"] = audio_segments + row["audio_segments"][first:last]
        if references:
            out["messages"][1]["content"] = "".join(prefix) + LIVE_AUDIO_PREFIX + "<audio>"
        if all(message.get("content") == "<|WAIT|>" for message in out["messages"] if message.get("role") == "assistant"):
            # A silence-only window in a valid, speech-containing meeting is
            # useful WAIT supervision. It must have independent timing evidence,
            # not merely an absence of text predictions or teacher failures.
            spans = row.get("source_speech_spans")
            if (isinstance(spans, list) and spans and out["window"]["safe_start"]
                    and out["window"]["safe_end"]
                    and not any(number(span.get("start_ms"), "speech.start_ms") < end
                                and number(span.get("end_ms"), "speech.end_ms") > start for span in spans)):
                out["window"]["validated_silence"] = True
                out["window"]["parent_has_transcription"] = True
        check_row(out, decode=False)
        outputs.append(out)
        for turn in row["turn_provenance"][first:last]:
            previous_speakers.update(s["speaker"] for s in turn["segments"])
            if turn["action"] == "DIRECT" and turn["segments"]:
                last_language = turn["segments"][-1]["language"]
                last_speaker = turn["segments"][-1]["speaker"]
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speaker-references", type=Path, help="JSONL: {id, references:[{audio,start_ms,end_ms,speaker,language,text,verified:true,clean:true,overlap:false}]}")
    parser.add_argument("--max-audio-seconds", type=float, default=20.)
    parser.add_argument("--max-reference-seconds", type=float, default=5.)
    parser.add_argument("--max-speakers", type=int, default=8)
    parser.add_argument("--max-total-reference-seconds", type=float, default=30.)
    args = parser.parse_args(argv)
    if (not 0 < args.max_reference_seconds <= 8 or args.max_audio_seconds <= 0 or args.max_speakers < 1
            or not 0 < args.max_total_reference_seconds < float("inf")):
        parser.error("Positive window/speaker budgets required; reference cap must be <=8 seconds")
    sidecar = Path(str(args.output) + ".rejected.jsonl")
    token_path = Path(str(args.output) + ".tokens.txt")
    sources = [args.input] + ([args.speaker_references] if args.speaker_references else [])
    all_paths = sources + [args.output, sidecar, token_path]
    if len(set(p.resolve() for p in all_paths)) != len(all_paths):
        parser.error("Input/reference/output/sidecar paths must be distinct")
    if any(p.exists() for p in (args.output, sidecar, token_path)):
        parser.error("Output/sidecars exist; choose a new path")
    reference_map, invalid_reference_ids = {}, set()
    if args.speaker_references:
        with args.speaker_references.open("rb") as handle:
            for ref_line, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                item = None
                try:
                    item = json.loads(line.decode("utf-8-sig"))
                    key = str(item["id"])
                    if key in reference_map or key in invalid_reference_ids:
                        raise ValueError(f"Duplicate reference record: {key}")
                    if not isinstance(item["references"], list):
                        raise ValueError("references must be a list")
                    reference_map[key] = item["references"]
                except Exception as exc:
                    if isinstance(item, dict) and "id" in item:
                        invalid_reference_ids.add(str(item["id"]))
                    print(f"[reference-reject] line={ref_line}: {exc}", file=sys.stderr)
    inventory = read_tokens(Path(str(args.input) + ".tokens.txt"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    kept, rejected = 0, 0
    with args.input.open("rb") as src, args.output.open("x", encoding="utf-8") as dst, sidecar.open("x", encoding="utf-8") as rejects:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line.decode("utf-8-sig"))
                if str(row.get("id")) in invalid_reference_ids:
                    raise ValueError("Speaker-reference record is invalid or duplicated")
                outputs = split_row(row, reference_map.get(str(row.get("id")), ()),
                                    args.max_audio_seconds, args.max_reference_seconds, args.max_speakers,
                                    args.max_total_reference_seconds)
                serialized = [json.dumps(out, ensure_ascii=False, allow_nan=False) for out in outputs]
                for value in serialized:
                    value.encode("utf-8")
                for out in outputs:
                    for ref in out.get("speaker_references", []):
                        inventory.update((f"<{ref['language']}>", f"<{ref['speaker']}>"))
            except Exception as exc:
                rejected += 1
                rejects.write(json.dumps({"line": line_no, "error": str(exc)}, ensure_ascii=False) + "\n")
                rejects.flush()
                continue
            for value in serialized:
                dst.write(value + "\n")
            dst.flush()
            kept += len(outputs)
    token_path.write_text("\n".join(sort_tokens(inventory)) + "\n", encoding="utf-8")
    print(json.dumps({"kept_windows": kept, "rejected_meetings": rejected,
                      "real_template_lengths_checked": False}))
    return 0 if kept else 1


if __name__ == "__main__":
    sys.exit(main())
