"""Read a Third events.jsonl and print the evidence around one audio turn."""

import argparse
import json
from pathlib import Path


def inspect(path, session, focus):
    turns = {}
    packets = {}
    packet_audio = []
    calls = []
    history = None
    options = {}
    malformed = 0
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except ValueError:
                malformed += 1
                continue
            if e.get("session_id") != session:
                continue
            event = e.get("event")
            request = e.get("request_id")
            if event == "turn_finished":
                result = e["turn_result"]
                if result.get("event") == "audio":
                    turns[e["turn"]] = (result, request)
            elif event == "request_finished":
                packets[request] = e
            elif event == "packet_audio":
                packet_audio.append((request, e["start_sample"], e["end_sample"], e["sample_rate"]))
            elif event == "session_started":
                options = e.get("options", {})
            elif e.get("turn") == focus:
                if event == "model_call" and e.get("kind") == "primary":
                    history = [m["content"] for m in e["messages"] if m["role"] == "assistant"]
                elif event in ("model_reply", "model_error"):
                    calls.append(e)
    if malformed:
        print(f"Skipped {malformed} incomplete or invalid JSON line(s).")
    if focus not in turns:
        raise SystemExit(f"No turn_finished for audio turn {focus} in session {session}.")

    target, request = turns[focus]
    ordered = sorted(turns)
    commits = [n for n in ordered if turns[n][0].get("display_text")]
    before = [n for n in commits if n < focus]
    after = [n for n in commits if n > focus]
    first = focus
    while first - 1 in turns and turns[first - 1][0].get("action") == "WAIT":
        first -= 1
    history_waits = 0
    for reply in reversed(history or []):
        if reply != "<|WAIT|>":
            break
        history_waits += 1

    print(f"session={session}  target={focus}  recorded_audio_turns={ordered[0]}..{ordered[-1]}")
    if options:
        print(f"Policy={options.get('context_policy')}  recovery={options.get('enable_recovery')}  "
              f"chunk_seconds={options.get('chunk_seconds')}")
    print(f"Previous two commits: {before[-2:]}   Next two commits: {after[:2]}")
    print(f"Consecutive WAITs directly logged before target: {focus - first}")
    if history is not None:
        print(f"Trailing WAITs in target model input: {history_waits}/{len(history)} archived replies")
    text = target.get("display_text", "")
    normalized = " ".join(text.casefold().split())
    duplicates = [n for n in before if " ".join(turns[n][0]["display_text"].casefold().split()) == normalized]
    print(f"Earlier commits matching the full target text (case/spacing ignored): {duplicates[-5:]}")
    print("Turn | recording seconds | action | turn seconds | committed text (first 140 chars)")
    nearby = {n for n in ordered if abs(n - focus) <= 4}
    for n in sorted(nearby | set(before[-2:] + after[:2])):
        r = turns[n][0]
        excerpt = " ".join(r.get("display_text", "").split())
        print(f"{n} | {r['start_seconds']:g}-{r['end_seconds']:g} | {r.get('action')} | "
              f"{r['generation_seconds']:.3f} | {excerpt[:140]}{'…' if len(excerpt) > 140 else ''}")
    for call in calls:
        raw = call.get("raw_text") or ""
        print(f"Target model: {call['event']} {call['kind']} {call.get('elapsed_ms')} ms "
              f"| {raw[:100]}")
    if request in packets:
        p = packets[request]
        print(f"Target HTTP: {p['status_code']} | {p['audio_seconds']}s audio | {p['turns']} turn(s) | "
              f"{p['duration_ms']} ms | {p['src_chars']} chars")
    for i, (packet_id, _, _, _) in enumerate(packet_audio[:-1]):
        if packet_id == request:
            _, start, end, rate = packet_audio[i + 1]
            print(f"Next logged packet: {start / rate:g}-{end / rate:g}s of recording ({(end - start) / rate:g}s)")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Per-meeting events.jsonl")
    parser.add_argument("--session", required=True)
    parser.add_argument("--turn", required=True, type=int)
    args = parser.parse_args()
    inspect(args.path, args.session, args.turn)
