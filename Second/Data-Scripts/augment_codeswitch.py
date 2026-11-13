"""Fill final bilingual/trilingual record quotas and write a timestamped training corpus."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
import heapq
from itertools import groupby, permutations
import json
import math
import os
from pathlib import Path
import random
import sys
from uuid import uuid4

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from second_asr.source import TAG, Unit, parse_record

RATE = 16000
SAMPLES_PER_MS = RATE // 1000
ENDINGS = tuple(".!?。！？،，,;；:：…؟।")
SCENARIOS = ("pair", "triple", "short-triple", "return", "short-return")
INPUT_ERRORS = (ValueError, KeyError, TypeError, IndexError, OverflowError, OSError, sf.LibsndfileError)


@dataclass(frozen=True)
class Clip:
    offset: int
    path: str
    start: int
    stop: int
    usage: str
    split: int = 0


def remap_frame(t, first, sr):
    return round((round(t * sr / 1000) - first) * RATE / sr)


def read_units(row, duration):
    """Keep finer attribution only where the input actually supplies its timing."""
    units = parse_record(row, duration)
    times = row["timestamps"]
    if "index" in times[0]:
        times = sorted(times, key=lambda t: t["index"])
    result = []
    for unit, stamp in zip(units, times):
        if stamp.get("spans"):
            for s in stamp["spans"]:
                body = s.get("text_plain")
                if body is None:
                    body = TAG.sub("", s["text"])
                result.append(Unit(s["start_ms"], s["end_ms"], ((s["language"], s["speaker"], body),)))
        else:
            result.append(unit)
    return result


class Catalog:
    """Reservoirs contain byte offsets and unit ranges, never the full corpus."""
    def __init__(self, args, rng):
        self.args, self.rng = args, rng
        self.pools = defaultdict(list)
        self.seen = Counter()
        self.routes = Counter()
        self.scenes = Counter()
        self.combinations = Counter()
        self.invalid_offsets = set()
        self.full_offsets = set()
        self.partial_offsets = set()
        self.error_count = 0
        self.bad_audio = set()

    def offer(self, key, clip):
        self.seen[key] += 1
        pool = self.pools[key]
        if key[0] != "short" or len(pool) < self.args.pool_size:
            pool.append(clip)
        else:
            pos = self.rng.randrange(self.seen[key])
            if pos < len(pool):
                pool[pos] = clip

    def count_routes(self, units):
        # Count actual consecutive language runs. Repeated semantic headers
        # and a speaker change within a language do not create a new language run.
        runs = []
        for i, unit in enumerate(units):
            for lang, _, _ in unit.spans:
                if runs and runs[-1][0] == lang:
                    runs[-1][2] = i
                else:
                    runs.append([lang, i, i])
        for n in (2, 3):
            for i in range(len(runs) - n + 1):
                part = runs[i:i+n]
                route = tuple(r[0] for r in part)
                self.routes[route] += 1
                if n == 2:
                    continue
                kind = "return" if route[0] == route[2] else "triple"
                mid = part[1]
                if units[mid[2]].end_ms - units[mid[1]].start_ms <= self.args.short_max_ms:
                    self.scenes["short-" + kind, route] += 1

    def score(self, scenario, route):
        if scenario in ("pair", "triple", "return"):
            existing = self.routes[route]
        else:
            existing = self.scenes[scenario, route]
        return existing, sum(self.routes[e] for e in zip(route, route[1:]))

    def add(self, units, offset, path):
        self.count_routes(units)
        langs = tuple(sorted({lang for u in units for lang, _, _ in u.spans}))
        self.combinations[langs] += 1
        if len(langs) == 1:
            # Monolingual whole donors may have several speakers. Both
            # annotations and the source audio before/after the words survive.
            cuts = [j for j in range(1, len(units)) if units[j-1].end_ms <= units[j].start_ms
                    and units[j-1].spans[-1][1] == units[j].spans[0][1]]
            natural = [j for j in cuts if units[j-1].spans[-1][2].rstrip().endswith(ENDINGS)]
            split = self.rng.choice(natural or cuts) if cuts else 0
            whole = Clip(offset, path, 0, len(units), "full", split)
            self.offer(("regular", langs[0]), whole)
            if split:
                self.offer(("split", langs[0]), whole)
        # Never crop through another annotated utterance. Adjacent units from
        # the same speaker may be grouped, preserving natural interior pauses.
        def state(i):
            u = units[i]
            if (len(u.spans) != 1 or not u.spans[0][2].strip()
                    or (i and units[i-1].end_ms > u.start_ms)
                    or (i+1 < len(units) and units[i+1].start_ms < u.end_ms)):
                return None
            return u.spans[0][:2]

        for identity, indices in groupby(range(len(units)), key=state):
            ids = list(indices)
            if identity is None:
                continue
            lang = identity[0]
            first, last = ids[0], ids[-1] + 1
            # Prefer full short clauses/turns. For a long unpunctuated clause
            # its individually timed semantic units are the finest legal cuts.
            begin = first
            for k in ids:
                if (k+1 == last or units[k].spans[0][2].rstrip().endswith(ENDINGS)
                        or units[k+1].start_ms - units[k].end_ms > self.args.max_internal_gap_ms):
                    if (k+1-begin < len(units)
                            and units[k].end_ms - units[begin].start_ms <= self.args.short_max_ms):
                        self.offer(("short", lang), Clip(offset, path, begin, k+1, "partial"))
                    elif begin < k:
                        for j in range(begin, k+1):
                            if units[j].end_ms - units[j].start_ms <= self.args.short_max_ms:
                                self.offer(("short", lang), Clip(offset, path, j, j+1, "partial"))
                    begin = k+1

    def commit_usage(self, row):
        for origin in row["synthesis"]["clips"]:
            offsets = self.full_offsets if origin["source_usage"] == "full" else self.partial_offsets
            offsets.add(origin["source_offset"])

    def scan(self, source):
        number = 0
        while True:
            offset = source.tell()
            raw = source.readline()
            if not raw:
                break
            number += 1
            try:
                row = json.loads(raw)
                # Avoid opening a million WAVs during indexing. Selected WAVs
                # are separately checked against their actual duration before use.
                end = max(t["end_ms"] for t in row["timestamps"])
                units = read_units(row, end)
                path = os.path.abspath(self.args.audio_root / row["audios"][0])
                self.add(units, offset, path)
            except INPUT_ERRORS as exc:
                self.error_count += 1
                self.invalid_offsets.add(offset)
                if self.error_count <= 8:
                    print(f"Skipped input line {number}: {exc}", file=sys.stderr)
            if number % 10000 == 0:
                print(f"Indexed {number:,} lines | rejected {self.error_count}", flush=True)
        print(f"Indexed {number:,} lines | rejected {self.error_count}", flush=True)


def available_routes(cat, scenario):
    langs = sorted({k[1] for k in cat.pools})
    for route in permutations(langs, 2 if scenario.endswith(("pair", "return")) else 3):
        a, b = route[:2]
        if ("regular", a) not in cat.pools:
            continue
        if scenario.endswith("return"):
            if ("split", a) not in cat.pools:
                continue
            route = (a, b, a)
        mid_role = "short" if scenario.startswith("short-") else "regular"
        if (mid_role, b) not in cat.pools:
            continue
        if len(route) == 3 and ("regular", route[2]) not in cat.pools:
            continue
        yield route


def draw(cat, scenario, route):
    a, b = route[:2]
    middle = "short" if scenario.startswith("short-") else "regular"
    if scenario.endswith("return"):
        keys = [("split", a), (middle, b)]
    else:
        keys = [("regular", a), (middle, b)]
        if len(route) == 3:
            keys.append(("regular", route[2]))
    clips = []
    for key in keys:
        pool = cat.pools[key]
        while pool:
            index = cat.rng.randrange(len(pool))
            clip = pool[index]
            opposite = cat.full_offsets if clip.usage == "partial" else cat.partial_offsets
            if clip.path in cat.bad_audio or clip.offset in opposite:
                pool[index] = pool[-1]
                pool.pop()
                continue
            if any(prior.path == clip.path for prior in clips):
                return None
            clips.append(clip)
            break
        else:
            return None
    if scenario.endswith("return"):
        outer, middle_clip = clips
        return (Clip(outer.offset, outer.path, 0, outer.split, "full"), middle_clip,
                Clip(outer.offset, outer.path, outer.split, outer.stop, "full"))
    return tuple(clips)


def crop(clip, all_units):
    units = all_units[clip.start:clip.stop]
    with sf.SoundFile(clip.path) as f:
        sr = f.samplerate
        total = len(f)
        first = round(units[0].start_ms * sr / 1000)
        if clip.usage == "full":
            if clip.start == 0:
                first = 0
            # ABA: at a splice both adjoining parts use the same source
            # sample. This preserves the inter-unit pause exactly once.
            last = total if clip.stop == len(all_units) else round(all_units[clip.stop].start_ms * sr / 1000)
        else:
            last = round(units[-1].end_ms * sr / 1000)
            if first == 0 and last == total:
                raise ValueError("short insert covers the full source waveform")
        if not 0 <= first < last <= total:
            raise ValueError("crop exceeds recording")
        f.seek(first)
        data = f.read(last-first, dtype="float32", always_2d=True)
    if len(data) != last-first or not np.isfinite(data).all():
        raise ValueError("truncated or non-finite audio")
    mono = data.mean(axis=1, dtype=np.float32)
    if sr != RATE:
        from scipy.signal import resample_poly
        gcd = math.gcd(sr, RATE)
        mono = resample_poly(mono, RATE // gcd, sr // gcd)
    # Map both ends from the same source sample grid used by the decoder.
    size = round((last-first) * RATE / sr)
    if size < 1:
        raise ValueError("empty resampled clip")
    bounds = []
    for u in units:
        start = remap_frame(u.start_ms, first, sr)
        end = remap_frame(u.end_ms, first, sr)
        if not 0 <= start < end <= size:
            raise ValueError("unit shorter than one sample or outside the crop")
        bounds.append((start, end))
    return mono[:size], bounds, (first, last, sr, total)


def assemble(clips, load, args, rng, destination, scenario, route, bad_audio):
    pieces, segments, times, origins = [], [], [], []
    speakers = {}
    frames = 0
    for i, clip in enumerate(clips):
        try:
            all_units = load(clip.offset, clip.path)
            pcm, bounds, (first, last, sr, total) = crop(clip, all_units)
            units = all_units[clip.start:clip.stop]
        except (sf.LibsndfileError, OSError):
            bad_audio.add(clip.path)
            raise
        gap = 0
        if i and rng.random() >= args.no_gap_prob:
            gap = rng.randint(args.gap_ms[0], args.gap_ms[1]) * SAMPLES_PER_MS
            pieces.append(np.zeros(gap, dtype=np.float32))
            frames += gap
        origins.append(dict(audio=clip.path, source_start_frame=first, source_end_frame=last,
                            source_frames=total, source_usage=clip.usage,
                            source_rate=sr, source_offset=clip.offset, start_frame=frames,
                            end_frame=frames+len(pcm), gap_before_ms=gap/SAMPLES_PER_MS))
        for u, (lo, hi) in zip(units, bounds):
            marked = []
            for lang, old, text in u.spans:
                speaker = speakers.setdefault((clip.path, old), f"spk{len(speakers)+1}")
                marked.append(f"<{lang}><{speaker}>{text}")
            tagged = "".join(marked)
            segments.append({"source_text": tagged, "action": "DIRECT"})
            start_ms, end_ms = (frames+lo)/SAMPLES_PER_MS, (frames+hi)/SAMPLES_PER_MS
            stamp = dict(start_ms=start_ms, end_ms=end_ms, index=len(times)+1, source_text=tagged)
            if len(u.spans) == 1:
                span = dict(start_ms=start_ms, end_ms=end_ms, text=tagged, text_plain=text,
                            language=lang, speaker=speaker)
                stamp.update(span, spans=[span])
            times.append(stamp)
        pieces.append(pcm)
        frames += len(pcm)
    annotation = {"segments": segments, "flush": False, "error": None}
    row = {"audios": [str(destination)], "messages": [
        {"role": "system", "content": "Transcribe the audio with language and speaker labels."},
        {"role": "user", "content": "<audio>**Entity Reference:** []"},
        {"role": "assistant", "content": "".join(s["source_text"] for s in segments)}],
        "output": json.dumps(annotation, ensure_ascii=False, separators=(",", ":")),
        "timestamps": times, "tags": ["<|DIRECT|>"] * len(times),
        "synthesis": {"scenario": scenario, "languages": route, "sample_rate": RATE, "clips": origins}}
    verified = parse_record(row, frames/SAMPLES_PER_MS)
    return np.concatenate(pieces), row, verified


class CombinationPlan:
    """Count records by exact unordered language set in the resulting training file."""
    def __init__(self, cat, args):
        from itertools import combinations

        self.cat = cat
        self.initial = cat.combinations.copy()
        self.counts = self.initial.copy()
        self.targets = {group: target for n, target in ((2, args.pair_target), (3, args.triple_target))
                        for group in combinations(sorted(args.languages), n)}
        self.choices = defaultdict(lambda: defaultdict(list))
        self.blocked = set()
        self.planned = Counter()
        for scene in args.scenarios:
            for route in available_routes(cat, scene):
                group = tuple(sorted(set(route)))
                if self.counts[group] < self.targets.get(group, 0):
                    self.choices[group][scene].append(route)
        self.required = sum(max(0, target-self.counts[g]) for g, target in self.targets.items())
        print(f"Exact language groups: {len(self.targets)} | new rows needed: {self.required:,}", flush=True)

    def jobs(self):
        active = list(self.choices)
        self.cat.rng.shuffle(active)
        while active:
            remaining = []
            for group in active:
                scenes = self.choices[group]
                # Mix ordinary conversations, short insertions and returns in
                # each group. Balance orders *within* the chosen scenario.
                scene = min(scenes, key=lambda name: (self.planned[group, name], self.cat.rng.random()))
                routes = scenes[scene]
                route = min(routes, key=lambda r: (self.cat.score(scene, r), self.cat.rng.random()))
                yield scene, route
                if (scene, route) in self.blocked:
                    routes.remove(route)
                    if not routes:
                        del scenes[scene]
                if scenes and self.counts[group] < self.targets[group]:
                    remaining.append(group)
            active = remaining

    def success(self, scene, route):
        group = tuple(sorted(set(route)))
        self.counts[group] += 1
        self.planned[group, scene] += 1

    def deficits(self):
        return {"+".join(group): target-self.counts[group] for group, target in self.targets.items()
                if self.counts[group] < target}


def supplemental_jobs(cat, args, blocked):
    queues = {}
    for scene in args.scenarios:
        queue = [(0, *cat.score(scene, r), cat.rng.random(), r) for r in available_routes(cat, scene)]
        if queue:
            heapq.heapify(queue)
            queues[scene] = queue
    while queues:
        for scene in list(queues):
            queue = queues[scene]
            while True:
                covered, existing, edges, tie, route = heapq.heappop(queue)
                current = cat.score(scene, route)
                if current == (existing, edges):
                    break
                heapq.heappush(queue, (covered, *current, tie, route))
            yield scene, route
            if (scene, route) not in blocked:
                heapq.heappush(queue, (1, *cat.score(scene, route), cat.rng.random(), route))
            if not queue:
                del queues[scene]


def generate(args):
    from merge_codeswitch import merge

    rng = random.Random(args.seed)
    cat = Catalog(args, rng)
    destination = args.output.resolve()
    audio_dir = args.audio_dir.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Choose a new output JSONL: {destination}")
    # Unique file namespace even when two independent jobs share one audio directory.
    prefix = uuid4().hex[:16]
    with args.input.open("rb") as source:
        cat.scan(source)

        @lru_cache(maxsize=64)
        def load(offset, path):
            source.seek(offset)
            info = sf.info(path)
            return read_units(json.loads(source.readline()), info.frames * 1000 / info.samplerate)

        plan = CombinationPlan(cat, args) if args.count is None else None
        blocked = plan.blocked if plan else set()
        jobs = plan.jobs() if plan else supplemental_jobs(cat, args, blocked)
        target = plan.required if plan else args.count
        produced, failures = Counter(), Counter()
        # Serialize successful rows incrementally; usable as a supplement if a
        # long run is interrupted. On successful merge only the final JSONL remains.
        supplement = (destination.with_name(f".{destination.name}.{prefix}.supplement.inprogress")
                      if plan else destination)
        print(f"Synthesized rows while running: {supplement}", flush=True)
        used = set()
        written = 0
        with supplement.open("x", encoding="utf-8", newline="\n") as out:
            for scene, route in jobs:
                path = audio_dir / f"{written//1000:05d}" / f"cs_{prefix}_{written:08d}.wav"
                for _ in range(args.attempts):
                    clips = draw(cat, scene, route)
                    if not clips or clips in used:
                        continue
                    try:
                        pcm, row, units = assemble(clips, load, args, rng, path, scene, route, cat.bad_audio)
                        line = json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                        line.encode("utf-8")
                    except INPUT_ERRORS as exc:
                        reason = type(exc).__name__
                        failures[reason] += 1
                        if failures[reason] <= 5:
                            print(f"Skipped candidate {route}: {exc}", file=sys.stderr)
                        continue
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("xb") as wav:
                        try:
                            sf.write(wav, pcm, RATE, format="WAV", subtype="PCM_16")
                        except BaseException:
                            wav.close()
                            path.unlink(missing_ok=True)
                            raise
                    out.write(line + "\n")
                    out.flush()
                    used.add(clips)
                    cat.commit_usage(row)
                    cat.count_routes(units)
                    if plan:
                        plan.success(scene, route)
                    produced[scene] += 1
                    written += 1
                    if written == 1 or written % 1000 == 0:
                        print(f"Wrote {written:,}/{target:,} | {dict(produced)}", flush=True)
                    break
                else:
                    blocked.add((scene, route))
                if written == target:
                    break
    result = {"requested": target, "written": written, "scenarios": produced, "candidate_errors": failures,
              "rejected_lines": cat.error_count, "unreadable_sources": len(cat.bad_audio)}
    if plan:
        result.update(merge(args.input, supplement, destination, args.audio_root, cat.invalid_offsets))
        supplement.unlink()
        result["missing_by_combination"] = plan.deficits()
        result["final_by_combination"] = {"+".join(g): plan.counts[g] for g in plan.targets}
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if written == target else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("input", type=Path, help="Source JSONL with per-unit timestamps")
    p.add_argument("--audio-dir", type=Path, required=True, help="Directory for synthesized WAVs")
    p.add_argument("--output", type=Path, help="New output JSONL (default: AUDIO_DIR/codeswitch.jsonl)")
    p.add_argument("--audio-root", type=Path, help="Base for relative source WAV paths (default: input directory)")
    p.add_argument("--count", type=int, help="Explicit supplemental-only mode: request N new recordings")
    p.add_argument("--pair-target", type=int, help="Minimum final records per unordered pair (default 10000)")
    p.add_argument("--triple-target", type=int, help="Minimum final records per unordered triple (default 3000)")
    p.add_argument("--languages", nargs="+", help="Quota languages (default: ja ru ko it de fr pt es en zh)")
    p.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    p.add_argument("--short-max-ms", type=int, default=900, help="Maximum annotated short middle duration")
    p.add_argument("--max-internal-gap-ms", type=int, default=350, help="Largest retained gap between units")
    p.add_argument("--no-gap-prob", type=float, default=0.5, help="Probability of zero inserted silence per join")
    p.add_argument("--gap-ms", nargs=2, type=int, default=(80, 400), metavar=("MIN", "MAX"),
                   help="Inclusive silence range for other joins")
    p.add_argument("--pool-size", type=int, default=6000, help="Maximum short insert candidates per language")
    p.add_argument("--attempts", type=int, default=60, help="Draw attempts before retiring an infeasible route")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    if args.count is not None and (args.count < 1 or any(x is not None for x in
                                  (args.pair_target, args.triple_target, args.languages))):
        p.error("--count must be positive and cannot be combined with quota options")
    args.pair_target = 10000 if args.pair_target is None else args.pair_target
    args.triple_target = 3000 if args.triple_target is None else args.triple_target
    args.languages = args.languages or "ja ru ko it de fr pt es en zh".split()
    if len(args.languages) < 2 or len(set(args.languages)) != len(args.languages):
        p.error("--languages needs at least two distinct language tags")
    if (min(args.short_max_ms, args.pool_size, args.attempts) < 1
            or min(args.pair_target, args.triple_target) < 0
            or args.max_internal_gap_ms < 0
            or not 0 <= args.no_gap_prob <= 1 or not 0 <= args.gap_ms[0] <= args.gap_ms[1]):
        p.error("Invalid count, clip, pool or gap settings")
    args.audio_root = args.audio_root or args.input.parent
    args.output = args.output or args.audio_dir / "codeswitch.jsonl"
    args.scenarios = list(dict.fromkeys(args.scenarios))
    return generate(args)


if __name__ == "__main__":
    raise SystemExit(main())
