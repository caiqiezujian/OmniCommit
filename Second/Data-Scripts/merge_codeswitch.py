"""Keep unused and partially reused source rows, replace fully reused ones, and append syntheses."""
import argparse
from collections import defaultdict
from functools import lru_cache
import json
import os
from pathlib import Path
import sys


def audio_path(row):
    paths = row.get("audios")
    if not isinstance(paths, list) or len(paths) != 1 or not isinstance(paths[0], str) or not paths[0].strip():
        raise ValueError("expected one audio path")
    return paths[0]


def relocate(row, root):
    path = audio_path(row)
    if not os.path.isabs(path):
        row["audios"] = [os.path.abspath(root / path)]
    return json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"


@lru_cache(maxsize=1024)
def legacy_frames(path):
    """Older supplement rows have no source duration. Missing evidence retains the original."""
    import soundfile as sf
    try:
        return sf.info(path).frames
    except (OSError, sf.LibsndfileError):
        return None


def fully_replaced(clips):
    """All ranges examined here belong to ONE source in ONE synthesized recording."""
    intervals = []
    declared_frames = None
    explicit_full = False
    for clip in clips:
        usage = clip.get("source_usage")
        if usage not in (None, "partial", "full"):
            raise ValueError(f"unknown source_usage {usage!r}")
        lo, hi = clip.get("source_start_frame"), clip.get("source_end_frame")
        if type(lo) is not int or type(hi) is not int or not 0 <= lo < hi:
            if usage is None and lo is None and hi is None:
                continue  # Legacy provenance without exact bounds.
            raise ValueError("invalid source frame range")
        n = clip.get("source_frames")
        if n is not None:
            if (type(n) is not int or n < hi or
                    (declared_frames is not None and n != declared_frames)):
                raise ValueError("inconsistent source_frames and source frame range")
            declared_frames = n
        if usage == "partial":
            continue
        if usage == "full":
            explicit_full = True
        intervals.append((lo, hi))
    covered = 0
    for lo, hi in sorted(intervals):
        if lo > covered:
            if explicit_full:
                raise ValueError("explicit full use leaves a gap in the source audio")
            return False
        covered = max(covered, hi)
    if covered == 0:
        return False
    total = declared_frames
    if total is None:
        total = legacy_frames(clips[0]["audio"])
    if total is None:
        return False
    if covered > total or (explicit_full and covered != total):
        raise ValueError("source frame coverage disagrees with complete audio length")
    return covered == total


def merge(original, supplement, output, audio_root=None, skip_offsets=()):
    original, supplement, output = map(Path, (original, supplement, output))
    if audio_root is not None:
        audio_root = Path(audio_root)
    root = audio_root or original.parent
    referenced = {}
    consumed = set()
    excluded = set(skip_offsets)
    kept = removed = partial = added = skipped = 0
    with supplement.open("rb") as extra, original.open("rb") as source:
        for number, raw in enumerate(extra, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            clips = row.get("synthesis", {}).get("clips")
            if not isinstance(clips, list) or not clips:
                raise ValueError(f"supplement line {number} has no source clips")
            by_source = defaultdict(list)
            for clip in clips:
                pos, path = clip["source_offset"], clip["audio"]
                if type(pos) is not int or pos < 0 or not isinstance(path, str) or not os.path.isabs(path):
                    raise ValueError(f"invalid source offset or audio at supplement line {number}")
                canonical = os.path.normcase(os.path.normpath(path))
                if referenced.setdefault(pos, canonical) != canonical:
                    raise ValueError(f"conflicting source paths for byte offset {pos}")
                by_source[pos].append(clip)
            for pos, source_clips in by_source.items():
                if fully_replaced(source_clips):
                    consumed.add(pos)
        # Only include the successfully scanned supplement, even if a producer
        # appends more records while the original is being streamed.
        end = extra.tell()
        extra.seek(0)
        output.parent.mkdir(parents=True, exist_ok=True)
        target = output.open("x", encoding="utf-8", newline="\n")
        try:
            with target:
                while True:
                    pos = source.tell()
                    raw = source.readline()
                    if not raw:
                        break
                    if pos in excluded:
                        skipped += 1
                        continue
                    if not raw.strip():
                        continue
                    row = json.loads(raw)
                    if pos in referenced:
                        actual = os.path.normcase(os.path.abspath(root / audio_path(row)))
                        if actual != referenced.pop(pos):
                            raise ValueError(f"source audio at offset {pos} does not match supplement")
                        if pos not in consumed:
                            partial += 1
                    if pos in consumed:
                        removed += 1
                    else:
                        target.write(relocate(row, root))
                        kept += 1
                if referenced:
                    raise ValueError(f"{len(referenced)} source offsets missing; use the original JSONL used for synthesis")
                while extra.tell() < end:
                    raw = extra.readline()
                    if not raw or extra.tell() > end:
                        raise ValueError("supplement changed during merge")
                    if raw.strip():
                        target.write(relocate(json.loads(raw), supplement.parent))
                        added += 1
        except BaseException:
            output.unlink(missing_ok=True)
            raise
    return dict(original_rows=kept+removed+skipped, skipped_source_rows=skipped,
                removed_source_rows=removed, retained_source_rows=kept,
                partial_source_rows_retained=partial, unused_source_rows=kept-partial,
                synthesized_rows=added, final_rows=kept+added)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("supplement", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audio-root", type=Path, help="Same relative source audio base used for augmentation")
    args = parser.parse_args()
    try:
        print(json.dumps(merge(args.original, args.supplement, args.output, args.audio_root)))
    except (ValueError, OSError, TypeError, KeyError, AttributeError) as exc:
        sys.exit(str(exc))
