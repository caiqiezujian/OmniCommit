"""Seek matching reference/trajectory JSONL records without retaining their payloads."""
import json
import math
import os
from pathlib import Path
import posixpath
import re


def pathkey(path, audio_root=None):
    value = os.fspath(path)
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError("Expected a nonempty source audio path")
    value = value.replace("\\", "/")
    if audio_root is not None and not (value.startswith("/") or re.match(r"^[A-Za-z]:/", value)):
        value = posixpath.join(os.fspath(audio_root).replace("\\", "/"), value)
    normalized = posixpath.normpath(value)
    if re.match(r"^[A-Za-z]:/", normalized) or normalized.startswith("//"):
        return normalized.casefold()
    return normalized


def _row_audio(row):
    if not isinstance(row, dict):
        raise ValueError("Expected a JSON object")
    paths = []
    if "audio" in row:
        paths.append(row["audio"])
    if "audios" in row:
        media = row["audios"]
        if not isinstance(media, list) or len(media) != 1:
            raise ValueError("Expected exactly one original recording in audios")
        paths.append(media[0])
    if not paths or not all(isinstance(path, str) and path.strip() for path in paths):
        raise ValueError("Missing original recording in audio/audios")
    return paths


class Index:
    def __init__(self, path, audio_root=None):
        self.path = Path(path)
        self.audio_root = audio_root
        self.issues = []
        self.offsets = {}
        self.stream = None

    def __enter__(self):
        self.stream = self.path.open("rb")
        try:
            offset = 0
            for line, raw in enumerate(self.stream, 1):
                here = offset
                offset += len(raw)
                if not raw.strip():
                    continue
                try:
                    row = json.loads(raw.decode("utf-8-sig"))
                    keys = {pathkey(p, self.audio_root) for p in _row_audio(row)}
                    if len(keys) != 1:
                        raise ValueError("audio and audios identify different recordings")
                    key, = keys
                except (TypeError, ValueError) as err:
                    self.issues.append({"line": line, "error": str(err)})
                    continue
                self.offsets.setdefault(key, []).append(here)
        except BaseException:
            self.stream.close()
            raise
        return self

    def __exit__(self, *_):
        self.stream.close()

    def get(self, audio):
        key = pathkey(audio, self.audio_root)
        rows = []
        for offset in self.offsets.get(key, []):
            self.stream.seek(offset)
            row = json.loads(self.stream.readline().decode("utf-8-sig"))
            if "audio" in row:
                row["audio"] = str(audio)
            if "audios" in row:
                row["audios"] = [str(audio)]
            rows.append(row)
        return rows


def _matches_turn(source, gold):
    if not isinstance(source, dict) or not (source.get("event", "audio") == gold["event"]
                                             and source.get("action") == gold["action"]):
        return False
    for key in ("start", "end"):
        value = source.get(key + "_ms")
        if (type(value) not in (int, float) or not math.isfinite(value)
                or abs(value / 1000 - gold[key + "_seconds"]) > 1e-6):
            return False
    fields = ("language", "speaker", "text")
    segments = source.get("segments")
    return (isinstance(segments, list) and len(segments) == len(gold["segments"])
            and all(isinstance(s, dict) and all(s.get(k) == g[k] for k in fields)
                    for s, g in zip(segments, gold["segments"])))


def _turn_window(row, parsed):
    source = row.get("turns")
    gold = parsed["turns"]
    if not isinstance(source, list) or len(source) < len(gold) or not gold:
        return None
    if parsed["complete"]:
        if len(source) != len(gold):
            return None
        starts = [0]
    else:
        starts = [i for i in range(len(source) - len(gold) + 1) if _matches_turn(source[i], gold[0])]
    if len(starts) != 1:
        return None
    start, = starts
    selected = source[start:start + len(gold)]
    return selected if all(_matches_turn(s, g) for s, g in zip(selected, gold)) else None


def match_trajectory(rows, parsed):
    """Attach metadata only for one source-identical, time/target-identical trajectory."""
    if not rows:
        return {}, None, "trajectory_not_found"
    matches = []
    for row in rows:
        if not isinstance(row, dict) or row.get("status") != "ok" or not isinstance(row.get("audio"), str):
            continue
        if pathkey(row["audio"]) != pathkey(parsed["audio"]):
            continue
        selected = _turn_window(row, parsed)
        if selected is not None:
            matches.append((row, selected))
    if len(matches) != 1:
        return {}, None, "ambiguous_matching_trajectories" if matches else "trajectory_grid_or_targets_do_not_match"
    row, selected = matches[0]
    provenance = {g["turn"]: s["provenance"] for s, g in zip(selected, parsed["turns"])
                  if isinstance(s.get("provenance"), dict)}
    aligned = None
    if isinstance(row.get("source_segments"), list) and isinstance(row.get("transcript"), str):
        aligned = {"kind": "aligned_asr", "schema_version": 2, "status": "ok", "audio": parsed["audio"],
                   "duration_ms": row.get("duration_ms"), "segments": row["source_segments"],
                   "transcript": row["transcript"]}
    return provenance, aligned, None
