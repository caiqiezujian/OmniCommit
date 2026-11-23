"""Match TSC/Byte audio to FLEX JSONL rows and checkpoint Fourth ASR results."""
from __future__ import annotations

import codecs
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

from filelock import FileLock


AUDIO_SUFFIXES = frozenset({
    ".wav", ".wave", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".oga", ".opus",
    ".wma", ".aiff", ".aif", ".aifc", ".amr", ".mp2", ".mka", ".caf", ".ac3",
})


def _vid_key(vid: str) -> str:
    suffix = Path(vid).suffix
    return vid[:-len(suffix)] if suffix.lower() in AUDIO_SUFFIXES else vid


@dataclass
class _Document:
    path: Path
    lines: list[bytes]
    snapshot: bytes

    def _check_unchanged(self) -> None:
        if self.path.is_symlink() or self.path.read_bytes() != self.snapshot:
            raise RuntimeError(f"JSONL changed since indexing: {self.path}. Reload before retrying.")

    def replace_row(self, index: int, row: dict) -> None:
        original = self.lines[index]
        body = original.rstrip(b"\r\n")
        ending = original[len(body):]
        bom = codecs.BOM_UTF8 if index == 0 and body.startswith(codecs.BOM_UTF8) else b""
        encoded = bom + json.dumps(row, ensure_ascii=False, allow_nan=False).encode("utf-8") + ending

        with FileLock(str(self.path) + ".lock"):
            self._check_unchanged()
            new_lines = self.lines.copy()
            new_lines[index] = encoded
            contents = b"".join(new_lines)
            mode = stat.S_IMODE(self.path.stat().st_mode)
            descriptor, name = tempfile.mkstemp(dir=self.path.parent, prefix=f".{self.path.name}.",
                                                suffix=".tmp")
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(contents)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.chmod(name, mode)
                self._check_unchanged()
                os.replace(name, self.path)
            finally:
                Path(name).unlink(missing_ok=True)
            self.lines, self.snapshot = new_lines, contents


@dataclass
class VidTarget:
    audio: Path
    vid: str
    document: _Document
    line_index: int
    row: dict
    prefix: str
    overwrite: bool

    @property
    def needs_inference(self) -> bool:
        return self.overwrite or not isinstance(self.row.get(self.prefix + "_asr"), str)

    @property
    def needs_backfill(self) -> bool:
        return not self.needs_inference and self.row.get(self.prefix + "_trans") != self.row["translation"]

    def save(self, asr: str | None = None) -> bool:
        """Persist both fields together, or repair only the copy of a reference."""
        asr_key = self.prefix + "_asr"
        if asr is None:
            if self.needs_inference:
                raise ValueError(f"Missing ASR result for {self.vid!r}")
            if not self.needs_backfill:
                return False
            asr = self.row[asr_key]
        elif not isinstance(asr, str):
            raise TypeError(f"ASR result must be a string for {self.vid!r}")
        elif not self.needs_inference and asr != self.row[asr_key]:
            raise ValueError(f"Recomputing {self.vid!r} requires overwrite=True")
        updated = dict(self.row)
        updated[asr_key] = asr
        updated[self.prefix + "_trans"] = self.row["translation"]
        changed = updated != self.row
        if changed:
            self.document.replace_row(self.line_index, updated)
        self.row = updated
        self.overwrite = False
        return changed


@dataclass
class VidPlan:
    targets: list[VidTarget]
    unmatched_audio: list[Path]
    unmatched_rows: int


def prepare_vid_update(audio_dir, jsonl_dir, prefix: str, *, recursive: bool = False,
                       relative_vid: bool = False, overwrite: bool = True) -> VidPlan:
    """Preflight all JSONLs before inference. Match vid only, stripping one audio extension."""
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("Specify a non-empty output field prefix")
    audio_root, manifest_root = Path(audio_dir).resolve(), Path(jsonl_dir).resolve()
    if not audio_root.is_dir() or not manifest_root.is_dir():
        raise ValueError(f"Expected existing audio and JSONL directories: {audio_root}, {manifest_root}")
    candidates = audio_root.rglob("*") if recursive else audio_root.iterdir()
    audio_paths = sorted((p for p in candidates if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES),
                         key=lambda p: p.relative_to(audio_root).as_posix())
    if not audio_paths:
        raise ValueError(f"No supported audio files found in {audio_root}")
    audio_by_id = {}
    for path in audio_paths:
        key = _vid_key(path.relative_to(audio_root).as_posix() if relative_vid else path.name)
        if key in audio_by_id:
            raise ValueError(f"Ambiguous audio stem {key!r}: {audio_by_id[key]} and {path}")
        audio_by_id[key] = path

    paths = sorted(p for p in manifest_root.iterdir() if p.is_file() and p.suffix.lower() == ".jsonl")
    if not paths:
        raise ValueError(f"No JSONL files directly in {manifest_root}")
    targets = {}
    unmatched_rows = 0
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"Expected a regular JSONL file: {path}")
        snapshot = path.read_bytes()
        document = _Document(path, snapshot.splitlines(keepends=True), snapshot)
        for index, line in enumerate(document.lines):
            location = f"{path}:{index + 1}"
            try:
                text = line.decode("utf-8-sig" if index == 0 else "utf-8")
                if not text.strip():
                    continue
                row = json.loads(text)
            except ValueError as exc:
                raise ValueError(f"Invalid JSON at {location}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {location}")
            vid = row.get("vid")
            key = _vid_key(vid) if isinstance(vid, str) else None
            if key not in audio_by_id:
                unmatched_rows += 1
                continue
            if key in targets:
                previous = targets[key]
                raise ValueError(f"Duplicate matching vid {vid!r}: "
                                 f"{previous.document.path}:{previous.line_index + 1} and {location}")
            if not isinstance(row.get("translation"), str):
                raise ValueError(f"Matched vid {vid!r} requires a string translation at {location}")
            try:
                json.dumps(row, ensure_ascii=False, allow_nan=False).encode("utf-8")
            except ValueError as exc:
                raise ValueError(f"Cannot save matched row at {location}: {exc}") from exc
            targets[key] = VidTarget(audio_by_id[key], vid, document, index, row, prefix, overwrite)
    return VidPlan([targets[k] for k in audio_by_id if k in targets],
                   [p for k, p in audio_by_id.items() if k not in targets], unmatched_rows)
