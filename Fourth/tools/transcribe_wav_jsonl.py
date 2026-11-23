#!/usr/bin/env python3
"""用 Fourth 流式转录 WAV，按 audio 的文件名原地更新单个 JSONL。"""
import json
import os
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
from pathlib import Path
import sys

from filelock import FileLock

from transcribe_manifests import (
    BatchBackend, FourthStreamOptions, UnparsedLine, decode_pass, load_model,
    reject_constant, save_manifest, wav_basename,
)


# 在服务器上改这里。JSONL_FILE 是要原地写入的现有清单。
AUDIO_DIR = Path("/data/yb/Code/OmniCommit/datasets/meeting_eval/wavs")
JSONL_FILE = Path("/data/yb/Code/OmniCommit/datasets/meeting_eval/test.jsonl")
OUTPUT_FIELD = "OmniCommit_Fourth"
MODEL = Path("/data/yb/Code/OmniCommit/output/fourth-merged-1s-v2/v0-20261121-063241/sft-checkpoint-2721")
GPU_IDS = "1"
BATCH_SIZE = 4
CHUNK_MS = 1000
WINDOW_SECONDS = 60
DISCARD_SECONDS = 50
SAVE_EVERY = 20  # 每新增多少条成功记录原子写回一次，最后不足此数也会写回。


def index_wavs(directory):
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    index = defaultdict(list)
    for path in directory.rglob("*"):
        if path.suffix.lower() == ".wav" and path.is_file():
            index[path.name].append(path)
    return index


def matching_rows(lines, index):
    jobs = defaultdict(list)
    failed = 0
    for i, line in enumerate(lines):
        body = line.raw.removeprefix("\ufeff") if i == 0 else line.raw
        if not body.strip():
            continue
        try:
            row = json.loads(body, parse_constant=reject_constant)
            audio = row.get("audio") if isinstance(row, dict) else None
            if not isinstance(audio, str) or not audio.strip():
                raise ValueError("缺少字符串字段 audio")
            name = wav_basename(audio)
            paths = index.get(name, [])
            if not paths:
                raise ValueError(f"音频目录中找不到 {name!r}")
            if len(paths) > 1:
                raise ValueError(f"同名 WAV 无法唯一匹配 {name!r}: {paths}")
            jobs[paths[0]].append(i)
        except (ValueError, RecursionError) as exc:
            failed += 1
            print(f"[ERROR] JSONL 第 {i + 1} 行: {exc}", file=sys.stderr, flush=True)
    return jobs, failed


def updated_line(original, field, transcript):
    # 只重新序列化成功行，同时保留 BOM、行尾和最后一行没有换行的情况。
    row = json.loads(original.removeprefix("\ufeff"), parse_constant=reject_constant)
    row[field] = transcript
    text = json.dumps(row, ensure_ascii=False, allow_nan=False)
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        text = json.dumps(row, ensure_ascii=True, allow_nan=False)
    ending = original[len(original.rstrip("\r\n")):]
    bom = "\ufeff" if original.startswith("\ufeff") else ""
    return UnparsedLine(bom + text + ending)


def run(audio_dir, jsonl_file, field, transcribe_audio, *, workers=1, save_every=1, prepare_model=None):
    manifest = Path(jsonl_file).resolve(strict=True)
    if not manifest.is_file() or manifest.suffix.lower() != ".jsonl":
        raise ValueError(f"需要一个已有的 JSONL 文件: {manifest}")
    if not isinstance(field, str) or not field.strip() or field == "audio":
        raise ValueError("请指定非空的结果字段名，audio 用于匹配音频")
    if workers < 1 or save_every < 1:
        raise ValueError("workers 和 save_every 必须为正数")
    index = index_wavs(Path(audio_dir).resolve())
    with FileLock(str(manifest) + ".lock"):
        # save_manifest writes UnparsedLine verbatim. Preserve failed and untouched rows.
        with manifest.open(encoding="utf-8", newline="") as source:
            lines = [UnparsedLine(line) for line in source]
        jobs, failed = matching_rows(lines, index)
        total = sum(map(len, jobs.values()))
        print(f"[匹配] {total} 行，{len(jobs)} 个不同 WAV，{failed} 行异常；目标字段 {field}", flush=True)
        if not jobs:
            return failed
        if prepare_model is not None:
            prepare_model()
        completed = saved = dirty = 0
        remaining = iter(jobs)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {}

            def enqueue():
                path = next(remaining, None)
                if path is not None:
                    pending[pool.submit(transcribe_audio, path)] = path

            for _ in range(min(workers, len(jobs))):
                enqueue()
            try:
                while pending:
                    finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in finished:
                        path = pending.pop(future)
                        indices = jobs[path]
                        try:
                            transcript = future.result()
                            if not isinstance(transcript, str):
                                raise TypeError("转录结果必须是字符串")
                        except Exception as exc:
                            failed += len(indices)
                            print(f"[ERROR] {path}，JSONL 行 {[i + 1 for i in indices]}: {exc}",
                                  file=sys.stderr, flush=True)
                        else:
                            for i in indices:
                                try:
                                    lines[i] = updated_line(lines[i].raw, field, transcript)
                                except (ValueError, RecursionError) as exc:
                                    failed += 1
                                    print(f"[ERROR] JSONL 第 {i + 1} 行: {exc}", file=sys.stderr, flush=True)
                                else:
                                    dirty += 1
                                    saved += 1
                        if dirty >= save_every:
                            save_manifest(manifest, lines)
                            dirty = 0
                        completed += 1
                        print(f"[{completed}/{len(jobs)} WAV] {path.name}，成功 {saved} 行，异常 {failed} 行",
                              flush=True)
                        enqueue()
            finally:
                for future in pending:
                    future.cancel()
                if dirty:
                    save_manifest(manifest, lines)
    return failed


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_IDS
    options = FourthStreamOptions(protocol="speaker", number_format="normalized",
                                 chunk_seconds=CHUNK_MS / 1000, context_policy="window",
                                 window_seconds=WINDOW_SECONDS, discard_seconds=DISCARD_SECONDS,
                                 enable_recovery=False)
    print(f"[Fourth ASR] model={MODEL} gpu={GPU_IDS} batch={BATCH_SIZE} chunk_ms={CHUNK_MS}\n"
          f"音频目录={AUDIO_DIR}\n原地写回={JSONL_FILE} 字段={OUTPUT_FIELD}", flush=True)
    with ExitStack() as resources:
        backend = None

        def prepare_model():
            nonlocal backend
            backend = BatchBackend(lambda: load_model(MODEL, gpu_ids=GPU_IDS, batch_size=BATCH_SIZE,
                                                       chunk_ms=CHUNK_MS, window_seconds=WINDOW_SECONDS),
                                   max_batch_size=BATCH_SIZE)
            resources.callback(backend.close)

        failed = run(AUDIO_DIR, JSONL_FILE, OUTPUT_FIELD,
                     lambda path: decode_pass(path, backend, options, CHUNK_MS),
                     workers=BATCH_SIZE, save_every=SAVE_EVERY, prepare_model=prepare_model)
    print(f"[完成] 失败行数 {failed}", flush=True)
    return int(bool(failed))


if __name__ == "__main__":
    sys.exit(main())
