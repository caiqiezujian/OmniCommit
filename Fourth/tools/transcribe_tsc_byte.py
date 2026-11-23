"""按 vid 将 Fourth 流式 ASR 及原行参考译文写回 TSC / Byte JSONL。"""
import argparse
import os
import subprocess
import sys
import tempfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS.parent))
sys.path.insert(0, str(TOOLS))
from fourth_asr.vid_manifest import prepare_vid_update


# 可填写实际目录后直接运行；也可按 README 在命令行指定一套数据。
ROOT = Path("/data/yb/Code/OmniCommit/datasets")
DATASETS = {
    "TSC": (ROOT / "TSC" / "audios", ROOT / "TSC" / "jsonl"),
    "Byte": (ROOT / "Byte" / "audios", ROOT / "Byte" / "jsonl"),
}
MODEL = Path("/data/yb/Code/OmniCommit/output/fourth-merged-1s-v2/v0-20261121-063241/sft-checkpoint-2721")
PREFIX = "OmniCommit_Fourth"
GPU_IDS = "0"
BATCH_SIZE = 4
CHUNK_MS = 1000
WINDOW_SECONDS = 60
DISCARD_SECONDS = 50


def run_dataset(audio_dir, jsonl_dir, prefix, transcribe_audio, *, overwrite=False, workers=1,
                recursive=False, relative_vid=False, check_only=False, prepare_model=None):
    """预检整个数据集后逐条落盘。普通音频失败计数返回，清单写入冲突直接报错。"""
    if workers < 1:
        raise ValueError("workers 必须大于 0")
    plan = prepare_vid_update(audio_dir, jsonl_dir, prefix, overwrite=overwrite,
                              recursive=recursive, relative_vid=relative_vid)
    todo = [target for target in plan.targets if target.needs_inference]
    repairs = [target for target in plan.targets if target.needs_backfill]
    print(f"[{jsonl_dir}] 匹配 {len(plan.targets)}，待转录 {len(todo)}，仅需同步译文 {len(repairs)}，"
          f"未匹配音频 {len(plan.unmatched_audio)}，未匹配记录 {plan.unmatched_rows}", flush=True)
    for path in plan.unmatched_audio[:5]:
        print(f"[SKIP] 没有对应 vid 的音频: {path}", flush=True)
    if not plan.targets:
        raise ValueError(f"没有匹配的 vid，请核对音频目录 {audio_dir} 和 JSONL 目录 {jsonl_dir}")
    if check_only:
        return 0
    for target in repairs:
        target.save()
    if not todo:
        return 0
    if prepare_model is not None:
        prepare_model()

    failures = 0
    completed = 0
    remaining = iter(todo)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {}

        def enqueue():
            target = next(remaining, None)
            if target is not None:
                pending[pool.submit(transcribe_audio, target.audio)] = target

        for _ in range(min(workers, len(todo))):
            enqueue()
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                target = pending.pop(future)
                try:
                    transcript = future.result()
                    if not isinstance(transcript, str):
                        raise TypeError("转录结果必须为字符串")
                except Exception as exc:
                    failures += 1
                    print(f"[ERROR] vid={target.vid!r} audio={target.audio}: {exc}", file=sys.stderr, flush=True)
                else:
                    target.save(transcript)
                completed += 1
                if completed % 20 == 0 or completed == len(todo):
                    print(f"[{jsonl_dir}] {completed}/{len(todo)}，失败 {failures}", flush=True)
                enqueue()
    return failures


def decode_file(path, backend, options):
    import soundfile as sf
    from transcribe_manifests import decode_pass

    try:
        sf.info(path)
    except (RuntimeError, OSError):
        # 某些 libsndfile 构建不能打开 M4A 等容器。仅转换输入音轨。
        with tempfile.TemporaryDirectory(prefix="fourth_asr_") as temp:
            wav = Path(temp) / "audio.wav"
            command = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0",
                       "-c:a", "pcm_f32le", str(wav)]
            result = subprocess.run(command, capture_output=True)
            if result.returncode:
                raise ValueError(result.stderr.decode(errors="replace").strip() or "FFmpeg 读取音频失败")
            return decode_pass(wav, backend, options, CHUNK_MS)
    return decode_pass(path, backend, options, CHUNK_MS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, help="处理单个数据集时指定音频目录")
    parser.add_argument("--jsonl-dir", type=Path, help="同一数据集的原始 JSONL 目录")
    parser.add_argument("--prefix", "--field-prefix", default=PREFIX, help="两个输出字段的公共前缀")
    parser.add_argument("--model", type=Path, default=MODEL, help="Fourth 合并后的 HF 目录")
    parser.add_argument("--recursive", action="store_true", help="递归查找音频")
    parser.add_argument("--vid-mode", choices=("filename", "relative"), default="filename")
    parser.add_argument("--check-only", action="store_true", help="检查匹配和字段，打印待处理数量")
    parser.add_argument("--overwrite", action="store_true", help="对已有同前缀 ASR 的记录也重新转录")
    args = parser.parse_args(argv)
    if (args.audio_dir is None) != (args.jsonl_dir is None):
        parser.error("--audio-dir 和 --jsonl-dir 需要一起指定")
    datasets = list(DATASETS.values()) if args.audio_dir is None else [(args.audio_dir, args.jsonl_dir)]

    gpu_ids = os.environ.setdefault("CUDA_VISIBLE_DEVICES", GPU_IDS)
    backend = options = None

    def prepare_model():
        nonlocal backend, options
        if backend is None:
            from transcribe_manifests import BatchBackend, FourthStreamOptions, load_model
            options = FourthStreamOptions(protocol="speaker", number_format="normalized",
                                         chunk_seconds=CHUNK_MS / 1000, context_policy="window",
                                         window_seconds=WINDOW_SECONDS, discard_seconds=DISCARD_SECONDS,
                                         enable_recovery=False)
            backend = BatchBackend(lambda: load_model(args.model, gpu_ids=gpu_ids, batch_size=BATCH_SIZE,
                                                      chunk_ms=CHUNK_MS, window_seconds=WINDOW_SECONDS),
                                   max_batch_size=BATCH_SIZE)

    failures = 0
    try:
        for audio_dir, jsonl_dir in datasets:
            failures += run_dataset(audio_dir, jsonl_dir, args.prefix, lambda path: decode_file(path, backend, options),
                                    overwrite=args.overwrite, workers=BATCH_SIZE, prepare_model=prepare_model,
                                    check_only=args.check_only, recursive=args.recursive,
                                    relative_vid=args.vid_mode == "relative")
    finally:
        if backend is not None:
            backend.close()
    return int(bool(failures))


if __name__ == "__main__":
    sys.exit(main())
