"""Evaluate Fourth streaming ASR and write plain transcripts to per-language JSON or JSONL manifests."""
import argparse
import json
import math
import os
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from filelock import FileLock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fourth_asr.batch_backend import BatchBackend
from fourth_asr.protocol import DIRECT, SPK_CHANGE, WAIT, get_protocol
from fourth_asr.runtime.inference_backend import multimodal_messages
from fourth_asr.runtime.streaming import GenerationTruncated
from fourth_asr.runtime.vllm_backend import VLLMBackend
from fourth_asr.service_backend import detect_model_protocol
from fourth_asr.service_session import FourthStreamOptions, FourthStreamSession


# 在服务器上按实际位置修改这里。启动时不需要传参数。
ROOT = Path("/data/yb/Code/OmniCommit/datasets/all_audios")
MODEL = Path("/data/yb/Code/OmniCommit/models/fourth-speaker-full-merged")
FIELD = "OmniCommit_Fourth"
MANIFEST_NAME = None  # 同时有两种格式时，填写要写入的文件名。
LANGUAGES = "ar de en es fr it ja ko pt ru th zh".split()
GPU_IDS = "0"
BATCH_SIZE = 4
CHUNK_MS = 1000
WINDOW_SECONDS = 60
DISCARD_SECONDS = 50
MAX_CONTEXT_TOKENS = 4352
MAX_NEW_TOKENS = 512
GPU_MEMORY_UTILIZATION = .85
SAVE_EVERY = 20
RATE = 16000

# These spellings are interpreted as media slots or chat boundaries when a
# previous assistant reply is rendered again. Quote them only in prompt history.
CHAT_OR_MEDIA_MARKER = re.compile(
    r"<audio>|<\|(?:im_start|im_end|endoftext|(?:audio|image|vision|video)_(?:pad|start|end))\|>"
)


class ManifestVLLMBackend(VLLMBackend):
    """Expose the tokens vLLM produced, including a length-capped completion."""

    def _decode_reply(self, output, expected):
        try:
            return super()._decode_reply(output, expected)
        except GenerationTruncated as cutoff:
            # The engine returned these tokens. Keep the whole capped hypothesis
            # for scoring, including any repetition the model actually produced.
            ids = output.outputs[0].token_ids
            return {
                "text": self.tokenizer.decode(ids, skip_special_tokens=False,
                                              clean_up_tokenization_spaces=False),
                "finish_reason": "length", "prompt_tokens": cutoff.prompt_tokens,
                "num_cached_tokens": getattr(output, "num_cached_tokens", None),
            }


class ManifestStreamSession(FourthStreamSession):
    """Score each generated turn while retaining the model's reply for the next chunk."""

    def _model_reply(self, reply, end_of_audio):
        if not isinstance(reply, str):
            raise TypeError("推理后端没有返回文本字符串")
        text = reply.replace(DIRECT, "").replace(WAIT, "").replace(SPK_CHANGE, "")
        if not text.strip():
            text = ""
        history_reply = CHAT_OR_MEDIA_MARKER.sub(lambda match: "&lt;" + match[0][1:], reply)
        # The action tracks whether any text was committed. An empty DIRECT
        # still appears as the model's original DIRECT in the next prompt.
        return {"text": history_reply, "display_text": text, "segments": [],
                "action": "DIRECT" if text else "WAIT"}

    def _decode_turn(self, history, pcm, end_of_audio):
        generated, parsed = self._generate_turn(history, pcm, end_of_audio)
        if isinstance(generated, dict) and generated.get("finish_reason") == "length":
            parsed["finish_reason"] = "length"
        return generated, parsed, history


def show_first_prompt(tokenizer):
    system = get_protocol("speaker").system_prompt("normalized")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": "<audio>"}]
    rendered = tokenizer.apply_chat_template(multimodal_messages(messages), tokenize=False,
                                             add_generation_prompt=True)
    print(f"[ASR prompt] model={MODEL}\n{rendered}", flush=True)
    if system not in rendered:
        raise ValueError(f"导出目录的 chat template 丢失了 Fourth system 提示词: {MODEL}")


def load_model():
    if detect_model_protocol(MODEL) != "speaker":
        raise ValueError(f"需要带 DIRECT/WAIT/SPK_CHANGE 的 Fourth 合并 HF 模型: {MODEL}")
    model = ManifestVLLMBackend(
        MODEL, dtype="bfloat16", tensor_parallel_size=len(GPU_IDS.split(",")),
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION, max_batch_size=BATCH_SIZE,
        max_context_tokens=MAX_CONTEXT_TOKENS, max_new_tokens=MAX_NEW_TOKENS,
        max_audio_turns=math.ceil(WINDOW_SECONDS * 1000 / CHUNK_MS),
        max_audio_seconds=max(.5, CHUNK_MS / 1000), enable_prefix_caching=True,
    )
    show_first_prompt(model.tokenizer)
    return model


def decode_pass(path, backend, options, chunk_ms):
    session = ManifestStreamSession(backend, options)
    pieces = []

    def append_turn(audio=None, *, final_chunk=False, end_of_audio=False):
        try:
            event = session.step(audio, final_chunk=final_chunk, end_of_audio=end_of_audio)
        except Exception as exc:
            kind = "end_of_audio" if end_of_audio else "audio"
            raise RuntimeError(f"{kind} turn={session.turn_count + 1}: {exc}") from exc
        pieces.append(event["display_text"])
        if event.get("finish_reason") == "length":
            print(f"[LENGTH] {path.name} {event['event']} turn={event['turn']} 已保留本轮生成的正文", flush=True)

    with sf.SoundFile(path) as wav:
        frames, sr = len(wav), wav.samplerate
        if not frames:
            raise ValueError("空 WAV")
        divisor = math.gcd(sr, RATE)
        pos = turn = 0
        while pos < frames:
            turn += 1
            # 从录音起点计算边界，避免各源采样率下的累计舍入误差。
            end = min(frames, round(turn * chunk_ms * sr / 1000))
            block = wav.read(end - pos, dtype="float32", always_2d=True)
            if not len(block) or len(block) != end - pos or not np.isfinite(block).all():
                raise ValueError(f"音频读取异常，起始采样点 {pos}")
            mono = block.mean(axis=1)
            if sr != RATE:
                from scipy.signal import resample_poly

                mono = resample_poly(mono, RATE // divisor, sr // divisor)
                count = max(1, min(round(chunk_ms * RATE / 1000),
                                   round(end * RATE / sr) - round(pos * RATE / sr)))
                mono = mono[:count]
                if len(mono) < count:
                    mono = np.pad(mono, (0, count - len(mono)))
            append_turn(np.clip(mono, -1, 1), final_chunk=end == frames)
            pos = end
    append_turn(end_of_audio=True)
    return "".join(pieces)


def wav_basename(path):
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def read_retry_list(path):
    """Read tab-separated language, original zero-based manifest index and WAV name."""
    targets = {}
    with Path(path).open(encoding="utf-8-sig") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 3:
                raise ValueError(f"{path}:{line_no} 需要用制表符分隔的语种、原行号和 WAV 文件名")
            lang, row_no, name = fields
            if (not re.fullmatch(r"[a-z]{2,3}", lang) or not re.fullmatch(r"[0-9]+", row_no)
                    or not name.lower().endswith(".wav") or wav_basename(name) != name):
                raise ValueError(f"{path}:{line_no} 无效的补跑记录")
            targets.setdefault(lang, {}).setdefault(name, set()).add(int(row_no))
    return targets


def match_retry_rows(rows, lang, targets):
    """Follow unique WAVs across manifest reorderings; disambiguate duplicates by logged index."""
    positions = {name: set() for name in targets}
    for i, row in enumerate(rows):
        audio = row.get("audio") if isinstance(row, dict) else None
        if isinstance(audio, str):
            name = wav_basename(audio)
            if name in positions:
                positions[name].add(i)

    indices = []
    unresolved = 0
    for name, logged in targets.items():
        found = positions[name]
        if len(found) == len(logged) == 1:
            indices.extend(found)
            continue
        indices.extend(found & logged)
        missing = logged - found
        if missing:
            unresolved += len(missing)
            print(f"[ERROR] {lang}/{name} 无法确认原行 {sorted(missing)}，当前同名行 {sorted(found)}",
                  file=sys.stderr, flush=True)
    return sorted(indices), unresolved


def transcribe(row, directory, backend, options, chunk_ms):
    name = wav_basename(row["audio"])
    if not name or not name.lower().endswith(".wav"):
        raise ValueError(f"无效的 WAV 路径: {row['audio']!r}")
    path = directory / "splited_audios" / name
    return decode_pass(path, backend, options, chunk_ms)


def resolve_manifest(directory, manifest_name):
    if manifest_name is not None:
        path = directory / manifest_name
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    paths = [directory / name for name in ("splited_manifest.json", "splited_manifest.jsonl")]
    present = [path for path in paths if path.is_file()]
    if not present:
        raise FileNotFoundError(f"{directory}: 缺少 splited_manifest.json 或 splited_manifest.jsonl")
    if len(present) > 1:
        raise ValueError(f"{directory}: 两个清单都存在，请用 MANIFEST_NAME 选择要写入的文件")
    return present[0]


@dataclass(frozen=True)
class UnparsedLine:
    raw: str
    error: str = ""


def reject_constant(value):
    raise ValueError(f"无效的 JSON 数值: {value}")


def read_manifest(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        if path.suffix.lower() == ".json":
            rows = json.load(source, parse_constant=reject_constant)
            if not isinstance(rows, list):
                raise ValueError("manifest 顶层必须是 JSON 数组")
            return rows
        rows = []
        for line in source:
            if not line.strip():
                rows.append(UnparsedLine(line))
                continue
            try:
                rows.append(json.loads(line, parse_constant=reject_constant))
            except ValueError as exc:
                # Preserve the physical line and its zero-based index for later retries.
                rows.append(UnparsedLine(line, str(exc)))
        return rows


def save_manifest(path, rows):
    # Replace only after the full checkpoint has been written in its original format.
    temp = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent,
                                         prefix=path.name + ".", delete=False) as out:
            temp = Path(out.name)
            if path.suffix.lower() == ".jsonl":
                for row in rows:
                    if isinstance(row, UnparsedLine):
                        out.write(row.raw)
                    else:
                        out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            else:
                json.dump(rows, out, ensure_ascii=False, indent=2, allow_nan=False)
                out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        temp.chmod(path.stat().st_mode & 0o777)
        os.replace(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def run(root, field, languages, make_backend, batch_size, chunk_ms, window_seconds, discard_seconds,
        *, retry_targets=None, manifest_name=None):
    if field in {"audio", "text", "language", "source", "entities", "zh_translation", "en_translation"} or not field.strip():
        raise ValueError("请指定用于模型转录的新字段名")
    if manifest_name is not None and (wav_basename(manifest_name) != manifest_name
                                      or Path(manifest_name).suffix.lower() not in {".json", ".jsonl"}):
        raise ValueError("MANIFEST_NAME 必须是语种目录内的 .json 或 .jsonl 文件名")
    options = FourthStreamOptions(protocol="speaker", number_format="normalized",
                                 chunk_seconds=chunk_ms / 1000, context_policy="window",
                                 window_seconds=window_seconds, discard_seconds=discard_seconds,
                                 enable_recovery=False)
    backend = None
    failures = 0
    if retry_targets is not None:
        languages = [lang for lang in languages if lang in retry_targets]
        disabled = sorted(retry_targets.keys() - set(languages))
        if disabled:
            print(f"[补跑] 配置未启用的名单语种: {', '.join(disabled)}", flush=True)
        count = sum(len(indices) for lang in languages for indices in retry_targets[lang].values())
        print(f"[补跑] 名单共 {count} 条，结果字段 {field}", flush=True)
    try:
        with ThreadPoolExecutor(max_workers=batch_size) as pool:
            for lang in languages:
                try:
                    path = resolve_manifest(Path(root) / lang, manifest_name)
                except (OSError, ValueError) as exc:
                    print(f"[ERROR] {lang}: {exc}", file=sys.stderr, flush=True)
                    failures += 1
                    continue
                with FileLock(str(path) + ".lock"):
                    try:
                        rows = read_manifest(path)
                    except (OSError, ValueError) as exc:
                        print(f"[ERROR] {path}: {exc}", file=sys.stderr, flush=True)
                        failures += 1
                        continue
                    candidates = range(len(rows))
                    if retry_targets is not None:
                        candidates, unresolved = match_retry_rows(rows, lang, retry_targets[lang])
                        failures += unresolved
                    indices = []
                    already_written = 0
                    for i in candidates:
                        row = rows[i]
                        if isinstance(row, UnparsedLine):
                            if row.error:
                                print(f"[ERROR] {lang}[{i}] JSONL 解析失败: {row.error}", file=sys.stderr, flush=True)
                                failures += 1
                            continue
                        if isinstance(row, dict) and isinstance(row.get(field), str):
                            already_written += 1
                            continue  # 已完成，包含成功识别为空串的情况。
                        if not isinstance(row, dict) or row.get(field) is not None:
                            print(f"[ERROR] {lang}[{i}] 记录或目标字段类型错误", file=sys.stderr, flush=True)
                            failures += 1
                            continue
                        indices.append(i)
                    if retry_targets is None:
                        print(f"[{lang}] 总计 {len(rows)}，待识别 {len(indices)}", flush=True)
                    else:
                        print(f"[{lang}] 名单匹配 {len(candidates)}，已写入 {already_written}，待补推 {len(indices)}", flush=True)
                    if not indices:
                        continue
                    if backend is None:
                        backend = BatchBackend(make_backend, max_batch_size=batch_size)
                    jobs = {pool.submit(transcribe, rows[i], path.parent, backend, options, chunk_ms): i
                            for i in indices}
                    dirty = 0
                    try:
                        for done, job in enumerate(as_completed(jobs), 1):
                            i = jobs[job]
                            try:
                                transcript = job.result()
                            except Exception as exc:
                                failures += 1
                                print(f"[ERROR] {lang}[{i}] {rows[i].get('audio')!r}: {exc}",
                                      file=sys.stderr, flush=True)
                            else:
                                rows[i][field] = transcript
                                dirty += 1
                            if dirty >= SAVE_EVERY:
                                save_manifest(path, rows)
                                dirty = 0
                            if done % SAVE_EVERY == 0 or done == len(jobs):
                                print(f"[{lang}] {done}/{len(jobs)}", flush=True)
                    finally:
                        for job in jobs:
                            job.cancel()
                        if dirty:
                            save_manifest(path, rows)
    finally:
        if backend is not None:
            backend.close()
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retry-list", type=Path, help="仅补推 TSV 名单中结果字段还未写入的 WAV")
    args = parser.parse_args()
    targets = read_retry_list(args.retry_list) if args.retry_list is not None else None
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_IDS
    print(f"[Fourth ASR] root={ROOT} field={FIELD} model={MODEL} "
          f"gpus={GPU_IDS} batch={BATCH_SIZE} chunk_ms={CHUNK_MS}", flush=True)
    failed = run(ROOT, FIELD, LANGUAGES, load_model, BATCH_SIZE, CHUNK_MS, WINDOW_SECONDS, DISCARD_SECONDS,
                 retry_targets=targets, manifest_name=MANIFEST_NAME)
    print(f"结束。处理失败数: {failed}", flush=True)
    sys.exit(1 if failed else 0)
