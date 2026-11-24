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
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

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

    def _run_requests(self, requests):
        try:
            return super()._run_requests(requests)
        except Exception as exc:
            raise RuntimeError(f"vLLM.generate(batch={len(requests)}): {type(exc).__name__}: {exc}") from exc

    def _decode_reply(self, output, expected):
        try:
            reply = super()._decode_reply(output, expected)
        except GenerationTruncated as cutoff:
            # The engine returned these tokens. Keep the whole capped hypothesis
            # for scoring, including any repetition the model actually produced.
            ids = output.outputs[0].token_ids
            reply = {
                "text": self.tokenizer.decode(ids, skip_special_tokens=False,
                                              clean_up_tokenization_spaces=False),
                "finish_reason": "length", "prompt_tokens": cutoff.prompt_tokens,
                "num_cached_tokens": getattr(output, "num_cached_tokens", None),
            }
        # Read on the engine's owner thread, before another group is dispatched.
        reply["request_batch_size"] = getattr(self, "last_batch_size", None)
        completion = output.outputs[0]
        reply.update(finish_reason=completion.finish_reason, stop_reason=getattr(completion, "stop_reason", None),
                     generated_tokens=len(completion.token_ids))
        return reply


def transcript_body(reply):
    text = reply.replace(DIRECT, "").replace(WAIT, "").replace(SPK_CHANGE, "")
    return text if text.strip() else ""


class ManifestStreamSession(FourthStreamSession):
    """Score each generated turn while retaining the model's reply for the next chunk."""

    def _model_reply(self, reply, end_of_audio):
        if not isinstance(reply, str):
            raise TypeError("推理后端没有返回文本字符串")
        text = transcript_body(reply)
        history_reply = CHAT_OR_MEDIA_MARKER.sub(lambda match: "&lt;" + match[0][1:], reply)
        # The action tracks whether any text was committed. An empty DIRECT
        # still appears as the model's original DIRECT in the next prompt.
        return {"text": history_reply, "display_text": text, "segments": [],
                "action": "DIRECT" if text else "WAIT"}

    def _decode_turn(self, history, pcm, end_of_audio):
        generated, parsed = self._generate_turn(history, pcm, end_of_audio)
        if isinstance(generated, dict):
            parsed["raw_reply"] = generated["text"]
            for key in ("request_batch_size", "finish_reason", "stop_reason", "generated_tokens"):
                if key in generated:
                    parsed[key] = generated[key]
        else:
            parsed["raw_reply"] = generated
        return generated, parsed, history


def show_first_prompt(tokenizer, model_path=MODEL):
    system = get_protocol("speaker").system_prompt("normalized")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": "<audio>"}]
    rendered = tokenizer.apply_chat_template(multimodal_messages(messages), tokenize=False,
                                             add_generation_prompt=True)
    print(f"[ASR prompt] model={model_path}\n{rendered}", flush=True)
    if system not in rendered:
        raise ValueError(f"导出目录的 chat template 丢失了 Fourth system 提示词: {model_path}")


def load_model(model_path=MODEL, *, gpu_ids=GPU_IDS, batch_size=BATCH_SIZE,
               chunk_ms=CHUNK_MS, window_seconds=WINDOW_SECONDS):
    if detect_model_protocol(model_path) != "speaker":
        raise ValueError(f"需要带 DIRECT/WAIT/SPK_CHANGE 的 Fourth 合并 HF 模型: {model_path}")
    model = ManifestVLLMBackend(
        model_path, dtype="bfloat16", tensor_parallel_size=len(gpu_ids.split(",")),
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION, max_batch_size=batch_size,
        max_context_tokens=MAX_CONTEXT_TOKENS, max_new_tokens=MAX_NEW_TOKENS,
        max_audio_turns=math.ceil(window_seconds * 1000 / chunk_ms),
        max_audio_seconds=max(.5, chunk_ms / 1000), enable_prefix_caching=True,
    )
    show_first_prompt(model.tokenizer, model_path)
    return model


def record_turn(trace, event):
    raw = event["raw_reply"]
    trace["completed_turns"] += 1
    trace["wait_replies"] += int(raw.strip() == WAIT)
    trace["empty_direct_replies"] += int(DIRECT in raw and not event["display_text"])
    trace["emitted_chars"] += len(event["display_text"])
    size = event.get("request_batch_size")
    if size:
        trace["largest_request_batch"] = max(size, trace.get("largest_request_batch", 0))
    trace["last_replies"].append({
        "turn": event["turn"], "event": event["event"], "end_seconds": event["end_seconds"], "raw": raw[:200],
        "request_batch_size": size, "finish_reason": event.get("finish_reason"),
        "stop_reason": event.get("stop_reason"), "generated_tokens": event.get("generated_tokens"),
        "prompt_tokens": event.get("prompt_tokens"), "num_cached_tokens": event.get("num_cached_tokens"),
    })
    del trace["last_replies"][:-4]


def decode_pass(path, backend, options, chunk_ms, *, trace=None):
    session = ManifestStreamSession(backend, options)
    pieces = []

    def append_turn(audio=None, *, final_chunk=False, end_of_audio=False):
        kind = "end_of_audio" if end_of_audio else "audio"
        if trace is not None:
            trace.update(stage="decode", event=kind, turn=session.turn_count + 1,
                         start_seconds=session.heard_seconds)
            if audio is not None:
                trace["peak_abs"] = max(trace.get("peak_abs", 0.), float(np.max(np.abs(audio))))
        try:
            event = session.step(audio, final_chunk=final_chunk, end_of_audio=end_of_audio)
        except Exception as exc:
            raise RuntimeError(f"{kind} turn={session.turn_count + 1}: {exc}") from exc
        pieces.append(event["display_text"])
        if trace is not None:
            record_turn(trace, event)
        if event.get("finish_reason") == "length":
            print(f"[LENGTH] {path.name} {event['event']} turn={event['turn']} 已保留本轮生成的正文", flush=True)

    with sf.SoundFile(path) as wav:
        frames, sr = len(wav), wav.samplerate
        if trace is not None:
            trace.update(stage="read_wav", duration_seconds=frames / sr, sample_rate=sr, channels=wav.channels)
        if not frames:
            raise ValueError("空 WAV")
        divisor = math.gcd(sr, RATE)
        pos = turn = 0
        while pos < frames:
            turn += 1
            if trace is not None:
                trace.update(stage="read_wav", event="audio", turn=turn, start_seconds=pos / sr)
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


def match_retry_rows(rows, lang, targets, *, on_unmatched=None):
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
            if on_unmatched is not None:
                for i in sorted(missing):
                    on_unmatched(i, name, sorted(found))
    return sorted(indices), unresolved


def transcribe(row, directory, backend, options, chunk_ms, *, trace=None):
    audio = row.get("audio")
    if not isinstance(audio, str):
        raise ValueError(f"audio 必须是 WAV 路径，实际为 {audio!r}")
    name = wav_basename(audio)
    if not name or not name.lower().endswith(".wav"):
        raise ValueError(f"无效的 WAV 路径: {row['audio']!r}")
    path = directory / "splited_audios" / name
    if trace is not None:
        trace.update(stage="open_wav", wav_path=str(path))
    if not path.is_file():
        raise FileNotFoundError(path)
    return decode_pass(path, backend, options, chunk_ms, trace=trace)


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


def transcribe_attempt(row, directory, backend, options, chunk_ms, number, workers):
    trace = {"attempt": number, "concurrent_wavs": workers, "stage": "resolve_audio",
             "completed_turns": 0, "wait_replies": 0, "empty_direct_replies": 0,
             "emitted_chars": 0, "last_replies": []}
    try:
        text = transcribe(row, directory, backend, options, chunk_ms, trace=trace)
    except Exception as exc:
        cause = exc
        while cause.__cause__ is not None:
            cause = cause.__cause__
        trace.update(status="error", error_type=type(cause).__name__, error=str(exc))
        return None, trace
    if text.strip():
        trace["status"] = "ok"
        return text, trace
    trace["status"] = ("all_wait" if trace["completed_turns"] and trace["wait_replies"] == trace["completed_turns"]
                       else "empty_transcript")
    return None, trace


def write_failure(report, context, **details):
    report.write(json.dumps({**context, **details}, ensure_ascii=False, allow_nan=False) + "\n")
    report.flush()


def run_rows(path, rows, indices, field, backend, options, batch_size, context, report):
    """Finish a pass before starting the next one. The enclosing manifest lock covers all three."""
    lang = context["language"]
    attempts = {}
    limits = (batch_size, max(1, batch_size // 2), 1)
    failed = 0
    for number, limit in enumerate(limits, 1):
        if not indices:
            break
        workers = min(limit, len(indices))
        print(f"[{lang}] 第 {number}/3 遍，待识别 {len(indices)}，并发 WAV {workers}", flush=True)
        pending = []
        dirty = succeeded = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = {pool.submit(transcribe_attempt, rows[i], path.parent, backend, options,
                                context["chunk_ms"], number, workers): i for i in indices}
            try:
                for done, job in enumerate(as_completed(jobs), 1):
                    i = jobs.pop(job)
                    text, trace = job.result()
                    if text is not None:
                        rows[i][field] = text
                        succeeded += 1
                        dirty += 1
                        previous = attempts.pop(i, [])
                        if previous:
                            last = previous[-1]
                            print(f"[恢复] {lang}[{i}] {rows[i]['audio']!r} 第 {number} 遍，"
                                  f"上次={last.get('error', last['status'])}，已返回回复中的最大请求 batch "
                                  f"{last.get('largest_request_batch', '未知')} -> "
                                  f"{trace.get('largest_request_batch', '未知')}", flush=True)
                    else:
                        history = attempts.setdefault(i, [])
                        history.append(trace)
                        reason = trace.get("error", trace["status"])
                        if number < len(limits):
                            pending.append(i)
                            print(f"[待重试] {lang}[{i}] 第 {number} 遍：{reason}", flush=True)
                        else:
                            audio = rows[i].get("audio")
                            write_failure(report, context, kind="transcription", index=i, audio=audio,
                                          attempts=history)
                            rows[i][field] = ""
                            dirty += 1
                            failed += 1
                            del attempts[i]
                            print(f"[ERROR] {lang}[{i}] {audio!r} 三遍未成功：{reason}", file=sys.stderr, flush=True)
                    if dirty >= SAVE_EVERY:
                        save_manifest(path, rows)
                        dirty = 0
                    if done % SAVE_EVERY == 0 or done == len(indices):
                        print(f"[{lang}] 第 {number} 遍 {done}/{len(indices)}，成功 {succeeded}", flush=True)
            finally:
                for job in jobs:
                    job.cancel()
                if dirty:
                    save_manifest(path, rows)
        indices = sorted(pending)
    return failed


def run(root, field, languages, make_backend, batch_size, chunk_ms, window_seconds, discard_seconds,
        *, retry_targets=None, manifest_name=None, resume=False, model_path=None):
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
    # One report for all languages in this invocation. Another checkpoint or a
    # targeted rerun must never truncate a previous run's diagnostics.
    prefix = f"failed-{quote(field, safe='')}-{datetime.now():%Y%m%d-%H%M%S}-"
    try:
        with tempfile.NamedTemporaryFile("w", dir=root, prefix=prefix, suffix=".jsonl", delete=False,
                                         encoding="utf-8", errors="backslashreplace") as report:
            print(f"[失败清单] {report.name}", flush=True)
            for lang in languages:
                directory = Path(root) / lang
                context = {"language": lang, "directory": str(directory), "field": field, "chunk_ms": chunk_ms,
                           "window_seconds": window_seconds, "discard_seconds": discard_seconds,
                           "batch_size": batch_size, "model": str(model_path) if model_path is not None else None}
                try:
                    path = resolve_manifest(directory, manifest_name)
                except (OSError, ValueError) as exc:
                    write_failure(report, context, kind="manifest", error=str(exc), attempts=[])
                    print(f"[ERROR] {lang}: {exc}", file=sys.stderr, flush=True)
                    failures += 1
                    continue
                context["manifest"] = str(path)
                with FileLock(str(path) + ".lock"):
                    try:
                        rows = read_manifest(path)
                    except (OSError, ValueError) as exc:
                        write_failure(report, context, kind="manifest", error=str(exc), attempts=[])
                        print(f"[ERROR] {path}: {exc}", file=sys.stderr, flush=True)
                        failures += 1
                        continue
                    candidates = range(len(rows))
                    if retry_targets is not None:
                        def report_unmatched(i, name, found):
                            write_failure(report, context, kind="unmatched_target", index=i, audio=name,
                                          matching_indices=found, error="无法定位补跑名单中的原记录", attempts=[])

                        candidates, unresolved = match_retry_rows(rows, lang, retry_targets[lang],
                                                                  on_unmatched=report_unmatched)
                        failures += unresolved
                    indices = []
                    for i in candidates:
                        row = rows[i]
                        if isinstance(row, UnparsedLine):
                            if row.error:
                                write_failure(report, context, kind="invalid_record", index=i,
                                              error=row.error, attempts=[])
                                print(f"[ERROR] {lang}[{i}] JSONL 解析失败: {row.error}", file=sys.stderr, flush=True)
                                failures += 1
                            continue
                        if not isinstance(row, dict):
                            write_failure(report, context, kind="invalid_record", index=i,
                                          error="记录必须是 JSON 对象", attempts=[])
                            print(f"[ERROR] {lang}[{i}] 记录必须是 JSON 对象", file=sys.stderr, flush=True)
                            failures += 1
                            continue
                        if resume and isinstance(row.get(field), str) and transcript_body(row[field]):
                            continue
                        indices.append(i)
                    if retry_targets is None:
                        print(f"[{lang}] 总计 {len(rows)}，待识别 {len(indices)}", flush=True)
                    else:
                        print(f"[{lang}] 名单匹配 {len(candidates)}，待识别 {len(indices)}", flush=True)
                    if not indices:
                        continue
                    if backend is None:
                        backend = BatchBackend(make_backend, max_batch_size=batch_size)
                    failures += run_rows(path, rows, indices, field, backend, options, batch_size, context, report)
            print(f"[失败清单] {report.name}（{failures} 条）", flush=True)
    finally:
        if backend is not None:
            backend.close()
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retry-list", type=Path, help="仅重新转录 TSV 名单中匹配的 WAV")
    parser.add_argument("--resume", action="store_true", help="只跳过已有非空正文，缺失或空结果继续识别")
    args = parser.parse_args()
    targets = read_retry_list(args.retry_list) if args.retry_list is not None else None
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_IDS
    print(f"[Fourth ASR] root={ROOT} field={FIELD} model={MODEL} "
          f"gpus={GPU_IDS} batch={BATCH_SIZE} chunk_ms={CHUNK_MS}", flush=True)
    failed = run(ROOT, FIELD, LANGUAGES, load_model, BATCH_SIZE, CHUNK_MS, WINDOW_SECONDS, DISCARD_SECONDS,
                 retry_targets=targets, manifest_name=MANIFEST_NAME, resume=args.resume, model_path=MODEL)
    print(f"结束。处理失败数: {failed}", flush=True)
    sys.exit(1 if failed else 0)
