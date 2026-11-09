#!/usr/bin/env python3
"""Read-only SFT sample preflight; no weights, teacher calls or GPU training."""
import argparse
import importlib.metadata
import io
import json
from pathlib import Path
import re
import sys

from omnicommit_audio import SAMPLE_RATE, decode_chunk_ref, chunk_wav_bytes
from prepare_sft import ACTION_TOKENS, LANGUAGE, SPEAKER, number, END_OF_AUDIO


def check_content(content, allow_newlines=False):
    if content == "<|WAIT|>":
        return {"<|WAIT|>"}
    if not isinstance(content, str) or not content.startswith("<|DIRECT|>"):
        raise ValueError("Assistant must return DIRECT with tagged text or exactly WAIT")
    tokens = {"<|DIRECT|>"}
    body = content[len("<|DIRECT|>"):]
    if allow_newlines:
        matches = list(re.finditer(r"<([^<>]+)><([^<>]+)>([^<>]+)", body))
        if not matches or "".join(m.group(0) for m in matches) != body:
            raise ValueError("Each DIRECT attribution must start with a full language/speaker pair")
        lines = [m.group(0) for m in matches]
    else:
        lines = body.split("\n")
    for line in lines:
        pattern = r"<([^<>]+)><([^<>]+)>([^<>]+)" if allow_newlines else r"<([^<>]+)><([^<>]+)>([^<>\r\n]+)"
        match = re.fullmatch(pattern, line)
        if not match:
            raise ValueError("Each DIRECT segment must be <language><speaker>nonempty-text")
        language, speaker, text = match.groups()
        if not LANGUAGE.fullmatch(language) or not SPEAKER.fullmatch(speaker) or not text.strip():
            raise ValueError("Invalid segment language, speaker or empty text")
        tokens.update((f"<{language}>", f"<{speaker}>"))
    return tokens


def check_row(row, decode=True, token_inventory=None):
    import soundfile as sf
    from omnicommit.runtime_protocol import speaker_reference_text, LIVE_AUDIO_PREFIX

    # Compact, directly trainable rows produced by build_asr_sft.py contain only
    # audios/messages.  Virtual refs already carry their exact input durations,
    # so derive those solely for preflight; never modify the dataset or infer
    # speaker references/windows that the compact row does not describe.
    if isinstance(row, dict) and set(row) == {"audios", "messages"}:
        refs = row.get("audios")
        if not isinstance(refs, list) or not refs:
            raise ValueError("audios must be a nonempty array of virtual references")
        row = dict(row, schema_version=2,
                   audio_chunk_durations=[decode_chunk_ref(ref)["input_seconds"] for ref in refs])

    if not isinstance(row, dict) or row.get("schema_version") not in (1, 2):
        raise ValueError("Expected a schema_version=1 or 2 SFT object")
    refs, durations, messages = (row.get(k) for k in ("audios", "audio_chunk_durations", "messages"))
    if not isinstance(refs, list) or not refs:
        raise ValueError("audios must be a nonempty array of virtual references")
    if not isinstance(durations, list) or len(refs) != len(durations):
        raise ValueError("audio_chunk_durations must match audios")
    references = row.get("speaker_references", [])
    if not isinstance(references, list):
        raise ValueError("speaker_references must be a list")
    n_reference = len(references)
    live_refs = refs[n_reference:]
    if not live_refs:
        raise ValueError("At least one live audio chunk is required")
    window = row.get("window")
    if window is not None:
        if not isinstance(window, dict) or row["schema_version"] != 2 or window.get("safe_start") is not True:
            raise ValueError("Invalid or unsafe training window")
        start_seconds = number(window.get("start_ms"), "window.start_ms") / 1000
        end_seconds = number(window.get("end_ms"), "window.end_ms") / 1000
        if start_seconds < 0 or end_seconds <= start_seconds or not isinstance(window.get("ends_meeting"), bool):
            raise ValueError("Invalid window bounds")
    else:
        start_seconds, end_seconds = 0., None
    eos_count = int(row["schema_version"] == 2 and (window is None or window["ends_meeting"]))
    if not isinstance(messages, list) or len(messages) != 1 + 2 * (len(live_refs) + eos_count):
        raise ValueError("Expected system, live audio user/assistant pairs and optional final EOS pair")
    if messages[0].get("role") != "system" or not isinstance(messages[0].get("content"), str) or not messages[0]["content"].strip():
        raise ValueError("Missing nonempty system message")
    source_path = decode_chunk_ref(live_refs[0])["path"]
    if not Path(source_path).is_absolute():
        raise ValueError("Source audio must use an absolute training-host path")
    info = sf.info(source_path)
    actual_seconds = info.frames / info.samplerate
    if end_seconds is None:
        end_seconds = actual_seconds
    if end_seconds > actual_seconds + 1e-6 or (eos_count and abs(end_seconds - actual_seconds) > 1e-6):
        raise ValueError("Window/EOS does not match source recording end")
    previous_end, direct_count = start_seconds, 0
    samples, used_tokens, prefixes, seen_speakers = [], set(), [], set()
    for i, ref in enumerate(refs):
        data = decode_chunk_ref(ref)
        if data["path"] != source_path or data["end"] > actual_seconds + 1e-8:
            raise ValueError(f"audio {i}: wrong source or range beyond recording")
        if abs(number(durations[i], "audio_chunk_durations") - data["input_seconds"]) > 1e-8:
            raise ValueError(f"audio {i}: virtual and declared durations differ")
        if i < n_reference:
            reference = references[i]
            if reference.get("verified") is not True or reference.get("overlap") is not False or reference.get("clean") is not True:
                raise ValueError("Speaker reference must be verified, clean and non-overlapping")
            if data["end"] > start_seconds + 1e-6 or data["start"] < 0:
                raise ValueError("Speaker reference must be entirely before the live window")
            if Path(reference.get("audio", "")).resolve() != Path(source_path).resolve():
                raise ValueError("Speaker reference metadata must identify the same meeting audio")
            if abs(number(reference.get("start_ms"), "reference.start_ms") / 1000 - data["start"]) > 1e-6 or abs(number(reference.get("end_ms"), "reference.end_ms") / 1000 - data["end"]) > 1e-6:
                raise ValueError("Speaker reference metadata interval differs from actual audio input")
            speaker, language = reference.get("speaker"), reference.get("language")
            if speaker in seen_speakers:
                raise ValueError("Only one reference per speaker is allowed")
            prefixes.append(speaker_reference_text(speaker, language, reference.get("text")))
            seen_speakers.add(speaker)
            used_tokens.update((f"<{speaker}>", f"<{language}>"))
        else:
            j = i - n_reference
            if abs(data["start"] - previous_end) > 1e-6:
                raise ValueError(f"turn {j}: audio ranges must be contiguous")
            user, assistant = messages[1 + 2*j:3 + 2*j]
            expected_user = "".join(prefixes) + LIVE_AUDIO_PREFIX + "<audio>" if j == 0 and prefixes else "<audio>"
            if user.get("role") != "user" or user.get("content") != expected_user:
                raise ValueError(f"turn {j}: live/reference audio placeholders or conditioning text mismatch")
            if assistant.get("role") != "assistant":
                raise ValueError(f"turn {j}: expected assistant")
            tokens = check_content(assistant.get("content"), allow_newlines=row["schema_version"] == 2)
            used_tokens.update(tokens)
            direct_count += "<|DIRECT|>" in tokens
            previous_end = data["end"]
        if decode:
            pcm, sr = sf.read(io.BytesIO(chunk_wav_bytes(ref)), dtype="float32")
            if sr != SAMPLE_RATE or pcm.ndim != 1 or len(pcm) != round(data["input_seconds"] * sr):
                raise ValueError(f"audio {i}: invalid materialized waveform")
            samples.append(len(pcm))
    if abs(previous_end - end_seconds) > 1e-6:
        raise ValueError("Final audio range must cover the window/recording end")
    if eos_count:
        user, assistant = messages[-2:]
        if user.get("role") != "user" or user.get("content") != END_OF_AUDIO or assistant.get("role") != "assistant":
            raise ValueError("schema_version=2 requires final text-only end_of_audio pair")
        tokens = check_content(assistant.get("content"), allow_newlines=row["schema_version"] == 2)
        used_tokens.update(tokens)
        direct_count += "<|DIRECT|>" in tokens
    if not direct_count:
        spans = row.get("source_speech_spans")
        silence = (window is not None and window.get("validated_silence") is True
                   and window.get("parent_has_transcription") is True
                   and window.get("safe_start") is True and window.get("safe_end") is True
                   and isinstance(spans, list) and bool(spans))
        if silence:
            for span in spans:
                begin = number(span.get("start_ms"), "speech.start_ms") / 1000
                finish = number(span.get("end_ms"), "speech.end_ms") / 1000
                if not 0 <= begin < finish <= actual_seconds + 1e-6:
                    raise ValueError("Invalid source speech evidence for silence window")
                if begin < end_seconds and finish > start_seconds:
                    raise ValueError("All-WAIT window overlaps actual source speech")
        else:
            raise ValueError("All-WAIT sample has no validated silence-window evidence")
    if token_inventory is not None and not used_tokens.issubset(token_inventory):
        raise ValueError(f"Missing special tokens: {sorted(used_tokens - token_inventory)}")
    return {"id": row.get("id"), "audio_turns": len(live_refs), "direct_turns": direct_count,
            "wait_turns": len(live_refs) + eos_count - direct_count, "eos_turns": eos_count,
            "reference_audios": n_reference, "decoded": decode,
            "input_seconds": sum(durations), "sample_counts": samples}


def read_tokens(path):
    tokens = path.read_text(encoding="utf-8-sig").splitlines()
    if tokens[:2] != ACTION_TOKENS or len(tokens) != len(set(tokens)):
        raise ValueError("Token file must begin WAIT, DIRECT and contain no duplicate tokens")
    for token in tokens[2:]:
        if not token.startswith("<") or not token.endswith(">") or not (LANGUAGE.fullmatch(token[1:-1]) or SPEAKER.fullmatch(token[1:-1])):
            raise ValueError(f"Invalid language/speaker special token: {token!r}")
    return set(tokens)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--tokens-file", type=Path)
    parser.add_argument("--framework", action="store_true", help="Import SWIFT plugin without loading weights")
    args = parser.parse_args(argv)
    if args.rows < 1:
        parser.error("--rows must be positive")
    try:
        inventory = read_tokens(args.tokens_file or Path(str(args.dataset) + ".tokens.txt"))
        count = 0
        with args.dataset.open(encoding="utf-8-sig") as source:
            for line_no, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    result = check_row(json.loads(line), token_inventory=inventory)
                except Exception as exc:
                    raise ValueError(f"line {line_no}: {exc}") from exc
                print(json.dumps(result, ensure_ascii=False), flush=True)
                count += 1
                if count == args.rows:
                    break
        if not count:
            raise ValueError("Dataset has no samples")
        if args.framework:
            import omnicommit_sft_plugin
            print(f"[check] ms-swift={importlib.metadata.version('ms-swift')} "
                  f"template={omnicommit_sft_plugin._meta.template_type}", flush=True)
        print(json.dumps({"ok": True, "sampled_rows_checked": count, "full_dataset_checked": False,
                          "framework_import_checked": args.framework,
                          "real_template_encoding_checked": False, "gpu_training_verified": False}))
        return 0
    except Exception as exc:
        print(f"[check] ERROR: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
