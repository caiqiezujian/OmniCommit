#!/usr/bin/env python3
"""Launch fresh multi-turn GSPO through Megatron-SWIFT, not static trajectory replay.

--dry-run imports no CUDA/framework modules and writes nothing. Arguments are
passed as a subprocess argv list, never executed through a shell.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

from asr_gspo.reward import REWARD_VERSION

ROOT = Path(__file__).resolve().parent
AUDITED_SWIFT_COMMIT = "f15bb91439d756d4a366b51956d72d1f088b7af6"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("model", help="Local merged SFT HF checkpoint, including its trained tokenizer")
    p.add_argument("dataset", help="Prepared train.jsonl, NOT trajectories.jsonl")
    p.add_argument("output", help="New run directory, or existing run with --resume-from")
    p.add_argument("--reward-mode", choices=("quality", "quality_latency"), default="quality_latency")
    p.add_argument("--switch-grace-turns", type=int, default=1, choices=range(5),
                   help="Scorer-only confirmation allowance at a true language/speaker boundary; never forces WAIT")
    p.add_argument("--gpu-ids", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3"),
                   help="Four or eight distinct local GPU IDs; eight GPUs with G8/micro2 require global batch >=64")
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--global-batch-size", type=int, help="Defaults to group size x DP x micro-batch")
    p.add_argument("--micro-batch-size", type=int, default=2)
    p.add_argument("--steps-per-generation", type=int, default=1)
    p.add_argument("--train-iters", type=int, default=2000)
    p.add_argument("--max-turns", type=int, default=65, help="Includes EOF for natural-end tasks; annotation_end needs only audio rounds")
    p.add_argument("--max-chunk-seconds", type=float, default=2.0, help="vLLM audio profiling length cap; must cover prepared input_seconds")
    p.add_argument("--image-slots", type=int, choices=(0, 1), default=1,
                   help="Keep one image profiling slot for accepted Omni initialization compatibility; requests stay audio-only")
    p.add_argument("--max-length", type=int, default=8192, help="Rollout prompt budget; vLLM total budget adds max-new-tokens, not a final training truncation cap")
    p.add_argument("--max-new-tokens", type=int, default=128, help="Per audio turn, not entire trajectory")
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--kl-beta", type=float, default=0.01)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--save-total-limit", type=int, default=6, help="Retain all 500-step checkpoints through 2500 steps")
    p.add_argument("--vllm-memory", type=float, default=0.40)
    p.add_argument("--vllm-max-num-seqs", type=int, default=16)
    p.add_argument("--vllm-mm-processor-cache-gb", type=float, default=0.0,
                   help="0 disables multimodal processor/transport caches to bypass the observed vLLM 0.19.1 cache-clear hang; KV prefix caching stays enabled")
    p.add_argument("--recompute-granularity", choices=("full", "selective"), default="full",
                   help="Keep full for the accepted recipe; selective is an opt-in speed/memory experiment")
    p.add_argument("--master-port", type=int, default=29581)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset-num-proc", type=int, default=4)
    p.add_argument("--rollout-log-every", type=int, default=5,
                   help="Log multi-turn collective/generation progress every N rounds (also first, idle, and final rounds)")
    p.add_argument("--resume-from", help="Native Megatron adapter checkpoint including optimizer/RNG; not HF merged weights")
    p.add_argument("--swift-source", help="Optional local audited ms-swift source checkout")
    p.add_argument("--dry-run", action="store_true", help="Print only: no file writes, scoring, or GPU use")
    return p


def parallel_layout(a: argparse.Namespace) -> dict[str, int]:
    """Single-node TP2/EP2 training, with independent four-GPU rollout engines."""
    gpu_ids = [x.strip() for x in a.gpu_ids.split(",")]
    if len(gpu_ids) not in (4, 8) or not all(gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("This recipe requires four or eight distinct GPU IDs, e.g. "
                         "--gpu-ids 0,1,2,3 or --gpu-ids 0,1,2,3,4,5,6,7")
    world_size = len(gpu_ids)
    return {"world_size": world_size, "tensor_parallel_size": 2, "expert_parallel_size": 2,
            "pipeline_parallel_size": 1, "context_parallel_size": 1,
            "data_parallel_size": world_size // 2,
            "vllm_tensor_parallel_size": 4, "vllm_replicas": world_size // 4}


def validate_options(a: argparse.Namespace) -> None:
    layout = parallel_layout(a)
    if a.global_batch_size is None:
        a.global_batch_size = a.group_size * layout["data_parallel_size"] * a.micro_batch_size
    for name in ("group_size", "global_batch_size", "micro_batch_size", "steps_per_generation", "train_iters",
                 "max_turns", "max_length", "max_new_tokens", "lora_rank", "save_steps", "save_total_limit", "vllm_max_num_seqs", "dataset_num_proc", "rollout_log_every"):
        if getattr(a, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if a.group_size < 2:
        raise ValueError("GSPO requires at least two candidates per group")
    if type(a.switch_grace_turns) is not int or not 0 <= a.switch_grace_turns <= 4:
        raise ValueError("switch-grace-turns must be in 0..4")
    generation_batch = a.global_batch_size * a.steps_per_generation
    if generation_batch % a.group_size:
        raise ValueError("global-batch-size * steps-per-generation must be divisible by group-size")
    # EP is not a further divisor of DP. TP2/PP1/CP1 => DP2 on four GPUs, DP4 on eight.
    dp_size = layout["data_parallel_size"]
    if (generation_batch // a.group_size) % (dp_size * a.micro_batch_size):
        raise ValueError(f"Prompt count per generation must be divisible by DP({dp_size}) * micro-batch-size. "
                         f"With G{a.group_size}/micro{a.micro_batch_size}, generation batch must be a multiple "
                         f"of {a.group_size * dp_size * a.micro_batch_size}; generation batch = "
                         "global-batch-size * steps-per-generation.")
    if a.global_batch_size % (dp_size * a.micro_batch_size):
        raise ValueError(f"global-batch-size must be divisible by DP({dp_size}) * micro-batch-size")
    if generation_batch < layout["world_size"] or generation_batch % layout["world_size"]:
        raise ValueError("Generation batch must be at least the GPU count and divisible by it, "
                         "so each rollout rank owns the same number of trajectories")
    if not 0 < a.vllm_memory < 0.9:
        raise ValueError("--vllm-memory must be between 0 and 0.9; GPUs are shared with training")
    if not all(math.isfinite(v) for v in (a.lr, a.temperature, a.kl_beta, a.lora_alpha, a.vllm_memory,
                                        a.max_chunk_seconds, a.vllm_mm_processor_cache_gb)):
        raise ValueError("All floating-point arguments must be finite (no NaN/Infinity)")
    if a.vllm_mm_processor_cache_gb < 0:
        raise ValueError("--vllm-mm-processor-cache-gb must be nonnegative; use 0 to disable multimodal caches")
    if not 0.1 <= a.max_chunk_seconds <= 8:
        raise ValueError("--max-chunk-seconds must be in 0.1..8 seconds")
    if a.lr <= 0 or a.temperature <= 0 or a.kl_beta < 0 or a.lora_alpha <= 0:
        raise ValueError("lr, temperature, lora-alpha must be positive; kl-beta must be nonnegative")
    if a.max_length <= a.max_new_tokens:
        raise ValueError("max-length must exceed per-turn max-new-tokens")
    if not 1024 <= a.master_port <= 65535:
        raise ValueError("master-port must be in 1024..65535")


def build_command(a: argparse.Namespace) -> list[str]:
    """Flags checked against the audited upstream implementation."""
    validate_options(a)
    layout = parallel_layout(a)
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(layout["world_size"]), "--nnodes", "1",
           "--master_port", str(a.master_port), "--module", "swift.cli._megatron.rlhf"]
    pairs = {
        "rlhf_type": "grpo", "model": a.model, "model_type": "qwen3_omni_moe",
        "template": "omnicommit_qwen3_omni", "dataset": a.dataset, "output_dir": a.output,
        "external_plugins": str(ROOT / "swift_plugin.py"), "multi_turn_scheduler": "asr_stream",
        "reward_funcs": "asr_online", "reward_weights": "1", "remove_unused_columns": "false",
        "load_from_cache_file": "false", "split_dataset_ratio": "0", "dataset_shuffle": "true",
        "train_iters": a.train_iters, "global_batch_size": a.global_batch_size,
        "micro_batch_size": a.micro_batch_size, "num_generations": a.group_size,
        "steps_per_generation": a.steps_per_generation, "num_iterations": 1,
        "tuner_type": "lora", "lora_rank": a.lora_rank, "lora_alpha": a.lora_alpha,
        "target_modules": "all-linear", "freeze_llm": "false", "freeze_vit": "true",
        "freeze_aligner": "true", "bf16": "true", "bridge_backend": "mcore-bridge",
        "tensor_model_parallel_size": layout["tensor_parallel_size"],
        "expert_model_parallel_size": layout["expert_parallel_size"],
        "pipeline_model_parallel_size": layout["pipeline_parallel_size"],
        "context_parallel_size": layout["context_parallel_size"],
        "sequence_parallel": "true", "padding_free": "false",
        "moe_grouped_gemm": "true", "moe_permute_fusion": "true", "moe_aux_loss_coeff": "0",
        "recompute_granularity": a.recompute_granularity,
        "gradient_accumulation_fusion": "false", "attention_backend": "flash",
        "lr": a.lr, "min_lr": a.lr * 0.1, "lr_warmup_fraction": "0.03", "clip_grad": "1",
        "beta": a.kl_beta, "importance_sampling_level": "sequence", "epsilon": "0.0003",
        "epsilon_high": "0.0004", "loss_type": "grpo", "scale_rewards": "none",
        "dynamic_sample": "false", "overlong_filter": "false", "truncation_strategy": "delete",
        "max_length": a.max_length, "max_completion_length": a.max_new_tokens,
        "completion_length_limit_scope": "per_round", "max_turns": a.max_turns,
        "temperature": a.temperature, "top_p": "1", "top_k": "-1", "repetition_penalty": "1",
        "use_vllm": "true", "vllm_mode": "colocate", "vllm_tensor_parallel_size": layout["vllm_tensor_parallel_size"],
        "vllm_enable_lora": "false", "vllm_gpu_memory_utilization": a.vllm_memory,
        "vllm_max_model_len": a.max_length + a.max_new_tokens,
        "vllm_max_num_seqs": a.vllm_max_num_seqs, "vllm_enforce_eager": "true",
        "vllm_limit_mm_per_prompt": json.dumps({"audio": {"count": a.max_turns, "length": round(a.max_chunk_seconds * 16000)},
                                                "image": a.image_slots, "video": 0}, separators=(",", ":")),
        "vllm_enable_prefix_caching": "true", "vllm_mm_processor_cache_gb": a.vllm_mm_processor_cache_gb,
        "sleep_level": "2", "offload_model": "true", "offload_optimizer": "true", "offload_bridge": "false",
        "save_safetensors": "true", "merge_lora": "true", "save_missing_weights": "true", "save_steps": a.save_steps,
        "no_save_optim": "false", "no_save_rng": "false", "save_total_limit": a.save_total_limit, "callbacks": "omnicommit_export", "async_save": "false",
        "finetune": "false" if a.resume_from else "true", "no_load_optim": "false", "no_load_rng": "false",
        # Upstream completion logging re-encodes messages without their audios.
        # Keep it disabled for this audio-only recipe. Our complete trajectory
        # and reward JSONL records remain enabled independently.
        "logging_steps": "1", "log_completions": "false", "report_to": "tensorboard",
        "dataset_num_proc": a.dataset_num_proc, "dataloader_num_workers": "0",
        "dataloader_persistent_workers": "false", "seed": a.seed,
    }
    # Selective recomputation rejects recompute_method in the audited framework.
    # Keep the original full-mode flags exactly; never pass them in selective mode.
    if a.recompute_granularity == "full":
        pairs.update({"recompute_method": "uniform", "recompute_num_layers": "1"})
    pairs.update({"advantage_estimator": "grpo", "kl_in_reward": "false"})
    if a.resume_from:
        pairs["mcore_adapter"] = a.resume_from
    for key, value in pairs.items():
        cmd.extend([f"--{key}", str(value)])
    return cmd


def build_environment(a: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": a.gpu_ids, "ENABLE_AUDIO_OUTPUT": "0", "PYTHONUNBUFFERED": "1",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1", "TOKENIZERS_PARALLELISM": "false",
                "ASR_GSPO_REWARD_MODE": a.reward_mode,
                "ASR_GSPO_SWITCH_GRACE_TURNS": str(a.switch_grace_turns),
                "ASR_GSPO_SFT_MODEL": str(Path(a.model).resolve()),
                "ASR_GSPO_LOG_DIR": str(Path(a.output).resolve() / "rewards"),
                "ASR_GSPO_ROLLOUT_LOG_EVERY": str(a.rollout_log_every)})
    paths = [str(ROOT), str(ROOT.parent / "Fourth")]
    if a.swift_source:
        paths.insert(0, str(Path(a.swift_source).resolve()))
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    # Do not force expandable_segments=True: incompatible with some CuMem pools.
    return env


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_framework(swift_source=None):
    """Verify the hooks actually used by this recipe before torchrun spawns workers."""
    if swift_source:
        swift = Path(swift_source) / "swift"
    else:
        spec = importlib.util.find_spec("swift")
        if not spec or not spec.submodule_search_locations:
            raise ValueError("Install the FLEX GSPO ms-swift environment or pass --swift-source")
        swift = Path(next(iter(spec.submodule_search_locations)))
    capabilities = {
        "megatron/trainers/rollout_mixin.py": ("run_multi_turn(", "on_trajectory_start("),
        "megatron/trainers/grpo_trainer.py": ("importance_sampling_level", "seq_level_log_weights"),
        "rl_core/data.py": ("def apply_rollout_output", "rollout_output.rollout_infos", "_MULTIMODAL_KEYS"),
        "rollout/multi_turn.py": ("def get_response_token_data", "def on_turn_end"),
        "megatron/callbacks/base.py": ("def on_save(self, output_dir)",),
        "template/templates/qwen.py": ("class Qwen3OmniTemplate",),
    }
    for relative, required in capabilities.items():
        file = swift / relative
        code = file.read_text(encoding="utf-8") if file.is_file() else ""
        if any(fragment not in code for fragment in required):
            raise ValueError(f"Incompatible Swift source at {file}; FLEX GSPO uses {AUDITED_SWIFT_COMMIT}")


def validate_local_inputs(a: argparse.Namespace) -> dict:
    from finalize_export import verify_weights
    from asr_gspo.audio import encode_chunk_ref
    from asr_gspo.protocol import initial_messages, total_turns
    from fourth_asr.service_backend import detect_model_protocol, validate_manifest
    from transformers import AutoTokenizer

    model = Path(a.model)
    if detect_model_protocol(model) != "speaker":
        raise ValueError("Use a merged Fourth DIRECT/WAIT/SPK_CHANGE checkpoint")
    token_manifest = json.loads((model / "omnicommit_tokens.json").read_text(encoding="utf-8"))
    validate_manifest(AutoTokenizer.from_pretrained(model, local_files_only=True), token_manifest)
    verify_weights(model, token_manifest["token_ids"].values())
    expected_messages = initial_messages()
    digest, count = hashlib.sha256(), 0
    timing_modes, timed_tasks = Counter(), 0
    print("[preflight] Checking prepared rows without decoding waveforms", flush=True)
    with Path(a.dataset).open("rb") as manifest:
        for n, raw in enumerate(manifest, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
                task = json.loads(row["asr_task"])
                msgs = row["messages"]
                if task.get("protocol") != "fourth_speaker_v1":
                    raise ValueError("Re-prepare data with the Fourth GSPO preparer")
                if msgs != expected_messages or row.get("audios") != [encode_chunk_ref(task, 0)]:
                    raise ValueError("expected Fourth prompt and first audio chunk only")
                chunks = task["chunks"]
                if not chunks or total_turns(task) > a.max_turns:
                    raise ValueError("task turn count exceeds --max-turns or input is empty")
                if any(c["input_seconds"] > a.max_chunk_seconds + 1e-8 for c in chunks):
                    raise ValueError("chunk exceeds --max-chunk-seconds")
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise ValueError(f"{a.dataset} line {n}: {exc}") from exc
            count += 1
            timing_modes[task.get("timing_mode", "aligned")] += 1
            timed_tasks += any(a.get("end_ms") is not None for a in task.get("alignment", []))
    if count < a.global_batch_size * a.steps_per_generation // a.group_size:
        raise ValueError("Dataset too small for one complete generation batch")
    if a.swift_source and not (Path(a.swift_source) / "swift" / "cli" / "_megatron" / "rlhf.py").is_file():
        raise ValueError("--swift-source is not an ms-swift checkout with Megatron RLHF")
    if a.resume_from and not Path(a.resume_from).is_dir():
        raise FileNotFoundError(f"Native resume checkpoint missing: {a.resume_from}")
    if a.resume_from:
        resume = Path(a.resume_from)
        tracker = resume / "latest_checkpointed_iteration.txt"
        if not tracker.is_file():
            raise ValueError("--resume-from must point to a native Megatron checkpoint containing latest_checkpointed_iteration.txt")
        iteration = int(tracker.read_text().strip())
        if iteration <= 0 or not (resume / f"iter_{iteration:07d}" / "common.pt").is_file():
            raise ValueError("Native checkpoint is incomplete (missing positive iteration or common.pt)")
        # Swift also copies tracker/common.pt/args.json into the HF -merged
        # export. Those files alone do NOT include trainable/optimizer shards.
        checkpoint_data = resume / f"iter_{iteration:07d}"
        metadata_only = {"common.pt", "metadata.json", ".metadata"}
        has_payload = any(path.is_file() and path.name not in metadata_only and path.stat().st_size > 0
                          for path in checkpoint_data.rglob("*"))
        if resume.name.endswith("-merged") or not has_payload:
            raise ValueError("Resume requires native sharded model/optimizer payloads, not checkpoint-N-merged "
                             "or a directory containing only tracker/common.pt metadata")
        checkpoint_args = resume / "args.json"
        if not checkpoint_args.is_file():
            raise ValueError("Resume checkpoint is missing args.json; cannot verify optimizer/RNG availability")
        saved = json.loads(checkpoint_args.read_text(encoding="utf-8"))
        if saved.get("no_save_optim", True) or saved.get("no_save_rng", True):
            raise ValueError("Checkpoint did not save optimizer/RNG: use accepted SFT weights to start a NEW run instead")
        if a.train_iters <= iteration:
            raise ValueError(f"--train-iters must exceed restored iteration {iteration}")
    output = Path(a.output)
    if output.exists() and any(output.iterdir()) and not a.resume_from:
        raise ValueError("Output directory is nonempty. Choose a new directory or use --resume-from; nothing overwritten.")
    print(f"[preflight] timing_modes={dict(timing_modes)} tasks_with_end_timing={timed_tasks}/{count}", flush=True)
    if not timed_tasks and a.reward_mode == "quality_latency":
        print("[preflight] No completion timing: CER/language/speaker/punctuation rewards are active; "
              "latency penalties and switch grace have no effect on these tasks.", flush=True)
    return {"sha256": digest.hexdigest(), "tasks": count, "timing_modes": dict(timing_modes), "timed_tasks": timed_tasks}


def run_identity(a: argparse.Namespace, manifest: dict) -> dict:
    """Bind resume to data, reward, model metadata and training/code settings.

    The 60+ GB model is not rehashed. Its path, index/config/tokenizer hashes,
    and all safetensors file sizes/mtimes are recorded; this is a metadata
    fingerprint, not a cryptographic content proof of all parameter values.
    """
    model = Path(a.model).resolve()
    metadata_files = ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "omnicommit_tokens.json",
                      "added_tokens.json", "model.safetensors.index.json", "preprocessor_config.json")
    model_metadata = {name: hash_file(model / name) for name in metadata_files if (model / name).is_file()}
    weights = {p.name: {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
               for p in sorted(model.glob("*.safetensors"))}
    ignored = {"output", "dataset", "resume_from", "dry_run", "gpu_ids", "master_port",
               "save_steps", "dataset_num_proc", "rollout_log_every"}
    # Include train_iters: changing the LR decay horizon mid-run is not exact resume.
    settings = {k: v for k, v in vars(a).items() if k not in ignored}
    source_files = [ROOT / "swift_plugin.py", ROOT / "train_gspo.py", ROOT / "finalize_export.py",
                    *sorted((ROOT / "asr_gspo").glob("*.py"))]
    fourth = ROOT.parent / "Fourth" / "fourth_asr"
    source_files.extend(fourth / name for name in ("training_audio.py", "chunk_refs.py", "protocol.py"))
    identity = {"schema": 1, "framework_commit": AUDITED_SWIFT_COMMIT, "model_path": str(model),
            # Physical device IDs may change; world size/DP may not change during exact resume.
            "parallelism": parallel_layout(a),
            "model_metadata_sha256": model_metadata, "weight_file_metadata": weights,
            "dataset_sha256": manifest["sha256"], "dataset_tasks": manifest["tasks"],
            "settings": settings,
            "code_sha256": {p.relative_to(ROOT.parent).as_posix(): hash_file(p) for p in source_files}}
    return identity


def verify_resume_identity(a: argparse.Namespace, identity: dict) -> None:
    if not a.resume_from:
        return
    # Swift may put its checkpoint below output/vN-.../checkpoint-N.
    resume = Path(a.resume_from).resolve()
    manifests = []
    for ancestor in list(resume.parents)[:3]:
        manifests = list(ancestor.glob("launch_*.json"))
        if manifests:
            break
    prior = []
    for path in sorted(manifests, reverse=True):
        document = json.loads(path.read_text(encoding="utf-8"))
        if "run_identity" in document:
            prior.append(document["run_identity"])
    if not prior:
        raise ValueError("Cannot verify resume: no launch_*.json run identity beside the original checkpoint/run")
    if not any(old == identity for old in prior):
        raise ValueError("Resume rejected: model/data/reward/code, parallelism or training settings differ from the original run. "
                         "Restore the original configuration, including train-iters/LR horizon, or start a new run.")


def main(argv=None):
    a = parser().parse_args(argv)
    try:
        cmd = build_command(a)
        env = build_environment(a)
        plan = {"framework_commit": AUDITED_SWIFT_COMMIT, "parallelism": parallel_layout(a),
                "reward_mode": a.reward_mode, "command": cmd, "group_size": a.group_size,
                "reward_version": REWARD_VERSION,
                "switch_grace_turns": a.switch_grace_turns,
                "prompts_per_generation": a.global_batch_size * a.steps_per_generation // a.group_size}
        print(json.dumps(plan, indent=2), flush=True)
        if a.dry_run:
            return 0
        validate_framework(a.swift_source)
        manifest = validate_local_inputs(a)
        plan["data_summary"] = manifest
        plan["run_identity"] = run_identity(a, manifest)
        verify_resume_identity(a, plan["run_identity"])
        output = Path(a.output)
        output.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        with (output / f"launch_{stamp}.json").open("x", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)
        print("[launch] " + shlex.join(cmd), flush=True)
        return subprocess.run(cmd, env=env, check=False).returncode
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        print(f"[launch] {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
