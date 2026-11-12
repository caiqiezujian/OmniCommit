#!/usr/bin/env bash
# Second protocol (WAIT / incremental text). ms-swift 3.12.1 Megatron reads an MCore base.
# The sibling omnicommit_sft_plugin.py registers omnicommit_second_qwen3_omni.
# Default: eight local GPUs, global batch 16, sequence length 8192.
# System prompts come from the prepared JSONL; no separate --system override.
set -euo pipefail

if (( $# < 3 )); then
  echo "Usage: bash train_sft.sh MCORE_MODEL TRAIN_JSONL OUTPUT_DIR [--gpu-ids 0,1,2,3,4,5,6,7] [--packed-no-flash | --packed-sdpa | --compat-old-driver] [--tokens-file FILE] [native Megatron options...]" >&2
  exit 2
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="$1"
DATASET_PATH="$2"
OUTPUT_PATH="$3"
shift 3
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
ATTENTION_MODE=""
TOKENS_PATH="${DATASET_PATH}.tokens.txt"
EXTRA_ARGS=()
while (( $# )); do
  case "$1" in
    --gpu-ids)
      if (( $# < 2 )); then echo "--gpu-ids requires a value" >&2; exit 2; fi
      GPU_LIST="$2"; shift 2 ;;
    --packed-no-flash|--packed-sdpa|--compat-old-driver)
      if [[ -n "$ATTENTION_MODE" && "$ATTENTION_MODE" != "$1" ]]; then
        echo "$ATTENTION_MODE and $1 select different attention backends; choose one." >&2
        exit 2
      fi
      ATTENTION_MODE="$1"; shift ;;
    --tokens-file)
      if (( $# < 2 )); then echo "--tokens-file requires a value" >&2; exit 2; fi
      TOKENS_PATH="$2"; shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done
IFS=',' read -r -a DEVICES <<< "$GPU_LIST"
GPU_COUNT=${#DEVICES[@]}
case "$GPU_COUNT" in
  8) MICRO_BATCH=2 ;;
  4) MICRO_BATCH=4 ;;
  *) echo "This recipe supports 8 or 4 local GPUs (global batch 16, EP=2). Got $GPU_COUNT." >&2; exit 2 ;;
esac
if [[ ! -d "$MODEL_PATH" || ! -f "$DATASET_PATH" || ! -f "$TOKENS_PATH" ]]; then
  echo "Missing model directory, training JSONL or token inventory. Paths must exist inside this container." >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export ENABLE_AUDIO_OUTPUT=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
ATTENTION_ARGS=(--attention_backend flash)
PACKING_ARGS=(--packing true)
unset OMNICOMMIT_PACKED_NO_FLASH
PACKED_NO_FLASH=0
if [[ -n "$ATTENTION_MODE" ]]; then
  # Keep the TE backend consistent with the sequence layout selected by each preset.
  for arg in "${EXTRA_ARGS[@]}"; do
    case "$arg" in
      --attn_impl|--attn_impl=*|--attention_backend|--attention_backend=*|--packing|--packing=*|--padding_free|--padding_free=*|--transformer_impl|--transformer_impl=*)
        echo "$ATTENTION_MODE controls $arg; remove the conflicting override." >&2
        exit 2 ;;
    esac
  done
  export OMNICOMMIT_SFT_ATTENTION_COMPAT=1
else
  unset OMNICOMMIT_SFT_ATTENTION_COMPAT
fi
if [[ "$ATTENTION_MODE" == "--packed-sdpa" ]]; then
  export NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0 NVTE_UNFUSED_ATTN=0
  ATTENTION_ARGS=(--attn_impl sdpa --attention_backend flash)
  PACKING_ARGS=(--packing true --padding_free true)
  echo "[train] Packing enabled: Omni audio/vision use segmented math SDPA; Megatron uses packed FlashAttention."
  python3 "$SCRIPT_DIR/check_packed_attention.py"
elif [[ "$ATTENTION_MODE" == "--packed-no-flash" || "$ATTENTION_MODE" == "--compat-old-driver" ]]; then
  export NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=0 NVTE_UNFUSED_ATTN=1
  export NVTE_MASKED_SOFTMAX_FUSION=0
  ATTENTION_ARGS=(--attn_impl sdpa --attention_backend unfused)
  MICRO_BATCH=1
  if [[ "$ATTENTION_MODE" == "--packed-no-flash" ]]; then
    PACKED_NO_FLASH=1
    PACKING_ARGS=(--packing true --padding_free true)
    echo "[train] Packing enabled: HF segmented math SDPA; packed Thinker uses per-example PyTorch SDPA, no FlashAttention."
    python3 "$SCRIPT_DIR/check_no_flash_packing.py"
  else
    PACKING_ARGS=(--packing false --padding_free false)
    echo "[train] Old-driver compatibility: HF math SDPA; TE unfused; packing/padding_free=false."
  fi
fi

python3 "$SCRIPT_DIR/check_sft.py" "$DATASET_PATH" --rows 2 --tokens-file "$TOKENS_PATH" --framework
echo "[train] MCore model=$MODEL_PATH | GPUs=$GPU_LIST | output=$OUTPUT_PATH"
OMNICOMMIT_PACKED_NO_FLASH="$PACKED_NO_FLASH" python3 -m torch.distributed.run \
  --nproc_per_node "$GPU_COUNT" --nnodes 1 --master_port "${MASTER_PORT:-29571}" \
  --module swift.cli._megatron.sft \
  --load "$MODEL_PATH" \
  --dataset "$DATASET_PATH" \
  --external_plugins "$SCRIPT_DIR/omnicommit_sft_plugin.py" \
  --template omnicommit_second_qwen3_omni \
  --train_type lora \
  --lora_rank 8 --lora_alpha 32 \
  --target_modules all-linear \
  --modules_to_save word_embeddings output_layer \
  --new_special_tokens "$TOKENS_PATH" \
  --loss_scale default \
  --sequence_parallel true "${PACKING_ARGS[@]}" \
  --freeze_llm false --freeze_vit true --freeze_aligner true \
  --load_from_cache_file false \
  --split_dataset_ratio 0.01 \
  --expert_model_parallel_size 2 \
  --moe_permute_fusion true --moe_grouped_gemm true \
  --moe_shared_expert_overlap true --moe_aux_loss_coeff 1e-3 \
  --micro_batch_size "$MICRO_BATCH" --global_batch_size 16 \
  --recompute_granularity full --recompute_method uniform --recompute_num_layers 1 \
  --finetune true --cross_entropy_loss_fusion false \
  --lr 1e-4 --lr_warmup_fraction 0.05 --min_lr 1e-5 \
  --max_epochs 3 \
  --save "$OUTPUT_PATH" \
  --save_safetensors true --merge_lora false \
  --eval_interval 1000000 --save_interval 3000 \
  --vit_gradient_checkpointing true --max_length 8192 \
  --num_workers 8 --dataset_num_proc 8 \
  --no_save_optim true --no_save_rng true "${ATTENTION_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
