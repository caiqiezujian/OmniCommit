#!/usr/bin/env bash
# Second protocol (WAIT / incremental text). ms-swift 4.1.3 Megatron reads an original HF base.
# The sibling omnicommit_sft_plugin.py registers omnicommit_second_qwen3_omni.
set -euo pipefail

if (( $# < 3 )); then
  echo "Usage: bash train_sft_swift4.sh HF_MODEL TRAIN_JSONL OUTPUT_DIR [--gpu-ids IDS] [--tokens-file FILE] [native megatron sft options...]" >&2
  exit 2
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HF_MODEL="$1"
DATASET="$2"
OUTPUT_DIR="$3"
shift 3
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TOKENS_FILE="${DATASET}.tokens.txt"
EPOCH_ARGS=(--num_train_epochs 1)
EXTRA_ARGS=()
while (( $# )); do
  case "$1" in
    --gpu-ids|--tokens-file)
      if (( $# < 2 )); then echo "Missing value for $1" >&2; exit 2; fi
      if [[ "$1" == "--gpu-ids" ]]; then GPU_LIST="$2"; else TOKENS_FILE="$2"; fi
      shift 2 ;;
    --train_iters|--train_iters=*|--num_train_epochs|--num_train_epochs=*)
      # SWIFT derives train_iters from the epoch count whenever both are set.
      EPOCH_ARGS=()
      EXTRA_ARGS+=("$1"); shift ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ ! -f "$HF_MODEL/config.json" ]]; then
  echo "Expected the original Hugging Face Qwen3-Omni model with config.json at: $HF_MODEL" >&2
  exit 2
fi
for file in "$DATASET" "$TOKENS_FILE"; do
  if [[ ! -f "$file" ]]; then echo "Missing file: $file" >&2; exit 2; fi
done
IFS=',' read -r -a DEVICES <<< "$GPU_LIST"
export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export NPROC_PER_NODE="${#DEVICES[@]}"
export ENABLE_AUDIO_OUTPUT=0 PYTHONUNBUFFERED=1
export MASTER_PORT="${MASTER_PORT:-29571}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# The SWIFT 3.x fallback patches must never be installed in the native 4.x path.
unset OMNICOMMIT_SFT_ATTENTION_COMPAT OMNICOMMIT_PACKED_NO_FLASH

python3 "$SCRIPT_DIR/check_sft.py" "$DATASET" --rows 2 --tokens-file "$TOKENS_FILE" --framework
echo "[train] HF model=$HF_MODEL | GPUs=$GPU_LIST | output=$OUTPUT_DIR"
exec megatron sft \
  --model "$HF_MODEL" --dataset "$DATASET" --output_dir "$OUTPUT_DIR" --add_version false \
  --external_plugins "$SCRIPT_DIR/omnicommit_sft_plugin.py" --template omnicommit_second_qwen3_omni \
  --tuner_type lora --lora_rank 8 --lora_alpha 32 --target_modules all-linear \
  --new_special_tokens "$TOKENS_FILE" --modules_to_save word_embeddings output_layer \
  --loss_scale default --freeze_llm false --freeze_vit true --freeze_aligner true \
  --packing true --max_length 8192 --attention_backend flash --sequence_parallel true \
  --expert_model_parallel_size 4 --moe_permute_fusion true --moe_grouped_gemm true \
  --moe_shared_expert_overlap true --moe_aux_loss_coeff 1e-3 \
  --micro_batch_size 4 --global_batch_size 32 \
  --recompute_granularity full --recompute_method uniform --recompute_num_layers 1 \
  --finetune true --cross_entropy_loss_fusion true --offload_bridge true \
  --lr 1e-4 --lr_warmup_fraction 0.05 --min_lr 1e-5 "${EPOCH_ARGS[@]}" \
  --save_safetensors true --merge_lora false --no_save_optim true --no_save_rng true \
  --save_steps 1000 --eval_steps 1000000 \
  --split_dataset_ratio 0.01 --load_from_cache_file false \
  --dataloader_pin_memory false --dataloader_num_workers 8 --dataset_num_proc 8 \
  "${EXTRA_ARGS[@]}"
