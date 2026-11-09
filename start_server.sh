#!/usr/bin/env bash
# FLEX-style launcher: first argument is the exported HF model directory.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if (( $# < 1 )); then
  printf 'Usage: bash start_server.sh /absolute/merged-model [additional service options...]\n' >&2
  exit 2
fi
model="$1"
shift
exec "${PYTHON:-python3}" tools/serve_web.py \
  --backend vllm --model "$model" \
  --host 0.0.0.0 --port 18889 \
  --chunk-seconds 0.5 --number-format normalized \
  --context-policy rolling --max-sessions 1 "$@"
