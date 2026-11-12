#!/usr/bin/env bash
# Serve Second sparse-attribution ASR. First argument: the exported HF model.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if (( $# < 1 )); then
  echo "Usage: bash start_server.sh MERGED_HF_MODEL [--chunk-ms MS] [other service options]" >&2
  exit 2
fi
model="$1"
shift
exec "${PYTHON:-python3}" tools/serve_web.py --model "$model" --host 0.0.0.0 "$@"
