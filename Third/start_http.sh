#!/usr/bin/env bash
# Usage: bash start_http.sh MERGED_HF_MODEL [--port 18892] [--batch-size 4]
set -euo pipefail
if (( $# < 1 )); then
  echo "Usage: bash start_http.sh MERGED_HF_MODEL [HTTP service options]" >&2
  exit 2
fi
model="$1"
shift
exec "${PYTHON:-python3}" "$(dirname -- "${BASH_SOURCE[0]}")/tools/serve_http.py" \
  --model "$model" --host 0.0.0.0 "$@"
