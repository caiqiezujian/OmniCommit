#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if (( $# < 1 )); then
  echo "Usage: bash Third/start_server.sh MERGED_HF_MODEL [--chunk-ms MS] [--window-seconds 60] [--discard-seconds 50] [--port PORT]" >&2
  exit 2
fi
model="$1"
shift
exec "${PYTHON:-python3}" tools/serve_web.py --model "$model" --host 0.0.0.0 "$@"
