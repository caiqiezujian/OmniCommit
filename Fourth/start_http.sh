#!/usr/bin/env bash
# No arguments: use http_config.sh. With a model path: use the caller's arguments.
set -euo pipefail
if (( $# == 0 )); then
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
  source ./http_config.sh
  export CUDA_VISIBLE_DEVICES
  case "${PACKET_PERIOD_TO_COMMA:-false}" in
    true) HTTP_ARGS+=(--packet-period-to-comma) ;;
    false) ;;
    *) echo 'PACKET_PERIOD_TO_COMMA must be true or false' >&2; exit 2 ;;
  esac
  if [[ -v PACKET_PERIOD_THRESHOLD_MS ]]; then
    HTTP_ARGS+=(--packet-period-threshold-ms "$PACKET_PERIOD_THRESHOLD_MS")
  fi
  exec "${PYTHON:-python3}" tools/serve_http.py --model "$ASR_MODEL" "${HTTP_ARGS[@]}"
fi
model="$1"
shift
exec "${PYTHON:-python3}" "$(dirname -- "${BASH_SOURCE[0]}")/tools/serve_http.py" \
  --model "$model" --host 0.0.0.0 "$@"
