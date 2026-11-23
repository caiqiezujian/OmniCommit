#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if (( $# == 0 )); then
  source ./web_config.sh
  export CUDA_VISIBLE_DEVICES
  if [[ -n "$SILERO_VAD_MODEL" ]]; then
    if [[ ! -f "$SILERO_VAD_MODEL" || ! -r "$SILERO_VAD_MODEL" ]]; then
      echo "Silero model is missing or unreadable: $SILERO_VAD_MODEL. Set its container path in Fourth/web_config.sh, or set SILERO_VAD_MODEL='' to run without the silence filter." >&2
      exit 2
    fi
    WEB_ARGS+=(--silero-vad-model "$SILERO_VAD_MODEL" --silero-quiet-threshold "$SILERO_QUIET_THRESHOLD")
  fi
  exec "${PYTHON:-python3}" tools/serve_web.py --model "$ASR_MODEL" "${WEB_ARGS[@]}"
fi
model="$1"
shift
exec "${PYTHON:-python3}" tools/serve_web.py --model "$model" --host 0.0.0.0 "$@"
