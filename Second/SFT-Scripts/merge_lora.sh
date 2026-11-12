#!/usr/bin/env bash
# Export a trained Second LoRA adapter using the original HF base and training token file.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" -u "$script_dir/merge_lora.py" "$@"
