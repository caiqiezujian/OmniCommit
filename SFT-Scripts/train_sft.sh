#!/usr/bin/env bash
set -euo pipefail
recipe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$recipe_dir/recipes/train_mcore.sh" \
  "$recipe_dir/check_sft.py" "$recipe_dir/omnicommit_sft_plugin.py" \
  omnicommit_qwen3_omni "$@"
