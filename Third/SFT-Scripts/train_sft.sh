#!/usr/bin/env bash
# Use the same optimizer, packing, distributed settings and checkpoint cadence as Second.
set -euo pipefail
recipe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OMNICOMMIT_SFT_CHECK="$recipe_dir/check_sft.py"
export OMNICOMMIT_SFT_PLUGIN="$recipe_dir/omnicommit_third_sft_plugin.py"
export OMNICOMMIT_SFT_TEMPLATE=omnicommit_third_qwen3_omni
exec bash "$recipe_dir/../../Second/SFT-Scripts/train_sft.sh" "$@"
