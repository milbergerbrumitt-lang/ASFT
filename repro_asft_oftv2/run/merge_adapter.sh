#!/bin/bash

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT_DIR/env.example.sh"

cat <<EOF
This repro bundle does not claim a fully revalidated OFTv2 merge workflow.

The parent repo currently contains:
  $ORIGINAL_REPO_ROOT/merge_lora.py

If you want to try the existing merge path manually, the template is:

python "$ORIGINAL_REPO_ROOT/merge_lora.py" merge_lora \\
  --base_model_path "$MODEL_PATH" \\
  --lora_model_path "$ADAPTER_OUTPUT_DIR" \\
  --save_path "$MERGED_OUTPUT_DIR"

Current recommendation:
  evaluate the adapter directly with run/eval_adapter.sh
EOF
