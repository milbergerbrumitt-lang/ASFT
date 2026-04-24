#!/bin/bash

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT_DIR/env.example.sh"

if [[ "$MODEL_PATH" == "/path/to/base-model" ]]; then
  echo "MODEL_PATH is not set."
  exit 1
fi

if [[ "$EVAL_DATA_PATH" == "/path/to/eval.jsonl" ]]; then
  echo "EVAL_DATA_PATH is not set."
  exit 1
fi

if [[ ! -d "$ADAPTER_OUTPUT_DIR" ]]; then
  echo "ADAPTER_OUTPUT_DIR does not exist: $ADAPTER_OUTPUT_DIR"
  exit 1
fi

python "$ORIGINAL_REPO_ROOT/eval/medeval/run_med_eval_adapter.py" \
  --base_model_path "$MODEL_PATH" \
  --adapter_path "$ADAPTER_OUTPUT_DIR" \
  --data_path "$EVAL_DATA_PATH"
