#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

MODEL_PATH=${MODEL_PATH:-"/root/autodl-tmp/models/Qwen2.5-7B-Instruct"}
DATA_PATH=${DATA_PATH:-"$REPO_DIR/tests/fixtures/smoke_sft.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_DIR/output/smoke_oftv2"}

python "$REPO_DIR/train_v2.py" \
  --model_name_or_path "$MODEL_PATH" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --mode sft \
  --use_adapter True \
  --adapter_type oftv2 \
  --target_modules q_proj,v_proj \
  --adapter_bias none \
  --oft_block_size 32 \
  --oft_r 0 \
  --oft_module_dropout 0.0 \
  --use_cayley_neumann True \
  --model_max_length 256 \
  --per_device_train_batch_size 1 \
  --global_batch_size 1 \
  --num_train_epochs 1 \
  --max_steps 1 \
  --learning_rate 5e-4 \
  --precision fp32 \
  --use_cpu True \
  --seed 42
