#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

MODEL_PATH=${MODEL_PATH:-"/root/autodl-tmp/models/Qwen2.5-7B-Instruct"}
DATA_PATH=${DATA_PATH:-"/root/autodl-tmp/data/train_medmcqa_alpaca_10k.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_DIR/output/sft_lora_baseline"}

python "$REPO_DIR/train_v2.py" \
  --model_name_or_path "$MODEL_PATH" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --mode sft \
  --use_adapter True \
  --adapter_type lora \
  --target_modules q_proj,v_proj \
  --adapter_bias none \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --model_max_length 512 \
  --per_device_train_batch_size 4 \
  --global_batch_size 64 \
  --num_train_epochs 3 \
  --max_steps -1 \
  --learning_rate 5e-4 \
  --precision bf16 \
  --seed 42
