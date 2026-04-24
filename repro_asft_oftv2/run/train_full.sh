#!/bin/bash

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT_DIR/env.example.sh"
source "$ROOT_DIR/configs/full.env"

if [[ "$MODEL_PATH" == "/path/to/base-model" ]]; then
  echo "MODEL_PATH is not set. Edit env.example.sh or export MODEL_PATH before running."
  exit 1
fi

if [[ "$DATA_PATH" == "/path/to/train.jsonl" ]]; then
  echo "DATA_PATH is not set. Edit env.example.sh or export DATA_PATH before running."
  exit 1
fi

mkdir -p "$ADAPTER_OUTPUT_DIR"

python "$ROOT_DIR/src/train_v2.py" \
  --model_name_or_path "$MODEL_PATH" \
  --data_path "$DATA_PATH" \
  --output_dir "$ADAPTER_OUTPUT_DIR" \
  --mode "$MODE" \
  --model_max_length "$MODEL_MAX_LENGTH" \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --global_batch_size "$GLOBAL_BATCH_SIZE" \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --learning_rate "$LEARNING_RATE" \
  --use_adapter "$USE_ADAPTER" \
  --adapter_type "$ADAPTER_TYPE" \
  --target_modules "$TARGET_MODULES" \
  --adapter_bias "$ADAPTER_BIAS" \
  --lora_r "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT" \
  --oft_block_size "$OFT_BLOCK_SIZE" \
  --oft_r "$OFT_R" \
  --oft_module_dropout "$OFT_MODULE_DROPOUT" \
  --use_cayley_neumann "$USE_CAYLEY_NEUMANN" \
  --use_oft_regularizer "$USE_OFT_REGULARIZER" \
  --lambda_oft "$LAMBDA_OFT" \
  --oft_regularizer_type "$OFT_REGULARIZER_TYPE"
