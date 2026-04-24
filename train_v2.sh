#!/bin/bash
# train_lora.sh - Script to run ASFT/LoRA training

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TRAIN_SCRIPT=${TRAIN_SCRIPT:-"$SCRIPT_DIR/train_v2.py"}

# ============ User Configurable Variables ============
# Select which GPUs to use (comma-separated)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"6"}

# Path to the pretrained model (make sure the model is downloaded)
MODEL_PATH=${MODEL_PATH:-"/path/to/model"}

# Path to the training data
DATA_PATH=${DATA_PATH:-"data/your-data.jsonl"}

# Output directory
OUTPUT_DIR=${OUTPUT_DIR:-"output/test_train_asft_lora"}

# Training parameters
MODE=${MODE:-"asft"} # Training mode: sft, dft, sft+kl, asft
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-512}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-3}
KL_WEIGHT=${KL_WEIGHT:-0.1}

# ============ LoRA Config ============
USE_LORA=${USE_LORA:-True}
USE_ADAPTER=${USE_ADAPTER:-"$USE_LORA"}
ADAPTER_TYPE=${ADAPTER_TYPE:-"lora"}
TARGET_MODULES=${TARGET_MODULES:-"q_proj,v_proj"}
ADAPTER_BIAS=${ADAPTER_BIAS:-"none"}
LORA_R=${LORA_R:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
OFT_BLOCK_SIZE=${OFT_BLOCK_SIZE:-32}
OFT_R=${OFT_R:-0}
OFT_MODULE_DROPOUT=${OFT_MODULE_DROPOUT:-0.0}
USE_CAYLEY_NEUMANN=${USE_CAYLEY_NEUMANN:-True}
USE_OFT_REGULARIZER=${USE_OFT_REGULARIZER:-False}
LAMBDA_OFT=${LAMBDA_OFT:-0.0}
OFT_REGULARIZER_TYPE=${OFT_REGULARIZER_TYPE:-identity}

# ============ Run Training ============
echo "Starting training with the following settings:"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "MODEL_PATH=$MODEL_PATH"
echo "DATA_PATH=$DATA_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "MODE=$MODE"
echo "MODEL_MAX_LENGTH=$MODEL_MAX_LENGTH"
echo "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE"
echo "LEARNING_RATE=$LEARNING_RATE"
echo "NUM_TRAIN_EPOCHS=$NUM_TRAIN_EPOCHS"
echo "KL_WEIGHT=$KL_WEIGHT"
echo "USE_LORA=$USE_LORA"
echo "USE_ADAPTER=$USE_ADAPTER"
echo "ADAPTER_TYPE=$ADAPTER_TYPE"
echo "TARGET_MODULES=$TARGET_MODULES"
echo "ADAPTER_BIAS=$ADAPTER_BIAS"
echo "LORA_R=$LORA_R"
echo "LORA_ALPHA=$LORA_ALPHA"
echo "LORA_DROPOUT=$LORA_DROPOUT"
echo "OFT_BLOCK_SIZE=$OFT_BLOCK_SIZE"
echo "OFT_R=$OFT_R"
echo "OFT_MODULE_DROPOUT=$OFT_MODULE_DROPOUT"
echo "USE_CAYLEY_NEUMANN=$USE_CAYLEY_NEUMANN"
echo "USE_OFT_REGULARIZER=$USE_OFT_REGULARIZER"
echo "LAMBDA_OFT=$LAMBDA_OFT"
echo "OFT_REGULARIZER_TYPE=$OFT_REGULARIZER_TYPE"

python "$TRAIN_SCRIPT" \
    --mode "$MODE" \
    --model_max_length "$MODEL_MAX_LENGTH" \
    --global_batch_size "$GLOBAL_BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --kl_weight "$KL_WEIGHT" \
    --model_name_or_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --use_adapter "$USE_ADAPTER" \
    --adapter_type "$ADAPTER_TYPE" \
    --target_modules "$TARGET_MODULES" \
    --adapter_bias "$ADAPTER_BIAS" \
    --use_lora "$USE_LORA" \
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
