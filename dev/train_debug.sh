#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
TRAIN_SCRIPT=${TRAIN_SCRIPT:-"$SCRIPT_DIR/train_debug.py"}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
export HF_HOME=${HF_HOME:-"$REPO_DIR/experiment/cache/hf"}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-"$REPO_DIR/experiment/cache/hf"}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-"$REPO_DIR/experiment/cache"}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-"$REPO_DIR/experiment/cache/triton"}
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR"

MODEL_PATH=${MODEL_PATH:-"/volume/demo/xlzhuang/zh/models/Llama-2-7b-hf"}
DATA_PATH=${DATA_PATH:-"$REPO_DIR/data/train_medmcqa_alpaca_10k.jsonl"}

MODE=${MODE:-"asft"}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-512}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-3}
KL_WEIGHT=${KL_WEIGHT:-0.05}

PRECISION=${PRECISION:-"bf16"}
RUN_TYPE=${RUN_TYPE:-"baseline"} # baseline | zero2 | zero3
TINY_RUN=${TINY_RUN:-"0"}
SEED=${SEED:-42}
NUM_GPUS=${NUM_GPUS:-8}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-"False"}
USE_ADAPTER=${USE_ADAPTER:-"False"}
ADAPTER_TYPE=${ADAPTER_TYPE:-"lora"}
TARGET_MODULES=${TARGET_MODULES:-"q_proj,v_proj"}
ADAPTER_BIAS=${ADAPTER_BIAS:-"none"}
LORA_R=${LORA_R:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
OFT_BLOCK_SIZE=${OFT_BLOCK_SIZE:-32}
OFT_R=${OFT_R:-0}
OFT_MODULE_DROPOUT=${OFT_MODULE_DROPOUT:-0.0}
USE_CAYLEY_NEUMANN=${USE_CAYLEY_NEUMANN:-"True"}
USE_OFT_REGULARIZER=${USE_OFT_REGULARIZER:-"False"}
LAMBDA_OFT=${LAMBDA_OFT:-0.0}
OFT_REGULARIZER_TYPE=${OFT_REGULARIZER_TYPE:-identity}

DS_CONFIG_BF16=${DS_CONFIG_BF16:-"$REPO_DIR/experiment/med_asft_debug/config/ds_zero2_bf16.json"}
DS_CONFIG_FP16=${DS_CONFIG_FP16:-"$REPO_DIR/experiment/med_asft_debug/config/ds_zero2_fp16.json"}
DS_CONFIG_Z3_BF16=${DS_CONFIG_Z3_BF16:-"$REPO_DIR/experiment/med_asft_debug/config/ds_zero3_bf16.json"}
DS_CONFIG_Z3_FP16=${DS_CONFIG_Z3_FP16:-"$REPO_DIR/experiment/med_asft_debug/config/ds_zero3_fp16.json"}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_NAME=${EXP_NAME:-"med_asft_${RUN_TYPE}_${PRECISION}"}
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_DIR/experiment/results/trained/${EXP_NAME}_${TIMESTAMP}"}
LOG_DIR=${LOG_DIR:-"$REPO_DIR/experiment/results/logs"}
mkdir -p "$LOG_DIR"

TRAIN_LOG="$LOG_DIR/${EXP_NAME}_${TIMESTAMP}.log"
EVAL_LOG="$LOG_DIR/${EXP_NAME}_${TIMESTAMP}_eval.log"
EVAL_JSON="$LOG_DIR/${EXP_NAME}_${TIMESTAMP}_eval.json"
EVAL_TASK=${EVAL_TASK:-"med"} # med | math | none
MATH_PROMPT_TYPE=${MATH_PROMPT_TYPE:-"qwen25-math-cot"}
MATH_N_SAMPLING=${MATH_N_SAMPLING:-1}
MATH_TEMPERATURE=${MATH_TEMPERATURE:-0}
MATH_OUTPUT_DIR="${LOG_DIR}/${EXP_NAME}_${TIMESTAMP}_math_eval"

MAX_STEPS=-1
if [[ "$TINY_RUN" == "1" ]]; then
  MAX_STEPS=2
  NUM_TRAIN_EPOCHS=1
fi

COMMON_ARGS=(
  --mode "$MODE"
  --model_max_length "$MODEL_MAX_LENGTH"
  --global_batch_size "$GLOBAL_BATCH_SIZE"
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
  --learning_rate "$LEARNING_RATE"
  --num_train_epochs "$NUM_TRAIN_EPOCHS"
  --kl_weight "$KL_WEIGHT"
  --model_name_or_path "$MODEL_PATH"
  --data_path "$DATA_PATH"
  --output_dir "$OUTPUT_DIR"
  --use_adapter "$USE_ADAPTER"
  --adapter_type "$ADAPTER_TYPE"
  --target_modules "$TARGET_MODULES"
  --adapter_bias "$ADAPTER_BIAS"
  --use_lora False
  --lora_r "$LORA_R"
  --lora_alpha "$LORA_ALPHA"
  --lora_dropout "$LORA_DROPOUT"
  --oft_block_size "$OFT_BLOCK_SIZE"
  --oft_r "$OFT_R"
  --oft_module_dropout "$OFT_MODULE_DROPOUT"
  --use_cayley_neumann "$USE_CAYLEY_NEUMANN"
  --use_oft_regularizer "$USE_OFT_REGULARIZER"
  --lambda_oft "$LAMBDA_OFT"
  --oft_regularizer_type "$OFT_REGULARIZER_TYPE"
  --precision "$PRECISION"
  --seed "$SEED"
  --max_steps "$MAX_STEPS"
  --gradient_checkpointing "$GRADIENT_CHECKPOINTING"
)

if [[ "$RUN_TYPE" == "zero2" || "$RUN_TYPE" == "zero3" ]]; then
  DS_CONFIG="$DS_CONFIG_BF16"
  if [[ "$PRECISION" == "fp16" ]]; then
    DS_CONFIG="$DS_CONFIG_FP16"
  fi
  if [[ "$RUN_TYPE" == "zero3" ]]; then
    DS_CONFIG="$DS_CONFIG_Z3_BF16"
    if [[ "$PRECISION" == "fp16" ]]; then
      DS_CONFIG="$DS_CONFIG_Z3_FP16"
    fi
  fi
  echo "[Train] DeepSpeed ${RUN_TYPE} with config: $DS_CONFIG"
  deepspeed --num_gpus "$NUM_GPUS" "$TRAIN_SCRIPT" \
    --deepspeed_config "$DS_CONFIG" \
    "${COMMON_ARGS[@]}" \
    >> "$TRAIN_LOG" 2>&1
else
  echo "[Train] Baseline DDP"
  if [[ "$NUM_GPUS" == "1" ]]; then
    python "$TRAIN_SCRIPT" \
      "${COMMON_ARGS[@]}" \
      >> "$TRAIN_LOG" 2>&1
  else
    torchrun --standalone --nproc_per_node "$NUM_GPUS" "$TRAIN_SCRIPT" \
      "${COMMON_ARGS[@]}" \
      >> "$TRAIN_LOG" 2>&1
  fi
fi

if [[ "$EVAL_TASK" == "med" ]]; then
  export CUDA_VISIBLE_DEVICES=7
  python "$REPO_DIR/eval/medeval/run_med_eval.py" \
    --model "$OUTPUT_DIR" \
    --test_data_dir "$REPO_DIR/eval/medeval/test_data" \
    --tensor_parallel_size 1 \
    --output_json "$EVAL_JSON" \
    >> "$EVAL_LOG" 2>&1
elif [[ "$EVAL_TASK" == "math" ]]; then
  export CUDA_VISIBLE_DEVICES=7
  (
    cd "$REPO_DIR/eval/math_evaluation" || exit 1
    bash "sh/eval.sh" \
      "$MATH_PROMPT_TYPE" \
      "$OUTPUT_DIR" \
      "$MATH_OUTPUT_DIR" \
      "$MATH_N_SAMPLING" \
      "$MATH_TEMPERATURE"
  ) >> "$EVAL_LOG" 2>&1
fi

echo "Train log: $TRAIN_LOG"
echo "Eval log: $EVAL_LOG"
echo "Eval json: $EVAL_JSON"
