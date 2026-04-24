#!/bin/bash

# Copy or source this file and override the paths below.

export REPRO_ROOT=${REPRO_ROOT:-/root/autodl-tmp/ASFT/repro_asft_oftv2}
export ORIGINAL_REPO_ROOT=${ORIGINAL_REPO_ROOT:-/root/autodl-tmp/ASFT}

export MODEL_PATH=${MODEL_PATH:-/path/to/base-model}
export DATA_PATH=${DATA_PATH:-/path/to/train.jsonl}
export EVAL_DATA_PATH=${EVAL_DATA_PATH:-/path/to/eval.jsonl}

export ADAPTER_OUTPUT_DIR=${ADAPTER_OUTPUT_DIR:-$REPRO_ROOT/outputs/asft_oftv2_run}
export MERGED_OUTPUT_DIR=${MERGED_OUTPUT_DIR:-$REPRO_ROOT/outputs/asft_oftv2_merged}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
