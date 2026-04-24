# ASFT + OFTv2 Reproducible Experiment

This folder is a minimal reproducible package for the current `ASFT + OFTv2` experiment. It does not add new training features. It only packages the already working training path, smoke test path, and evaluation entrypoints into a smaller repo-style layout.

## What This Experiment Does

This experiment trains the existing `train_v2.py` SFT/ASFT pipeline with PEFT adapters, including:

- `adapter_type=lora`
- `adapter_type=oft` / `adapter_type=oftv2`
- optional OFTv2 identity regularization in `mode=asft`

The goal of this folder is reproducibility, not further algorithm changes.

## Directory Layout

```text
repro_asft_oftv2/
├── README.md
├── requirements.txt
├── env.example.sh
├── run/
│   ├── train_smoke.sh
│   ├── train_full.sh
│   ├── eval_adapter.sh
│   └── merge_adapter.sh
├── configs/
│   ├── smoke.env
│   └── full.env
├── src/
│   ├── train_v2.py
│   └── train_debug.py
├── tests/
│   ├── run_smoke.py
│   ├── test_train_v2_adapters.py
│   └── fixtures/
│       └── smoke_sft.jsonl
└── notes/
    ├── CHANGELOG.md
    └── KNOWN_ISSUES.md
```

## Environment

- Python: `3.10+`
- Recommended GPU: at least one CUDA GPU for full training
- Smoke test can run on CPU because it creates a tiny local model
- Key packages:
  - `torch`
  - `transformers`
  - `peft`
  - `accelerate`
  - `fire`
  - `tokenizers`
  - `sentencepiece`
  - `safetensors`

Install the minimal runtime:

```bash
cd /root/autodl-tmp/ASFT/repro_asft_oftv2
pip install -r requirements.txt
```

## Data And Model Paths

For full training and evaluation you must set these paths:

- `MODEL_PATH`: base model path
- `DATA_PATH`: training jsonl path
- `ADAPTER_OUTPUT_DIR`: output adapter directory

The smoke path does not require an external base model. It generates a tiny local LLaMA checkpoint automatically.

## Minimal Smoke Test

Command:

```bash
cd /root/autodl-tmp/ASFT/repro_asft_oftv2
bash run/train_smoke.sh
```

What it does:

- creates a tiny local LLaMA checkpoint
- runs OFTv2 + ASFT + identity regularizer
- runs OFTv2 plain SFT
- runs LoRA plain SFT
- saves adapter checkpoints

Expected pass signals:

- output contains `oft_identity_regularizer tensor(`
- output contains `adapter_type: oftv2`
- output contains `use_oft_regularizer: True`
- output ends with:
  - `oftv2_reg_adapter_config True`
  - `oftv2_plain_adapter_config True`
  - `lora_adapter_config True`

## Full Training

1. Edit paths in `env.example.sh` or export them in your shell.
2. Optionally edit `configs/full.env`.
3. Run:

```bash
cd /root/autodl-tmp/ASFT/repro_asft_oftv2
bash run/train_full.sh
```

Default full training template uses:

- `mode=asft`
- `adapter_type=oftv2`
- `target_modules=q_proj,v_proj`
- `use_oft_regularizer=true`
- `oft_regularizer_type=identity`

The main output directory is:

- `${ADAPTER_OUTPUT_DIR}`

## Evaluation

Recommended path right now: evaluate the adapter directly instead of merging first.

```bash
cd /root/autodl-tmp/ASFT/repro_asft_oftv2
bash run/eval_adapter.sh
```

This wrapper calls the existing medical evaluation adapter entrypoint from the parent ASFT repo:

- `/root/autodl-tmp/ASFT/eval/medeval/run_med_eval_adapter.py`

You must set:

- `MODEL_PATH`
- `ADAPTER_OUTPUT_DIR`
- `EVAL_DATA_PATH`

## Merge

There is an existing merge script in the parent repo:

- `/root/autodl-tmp/ASFT/merge_lora.py`

This repro bundle does not claim that merge is fully validated for OFTv2. The provided `run/merge_adapter.sh` is a documented wrapper template, not a newly validated OFTv2 merge workflow. If you want the lowest-risk path today, evaluate the adapter directly.

## Known Limitations

- Evaluation still depends on the original repo eval entrypoint.
- Merge is not revalidated here for OFTv2.
- The smoke test uses a synthetic tiny model and synthetic jsonl, so it validates code path correctness, not task quality.
- `target_modules` defaults to `q_proj,v_proj`, which is the current experiment assumption.
- OFTv2 target layers must satisfy the `in_features` divisibility checks enforced by `src/train_v2.py`.

## Result Recording Suggestions

For each run, keep at least:

- the exact `configs/*.env` used
- the shell command
- `training_args.bin`
- `adapter_config.json`
- the training log
- evaluation json outputs
- the base model path and dataset path

## Quick Start

Minimal path for someone else:

1. Read this README.
2. Run `bash run/train_smoke.sh`.
3. Set `MODEL_PATH` and `DATA_PATH`.
4. Run `bash run/train_full.sh`.
5. Set `EVAL_DATA_PATH`.
6. Run `bash run/eval_adapter.sh`.
