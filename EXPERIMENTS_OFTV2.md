# OFTv2 Reproducible SFT Runs

This repository uses a custom `transformers.Trainer`-based SFT/ASFT loop in `train_v2.py` with PEFT adapters. OFTv2 is integrated through PEFT's `OFTConfig`, while the existing LoRA path remains available through `LoraConfig`.

## Environment

- Python 3.10+ with the packages from `requirements.txt`
- `transformers==4.57.1` and `peft==0.17.1` were the local versions used during integration
- For GPU training, use a PyTorch build compatible with your GPU architecture
- For DeepSpeed runs, reuse `scripts/ds_zero2_bf16.json` or `scripts/ds_zero3_bf16.json` as needed

## Data Paths

- Smoke test data in repo: `tests/fixtures/smoke_sft.jsonl`
- Local medical SFT dataset detected on this machine:
  - `/root/autodl-tmp/data/train_medmcqa_alpaca_10k.jsonl`
  - `/root/autodl-tmp/data/train_medmcqa_alpaca_30k.jsonl`
  - `/root/autodl-tmp/data/train_medmcqa_alpaca_100k.jsonl`

## Model Path

- Default model path used in the new scripts: `/root/autodl-tmp/models/Qwen2.5-7B-Instruct`
- Override with `MODEL_PATH=/path/to/your/model`

## Reproducible Commands

### 1. OFTv2 Smoke Test

Purpose: verify the OFTv2 path can initialize, run one forward/backward step, and save the final adapter checkpoint.

```bash
bash scripts/train_smoke_oftv2.sh
```

Fixed parameters:

- `mode=sft`
- `adapter_type=oftv2`
- `target_modules=q_proj,v_proj`
- `oft_block_size=32`
- `use_cayley_neumann=True`
- `max_steps=1`
- `seed=42`
- `use_cpu=True`
- `data_path=tests/fixtures/smoke_sft.jsonl`

Expected output directory:

- `output/smoke_oftv2`

### 2. Baseline LoRA SFT

Purpose: baseline SFT run that is directly comparable to OFTv2 on the same model, dataset, seed, batch settings, and epoch budget.

```bash
bash scripts/train_baseline_lora.sh
```

Fixed parameters:

- `mode=sft`
- `adapter_type=lora`
- `lora_r=8`
- `lora_alpha=16`
- `lora_dropout=0.05`
- `learning_rate=5e-4`
- `seed=42`
- `data_path=/root/autodl-tmp/data/train_medmcqa_alpaca_10k.jsonl`

Expected output directory:

- `output/sft_lora_baseline`

### 3. OFTv2 Formal SFT

Purpose: formal OFTv2 SFT run using the same core setup as the baseline LoRA run.

```bash
bash scripts/train_oftv2.sh
```

Fixed parameters:

- `mode=sft`
- `adapter_type=oftv2`
- `target_modules=q_proj,v_proj`
- `adapter_bias=none`
- `oft_block_size=32`
- `oft_r=0`
- `oft_module_dropout=0.0`
- `use_cayley_neumann=True`
- `learning_rate=5e-4`
- `seed=42`
- `data_path=/root/autodl-tmp/data/train_medmcqa_alpaca_10k.jsonl`

Expected output directory:

- `output/sft_oftv2`

## How to Compare Baseline and OFTv2

Keep the following fixed between `scripts/train_baseline_lora.sh` and `scripts/train_oftv2.sh`:

- `MODEL_PATH`
- `DATA_PATH`
- `seed`
- `mode=sft`
- `model_max_length`
- `per_device_train_batch_size`
- `global_batch_size`
- `num_train_epochs`
- `max_steps`
- `precision`

Only vary the adapter-specific settings:

- LoRA: `adapter_type=lora`, `lora_r`, `lora_alpha`, `lora_dropout`
- OFTv2: `adapter_type=oftv2`, `oft_block_size`, `oft_r`, `oft_module_dropout`, `use_cayley_neumann`

For a fair comparison, evaluate both checkpoints on the same downstream benchmark and report:

- training loss curve
- final adapter checkpoint path
- task metrics from the same evaluation script and prompt template

## Resume or Evaluate from Checkpoint

The training script saves the final model at `output_dir` via `trainer.save_model(output_dir)`.

Resume training from an adapter checkpoint:

```bash
python train_v2.py \
  --model_name_or_path /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
  --data_path /root/autodl-tmp/data/train_medmcqa_alpaca_10k.jsonl \
  --output_dir output/resumed_oftv2 \
  --mode sft \
  --use_adapter True \
  --adapter_type oftv2 \
  --target_modules q_proj,v_proj \
  --oft_block_size 32 \
  --use_cayley_neumann True \
  --resume_from_checkpoint /path/to/previous/checkpoint
```

Evaluate a saved adapter checkpoint with existing evaluation scripts by pointing the evaluator at the saved output directory, for example:

```bash
python eval/medeval/run_med_eval.py \
  --model output/sft_oftv2 \
  --test_data_dir eval/medeval/test_data
```

Adjust the evaluation entrypoint to match the benchmark you are running.
