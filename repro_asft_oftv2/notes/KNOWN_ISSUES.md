# Known Issues

- `run/eval_adapter.sh` depends on the parent repo evaluation entrypoint:
  - `/root/autodl-tmp/ASFT/eval/medeval/run_med_eval_adapter.py`
- `run/merge_adapter.sh` is only a documented wrapper template.
- The smoke test validates code path correctness with a tiny local synthetic checkpoint, not downstream quality.
- Full training still requires you to provide a local base model and local training jsonl.
- OFTv2 requires target-layer dimensional compatibility enforced by `src/train_v2.py`.
