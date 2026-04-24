import json
import os
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from train_v2 import oft_identity_regularization, train


def build_tokenizer(tokenizer_dir: Path):
    vocab = {
        "[PAD]": 0,
        "</s>": 1,
        "<unk>": 2,
        "Below": 3,
        "is": 4,
        "an": 5,
        "instruction": 6,
        "that": 7,
        "describes": 8,
        "a": 9,
        "task.": 10,
        "Write": 11,
        "response": 12,
        "appropriately": 13,
        "completes": 14,
        "the": 15,
        "request.": 16,
        "###": 17,
        "Instruction:": 18,
        "Response:": 19,
        "hello": 20,
        "world": 21,
        "foo": 22,
        "bar": 23,
        "baz": 24,
    }
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        eos_token="</s>",
        bos_token="</s>",
        unk_token="<unk>",
    )
    fast.save_pretrained(tokenizer_dir)


def build_tiny_checkpoint(model_dir: Path):
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
        )
    )
    model.save_pretrained(model_dir)
    build_tokenizer(model_dir)


def build_tiny_data(data_path: Path):
    rows = [
        {"instruction": "hello world", "response": "foo bar"},
        {"instruction": "foo bar", "response": "hello baz"},
    ]
    with open(data_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def build_tiny_oftv2_model():
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
        )
    )
    from peft import OFTConfig, TaskType, get_peft_model

    peft_model = get_peft_model(
        model,
        OFTConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj"],
            oft_block_size=8,
            r=0,
            module_dropout=0.0,
            bias="none",
            use_cayley_neumann=True,
        ),
    )
    return peft_model


def run_tiny_train(root: Path, adapter_type: str, mode: str = "sft", use_oft_regularizer: bool = False):
    model_dir = root / "tiny_llama"
    output_dir = root / f"output_{adapter_type}_{mode}_{int(use_oft_regularizer)}"
    data_path = root / "tiny_data.jsonl"
    model_dir.mkdir(parents=True, exist_ok=True)
    build_tiny_checkpoint(model_dir)
    build_tiny_data(data_path)

    os.environ["WANDB_DISABLED"] = "true"
    train(
        model_name_or_path=str(model_dir),
        data_path=str(data_path),
        output_dir=str(output_dir),
        mode=mode,
        model_max_length=64,
        per_device_train_batch_size=1,
        global_batch_size=1,
        num_train_epochs=1,
        learning_rate=1e-4,
        use_adapter=True,
        adapter_type=adapter_type,
        target_modules="q_proj,v_proj",
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        oft_block_size=8,
        oft_r=0,
        oft_module_dropout=0.0,
        use_cayley_neumann=True,
        use_oft_regularizer=use_oft_regularizer,
        lambda_oft=0.5 if use_oft_regularizer else 0.0,
        oft_regularizer_type="identity",
        no_cuda=True,
        report_to=[],
        disable_tqdm=True,
        save_strategy="no",
    )
    return output_dir


def test_oft_identity_regularizer_returns_scalar():
    reg = oft_identity_regularization(build_tiny_oftv2_model())
    assert reg.ndim == 0
    assert torch.isfinite(reg)


def test_lora_and_oftv2_training_paths(tmp_path: Path):
    oft_reg_dir = run_tiny_train(tmp_path / "oft_reg", "oftv2", mode="asft", use_oft_regularizer=True)
    oft_plain_dir = run_tiny_train(tmp_path / "oft_plain", "oftv2", mode="sft", use_oft_regularizer=False)
    lora_dir = run_tiny_train(tmp_path / "lora", "lora", mode="sft", use_oft_regularizer=False)

    assert (oft_reg_dir / "adapter_config.json").exists()
    assert (oft_plain_dir / "adapter_config.json").exists()
    assert (lora_dir / "adapter_config.json").exists()
