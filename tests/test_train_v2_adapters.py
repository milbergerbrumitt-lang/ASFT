import json

from peft import OFTConfig, TaskType, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from train_v2 import oft_identity_regularization, train, validate_oftv2_target_modules


def create_tiny_llama_checkpoint(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    model_dir = tmp_path / "tiny_llama"
    model_dir.mkdir()

    config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = LlamaForCausalLM(config)
    model.save_pretrained(model_dir)

    vocab = {
        "[PAD]": 0,
        "<s>": 1,
        "</s>": 2,
        "<unk>": 3,
        "Below": 4,
        "is": 5,
        "an": 6,
        "instruction": 7,
        "that": 8,
        "describes": 9,
        "a": 10,
        "task": 11,
        "Write": 12,
        "response": 13,
        "appropriately": 14,
        "completes": 15,
        "the": 16,
        "request": 17,
        "###": 18,
        "Instruction": 19,
        "Response": 20,
        ":": 21,
        "hello": 22,
        "world": 23,
        "short": 24,
        "answer": 25,
        "one": 26,
        "two": 27,
        "test": 28,
        "sample": 29,
        "tiny": 30,
        "train": 31,
    }
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="[PAD]",
    )
    fast_tokenizer.save_pretrained(model_dir)

    dataset_path = tmp_path / "tiny_data.jsonl"
    records = [
        {"instruction": "hello world", "response": "short answer"},
        {"instruction": "tiny test", "response": "sample answer"},
    ]
    with dataset_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    return model_dir, dataset_path


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
    config = OFTConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"],
        oft_block_size=8,
        r=0,
        module_dropout=0.0,
        bias="none",
        use_cayley_neumann=True,
    )
    return get_peft_model(model, config)


def run_tiny_train(tmp_path, adapter_type, mode="sft", use_oft_regularizer=False):
    model_dir, dataset_path = create_tiny_llama_checkpoint(tmp_path)
    output_dir = tmp_path / f"output_{adapter_type}"

    kwargs = dict(
        model_name_or_path=str(model_dir),
        data_path=str(dataset_path),
        output_dir=str(output_dir),
        mode=mode,
        model_max_length=64,
        per_device_train_batch_size=1,
        global_batch_size=1,
        num_train_epochs=1,
        learning_rate=1e-4,
        max_steps=1,
        precision="fp32",
        use_cpu=True,
        use_adapter=True,
        adapter_type=adapter_type,
        target_modules="q_proj,v_proj",
        seed=123,
        use_oft_regularizer=use_oft_regularizer,
        lambda_oft=0.5 if use_oft_regularizer else 0.0,
        oft_regularizer_type="identity",
    )
    if adapter_type == "lora":
        kwargs.update(lora_r=4, lora_alpha=8, lora_dropout=0.0)
    else:
        kwargs.update(oft_block_size=8, oft_r=0, oft_module_dropout=0.0, use_cayley_neumann=True)

    train(**kwargs)
    return output_dir


def test_lora_and_oftv2_training_paths(tmp_path):
    lora_dir = run_tiny_train(tmp_path / "lora_case", "lora")
    oftv2_dir = run_tiny_train(tmp_path / "oftv2_case", "oftv2")
    oftv2_reg_dir = run_tiny_train(tmp_path / "oftv2_reg_case", "oftv2", mode="asft", use_oft_regularizer=True)

    assert (lora_dir / "adapter_config.json").exists()
    assert (oftv2_dir / "adapter_config.json").exists()
    assert (oftv2_reg_dir / "adapter_config.json").exists()


def test_oft_identity_regularizer_returns_scalar():
    model = build_tiny_oftv2_model()
    regularizer = oft_identity_regularization(model)

    assert regularizer.dim() == 0
    assert regularizer.requires_grad


def test_oftv2_validation_error_includes_layer_details():
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=30,
            intermediate_size=60,
            num_hidden_layers=1,
            num_attention_heads=5,
            num_key_value_heads=5,
            max_position_embeddings=64,
        )
    )

    try:
        validate_oftv2_target_modules(
            model=model,
            target_modules=["q_proj"],
            oft_block_size=8,
            oft_r=0,
        )
    except ValueError as exc:
        message = str(exc)
        assert "q_proj" in message
        assert "in_features=30" in message
        assert "oft_block_size=8" in message
    else:
        raise AssertionError("Expected OFTv2 validation to fail for incompatible hidden size.")
