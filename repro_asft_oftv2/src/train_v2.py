import os
import copy
import logging
import json
import numpy as np
import fire
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Union
import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
from peft import LoraConfig, OFTConfig, TaskType, get_peft_model
os.environ["WANDB_MODE"] = "offline"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

IGNORE_INDEX = -100

def load_jsonl(file_path: str):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
PROMPT_DICT = {
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="models/Llama-2-7b")

@dataclass
class DataArguments:
    data_path: str = field(default="alpaca_data.json")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512)
    output_dir: str = field(default="./output")
    per_device_train_batch_size: int = field(default=4)
    num_train_epochs: float = field(default=3.0)
    learning_rate: float = field(default=2e-5)


SUPPORTED_ADAPTER_TYPES = {"lora", "oft", "oftv2"}


def parse_target_modules(target_modules: Optional[Union[str, Sequence[str]]]):
    if target_modules is None:
        return ["q_proj", "v_proj"]
    if isinstance(target_modules, str):
        normalized = target_modules.strip()
        if normalized.lower() == "all-linear":
            return "all-linear"
        return [item.strip() for item in normalized.split(",") if item.strip()]
    return [item for item in target_modules if item]


def iter_target_linear_modules(model: torch.nn.Module, target_modules):
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if target_modules == "all-linear":
            yield name, module
            continue
        if any(name.endswith(target_name) for target_name in target_modules):
            yield name, module


def validate_oftv2_target_modules(
    model: torch.nn.Module,
    target_modules,
    oft_block_size: int,
    oft_r: int,
):
    if oft_block_size > 0 and oft_r > 0:
        raise ValueError(
            "OFTv2 expects only one of oft_block_size or oft_r to be set. "
            f"Received oft_block_size={oft_block_size}, oft_r={oft_r}."
        )
    if oft_block_size <= 0 and oft_r <= 0:
        raise ValueError(
            "OFTv2 requires a positive oft_block_size or a positive oft_r."
        )

    matched = list(iter_target_linear_modules(model, target_modules))
    if not matched:
        raise ValueError(f"No target linear modules matched target_modules={target_modules}.")

    for name, module in matched:
        if oft_block_size > 0 and module.in_features % oft_block_size != 0:
            raise ValueError(
                f"OFTv2 target module '{name}' has in_features={module.in_features}, "
                f"which is incompatible with oft_block_size={oft_block_size}. "
                "Expected in_features % oft_block_size == 0 so that "
                "r = in_features / oft_block_size is an integer."
            )
        if oft_r > 0 and module.in_features % oft_r != 0:
            raise ValueError(
                f"OFTv2 target module '{name}' has in_features={module.in_features}, "
                f"which is incompatible with oft_r={oft_r}. "
                "Expected in_features % oft_r == 0 so that "
                "oft_block_size = in_features / oft_r is an integer."
            )


def build_adapter_config(
    model: torch.nn.Module,
    adapter_type: str,
    target_modules,
    adapter_bias: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    oft_block_size: int,
    oft_r: int,
    oft_module_dropout: float,
    use_cayley_neumann: bool,
):
    normalized = adapter_type.lower()
    if normalized == "lora":
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias=adapter_bias,
        )
    if normalized not in {"oft", "oftv2"}:
        raise ValueError(f"Unsupported adapter_type={adapter_type}. Expected one of {SUPPORTED_ADAPTER_TYPES}.")

    validate_oftv2_target_modules(
        model=model,
        target_modules=target_modules,
        oft_block_size=oft_block_size,
        oft_r=oft_r,
    )
    return OFTConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
        r=oft_r,
        oft_block_size=oft_block_size,
        module_dropout=oft_module_dropout,
        bias=adapter_bias,
        use_cayley_neumann=use_cayley_neumann,
    )


def iter_oft_rotation_modules(model: torch.nn.Module):
    for name, module in model.named_modules():
        if module.__class__.__name__ == "OFTRotationModule" and hasattr(module, "get_weight"):
            yield name, module


def oft_identity_regularization(model: torch.nn.Module) -> torch.Tensor:
    penalties = []
    for _, rotation_module in iter_oft_rotation_modules(model):
        rotation = rotation_module.get_weight()
        identity = torch.eye(rotation.size(-1), device=rotation.device, dtype=rotation.dtype)
        penalties.append((rotation - identity).pow(2).sum())

    if not penalties:
        raise ValueError(
            "No OFTv2 rotation modules were found when computing oft_identity_regularization()."
        )

    return torch.stack(penalties).mean()

class EnhancedTrainer(Trainer):
    def __init__(
        self,
        mode="sft",
        kl_weight=0.1,
        clip_min=0.1,
        clip_max=2.0,
        alpha=0.1,
        original_model=None,
        use_oft_regularizer=False,
        lambda_oft=0.0,
        oft_regularizer_type="identity",
        adapter_type=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.kl_weight = kl_weight
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.alpha = alpha
        self.original_model = original_model
        self.use_oft_regularizer = use_oft_regularizer
        self.lambda_oft = lambda_oft
        self.oft_regularizer_type = oft_regularizer_type
        self.adapter_type = adapter_type
        if original_model is not None:
            self.original_model.eval()
        print(f"Training mode: {mode}, alpha: {alpha}")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            valid_mask = shift_labels != IGNORE_INDEX
            
            if valid_mask.sum() == 0:
                loss = torch.tensor(0.0, device=shift_logits.device, requires_grad=True)
            else:
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                token_losses = loss_fct(shift_logits, shift_labels)
                
                if self.mode == "sft":
                    weighted_losses = token_losses
                    
                elif self.mode == "dft":
                    probs = torch.softmax(shift_logits, dim=-1)
                    valid_labels = torch.clamp(shift_labels, min=0, max=probs.size(-1)-1)
                    weights = probs.gather(1, valid_labels.unsqueeze(-1)).squeeze(-1).detach()
                    weighted_losses = token_losses * weights
                    
                elif self.mode == "sft+kl":
                    if self.original_model is not None:
                        with torch.no_grad():
                            orig_outputs = self.original_model(**inputs)
                            orig_logits = orig_outputs.get("logits")[..., :-1, :].contiguous()
                            # Ensure sequence lengths match
                            orig_logits = orig_logits.view(-1, orig_logits.size(-1))[:shift_logits.size(0)]
                        if orig_logits.size(0) == shift_logits.size(0):
                            kl_div = F.kl_div(
                                F.log_softmax(shift_logits, dim=-1),
                                F.softmax(orig_logits, dim=-1),
                                reduction='none'
                            ).sum(dim=-1)
                            weighted_losses = token_losses + self.kl_weight * kl_div
                        else:
                            weighted_losses = token_losses
                    else:
                        weighted_losses = token_losses

                elif self.mode == "asft":
                    probs = torch.softmax(shift_logits, dim=-1)
                    valid_labels = torch.clamp(shift_labels, min=0, max=probs.size(-1)-1)
                    weights = probs.gather(1, valid_labels.unsqueeze(-1)).squeeze(-1).detach()
                    dft_losses = token_losses * weights
                    if self.use_oft_regularizer:
                        base_loss = dft_losses[valid_mask].mean()
                        loss = base_loss + self.lambda_oft * oft_identity_regularization(model)
                        return (loss, outputs) if return_outputs else loss
                    if self.original_model is not None:
                        with torch.no_grad():
                            orig_outputs = self.original_model(**inputs)
                            orig_logits = orig_outputs.get("logits")[..., :-1, :].contiguous()
                            orig_logits = orig_logits.view(-1, orig_logits.size(-1))[:shift_logits.size(0)]
                        if orig_logits.size(0) == shift_logits.size(0):
                            kl_div = F.kl_div(F.log_softmax(shift_logits, dim=-1), F.softmax(orig_logits, dim=-1), reduction='none').sum(dim=-1)
                            weighted_losses = dft_losses + self.kl_weight * kl_div
                        else:
                            weighted_losses = dft_losses
                    else:
                        weighted_losses = dft_losses
                else:
                    raise ValueError(f"Unsupported mode={self.mode}")

                loss = weighted_losses[valid_mask].mean()
        else:
            loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss

def smart_tokenizer_and_embedding_resize(special_tokens_dict: Dict, tokenizer: transformers.PreTrainedTokenizer, model: transformers.PreTrainedModel):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    tokenized_list = [
        tokenizer(text, return_tensors="pt", padding="longest", max_length=tokenizer.model_max_length, truncation=True)
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list]
    return dict(input_ids=input_ids, labels=labels, input_ids_lens=input_ids_lens, labels_lens=labels_lens)

def preprocess(sources: Sequence[str], targets: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

class SupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        list_data_dict = load_jsonl(data_path)
        prompt_no_input = PROMPT_DICT["prompt_no_input"]
        sources = [prompt_no_input.format_map(example) for example in list_data_dict]
        targets = [f"{example['response']}{tokenizer.eos_token}" for example in list_data_dict]
        data_dict = preprocess(sources, targets, tokenizer)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.sources = sources
        self.targets = targets

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)



def show_first_example(data_path: str, tokenizer: transformers.PreTrainedTokenizer):
    """Show first training example with tokenization details"""
    print("\n" + "="*50)
    print("FIRST TRAINING EXAMPLE")
    print("="*50)
    
    # Load first example
    data = load_jsonl(data_path)
    if not data:
        print("No data found")
        return
    
    example = data[0]
    instruction = example.get('instruction', '')
    response = example.get('response', '')
    
    print(f"Instruction: {instruction}")
    print(f"Response: {response}")
    
    # Format with prompt
    prompt = PROMPT_DICT["prompt_no_input"].format_map(example)
    full_text = prompt + response + tokenizer.eos_token
    
    print(f"\nFull prompt:\n{prompt}")
    print(f"Target text: {response}{tokenizer.eos_token}")
    
    # Tokenize
    tokenized = tokenizer(full_text, return_tensors="pt", max_length=tokenizer.model_max_length, truncation=True)
    input_ids = tokenized.input_ids[0]
    
    # Calculate instruction length for masking
    instruction_tokenized = tokenizer(prompt, return_tensors="pt", max_length=tokenizer.model_max_length, truncation=True)
    instruction_len = instruction_tokenized.input_ids[0].shape[0]
    
    print(f"\nTokenization:")
    print(f"Total tokens: {len(input_ids)}")
    print(f"Instruction tokens: {instruction_len}")
    print(f"Response tokens: {len(input_ids) - instruction_len}")
    
    # Show loss computation part
    labels = input_ids.clone()
    labels[:instruction_len] = IGNORE_INDEX
    
    print(f"\nLoss computation tokens (response part):")
    loss_tokens = input_ids[instruction_len:]
    decoded_loss_part = tokenizer.decode(loss_tokens, skip_special_tokens=False)
    print(f"Tokens for loss: {decoded_loss_part}")
    print(f"Token IDs: {loss_tokens.tolist()}")
    print("="*50 + "\n")


def train(
    model_name_or_path: str = "models/Llama-2-7b",
    data_path: str = "data/train_medmcqa_alpaca_10k.jsonl",
    cache_dir: str = None,
    model_max_length: int = 512,
    per_device_train_batch_size: int = 4,
    num_train_epochs: float = 3.0,
    learning_rate: float = 2e-5,
    global_batch_size: int = 64,
    mode: str = "sft",  # sft, dft, sft+kl, asft
    kl_weight: float = 0.1,
    alpha: float = 0.1,
    clip_min: float = 0.1,
    clip_max: float = 2.0,
    use_adapter: bool = False,
    adapter_type: str = "lora",
    target_modules: Optional[str] = None,
    adapter_bias: str = "none",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    oft_block_size: int = 32,
    oft_r: int = 0,
    oft_module_dropout: float = 0.0,
    use_cayley_neumann: bool = True,
    use_oft_regularizer: bool = False,
    lambda_oft: float = 0.0,
    oft_regularizer_type: str = "identity",
    output_dir: str = None,
    **kwargs
):
    """Enhanced training with multiple DFT variants"""

    model_args = ModelArguments(model_name_or_path=model_name_or_path)
    data_args = DataArguments(data_path=data_path)

    print("==== ModelArguments ====")
    print(model_args)
    print("========================")

    print("==== DataArguments ====")
    print(data_args)
    print("=======================")

    if output_dir is None:
        output_dir: str = f"./output/{mode}/{os.path.basename(model_name_or_path)}"

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    gradient_accumulation_steps = max(1, global_batch_size // (per_device_train_batch_size * world_size))

    print("==== Training Parameters ====")
    print(f"model_name_or_path: {model_name_or_path}")
    print(f"data_path: {data_path}")
    print(f"mode: {mode}")
    print(f"global_batch_size: {global_batch_size}")
    print(f"per_device_train_batch_size: {per_device_train_batch_size}")
    print(f"num_train_epochs: {num_train_epochs}")
    print(f"learning_rate: {learning_rate}")
    print(f"world_size: {world_size}")
    print(f"gradient_accumulation_steps: {gradient_accumulation_steps}")
    print(f"use_adapter: {use_adapter}")
    print(f"adapter_type: {adapter_type}")
    print(f"target_modules: {target_modules}")
    print(f"use_oft_regularizer: {use_oft_regularizer}")
    print(f"lambda_oft: {lambda_oft}")
    print(f"oft_regularizer_type: {oft_regularizer_type}")
    print("=============================")

    normalized_adapter_type = adapter_type.lower()
    if use_adapter and normalized_adapter_type not in SUPPORTED_ADAPTER_TYPES:
        raise ValueError(
            f"Unsupported adapter_type={adapter_type}. Expected one of {SUPPORTED_ADAPTER_TYPES}."
        )
    if use_oft_regularizer:
        if not use_adapter or normalized_adapter_type not in {"oft", "oftv2"}:
            raise ValueError(
                "use_oft_regularizer=True requires use_adapter=True and adapter_type=oftv2."
            )
        if mode != "asft":
            raise ValueError("use_oft_regularizer=True is only supported with mode='asft'.")
        if oft_regularizer_type != "identity":
            raise ValueError(
                f"Unsupported oft_regularizer_type={oft_regularizer_type}. Only 'identity' is supported."
            )

    training_kwargs = dict(kwargs)
    training_kwargs.setdefault("logging_steps", 1)
    training_kwargs.setdefault("save_strategy", "epoch")
    training_kwargs.setdefault("save_total_limit", 50)
    training_kwargs.setdefault("dataloader_num_workers", 0)
    training_kwargs.setdefault("dataloader_pin_memory", False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        cache_dir=cache_dir,
        model_max_length=model_max_length,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        **training_kwargs
    )

    print("==== Transformers TrainingArguments ====")
    print(training_args)
    print("=============================")

    if training_kwargs:
        print("==== Extra kwargs passed to TrainingArguments ====")
        print(training_kwargs)
        print("===============================================")

    # Detect if we're in distributed training mode
    is_distributed = world_size > 1

    if is_distributed:
        print(f"Distributed training mode detected: {world_size} GPUs")
    else:
        print("Single GPU training mode")

    # Load model
    model_kwargs = {
        "cache_dir": training_args.cache_dir,
        "torch_dtype": torch.bfloat16,
    }
    # Only use device_map in single-GPU mode
    if not is_distributed and not training_args.no_cuda:
        model_kwargs["device_map"] = "auto"

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs
    )

    # In distributed mode, move model to the correct device
    if is_distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        model = model.to(f"cuda:{local_rank}")

    if use_adapter:
        parsed_target_modules = parse_target_modules(target_modules)
        adapter_config = build_adapter_config(
            model=model,
            adapter_type=normalized_adapter_type,
            target_modules=parsed_target_modules,
            adapter_bias=adapter_bias,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            oft_block_size=oft_block_size,
            oft_r=oft_r,
            oft_module_dropout=oft_module_dropout,
            use_cayley_neumann=use_cayley_neumann,
        )
        model = get_peft_model(model, adapter_config)
        model.print_trainable_parameters()


    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )

    model_name_lower = model_args.model_name_or_path.lower()
    llama3_markers = [
        "llama-3",
        "llama3",
        "llama 3",
        "llama_3",
        "llama3.",
        "llama-3.",
    ]
    is_llama3_family = any(marker in model_name_lower for marker in llama3_markers)
    if is_llama3_family:
        logger.info(
            "Detected LLaMA 3 family model (%s); skipping manual special token overrides.",
            model_args.model_name_or_path,
        )
    elif "llama" in model_name_lower:
        tokenizer.add_special_tokens({
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        })

    # Show first example
    show_first_example(data_args.data_path, tokenizer)

    # Load original model for KL modes
    original_model = None
    if ("kl" in mode or mode == "asft") and not use_oft_regularizer:
        print("Loading original model for KL divergence...")

        original_model_kwargs = {
            "cache_dir": training_args.cache_dir,
            "torch_dtype": torch.bfloat16,
        }
        # Only use device_map in single-GPU mode
        if not is_distributed and not training_args.no_cuda:
            original_model_kwargs["device_map"] = "auto"

        original_model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            **original_model_kwargs
        )

        # In distributed mode, move model to the correct device
        if is_distributed:
            original_model = original_model.to(f"cuda:{local_rank}")

        original_tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

        if original_tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=original_tokenizer,
                model=original_model,
            )

        for param in original_model.parameters():
            param.requires_grad = False
        original_model.eval()

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = EnhancedTrainer(
        mode=mode,
        kl_weight=kl_weight,
        alpha=alpha,
        clip_min=clip_min,
        clip_max=clip_max,
        original_model=original_model,
        use_oft_regularizer=use_oft_regularizer,
        lambda_oft=lambda_oft,
        oft_regularizer_type=oft_regularizer_type,
        adapter_type=normalized_adapter_type if use_adapter else None,
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )
    
    trainer.train()
    trainer.save_model(training_args.output_dir)
    # tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    fire.Fire(train)
    
