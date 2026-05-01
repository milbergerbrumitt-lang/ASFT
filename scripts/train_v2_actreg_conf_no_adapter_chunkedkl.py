import os
import copy
import logging
import json
import random
import re
import numpy as np
import fire
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, set_seed
from peft import LoraConfig, OFTConfig, get_peft_model, TaskType
os.environ["WANDB_MODE"] = "offline"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

IGNORE_INDEX = -100
SUPPORTED_ADAPTER_TYPES = {"lora", "oft", "oftv2"}

def set_random_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    model_name_or_path: Optional[str] = field(default=None)

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

class EnhancedTrainer(Trainer):
    def __init__(
        self,
        mode="sft",
        kl_weight=0.1,
        kl_chunk_size=256,
        clip_min=0.1,
        clip_max=2.0,
        alpha=0.1,
        original_model=None,
        use_oft_regularizer=False,
        lambda_oft=0.0,
        oft_regularizer_type="identity",
        adapter_type=None,
        use_geometry_regularizer=False,
        lambda_geom=0.0,
        geometry_regularizer_type="orthogonality",
        geometry_target_modules=None,
        disable_kl_in_geometry_asft=False,
        use_activation_regularizer=False,
        lambda_act=0.0,
        activation_regularizer_type="offdiag_cov",
        activation_confidence_power=1.0,
        activation_confidence_min=0.0,
        activation_target_modules="v_proj",
        activation_target_last_k_layers=4,
        activation_answer_only=True,
        use_adapter=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.kl_weight = kl_weight
        if kl_chunk_size <= 0:
            raise ValueError(f"kl_chunk_size must be positive, received {kl_chunk_size}.")
        self.kl_chunk_size = kl_chunk_size
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.alpha = alpha
        self.original_model = original_model
        self.use_oft_regularizer = use_oft_regularizer
        self.lambda_oft = lambda_oft
        self.oft_regularizer_type = oft_regularizer_type
        self.adapter_type = adapter_type
        self.use_geometry_regularizer = use_geometry_regularizer
        self.lambda_geom = lambda_geom
        self.geometry_regularizer_type = geometry_regularizer_type
        self.geometry_target_modules = geometry_target_modules
        self.disable_kl_in_geometry_asft = disable_kl_in_geometry_asft
        self.use_activation_regularizer = use_activation_regularizer
        self.lambda_act = lambda_act
        self.activation_regularizer_type = activation_regularizer_type.lower()
        if activation_confidence_power < 0.0:
            raise ValueError(
                f"activation_confidence_power must be >= 0.0, got {activation_confidence_power}."
            )
        if not 0.0 <= activation_confidence_min <= 1.0:
            raise ValueError(
                f"activation_confidence_min must be in [0.0, 1.0], got {activation_confidence_min}."
            )
        self.activation_confidence_power = activation_confidence_power
        self.activation_confidence_min = activation_confidence_min
        self.activation_target_modules = parse_activation_target_modules(activation_target_modules)
        self.activation_target_last_k_layers = activation_target_last_k_layers
        self.activation_answer_only = activation_answer_only
        self.use_adapter = use_adapter
        self._activation_cache = {}
        self._activation_hook_handles = []
        self.activation_target_module_names = []
        if original_model is not None:
            self.original_model.eval()
        if self.use_activation_regularizer:
            supported_activation_regularizer_types = {"offdiag_cov", "conf_offdiag_cov"}
            if self.activation_regularizer_type not in supported_activation_regularizer_types:
                raise ValueError(
                    f"Unsupported activation_regularizer_type={self.activation_regularizer_type!r}. "
                    f"Supported values: {sorted(supported_activation_regularizer_types)}."
                )
            self._register_activation_hooks()
        print(f"Training mode: {mode}, alpha: {alpha}")

    def _clear_activation_cache(self):
        self._activation_cache.clear()

    def _make_activation_hook(self, module_name):
        def hook(module, module_inputs, module_output):
            activation = module_output[0] if isinstance(module_output, (tuple, list)) else module_output
            self._activation_cache[module_name] = activation

        return hook

    def _register_activation_hooks(self):
        matched_modules = collect_activation_target_linear_modules(
            model=self.model,
            target_modules=self.activation_target_modules,
            last_k_layers=self.activation_target_last_k_layers,
        )
        self.activation_target_module_names = [module_name for module_name, _, _ in matched_modules]
        for module_name, module, _ in matched_modules:
            self._activation_hook_handles.append(module.register_forward_hook(self._make_activation_hook(module_name)))
        print(
            "[activation_asft] "
            f"registered_hooks={len(self.activation_target_module_names)} "
            f"target_modules={self.activation_target_modules} "
            f"target_module_names={self.activation_target_module_names}"
        )
    
    def get_reference_logits(self, model, inputs):
        """
        If LoRA is enabled:
            reference = base model (adapter disabled)
        Else:
            reference = self.original_model
        """
        if hasattr(model, "disable_adapter"):
            with model.disable_adapter():
                ref_outputs = model(**inputs)
                ref_logits = ref_outputs.logits
        else:
            with torch.no_grad():
                ref_outputs = self.original_model(**inputs)
                ref_logits = ref_outputs.logits

        return ref_logits
    
    def _chunked_valid_kl_mean(self, valid_shift_logits, valid_ref_logits):
        kl_sum = valid_shift_logits.new_zeros(())
        num_valid_tokens = valid_shift_logits.size(0)
        for start in range(0, num_valid_tokens, self.kl_chunk_size):
            end = min(start + self.kl_chunk_size, num_valid_tokens)
            chunk_kl_sum = F.kl_div(
                F.log_softmax(valid_shift_logits[start:end], dim=-1),
                F.softmax(valid_ref_logits[start:end], dim=-1),
                reduction="sum",
            )
            kl_sum = kl_sum + chunk_kl_sum
        return kl_sum / num_valid_tokens


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        if self.use_activation_regularizer:
            self._clear_activation_cache()
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels_2d = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels_2d.view(-1)
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
                    # if self.original_model is not None:
                    #     with torch.no_grad():
                    #         orig_outputs = self.original_model(**inputs)
                    #         orig_logits = orig_outputs.get("logits")[..., :-1, :].contiguous()
                    #         # Ensure sequence lengths match
                    #         orig_logits = orig_logits.view(-1, orig_logits.size(-1))[:shift_logits.size(0)]
                    #     if orig_logits.size(0) == shift_logits.size(0):
                    #         kl_div = F.kl_div(F.log_softmax(shift_logits, dim=-1), F.softmax(orig_logits, dim=-1), reduction='none').sum(dim=-1)
                    #         weighted_losses = token_losses + self.kl_weight * kl_div
                    #     else:
                    #         weighted_losses = token_losses
                    # else:
                    with torch.no_grad():
                        ref_logits = self.get_reference_logits(model, inputs)
                        ref_logits = ref_logits[..., :-1, :].contiguous()
                        ref_logits = ref_logits.view(-1, ref_logits.size(-1))[:shift_logits.size(0)]

                    kl_div = F.kl_div(
                        F.log_softmax(shift_logits, dim=-1),
                        F.softmax(ref_logits, dim=-1),
                        reduction="none",
                    ).sum(dim=-1)

                    weighted_losses = token_losses + self.kl_weight * kl_div
                                            
                elif self.mode == "asft":
                    probs = torch.softmax(shift_logits, dim=-1)
                    valid_labels = torch.clamp(shift_labels, min=0, max=probs.size(-1)-1)
                    confidence = probs.gather(1, valid_labels.unsqueeze(-1)).squeeze(-1)
                    confidence = confidence.detach()
                    weights = confidence
                    dft_losses = token_losses * weights
                    dft_loss = dft_losses[valid_mask].sum() / valid_mask.sum()
                    if self.use_oft_regularizer:
                        if self.oft_regularizer_type != "identity":
                            raise ValueError(
                                f"Unsupported oft_regularizer_type={self.oft_regularizer_type!r}. "
                                "Only 'identity' is currently supported."
                            )
                        if self.adapter_type != "oftv2":
                            raise ValueError(
                                "use_oft_regularizer=True requires adapter_type='oftv2'. "
                                f"Received adapter_type={self.adapter_type!r}."
                            )
                        loss = dft_loss + self.lambda_oft * oft_identity_regularization(model)
                    else:
                        if self.use_activation_regularizer:
                            if self.use_geometry_regularizer:
                                raise ValueError(
                                    "Do not enable static geometry regularizer and dynamic activation regularizer "
                                    "at the same time in this first implementation; the experiment meaning would be mixed."
                                )
                            with torch.no_grad():
                                ref_logits = self.get_reference_logits(model, inputs)
                                ref_logits = ref_logits[..., :-1, :].contiguous()
                                ref_logits = ref_logits.view(-1, ref_logits.size(-1))[:shift_logits.size(0)]
                                valid_ref_logits = ref_logits[valid_mask]
                                del ref_logits

                            valid_shift_logits = shift_logits[valid_mask]
                            kl_mean = self._chunked_valid_kl_mean(valid_shift_logits, valid_ref_logits)
                            kl_term = self.kl_weight * kl_mean
                            del valid_shift_logits, valid_ref_logits
                            token_confidences = None
                            confidence_mean = shift_logits.new_zeros(())
                            confidence_min = shift_logits.new_zeros(())
                            confidence_max = shift_logits.new_zeros(())
                            confidence_weight_sum = shift_logits.new_zeros(())
                            if self.activation_regularizer_type == "conf_offdiag_cov":
                                token_confidences = confidence.view_as(shift_labels_2d)
                                valid_confidences = token_confidences[shift_labels_2d.ne(IGNORE_INDEX)]
                                if valid_confidences.numel() > 0:
                                    confidence_mean = valid_confidences.float().mean()
                                    confidence_min = valid_confidences.float().min()
                                    confidence_max = valid_confidences.float().max()
                                    confidence_weights = torch.clamp(
                                        valid_confidences.float(),
                                        min=self.activation_confidence_min,
                                    ).pow(self.activation_confidence_power)
                                    confidence_weight_sum = confidence_weights.sum()
                            activation_reg = activation_regularization(
                                activation_cache=self._activation_cache,
                                target_module_names=self.activation_target_module_names,
                                labels=labels,
                                attention_mask=inputs.get("attention_mask"),
                                regularizer_type=self.activation_regularizer_type,
                                answer_only=self.activation_answer_only,
                                token_confidences=token_confidences,
                                activation_confidence_power=self.activation_confidence_power,
                                activation_confidence_min=self.activation_confidence_min,
                            )
                            activation_term = self.lambda_act * activation_reg
                            loss = dft_loss + kl_term + activation_term
                            if self.activation_regularizer_type == "conf_offdiag_cov":
                                print(
                                    "[conf_activation_asft] "
                                    f"dft_loss={dft_loss.detach().float().item():.6f} "
                                    f"kl_term={kl_term.detach().float().item():.6f} "
                                    f"activation_reg={activation_reg.detach().float().item():.6f} "
                                    f"lambda_act_times_activation={activation_term.detach().float().item():.6f} "
                                    f"total_loss={loss.detach().float().item():.6f} "
                                    f"confidence_mean={confidence_mean.detach().float().item():.6f} "
                                    f"confidence_min={confidence_min.detach().float().item():.6f} "
                                    f"confidence_max={confidence_max.detach().float().item():.6f} "
                                    f"confidence_weight_sum={confidence_weight_sum.detach().float().item():.6f}"
                                )
                            else:
                                print(
                                    "[activation_asft] "
                                    f"dft_loss={dft_loss.detach().float().item():.6f} "
                                    f"kl_term={kl_term.detach().float().item():.6f} "
                                    f"activation_reg={activation_reg.detach().float().item():.6f} "
                                    f"lambda_act_times_activation={activation_term.detach().float().item():.6f} "
                                    f"total_loss={loss.detach().float().item():.6f}"
                                )
                        else:
                            include_kl_in_geometry_asft = not (
                                self.use_geometry_regularizer and self.disable_kl_in_geometry_asft
                            )
                            if include_kl_in_geometry_asft:
                                with torch.no_grad():
                                    ref_logits = self.get_reference_logits(model, inputs)
                                    ref_logits = ref_logits[..., :-1, :].contiguous()
                                    ref_logits = ref_logits.view(-1, ref_logits.size(-1))[:shift_logits.size(0)]

                                kl_div = F.kl_div(
                                    F.log_softmax(shift_logits, dim=-1),
                                    F.softmax(ref_logits, dim=-1),
                                    reduction="none",
                                ).sum(dim=-1)

                                kl_term = self.kl_weight * (kl_div[valid_mask].sum() / valid_mask.sum())
                                weighted_losses = dft_losses + self.kl_weight * kl_div
                                loss = weighted_losses[valid_mask].sum() / valid_mask.sum()
                            else:
                                kl_term = torch.zeros((), device=dft_loss.device, dtype=dft_loss.dtype)
                                loss = dft_loss

                            if self.use_geometry_regularizer:
                                geometry_reg = geometry_regularization(
                                    model=model,
                                    target_modules=self.geometry_target_modules,
                                    regularizer_type=self.geometry_regularizer_type,
                                )
                                geometry_term = self.lambda_geom * geometry_reg
                                loss = loss + geometry_term
                                print(
                                    "[geometry_asft] "
                                    f"include_kl={include_kl_in_geometry_asft} "
                                    f"dft_loss={dft_loss.detach().float().item():.6f} "
                                    f"kl_term={kl_term.detach().float().item():.6f} "
                                    f"geometry_reg={geometry_reg.detach().float().item():.6f} "
                                    f"lambda_geom_times_geometry={geometry_term.detach().float().item():.6f} "
                                    f"total_loss={loss.detach().float().item():.6f}"
                                )

                if not (self.mode == "asft" and (self.use_oft_regularizer or self.use_geometry_regularizer or self.use_activation_regularizer)):
                    loss = (weighted_losses[valid_mask].sum() / valid_mask.sum())

        else:
            loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss


def parse_target_modules(target_modules):
    if target_modules is None:
        return ["q_proj", "v_proj"]

    if isinstance(target_modules, str):
        target_modules = target_modules.strip()
        if not target_modules:
            return ["q_proj", "v_proj"]
        if target_modules.lower() == "all-linear":
            return "all-linear"
        if "," in target_modules:
            parsed_modules = [module.strip() for module in target_modules.split(",") if module.strip()]
            if not parsed_modules:
                raise ValueError("target_modules is empty after parsing the comma-separated string.")
            return parsed_modules

    return target_modules


def match_target_module(name: str, target_modules) -> bool:
    if target_modules == "all-linear":
        return True

    if isinstance(target_modules, str):
        return re.search(target_modules, name) is not None

    return any(name == module_name or name.endswith(f".{module_name}") or name.endswith(module_name) for module_name in target_modules)


def collect_target_linear_modules(model, target_modules):
    matched_modules = []
    for module_name, module in model.named_modules():
        if module_name == "":
            continue

        if target_modules == "all-linear":
            if isinstance(module, torch.nn.Linear) and not module_name.endswith("lm_head"):
                matched_modules.append((module_name, module))
            continue

        if match_target_module(module_name, target_modules):
            matched_modules.append((module_name, module))

    return matched_modules


def parse_activation_target_modules(target_modules):
    if target_modules is None:
        return ["v_proj"]

    if isinstance(target_modules, str):
        target_modules = target_modules.strip()
        if not target_modules:
            return ["v_proj"]
        if target_modules.lower() == "all-linear":
            raise ValueError("activation_target_modules currently only supports explicit v_proj modules.")
        parsed_modules = [module.strip() for module in target_modules.split(",") if module.strip()]
        if not parsed_modules:
            raise ValueError("activation_target_modules is empty after parsing.")
        return parsed_modules

    if isinstance(target_modules, (list, tuple)):
        parsed_modules = [str(module).strip() for module in target_modules if str(module).strip()]
        if not parsed_modules:
            raise ValueError("activation_target_modules is empty after parsing.")
        return parsed_modules

    raise ValueError(
        f"Unsupported activation_target_modules={target_modules!r}. "
        "Use 'v_proj' or a comma-separated string such as 'v_proj'."
    )


def _matches_activation_target(module_name: str, target_modules) -> bool:
    return any(module_name == target_module or module_name.endswith(f".{target_module}") for target_module in target_modules)


def _parse_qwen_activation_layer_index(module_name: str, target_modules):
    for target_module in target_modules:
        pattern = rf"(?:^|\.)model\.layers\.(\d+)\.self_attn\.{re.escape(target_module)}$"
        match = re.search(pattern, module_name)
        if match is not None:
            return int(match.group(1))
    return None


def collect_activation_target_linear_modules(model, target_modules, last_k_layers: int):
    matched_modules = []
    unparsed_module_names = []

    for module_name, module in model.named_modules():
        if module_name == "" or not _matches_activation_target(module_name, target_modules):
            continue

        layer_index = _parse_qwen_activation_layer_index(module_name, target_modules)
        if layer_index is None:
            unparsed_module_names.append(module_name)
            continue

        if not isinstance(module, torch.nn.Linear):
            raise ValueError(
                f"Activation regularizer target module '{module_name}' is a {type(module).__name__}, "
                "but only torch.nn.Linear v_proj modules are supported."
            )

        matched_modules.append((module_name, module, layer_index))

    if unparsed_module_names:
        raise ValueError(
            "Matched activation target modules whose layer index could not be parsed: "
            f"{unparsed_module_names}. Expected Qwen-style names like "
            "'model.layers.{i}.self_attn.v_proj'."
        )

    if not matched_modules:
        raise ValueError(
            f"No target modules matched for activation_target_modules={target_modules!r}. "
            "Expected Qwen-style modules like 'model.layers.{i}.self_attn.v_proj'."
        )

    layer_indices = sorted({layer_index for _, _, layer_index in matched_modules})
    selected_layer_indices = set(layer_indices[-last_k_layers:])
    selected_modules = [
        (module_name, module, layer_index)
        for module_name, module, layer_index in matched_modules
        if layer_index in selected_layer_indices
    ]

    if not selected_modules:
        raise ValueError(
            f"No activation target modules remained after selecting the last {last_k_layers} layers. "
            f"Matched layer indices were {layer_indices}."
        )

    return selected_modules


def validate_activation_target_modules(model, target_modules, last_k_layers: int):
    collect_activation_target_linear_modules(
        model=model,
        target_modules=target_modules,
        last_k_layers=last_k_layers,
    )


def validate_oftv2_target_modules(model, target_modules, oft_block_size: int, oft_r: int):
    if oft_block_size < 0:
        raise ValueError(f"oft_block_size must be >= 0, got {oft_block_size}.")
    if oft_r < 0:
        raise ValueError(f"oft_r must be >= 0, got {oft_r}.")
    if (oft_block_size == 0) == (oft_r == 0):
        raise ValueError(
            "OFTv2 requires exactly one of oft_block_size or oft_r to be set. "
            f"Received oft_block_size={oft_block_size}, oft_r={oft_r}."
        )

    matched_modules = collect_target_linear_modules(model, target_modules)
    if not matched_modules:
        raise ValueError(f"No target modules matched for OFTv2 target_modules={target_modules!r}.")

    for module_name, module in matched_modules:
        if not hasattr(module, "in_features"):
            raise ValueError(
                f"OFTv2 target module '{module_name}' does not expose in_features; "
                "only Linear-like modules with in_features are currently supported by this integration."
            )

        in_features = module.in_features
        if oft_block_size > 0 and in_features % oft_block_size != 0:
            raise ValueError(
                f"OFTv2 target module '{module_name}' has in_features={in_features}, which is incompatible with "
                f"oft_block_size={oft_block_size}. Expected in_features % oft_block_size == 0 so that "
                f"r = in_features / oft_block_size is an integer."
            )

        if oft_r > 0 and in_features % oft_r != 0:
            raise ValueError(
                f"OFTv2 target module '{module_name}' has in_features={in_features}, which is incompatible with "
                f"oft_r={oft_r}. Expected in_features % oft_r == 0 so that "
                f"oft_block_size = in_features / oft_r is an integer."
            )


def build_adapter_config(
    model,
    adapter_type: str,
    target_modules,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    oft_block_size: int,
    oft_r: int,
    oft_module_dropout: float,
    use_cayley_neumann: bool,
    adapter_bias: str,
):
    adapter_type = adapter_type.lower()
    if adapter_type not in SUPPORTED_ADAPTER_TYPES:
        raise ValueError(
            f"Unsupported adapter_type={adapter_type!r}. Supported values: {sorted(SUPPORTED_ADAPTER_TYPES)}."
        )

    if adapter_type == "lora":
        logger.info(
            "Using LoRA with r=%s, alpha=%s, dropout=%s, target_modules=%s",
            lora_r,
            lora_alpha,
            lora_dropout,
            target_modules,
        )
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias=adapter_bias,
        )

    validate_oftv2_target_modules(
        model=model,
        target_modules=target_modules,
        oft_block_size=oft_block_size,
        oft_r=oft_r,
    )
    logger.info(
        "Using OFTv2 with oft_block_size=%s, oft_r=%s, module_dropout=%s, use_cayley_neumann=%s, target_modules=%s",
        oft_block_size,
        oft_r,
        oft_module_dropout,
        use_cayley_neumann,
        target_modules,
    )
    return OFTConfig(
        task_type=TaskType.CAUSAL_LM,
        r=oft_r,
        oft_block_size=oft_block_size,
        module_dropout=oft_module_dropout,
        target_modules=target_modules,
        bias=adapter_bias,
        use_cayley_neumann=use_cayley_neumann,
    )


def get_active_adapter_names(model):
    active_adapters = getattr(model, "active_adapters", None)
    if active_adapters is None:
        active_adapters = getattr(model, "active_adapter", "default")
    if isinstance(active_adapters, str):
        return [active_adapters]
    return list(active_adapters)


def iter_oft_rotation_modules(model):
    active_adapter_names = get_active_adapter_names(model)
    for module_name, module in model.named_modules():
        oft_modules = getattr(module, "oft_R", None)
        if not isinstance(oft_modules, torch.nn.ModuleDict) or len(oft_modules) == 0:
            continue

        for adapter_name in active_adapter_names:
            if adapter_name in oft_modules:
                yield module_name, adapter_name, oft_modules[adapter_name]


def oft_identity_regularization(model) -> torch.Tensor:
    penalties = []
    for module_name, adapter_name, rotation_module in iter_oft_rotation_modules(model):
        if not hasattr(rotation_module, "get_weight"):
            raise ValueError(
                f"OFTv2 rotation module '{module_name}' for adapter '{adapter_name}' does not expose get_weight(), "
                "so the explicit transform matrix R cannot be constructed."
            )

        rotation = rotation_module.get_weight()
        if rotation.dim() != 2 or rotation.shape[0] != rotation.shape[1]:
            raise ValueError(
                f"OFTv2 rotation module '{module_name}' for adapter '{adapter_name}' produced an invalid R with "
                f"shape {tuple(rotation.shape)}. Expected a square 2D matrix."
            )

        identity = torch.eye(rotation.size(0), device=rotation.device, dtype=rotation.dtype)
        penalties.append(torch.norm(rotation - identity, p="fro").pow(2))

    if not penalties:
        raise ValueError(
            "No OFTv2 target modules were found when computing the OFTv2 identity regularizer. "
            "Check adapter_type, target_modules, and whether the model is wrapped with OFTv2."
        )

    return torch.stack(penalties).mean()


def validate_geometry_target_modules(model, target_modules):
    matched_modules = collect_target_linear_modules(model, target_modules)
    if not matched_modules:
        raise ValueError(f"No target modules matched for geometry_target_modules={target_modules!r}.")

    for module_name, module in matched_modules:
        if not isinstance(module, torch.nn.Linear):
            raise ValueError(
                f"Geometry regularizer target module '{module_name}' is a {type(module).__name__}, "
                "but only torch.nn.Linear modules are supported."
            )


def _semi_orthogonality_penalty(weight: torch.Tensor) -> torch.Tensor:
    if weight.dim() != 2:
        raise ValueError(f"Expected a 2D weight matrix, got shape={tuple(weight.shape)}.")

    rows, cols = weight.shape
    if rows <= cols:
        gram = weight @ weight.transpose(0, 1)
        identity = torch.eye(rows, device=weight.device, dtype=weight.dtype)
        return torch.norm(gram - identity, p="fro").pow(2) / rows

    gram = weight.transpose(0, 1) @ weight
    identity = torch.eye(cols, device=weight.device, dtype=weight.dtype)
    return torch.norm(gram - identity, p="fro").pow(2) / cols


def geometry_regularization(model, target_modules, regularizer_type: str = "orthogonality") -> torch.Tensor:
    # Unlike oft_identity_regularization(), this penalty is computed directly from the
    # selected Linear layer weights W, so it does not rely on OFTv2-specific oft_R modules.
    penalties = []
    for module_name, module in collect_target_linear_modules(model, target_modules):
        if not isinstance(module, torch.nn.Linear):
            raise ValueError(
                f"Geometry regularizer target module '{module_name}' is a {type(module).__name__}, "
                "but only torch.nn.Linear modules are supported."
            )

        if regularizer_type == "orthogonality":
            penalties.append(_semi_orthogonality_penalty(module.weight))
        else:
            raise ValueError(
                f"Unsupported geometry_regularizer_type={regularizer_type!r}. "
                "Only 'orthogonality' is currently supported."
            )

    if not penalties:
        raise ValueError(
            "No target modules were found when computing the geometry regularizer. "
            "Check geometry_target_modules or target_modules."
        )

    return torch.stack(penalties).mean()


def _offdiag_covariance_penalty(activations: torch.Tensor) -> torch.Tensor:
    if activations.size(0) < 2 or activations.size(1) < 2:
        return activations.new_zeros(())

    x = activations.float()
    x = x - x.mean(dim=0, keepdim=True)
    variance = x.pow(2).mean(dim=0, keepdim=True)
    x = x * torch.rsqrt(variance.clamp_min(1e-6))
    covariance = x.transpose(0, 1).matmul(x) / (x.size(0) - 1)
    diagonal = torch.diagonal(covariance)
    offdiag_count = covariance.numel() - diagonal.numel()
    if offdiag_count <= 0:
        return covariance.new_zeros(())
    offdiag_squared_sum = covariance.pow(2).sum() - diagonal.pow(2).sum()
    return offdiag_squared_sum.clamp_min(0.0) / offdiag_count


def _weighted_offdiag_covariance_penalty(
    activations: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    if activations.size(0) < 2 or activations.size(1) < 2:
        return activations.new_zeros(())
    if weights.dim() != 1 or weights.size(0) != activations.size(0):
        raise ValueError(
            "Weighted offdiag covariance requires weights with shape [num_tokens], "
            f"got activations={tuple(activations.shape)} and weights={tuple(weights.shape)}."
        )

    z = activations.float()
    w = weights.detach().to(device=z.device).float()
    if torch.count_nonzero(w > 0.0).detach().item() < 2:
        return activations.new_zeros(())

    weight_sum = w.sum()
    if weight_sum.detach().le(eps).item():
        return activations.new_zeros(())

    weighted_mean = (w.unsqueeze(1) * z).sum(dim=0, keepdim=True) / (weight_sum + eps)
    centered = z - weighted_mean
    covariance = (w.unsqueeze(1) * centered).transpose(0, 1).matmul(centered) / (weight_sum + eps)
    diagonal = torch.diagonal(covariance)
    offdiag_count = covariance.numel() - diagonal.numel()
    if offdiag_count <= 0:
        return covariance.new_zeros(())
    offdiag_squared_sum = covariance.pow(2).sum() - diagonal.pow(2).sum()
    return offdiag_squared_sum.clamp_min(0.0) / offdiag_count


def activation_regularization(
    activation_cache,
    target_module_names,
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    regularizer_type: str = "offdiag_cov",
    answer_only: bool = True,
    token_confidences: Optional[torch.Tensor] = None,
    activation_confidence_power: float = 1.0,
    activation_confidence_min: float = 0.0,
) -> torch.Tensor:
    supported_activation_regularizer_types = {"offdiag_cov", "conf_offdiag_cov"}
    if regularizer_type not in supported_activation_regularizer_types:
        raise ValueError(
            f"Unsupported activation_regularizer_type={regularizer_type!r}. "
            f"Supported values: {sorted(supported_activation_regularizer_types)}."
        )
    if activation_confidence_power < 0.0:
        raise ValueError(
            f"activation_confidence_power must be >= 0.0, got {activation_confidence_power}."
        )
    if not 0.0 <= activation_confidence_min <= 1.0:
        raise ValueError(
            f"activation_confidence_min must be in [0.0, 1.0], got {activation_confidence_min}."
        )
    if labels is None:
        raise ValueError("activation_regularization requires labels for next-token answer masking.")
    if not target_module_names:
        raise ValueError("No activation target module names were registered.")
    if regularizer_type == "conf_offdiag_cov":
        if token_confidences is None:
            raise ValueError("conf_offdiag_cov requires token_confidences.")
        expected_confidence_shape = labels[:, 1:].shape
        if token_confidences.shape != expected_confidence_shape:
            raise ValueError(
                "token_confidences must align with labels[:, 1:] for conf_offdiag_cov, "
                f"got token_confidences={tuple(token_confidences.shape)} and "
                f"labels[:, 1:]={tuple(expected_confidence_shape)}."
            )

    penalties = []
    missing_module_names = [module_name for module_name in target_module_names if module_name not in activation_cache]
    if missing_module_names:
        raise ValueError(
            "Activation cache is missing outputs for target modules: "
            f"{missing_module_names}. Ensure hooks were registered before the forward pass."
        )

    for module_name in target_module_names:
        activation = activation_cache[module_name]
        if isinstance(activation, (tuple, list)):
            activation = activation[0]
        if not torch.is_tensor(activation):
            raise ValueError(
                f"Activation hook for module '{module_name}' returned {type(activation).__name__}, "
                "expected a tensor."
            )
        if activation.dim() != 3:
            raise ValueError(
                f"Activation for module '{module_name}' has shape={tuple(activation.shape)}; "
                "expected [batch, sequence, hidden]."
            )
        if activation.size(0) != labels.size(0) or activation.size(1) != labels.size(1):
            raise ValueError(
                f"Activation shape {tuple(activation.shape)} for module '{module_name}' does not align with "
                f"labels shape {tuple(labels.shape)}. Expected the same batch and sequence dimensions."
            )

        prediction_activations = activation[:, :-1, :]
        if regularizer_type == "conf_offdiag_cov":
            token_mask = labels[:, 1:].ne(IGNORE_INDEX)
        elif answer_only:
            token_mask = labels[:, 1:].ne(IGNORE_INDEX)
        elif attention_mask is not None:
            token_mask = attention_mask[:, 1:].bool()
        else:
            token_mask = torch.ones(
                prediction_activations.shape[:2],
                device=prediction_activations.device,
                dtype=torch.bool,
            )
        token_mask = token_mask.to(device=prediction_activations.device, dtype=torch.bool)

        if token_mask.shape != prediction_activations.shape[:2]:
            raise ValueError(
                f"Activation mask shape {tuple(token_mask.shape)} does not align with prediction activations "
                f"shape {tuple(prediction_activations.shape)} for module '{module_name}'."
            )

        selected_activations = prediction_activations[token_mask]
        if regularizer_type == "conf_offdiag_cov":
            confidence_values = token_confidences.to(device=prediction_activations.device)[token_mask]
            confidence_values = confidence_values.detach().float()
            confidence_weights = torch.clamp(
                confidence_values,
                min=activation_confidence_min,
            ).pow(activation_confidence_power)
            penalties.append(_weighted_offdiag_covariance_penalty(selected_activations, confidence_weights))
        else:
            penalties.append(_offdiag_covariance_penalty(selected_activations))

    return torch.stack(penalties).mean()


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


# TODO: 这个应该show两个例子，一个bs;
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
    model_name_or_path: Optional[str] = None,
    data_path: str = "data/train_medmcqa_alpaca_10k.jsonl",
    cache_dir: str = None,
    model_max_length: int = 512,
    per_device_train_batch_size: int = 4,
    num_train_epochs: float = 3.0,
    learning_rate: float = 2e-5,
    global_batch_size: int = 64,
    mode: str = "sft",  # sft, dft, sft+kl, asft, dft+sft
    kl_weight: float = 0.1,
    kl_chunk_size: int = 256,
    alpha: float = 0.1,
    clip_min: float = 0.1,
    clip_max: float = 2.0,
    output_dir: str = None,
    use_adapter: bool = False,
    adapter_type: str = "lora",
    target_modules=None,
    adapter_bias: str = "none",
    use_lora: bool = False,
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
    use_geometry_regularizer: bool = False,
    lambda_geom: float = 0.0,
    geometry_regularizer_type: str = "orthogonality",
    geometry_target_modules=None,
    disable_kl_in_geometry_asft: bool = False,
    use_activation_regularizer: bool = False,
    lambda_act: float = 0.0,
    activation_regularizer_type: str = "offdiag_cov",
    activation_confidence_power: float = 1.0,
    activation_confidence_min: float = 0.0,
    activation_target_modules="v_proj",
    activation_target_last_k_layers: int = 4,
    activation_answer_only: bool = True,
    deepspeed_config: Optional[str] = None,
    precision: str = "bf16",
    gradient_checkpointing: bool = False,
    seed: int = 42,
    max_steps: int = -1,
    **kwargs
):
    """Enhanced training with multiple DFT variants"""

    if not model_name_or_path:
        raise ValueError(
            "model_name_or_path must be provided explicitly. "
            "Pass it with --model_name_or_path /path/to/model."
        )

    model_args = ModelArguments(model_name_or_path=model_name_or_path)
    data_args = DataArguments(data_path=data_path)

    set_random_seed(seed)

    print("==== ModelArguments ====")
    print(model_args)
    print("========================")

    print("==== DataArguments ====")
    print(data_args)
    print("=======================")

    if output_dir is None:
        output_dir: str = f"./output/{mode}/{os.path.basename(model_name_or_path)}"

    adapter_enabled = use_adapter or use_lora
    normalized_target_modules = parse_target_modules(target_modules)
    normalized_adapter_type = adapter_type.lower()
    normalized_oft_regularizer_type = oft_regularizer_type.lower()
    normalized_geometry_regularizer_type = geometry_regularizer_type.lower()
    normalized_geometry_target_modules = parse_target_modules(
        geometry_target_modules if geometry_target_modules is not None else normalized_target_modules
    )
    normalized_activation_regularizer_type = activation_regularizer_type.lower()
    normalized_activation_target_modules = parse_activation_target_modules(activation_target_modules)

    if use_oft_regularizer:
        if normalized_adapter_type != "oftv2":
            raise ValueError(
                "use_oft_regularizer=True requires adapter_type='oftv2'. "
                f"Received adapter_type={normalized_adapter_type!r}."
            )
        if not adapter_enabled:
            raise ValueError("use_oft_regularizer=True requires OFTv2 adapter injection to be enabled.")
        if mode != "asft":
            raise ValueError(
                "use_oft_regularizer=True is currently only supported for mode='asft', "
                f"received mode={mode!r}."
            )
        if normalized_oft_regularizer_type != "identity":
            raise ValueError(
                f"Unsupported oft_regularizer_type={oft_regularizer_type!r}. Only 'identity' is supported."
            )

    if use_geometry_regularizer:
        if mode != "asft":
            raise ValueError(
                "use_geometry_regularizer=True is currently only supported for mode='asft', "
                f"received mode={mode!r}."
            )
        if adapter_enabled:
            raise ValueError(
                "use_geometry_regularizer=True currently requires use_adapter=False and use_lora=False. "
                "This geometry penalty is designed for direct weight updates without PEFT adapters."
            )
        if use_oft_regularizer:
            raise ValueError(
                "use_geometry_regularizer=True cannot be combined with use_oft_regularizer=True. "
                "The OFTv2 reg_only branch is kept unchanged in this file."
            )
        if lambda_geom < 0.0:
            raise ValueError(f"lambda_geom must be >= 0.0, got {lambda_geom}.")
        if normalized_geometry_regularizer_type != "orthogonality":
            raise ValueError(
                f"Unsupported geometry_regularizer_type={geometry_regularizer_type!r}. "
                "Only 'orthogonality' is supported."
            )
    elif disable_kl_in_geometry_asft:
        raise ValueError(
            "disable_kl_in_geometry_asft=True requires use_geometry_regularizer=True."
        )

    if activation_target_last_k_layers <= 0:
        raise ValueError(
            f"activation_target_last_k_layers must be > 0, got {activation_target_last_k_layers}."
        )
    if activation_confidence_power < 0.0:
        raise ValueError(
            f"activation_confidence_power must be >= 0.0, got {activation_confidence_power}."
        )
    if not 0.0 <= activation_confidence_min <= 1.0:
        raise ValueError(
            f"activation_confidence_min must be in [0.0, 1.0], got {activation_confidence_min}."
        )

    if use_activation_regularizer:
        if mode != "asft":
            raise ValueError(
                "use_activation_regularizer=True is currently only supported for mode='asft', "
                f"received mode={mode!r}."
            )
        if adapter_enabled:
            raise ValueError(
                "use_activation_regularizer=True currently requires use_adapter=False and use_lora=False. "
                "This first activation regularizer implementation is for direct weight updates without PEFT adapters."
            )
        if use_oft_regularizer:
            raise ValueError(
                "use_activation_regularizer=True cannot be combined with use_oft_regularizer=True."
            )
        if use_geometry_regularizer:
            raise ValueError(
                "Do not enable static geometry regularizer and dynamic activation regularizer at the same time "
                "in this first implementation; the experiment meaning would be mixed."
            )
        if lambda_act < 0.0:
            raise ValueError(f"lambda_act must be >= 0.0, got {lambda_act}.")
        supported_activation_regularizer_types = {"offdiag_cov", "conf_offdiag_cov"}
        if normalized_activation_regularizer_type not in supported_activation_regularizer_types:
            raise ValueError(
                f"Unsupported activation_regularizer_type={activation_regularizer_type!r}. "
                f"Supported values: {sorted(supported_activation_regularizer_types)}."
            )
        unsupported_activation_targets = [
            target_module for target_module in normalized_activation_target_modules if target_module != "v_proj"
        ]
        if unsupported_activation_targets:
            raise ValueError(
                "activation_target_modules currently only supports 'v_proj'. "
                f"Unsupported targets: {unsupported_activation_targets}."
            )
        if gradient_checkpointing:
            raise ValueError(
                "First-version activation regularizer does not support gradient_checkpointing=True because "
                "forward hooks and checkpointing repeated forwards can contaminate the activation cache. "
                "Please use gradient_checkpointing=False."
            )

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
    print(f"kl_chunk_size: {kl_chunk_size}")
    print(f"world_size: {world_size}")
    print(f"gradient_accumulation_steps: {gradient_accumulation_steps}")
    print(f"adapter_enabled: {adapter_enabled}")
    print(f"adapter_type: {normalized_adapter_type}")
    print(f"target_modules: {normalized_target_modules}")
    print(f"use_oft_regularizer: {use_oft_regularizer}")
    print(f"lambda_oft: {lambda_oft}")
    print(f"oft_regularizer_type: {normalized_oft_regularizer_type}")
    print(f"use_geometry_regularizer: {use_geometry_regularizer}")
    print(f"lambda_geom: {lambda_geom}")
    print(f"geometry_regularizer_type: {normalized_geometry_regularizer_type}")
    print(f"geometry_target_modules: {normalized_geometry_target_modules}")
    print(f"disable_kl_in_geometry_asft: {disable_kl_in_geometry_asft}")
    print(f"use_activation_regularizer: {use_activation_regularizer}")
    print(f"lambda_act: {lambda_act}")
    print(f"activation_regularizer_type: {normalized_activation_regularizer_type}")
    print(f"activation_confidence_power: {activation_confidence_power}")
    print(f"activation_confidence_min: {activation_confidence_min}")
    print(f"activation_target_modules: {normalized_activation_target_modules}")
    print(f"activation_target_last_k_layers: {activation_target_last_k_layers}")
    print(f"activation_answer_only: {activation_answer_only}")
    print("=============================")

    precision = precision.lower()
    if precision not in {"bf16", "fp16", "fp32"}:
        raise ValueError(f"Unsupported precision: {precision}. Use bf16, fp16, or fp32.")
    use_bf16 = precision == "bf16"
    use_fp16 = precision == "fp16"
    torch_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

    # Detect if we're in distributed training mode
    is_distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) if is_distributed else 0
    deepspeed_enabled = deepspeed_config is not None

    if is_distributed:
        print(f"Distributed training mode detected: {world_size} GPUs")
    else:
        print("Single GPU training mode")

    if deepspeed_enabled:
        os.environ.setdefault("DEEPSPEED_ZERO_INIT", "0")
        os.environ.setdefault("DEEPSPEED_ZERO3_INIT", "0")

    # Load model
    model_kwargs = {
        "cache_dir": cache_dir,
        "torch_dtype": torch_dtype,
    }
    # Only use device_map in single-GPU mode
    if not is_distributed and not deepspeed_enabled:
        model_kwargs["device_map"] = "auto"

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs
    )

    if use_geometry_regularizer:
        validate_geometry_target_modules(model, normalized_geometry_target_modules)
    if use_activation_regularizer:
        validate_activation_target_modules(
            model=model,
            target_modules=normalized_activation_target_modules,
            last_k_layers=activation_target_last_k_layers,
        )
    
    if adapter_enabled:
        adapter_config = build_adapter_config(
            model=model,
            adapter_type=normalized_adapter_type,
            target_modules=normalized_target_modules,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            oft_block_size=oft_block_size,
            oft_r=oft_r,
            oft_module_dropout=oft_module_dropout,
            use_cayley_neumann=use_cayley_neumann,
            adapter_bias=adapter_bias,
        )

        model = get_peft_model(model, adapter_config)
        model.print_trainable_parameters()

    # In distributed mode, move model to the correct device
    if is_distributed and not deepspeed_enabled:
        model = model.to(f"cuda:{local_rank}")

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=cache_dir,
        model_max_length=model_max_length,
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
    needs_reference_model = ("kl" in mode) or (
        mode == "asft" and not (use_geometry_regularizer and disable_kl_in_geometry_asft)
    )
    if needs_reference_model and not adapter_enabled:
        print("Loading original model for KL divergence...")

        original_model_kwargs = {
            "cache_dir": cache_dir,
            "torch_dtype": torch_dtype,
        }
        # Only use device_map in single-GPU mode
        if not is_distributed and not deepspeed_enabled:
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
            cache_dir=cache_dir,
            model_max_length=model_max_length,
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

    training_args = TrainingArguments(
        output_dir=output_dir,
        cache_dir=cache_dir,
        model_max_length=model_max_length,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        max_steps=max_steps,
        logging_steps=1,
        save_strategy="no",
        save_total_limit=1,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        seed=seed,
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=gradient_checkpointing,
        deepspeed=deepspeed_config,
        report_to="none",
        # logging_steps=1,
        **kwargs
    )

    print("==== Transformers TrainingArguments ====")
    print(training_args)
    print("=============================")

    if kwargs:
        print("==== Extra kwargs passed to TrainingArguments ====")
        print(kwargs)
        print("===============================================")

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = EnhancedTrainer(
        mode=mode,
        kl_weight=kl_weight,
        kl_chunk_size=kl_chunk_size,
        alpha=alpha,
        clip_min=clip_min,
        clip_max=clip_max,
        original_model=original_model,
        use_oft_regularizer=use_oft_regularizer,
        lambda_oft=lambda_oft,
        oft_regularizer_type=normalized_oft_regularizer_type,
        adapter_type=normalized_adapter_type,
        use_geometry_regularizer=use_geometry_regularizer,
        lambda_geom=lambda_geom,
        geometry_regularizer_type=normalized_geometry_regularizer_type,
        geometry_target_modules=normalized_geometry_target_modules,
        disable_kl_in_geometry_asft=disable_kl_in_geometry_asft,
        use_activation_regularizer=use_activation_regularizer,
        lambda_act=lambda_act,
        activation_regularizer_type=normalized_activation_regularizer_type,
        activation_confidence_power=activation_confidence_power,
        activation_confidence_min=activation_confidence_min,
        activation_target_modules=normalized_activation_target_modules,
        activation_target_last_k_layers=activation_target_last_k_layers,
        activation_answer_only=activation_answer_only,
        use_adapter=adapter_enabled,
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
    
