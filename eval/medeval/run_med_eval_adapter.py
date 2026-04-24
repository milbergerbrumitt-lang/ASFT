import os
import re
import json
import argparse
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def get_field(sample: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in sample and sample[k] is not None:
            return sample[k]
    return default


def normalize_option_key(x: str) -> str:
    x = str(x).strip()
    m = re.match(r"^([A-Ea-e])[\).\s:：-]*", x)
    if m:
        return m.group(1).upper()
    if len(x) == 1 and x.upper() in {"A", "B", "C", "D", "E"}:
        return x.upper()
    return x


def extract_answer_letter(text: str) -> Optional[str]:
    if not text:
        return None

    patterns = [
        r"(?i)(?:answer|final answer|correct answer|答案)\s*[:：]?\s*[\(\[]?\s*([A-E])\s*[\)\]]?",
        r"(?i)\boption\s*([A-E])\b",
        r"(?i)\bchoice\s*([A-E])\b",
        r"\b([A-E])\b",
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(1).upper()

    text = text.strip()
    if text and text[0].upper() in {"A", "B", "C", "D", "E"}:
        return text[0].upper()

    return None


def build_options(sample: Dict[str, Any]) -> Dict[str, str]:
    # Common schemas
    options = get_field(sample, ["options", "choices"], None)

    if isinstance(options, dict):
        out = {}
        for k, v in options.items():
            nk = normalize_option_key(str(k))
            if nk in {"A", "B", "C", "D", "E"}:
                out[nk] = str(v)
        if out:
            return dict(sorted(out.items()))

    if isinstance(options, list):
        out = {}
        for idx, v in enumerate(options):
            key = chr(ord("A") + idx)
            if key <= "E":
                out[key] = str(v)
        if out:
            return out

    # Flat fields
    out = {}
    candidate_keys = {
        "A": ["A", "a", "opa", "option_a", "choice_a"],
        "B": ["B", "b", "opb", "option_b", "choice_b"],
        "C": ["C", "c", "opc", "option_c", "choice_c"],
        "D": ["D", "d", "opd", "option_d", "choice_d"],
        "E": ["E", "e", "ope", "option_e", "choice_e"],
    }
    for label, keys in candidate_keys.items():
        val = get_field(sample, keys, None)
        if val is not None:
            out[label] = str(val)

    return out


def get_gold_answer(sample: Dict[str, Any]) -> Optional[str]:
    answer = get_field(
        sample,
        ["answer", "label", "target", "gold", "correct_answer", "output"],
        None,
    )
    if answer is None:
        return None

    if isinstance(answer, int):
        if 0 <= answer <= 4:
            return chr(ord("A") + answer)

    answer = str(answer).strip()
    answer = normalize_option_key(answer)

    if answer in {"A", "B", "C", "D", "E"}:
        return answer

    # Sometimes label is option text rather than option letter
    options = build_options(sample)
    for k, v in options.items():
        if str(v).strip() == str(answer).strip():
            return k

    return None


def build_prompt(sample: Dict[str, Any], dataset_name: str) -> str:
    question = get_field(
        sample,
        ["question", "query", "instruction", "input", "problem", "stem"],
        "",
    )
    question = str(question).strip()

    options = build_options(sample)

    prompt_lines = [
        "You are a medical expert. Answer the following multiple-choice question.",
        "Return only the option letter (A, B, C, D, or E).",
        "",
        f"Question: {question}",
        "",
        "Options:",
    ]

    for k, v in options.items():
        prompt_lines.append(f"{k}. {v}")

    prompt_lines += [
        "",
        "Answer:"
    ]
    return "\n".join(prompt_lines)


class HFLocalLLM:
    def __init__(self, base_model_path: str, adapter_path: Optional[str] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )

        if adapter_path:
            model = PeftModel.from_pretrained(model, adapter_path)

        self.model = model.eval()

    def generate_one(self, prompt: str, max_new_tokens: int = 32, temperature: float = 0.0) -> str:
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip()


def test_dataset(
    llm: HFLocalLLM,
    data: List[Dict[str, Any]],
    dataset_name: str,
    max_new_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    correct = 0
    total = 0
    records = []

    for i, sample in enumerate(data):
        gold = get_gold_answer(sample)
        if gold is None:
            continue

        prompt = build_prompt(sample, dataset_name)
        pred_text = llm.generate_one(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        pred = extract_answer_letter(pred_text)

        is_correct = pred == gold
        correct += int(is_correct)
        total += 1

        records.append(
            {
                "index": i,
                "gold": gold,
                "pred": pred,
                "correct": is_correct,
                "raw_output": pred_text,
            }
        )

        if total % 20 == 0:
            print(f"[{dataset_name}] {total} samples, accuracy={correct/total:.4f}")

    accuracy = (correct / total) if total > 0 else 0.0
    print(f"{dataset_name:>10}: {accuracy:.4f} ({correct}/{total})")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "details": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", required=True, help="Base model path")
    parser.add_argument("--adapter_path", default=None, help="Optional PEFT adapter path")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--test_data_dir", default="eval/medeval/test_data")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    dataset_paths = {
        "medqa": os.path.join(args.test_data_dir, "medqa_test.jsonl"),
        "mmlu": os.path.join(args.test_data_dir, "mmlu_medical_test.jsonl"),
        "medmcqa": os.path.join(args.test_data_dir, "medmcqa_test.jsonl"),
    }

    model_name = args.adapter_path if args.adapter_path else args.base_model_path
    print(f"\nTesting model: {model_name}")

    llm = HFLocalLLM(
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path,
    )

    results = {}
    for dataset_name, dataset_path in dataset_paths.items():
        if not os.path.exists(dataset_path):
            print(f"Skip {dataset_name}: not found at {dataset_path}")
            continue
        data = load_jsonl(dataset_path)
        results[dataset_name] = test_dataset(
            llm,
            data,
            dataset_name,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

    # overall
    total_correct = sum(v["correct"] for v in results.values())
    total_count = sum(v["total"] for v in results.values())
    overall_accuracy = (total_correct / total_count) if total_count > 0 else 0.0
    results["overall"] = {
        "accuracy": overall_accuracy,
        "correct": total_correct,
        "total": total_count,
    }

    print("\nSummary:")
    for dataset_name, result in results.items():
        if isinstance(result, dict) and "accuracy" in result and "details" not in result:
            print(f"{dataset_name:>10}: {result['accuracy']:.4f} ({result['correct']}/{result['total']})")
    print(f"{'overall':>10}: {overall_accuracy:.4f} ({total_correct}/{total_count})")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
