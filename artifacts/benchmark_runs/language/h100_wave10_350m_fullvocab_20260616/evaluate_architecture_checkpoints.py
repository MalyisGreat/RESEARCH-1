from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
from collections import Counter
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer


HERE = Path(__file__).resolve().parent
TRAINER_PATH = HERE / "h100_wave10_fullvocab_train.py"
SPEC = importlib.util.spec_from_file_location("h100_wave10_train", TRAINER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {TRAINER_PATH}")
trainer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = trainer
SPEC.loader.exec_module(trainer)


PROMPTS = (
    "The history of computing began",
    "In a controlled scientific experiment, researchers found",
    "Python is a programming language that",
    "The capital of France is",
    "To solve the equation, first",
    "Once upon a time in a distant city",
    "The following function returns the sum of two numbers:\n",
    "Question: Why does ice float on water?\nAnswer:",
    "A healthy way to handle disagreement is",
    "The main risk of this engineering design is",
    "Alice gave Bob the blue notebook. Later, Bob",
    "The password is K7mQ-42. The password is",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260818)
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config_payload = dict(payload["config"])
    allowed = {field.name for field in fields(trainer.base.TrainConfig)}
    config_payload = {key: value for key, value in config_payload.items() if key in allowed}
    for key in ("cache_path", "output_dir", "resume_checkpoint"):
        if config_payload.get(key) is not None:
            config_payload[key] = Path(config_payload[key])
    config = trainer.base.TrainConfig(**config_payload)
    architecture = payload.get("h100_args", {}).get("architecture", "wave")
    model = trainer.base.CausalConvFactorizedLM(config)
    model = trainer.apply_architecture(model, architecture)
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    return model, config, {"architecture": architecture, "tokens_seen": payload.get("tokens_seen")}


def repetition_metrics(tokens: list[int]) -> dict[str, float]:
    if not tokens:
        return {"unique_token_ratio": 0.0, "repeat_4gram_fraction": 0.0, "max_token_run": 0.0}
    ngrams = [tuple(tokens[index : index + 4]) for index in range(max(0, len(tokens) - 3))]
    repeated = sum(count - 1 for count in Counter(ngrams).values() if count > 1)
    max_run = 1
    run = 1
    for previous, current in zip(tokens, tokens[1:]):
        run = run + 1 if current == previous else 1
        max_run = max(max_run, run)
    return {
        "unique_token_ratio": len(set(tokens)) / len(tokens),
        "repeat_4gram_fraction": repeated / max(len(ngrams), 1),
        "max_token_run": float(max_run),
    }


@torch.inference_mode()
def generate(
    model: torch.nn.Module,
    config: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    generator: torch.Generator,
    device: torch.device,
) -> dict[str, Any]:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated: list[int] = []
    entropies: list[float] = []
    for _ in range(max_new_tokens):
        context = input_ids[:, -config.sequence_length :]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            features = model.features(context)[:, -1, :]
            hidden = model.factor_down(features)
            logits = model.factor_up(hidden).float().squeeze(0)
        probabilities = torch.softmax(logits, dim=-1)
        entropies.append(float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum().item()))
        scaled = logits / max(temperature, 1e-6)
        values, indices = torch.topk(scaled, k=min(top_k, scaled.numel()))
        sampled = torch.multinomial(torch.softmax(values, dim=-1), 1, generator=generator)
        next_id = int(indices[sampled].item())
        generated.append(next_id)
        input_ids = torch.cat((input_ids, input_ids.new_tensor([[next_id]])), dim=1)
    completion = tokenizer.decode(generated, skip_special_tokens=True)
    metrics = repetition_metrics(generated)
    metrics["mean_next_token_entropy"] = statistics.mean(entropies)
    return {"prompt": prompt, "completion": completion, "token_ids": generated, "metrics": metrics}


def summarize(samples: list[dict[str, Any]]) -> dict[str, float]:
    keys = samples[0]["metrics"].keys()
    summary = {key: statistics.mean(sample["metrics"][key] for sample in samples) for key in keys}
    password = next(sample for sample in samples if "password" in sample["prompt"].lower())
    summary["password_exact_match"] = float("K7mQ-42" in password["completion"])
    return summary


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    results = []
    for checkpoint in args.checkpoint:
        model, config, metadata = load_model(checkpoint, device)
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed)
        samples = [
            generate(
                model,
                config,
                tokenizer,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                generator=generator,
                device=device,
            )
            for prompt in PROMPTS
        ]
        results.append(
            {
                "checkpoint": str(checkpoint),
                **metadata,
                "summary": summarize(samples),
                "samples": samples,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"results": results}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([{"architecture": row["architecture"], **row["summary"]} for row in results], indent=2))


if __name__ == "__main__":
    main()
