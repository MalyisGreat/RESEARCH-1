from __future__ import annotations

import argparse
import json
import random
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

from standalone_longseq_anchor_train import CausalConvFactorizedLM, TrainConfig


PROMPTS = [
    "The future of artificial intelligence is",
    "In a surprising result, the experiment showed that",
    "Question: What is machine learning? Answer:",
    "Question: Why do plants need sunlight? Answer:",
    "Explain the difference between training loss and validation loss.",
    "A small neural network can learn",
    "Once upon a time in a small town",
    "The old library at the edge of the city",
    "The captain looked at the map and realized",
    "During the experiment, the scientist noticed",
    "Python function to add two numbers:",
    "Here is a short recipe for tomato soup:",
    "The main causes of climate change include",
    "A good way to test a hypothesis is",
    "The patient reported mild symptoms and",
    "The history of mathematics includes",
    "If a ball is dropped from a tower",
    "The team built a simple robot that",
    "News report: Local officials announced",
    "The meaning of life is",
    "Step 1: collect the data. Step 2:",
    "A concise summary of this article:",
    "To debug a failing training run, first",
    "The most important limitation of this model is",
    "A careful scientist should avoid",
    "The city council voted to approve",
    "The following table compares the results:",
    "Write a polite email asking for more information:",
    "The theorem can be explained with a simple example:",
    "When the spacecraft entered orbit,",
    "A farmer wants to increase crop yield by",
    "The data suggest that the original hypothesis",
    "The program crashed because",
    "In plain English, validation loss measures",
    "A child asked why the sky is blue. The teacher said",
    "The restaurant menu included",
    "The economic report found that",
    "A useful checklist for evaluating model samples is",
    "The river flowed through the valley and",
    "The engineer opened the log file and saw",
]


def load_config(raw: dict[str, Any], checkpoint_path: Path) -> TrainConfig:
    defaults = {
        "cache_path": checkpoint_path.parent / "unused_cache.pt",
        "output_dir": checkpoint_path.parent,
        "run_name": checkpoint_path.parent.name,
    }
    raw = {**defaults, **raw}
    names = {field.name for field in fields(TrainConfig)}
    clean: dict[str, Any] = {}
    for name in names:
        if name not in raw:
            continue
        value = raw[name]
        if name in {"cache_path", "output_dir", "resume_checkpoint"} and value is not None:
            value = Path(value)
        clean[name] = value
    return TrainConfig(**clean)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.inference_mode()
def sample_one(
    model: CausalConvFactorizedLM,
    tokenizer: Any,
    prompt: str,
    *,
    config: TrainConfig,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids:
        ids = [tokenizer.eos_token_id]
    generated = torch.tensor([ids], dtype=torch.long, device=device)
    limit = max(1, int(config.sequence_length))
    for _ in range(max_new_tokens):
        window = generated[:, -limit:]
        hidden = model.factor_down(model.features(window))
        logits = model.factor_up(hidden[:, -1, :]).float().squeeze(0)
        if temperature > 0:
            logits = logits / temperature
        if top_k > 0 and top_k < logits.numel():
            values, indices = torch.topk(logits, top_k)
            probs = torch.softmax(values, dim=-1)
            next_token = indices[torch.multinomial(probs, num_samples=1)]
        else:
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token.view(1, 1)], dim=1)
        if int(next_token.item()) == int(tokenizer.eos_token_id):
            break
    return tokenizer.decode(generated.squeeze(0).tolist(), skip_special_tokens=False)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = load_config(dict(checkpoint.get("config", {})), args.checkpoint)
    model = CausalConvFactorizedLM(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True, local_files_only=True)
    tokenizer.model_max_length = int(1e9)

    prompts = [PROMPTS[index % len(PROMPTS)] for index in range(max(1, args.count))]
    temperatures = [0.65, 0.75, 0.85, 0.95]
    top_ks = [40, 50, 64, 80]
    samples: list[dict[str, Any]] = []
    for index, prompt in enumerate(prompts, start=1):
        temp = temperatures[(index - 1) % len(temperatures)]
        top_k = top_ks[(index - 1) % len(top_ks)]
        text = sample_one(
            model,
            tokenizer,
            prompt,
            config=config,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=temp,
            top_k=top_k,
        )
        samples.append({"index": index, "prompt": prompt, "temperature": temp, "top_k": top_k, "text": text})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "checkpoint": str(args.checkpoint),
        "step": int(checkpoint.get("step", 0)),
        "tokens_seen": int(checkpoint.get("tokens_seen", 0)),
        "count": len(samples),
        "max_new_tokens": args.max_new_tokens,
        "device": str(device),
    }
    lines = ["LongSeq checkpoint expanded samples", json.dumps(header, indent=2), ""]
    for sample in samples:
        lines.append(
            f"=== sample {sample['index']:02d} temp={sample['temperature']} top_k={sample['top_k']} "
            f"prompt={sample['prompt']!r} ==="
        )
        lines.append(sample["text"])
        lines.append("")
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"SAMPLE_OUTPUT={args.output}", flush=True)
    print(f"SAMPLE_COUNT={len(samples)}", flush=True)


if __name__ == "__main__":
    main()
