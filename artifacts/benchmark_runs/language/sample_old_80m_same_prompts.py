from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
ARC_TACTIC3 = REPO_ROOT / "arc_tactic3"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ARC_TACTIC3))

from language_longseq_replay_probe import LongSeqReplayProbeConfig, _build_model  # noqa: E402


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
]


def clean_config(raw: dict[str, Any]) -> LongSeqReplayProbeConfig:
    names = LongSeqReplayProbeConfig.__dataclass_fields__.keys()
    clean: dict[str, Any] = {}
    for name in names:
        if name not in raw:
            continue
        value = raw[name]
        if name in {"output_dir", "cache_path", "resolved_cache_path", "variant_artifact_dir"} and value is not None:
            value = Path(value)
        if name == "variants" and isinstance(value, list):
            value = tuple(value)
        clean[name] = value
    return LongSeqReplayProbeConfig(**clean)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.inference_mode()
def sample_one(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    *,
    sequence_length: int,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids:
        ids = [tokenizer.eos_token_id]
    generated = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        window = generated[:, -sequence_length:]
        hidden = model.factor_down(model.features(window))
        logits = model.factor_up(hidden[:, -1, :]).float().squeeze(0)
        if temperature > 0:
            logits = logits / temperature
        if 0 < top_k < logits.numel():
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
    config = clean_config(dict(checkpoint["config"]))
    model = _build_model(
        checkpoint["variant"],
        config,
        vocab_size=50_257,
        partial_token_ids=torch.empty(0, dtype=torch.long),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True, local_files_only=True)
    tokenizer.model_max_length = int(1e9)

    temperatures = [0.65, 0.75, 0.85, 0.95]
    top_ks = [40, 50, 64, 80]
    samples: list[dict[str, Any]] = []
    prompts = [PROMPTS[index % len(PROMPTS)] for index in range(max(1, args.count))]
    for index, prompt in enumerate(prompts, start=1):
        temperature = temperatures[(index - 1) % len(temperatures)]
        top_k = top_ks[(index - 1) % len(top_ks)]
        text = sample_one(
            model,
            tokenizer,
            prompt,
            sequence_length=int(config.sequence_length),
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        samples.append(
            {
                "index": index,
                "prompt": prompt,
                "temperature": temperature,
                "top_k": top_k,
                "text": text,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "checkpoint": str(args.checkpoint),
        "variant": str(checkpoint["variant"]),
        "step": int(checkpoint["step"]),
        "tokens_seen": int(checkpoint["tokens_seen"]),
        "count": len(samples),
        "max_new_tokens": args.max_new_tokens,
        "device": str(device),
    }
    lines = ["80M 2B checkpoint samples, same prompts as lowrank76M", json.dumps(header, indent=2), ""]
    for sample in samples:
        lines.append(
            f"=== sample {sample['index']:02d} temp={sample['temperature']} top_k={sample['top_k']} "
            f"prompt={sample['prompt']!r} ==="
        )
        lines.append(sample["text"])
        lines.append("")
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"OLD80M_SAMPLE_OUTPUT={args.output}", flush=True)
    print(f"OLD80M_SAMPLE_COUNT={len(samples)}", flush=True)


if __name__ == "__main__":
    main()
