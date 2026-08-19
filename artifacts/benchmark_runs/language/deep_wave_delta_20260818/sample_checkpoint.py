from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from deep_wave_delta import DeepWaveDeltaConfig, DeepWaveDeltaLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample Deep Wave-Delta using its recurrent inference cache")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--architecture-config", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def sample(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    logits = logits.float() / max(temperature, 1e-5)
    if top_k > 0:
        values, indices = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
        selected = torch.multinomial(torch.softmax(values, dim=-1), 1)
        return indices.gather(-1, selected).squeeze(-1)
    return torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(-1)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    payload = json.loads(args.architecture_config.read_text(encoding="utf-8"))
    config = DeepWaveDeltaConfig(**payload["architecture"])
    model = DeepWaveDeltaLM(config).to(device=device, dtype=dtype).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True, local_files_only=True)
    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False) or [tokenizer.eos_token_id]
    torch.manual_seed(args.seed)
    cache = None
    generated = list(prompt_ids)
    with torch.inference_mode():
        logits = None
        for token_id in prompt_ids:
            logits, cache = model.step(torch.tensor([token_id], device=device), cache)
        assert logits is not None
        for _ in range(args.max_new_tokens):
            next_token = sample(logits[:, -1], args.temperature, args.top_k)
            generated.append(int(next_token.item()))
            if int(next_token.item()) == tokenizer.eos_token_id:
                break
            logits, cache = model.step(next_token, cache)
    print(tokenizer.decode(generated, skip_special_tokens=False))


if __name__ == "__main__":
    main()

