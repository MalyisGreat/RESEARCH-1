from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

from standalone_longseq_anchor_train import CausalConvFactorizedLM, TrainConfig


PROMPTS = [
    "The future of artificial intelligence is",
    "Question: What is machine learning? Answer:",
    "Explain the difference between training loss and validation loss.",
    "To debug a failing training run, first",
    "The engineer opened the log file and saw",
    "A careful scientist should avoid",
    "Python function to add two numbers:",
    "Here is a short recipe for tomato soup:",
    "The city council voted to approve",
    "The captain looked at the map and realized",
    "In plain English, validation loss measures",
    "A farmer wants to increase crop yield by",
    "The old library at the edge of the city",
    "During the experiment, the scientist noticed",
    "The program crashed because",
    "A useful checklist for evaluating model samples is",
    "Write a polite email asking for more information:",
    "The main causes of climate change include",
    "If a ball is dropped from a tower",
    "The most important limitation of this model is",
    "A concise summary of this article:",
    "The data suggest that the original hypothesis",
    "News report: Local officials announced",
    "A child asked why the sky is blue. The teacher said",
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
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-checkpoints", type=int, default=8)
    parser.add_argument("--prompts", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int, default=140)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def choose_checkpoints(run_dirs: list[Path], limit: int) -> list[Path]:
    by_tokens: dict[int, Path] = {}
    metadata: list[tuple[int, int, Path]] = []
    for run_dir in run_dirs:
        for path in run_dir.glob("checkpoint*.pt"):
            try:
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                tokens = int(checkpoint.get("tokens_seen", -1))
                step = int(checkpoint.get("step", -1))
            except Exception:
                continue
            if tokens <= 0:
                continue
            by_tokens[tokens] = path
            metadata.append((tokens, step, path))
    metadata = sorted({(tokens, step, path) for tokens, step, path in metadata})
    if len(metadata) <= limit:
        return [path for _, _, path in metadata]
    # Evenly sample across progress, always including first and latest.
    selected: set[int] = {0, len(metadata) - 1}
    for index in range(limit):
        pos = round(index * (len(metadata) - 1) / max(1, limit - 1))
        selected.add(pos)
    return [metadata[index][2] for index in sorted(selected)]


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
) -> tuple[str, list[int]]:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids:
        ids = [tokenizer.eos_token_id]
    generated = torch.tensor([ids], dtype=torch.long, device=device)
    prompt_len = len(ids)
    limit = max(1, int(config.sequence_length))
    for _ in range(max_new_tokens):
        window = generated[:, -limit:]
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
    ids_out = generated.squeeze(0).tolist()
    return tokenizer.decode(ids_out, skip_special_tokens=False), ids_out[prompt_len:]


def repetition_metrics(text: str, token_ids: list[int]) -> dict[str, Any]:
    generated_text = text.strip()
    words = re.findall(r"[A-Za-z0-9']+", generated_text.lower())
    word_count = len(words)
    unique_word_ratio = len(set(words)) / max(1, word_count)
    token_count = len(token_ids)
    unique_token_ratio = len(set(token_ids)) / max(1, token_count)
    bigrams = list(zip(words, words[1:]))
    bigram_repetition = 0.0
    if bigrams:
        counts = Counter(bigrams)
        repeated = sum(count - 1 for count in counts.values() if count > 1)
        bigram_repetition = repeated / len(bigrams)
    longest_run = 1
    current = 1
    for left, right in zip(token_ids, token_ids[1:]):
        if left == right:
            current += 1
            longest_run = max(longest_run, current)
        else:
            current = 1
    non_ascii_ratio = sum(ord(ch) > 127 for ch in generated_text) / max(1, len(generated_text))
    return {
        "chars": len(generated_text),
        "words": word_count,
        "generated_tokens": token_count,
        "unique_word_ratio": unique_word_ratio,
        "unique_token_ratio": unique_token_ratio,
        "bigram_repetition": bigram_repetition,
        "longest_same_token_run": longest_run,
        "non_ascii_ratio": non_ascii_ratio,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = choose_checkpoints(args.run_dir, args.max_checkpoints)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True, local_files_only=True)
    tokenizer.model_max_length = int(1e9)
    prompts = PROMPTS[: max(1, min(args.prompts, len(PROMPTS)))]

    rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    samples_path = args.output_dir / "samples.md"
    with samples_path.open("w", encoding="utf-8") as sample_file:
        sample_file.write("# Checkpoint Samples\n\n")
        for checkpoint_path in checkpoints:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            config = load_config(dict(checkpoint.get("config", {})), checkpoint_path)
            model = CausalConvFactorizedLM(config).to(device)
            model.load_state_dict(checkpoint["model_state"], strict=True)
            model.eval()

            step = int(checkpoint.get("step", -1))
            tokens_seen = int(checkpoint.get("tokens_seen", -1))
            manifest.append({"checkpoint": str(checkpoint_path), "step": step, "tokens_seen": tokens_seen})
            sample_file.write(f"## checkpoint step={step} tokens={tokens_seen}\n\n")
            for prompt_index, prompt in enumerate(prompts, start=1):
                # Deterministic but different per checkpoint/prompt.
                torch.manual_seed(args.seed + step * 31 + prompt_index)
                text, generated_ids = sample_one(
                    model,
                    tokenizer,
                    prompt,
                    config=config,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                )
                metrics = repetition_metrics(text[len(prompt) :], generated_ids)
                row = {
                    "checkpoint": str(checkpoint_path),
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "tokens_b": tokens_seen / 1e9,
                    "prompt_index": prompt_index,
                    "prompt": prompt,
                    "text": text,
                    **metrics,
                }
                rows.append(row)
                sample_file.write(f"### prompt {prompt_index}: {prompt!r}\n\n")
                sample_file.write(text + "\n\n")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    csv_path = args.output_dir / "sample_metrics.csv"
    fieldnames = [
        "checkpoint",
        "step",
        "tokens_seen",
        "tokens_b",
        "prompt_index",
        "prompt",
        "chars",
        "words",
        "generated_tokens",
        "unique_word_ratio",
        "unique_token_ratio",
        "bigram_repetition",
        "longest_same_token_run",
        "non_ascii_ratio",
        "text",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    by_ckpt: list[dict[str, Any]] = []
    for item in manifest:
        subset = [row for row in rows if row["tokens_seen"] == item["tokens_seen"]]
        if not subset:
            continue
        by_ckpt.append(
            {
                **item,
                "sample_count": len(subset),
                "avg_words": sum(float(row["words"]) for row in subset) / len(subset),
                "avg_unique_word_ratio": sum(float(row["unique_word_ratio"]) for row in subset) / len(subset),
                "avg_unique_token_ratio": sum(float(row["unique_token_ratio"]) for row in subset) / len(subset),
                "avg_bigram_repetition": sum(float(row["bigram_repetition"]) for row in subset) / len(subset),
                "max_same_token_run": max(int(row["longest_same_token_run"]) for row in subset),
                "avg_non_ascii_ratio": sum(float(row["non_ascii_ratio"]) for row in subset) / len(subset),
            }
        )
    summary = {
        "checkpoints": by_ckpt,
        "samples_path": str(samples_path),
        "csv_path": str(csv_path),
        "device": str(device),
        "temperature": args.temperature,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "prompts": prompts,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("SAMPLE_EVAL_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
