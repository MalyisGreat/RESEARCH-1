from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from arc_tactic3.language_fastlearn_benchmark import set_global_seed
from arc_tactic3.language_realtext_microbench import (
    RealTextConfig,
    TokenBlockDataset,
    build_models,
    fairness_summary,
    train_microbenchmark,
)


@dataclass(frozen=True, slots=True)
class FineWebCompareConfig:
    seed: int = 13
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    split: str = "train"
    text_column: str = "text"
    tokenizer_name: str = "gpt2"
    total_tokens: int = 20_000_000
    train_tokens: int = 18_000_000
    val_tokens: int = 2_000_000
    sequence_length: int = 127
    batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    recurrent_embedding_dim: int = 192
    recurrent_hidden_dim: int = 288
    recurrent_memory_dim: int = 168
    gpt_d_model: int = 184
    gpt_heads: int = 4
    gpt_layers: int = 2
    gpt_ff_dim: int = 640
    dropout: float = 0.1
    prompt_count: int = 4
    generation_tokens: int = 48
    cache_path: Path | None = None


def _config_payload(config: FineWebCompareConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["cache_path"] = str(config.cache_path) if config.cache_path is not None else None
    return payload


def _validate_token_budget(config: FineWebCompareConfig) -> None:
    if config.train_tokens + config.val_tokens != config.total_tokens:
        raise ValueError("train_tokens + val_tokens must equal total_tokens.")
    block_size = config.sequence_length + 1
    if config.total_tokens % block_size != 0:
        raise ValueError("total_tokens must be divisible by sequence_length + 1 for exact fixed-block packing.")
    if config.train_tokens % block_size != 0 or config.val_tokens % block_size != 0:
        raise ValueError("train_tokens and val_tokens must each be divisible by sequence_length + 1.")


def _fill_token_buffer(
    texts: Iterable[str],
    *,
    tokenizer,
    total_tokens: int,
) -> torch.Tensor:
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError(f"Tokenizer {tokenizer.name_or_path} does not expose eos_token_id.")
    buffer = torch.empty(total_tokens, dtype=torch.int32)
    cursor = 0
    for text in texts:
        if cursor >= total_tokens:
            break
        if not text or not text.strip():
            continue
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        token_ids.append(eos_id)
        remaining = total_tokens - cursor
        if len(token_ids) > remaining:
            token_ids = token_ids[:remaining]
        chunk = torch.tensor(token_ids, dtype=torch.int32)
        buffer[cursor : cursor + chunk.numel()] = chunk
        cursor += chunk.numel()
    if cursor != total_tokens:
        raise RuntimeError(f"Token stream ended early: expected {total_tokens} tokens, got {cursor}.")
    return buffer


def _buffer_to_dataset(token_buffer: torch.Tensor, *, sequence_length: int) -> TokenBlockDataset:
    block_size = sequence_length + 1
    blocks = token_buffer.long().view(-1, block_size)
    return TokenBlockDataset(blocks[:, :-1].contiguous(), blocks[:, 1:].contiguous())


def load_fineweb_token_budget(config: FineWebCompareConfig) -> tuple[TokenBlockDataset, TokenBlockDataset, int]:
    _validate_token_budget(config)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, use_fast=True)
    tokenizer.model_max_length = int(1e9)
    if config.cache_path is not None and config.cache_path.exists():
        payload = torch.load(config.cache_path)
        train_tokens = payload["train_tokens"]
        val_tokens = payload["val_tokens"]
        vocab_size = int(payload["vocab_size"])
    else:
        stream = load_dataset(config.dataset_name, split=config.split, streaming=True)
        token_buffer = _fill_token_buffer(
            (row[config.text_column] for row in stream),
            tokenizer=tokenizer,
            total_tokens=config.total_tokens,
        )
        train_tokens = token_buffer[: config.train_tokens].clone()
        val_tokens = token_buffer[config.train_tokens : config.train_tokens + config.val_tokens].clone()
        vocab_size = tokenizer.vocab_size
        if config.cache_path is not None:
            config.cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "train_tokens": train_tokens,
                    "val_tokens": val_tokens,
                    "vocab_size": vocab_size,
                },
                config.cache_path,
            )
    train_dataset = _buffer_to_dataset(train_tokens, sequence_length=config.sequence_length)
    val_dataset = _buffer_to_dataset(val_tokens, sequence_length=config.sequence_length)
    return train_dataset, val_dataset, vocab_size


def _realtext_config_from_fineweb(config: FineWebCompareConfig, *, train_dataset: TokenBlockDataset) -> RealTextConfig:
    steps_per_epoch = len(train_dataset) // config.batch_size
    eval_interval = max(steps_per_epoch // 4, 1)
    return RealTextConfig(
        seed=config.seed,
        dataset_name=config.dataset_name,
        dataset_config="streaming_first_20m_tokens",
        train_split=config.split,
        validation_split=config.split,
        text_column=config.text_column,
        tokenizer_name=config.tokenizer_name,
        sequence_length=config.sequence_length,
        max_train_sequences=len(train_dataset),
        max_eval_sequences=config.val_tokens // (config.sequence_length + 1),
        train_steps=steps_per_epoch,
        eval_interval=eval_interval,
        batch_size=config.batch_size,
        eval_batch_size=config.eval_batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=config.device,
        recurrent_embedding_dim=config.recurrent_embedding_dim,
        recurrent_hidden_dim=config.recurrent_hidden_dim,
        recurrent_memory_dim=config.recurrent_memory_dim,
        gpt_d_model=config.gpt_d_model,
        gpt_heads=config.gpt_heads,
        gpt_layers=config.gpt_layers,
        gpt_ff_dim=config.gpt_ff_dim,
        dropout=config.dropout,
    )


def _greedy_generate(model: torch.nn.Module, prompt_ids: list[int], *, sequence_length: int, device: str, total_steps: int) -> list[int]:
    sequence = prompt_ids[:]
    model.eval()
    device_obj = torch.device(device)
    with torch.no_grad():
        for _ in range(total_steps):
            window = sequence[-sequence_length:]
            input_ids = torch.tensor(window, dtype=torch.long, device=device_obj).unsqueeze(0)
            logits = model(input_ids)
            sequence.append(int(logits[0, -1].argmax().item()))
    return sequence


def run_compare(config: FineWebCompareConfig) -> dict[str, object]:
    set_global_seed(config.seed)
    start = time.perf_counter()
    train_dataset, val_dataset, vocab_size = load_fineweb_token_budget(config)
    train_config = _realtext_config_from_fineweb(config, train_dataset=train_dataset)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, use_fast=True)
    tokenizer.model_max_length = int(1e9)

    reports: dict[str, dict[str, object]] = {}
    trained_models: dict[str, torch.nn.Module] = {}
    model_runtimes: dict[str, float] = {}
    for name, model in build_models(train_config, vocab_size=vocab_size).items():
        model_start = time.perf_counter()
        report = train_microbenchmark(model, train_dataset, val_dataset, config=train_config)
        reports[name] = report
        trained_models[name] = model.to(config.device)
        model_runtimes[name] = time.perf_counter() - model_start

    samples: list[dict[str, str]] = []
    for index in range(min(config.prompt_count, len(val_dataset))):
        prompt_ids = val_dataset[index]["input_ids"][: min(32, config.sequence_length)].tolist()
        entry = {"prompt": tokenizer.decode(prompt_ids)}
        for name, model in trained_models.items():
            generated_ids = _greedy_generate(
                model,
                prompt_ids,
                sequence_length=config.sequence_length,
                device=config.device,
                total_steps=config.generation_tokens,
            )
            entry[name] = tokenizer.decode(generated_ids)
        samples.append(entry)

    model_summaries = {}
    for name, report in reports.items():
        initial_loss = float(report["initial_val_loss"])
        final_loss = float(report["final_val_loss"])
        model_summaries[name] = {
            "parameter_count": int(report["parameter_count"]),
            "initial_val_loss": initial_loss,
            "final_val_loss": final_loss,
            "loss_delta": initial_loss - final_loss,
            "relative_loss_reduction": (initial_loss - final_loss) / initial_loss if initial_loss else 0.0,
            "final_val_perplexity": math.exp(min(final_loss, 20.0)),
            "tokens_seen": float(report["history"][-1]["tokens_seen"]),
            "sequences_seen": float(report["history"][-1]["sequences_seen"]),
            "history": report["history"],
            "runtime_seconds": model_runtimes[name],
        }

    fairness = fairness_summary({name: {"parameter_count": report["parameter_count"]} for name, report in reports.items()})
    recurrent = model_summaries["associative_recurrent"]
    gpt = model_summaries["gpt2_like"]
    return {
        "benchmark": "language_fineweb20m_compare",
        "config": _config_payload(config),
        "dataset": {
            "name": config.dataset_name,
            "split": config.split,
            "tokenizer_name": config.tokenizer_name,
            "train_sequences": len(train_dataset),
            "validation_sequences": len(val_dataset),
            "sequence_length": config.sequence_length,
            "train_tokens": config.train_tokens,
            "val_tokens": config.val_tokens,
            "total_tokens": config.total_tokens,
        },
        "fairness": fairness,
        "models": model_summaries,
        "comparison": {
            "final_val_loss_gap": gpt["final_val_loss"] - recurrent["final_val_loss"],
            "relative_final_loss_advantage_vs_gpt": (gpt["final_val_loss"] - recurrent["final_val_loss"]) / gpt["final_val_loss"],
            "recurrent_better_by_loss": recurrent["final_val_loss"] < gpt["final_val_loss"],
            "same_tokens_seen": recurrent["tokens_seen"] == gpt["tokens_seen"],
        },
        "samples": samples,
        "winner_by_final_val_loss": min(model_summaries.items(), key=lambda item: item[1]["final_val_loss"])[0],
        "runtime_seconds_total": time.perf_counter() - start,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare associative recurrent vs GPT2-like on first 20M FineWeb-Edu tokens.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = FineWebCompareConfig(device=args.device, cache_path=args.cache_path)
    payload = run_compare(config)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
