from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from itertools import cycle
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from arc_tactic3.language_fastlearn_benchmark import count_parameters, set_global_seed
from arc_tactic3.language_realtext_microbench import (
    RealTextConfig,
    TokenBlockDataset,
    _build_optimizer,
    _dataset_tensors,
    _iter_tensor_batches,
    _loss_and_tokens,
    _move_batch,
    build_models,
    evaluate_loss,
)


@dataclass(frozen=True, slots=True)
class NanoStyleCompareConfig:
    cache_path: Path
    tokenizer_name: str = "gpt2"
    train_blocks: int = 2048
    val_blocks: int = 256
    sequence_length: int = 127
    batch_size: int = 16
    eval_batch_size: int = 32
    train_steps: int = 64
    eval_interval: int = 16
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    seed: int = 13
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = torch.cuda.is_available()
    pin_memory: bool = torch.cuda.is_available()
    use_fused_adamw: bool = torch.cuda.is_available()
    tensor_batching: bool = False
    cache_dataset_on_device: bool = False
    compute_val_bpb: bool = True
    recurrent_embedding_dim: int = 96
    recurrent_hidden_dim: int = 192
    recurrent_memory_dim: int = 96
    gpt_d_model: int = 96
    gpt_heads: int = 4
    gpt_layers: int = 2
    gpt_ff_dim: int = 384


def _load_cached_datasets(config: NanoStyleCompareConfig) -> tuple[TokenBlockDataset, TokenBlockDataset, int]:
    payload = torch.load(config.cache_path, map_location="cpu", weights_only=False)
    block_size = config.sequence_length + 1
    train_blocks = payload["train_tokens"].long().view(-1, block_size)
    val_blocks = payload["val_tokens"].long().view(-1, block_size)
    train_dataset = TokenBlockDataset(
        train_blocks[: config.train_blocks, :-1].contiguous(),
        train_blocks[: config.train_blocks, 1:].contiguous(),
    )
    val_dataset = TokenBlockDataset(
        val_blocks[: config.val_blocks, :-1].contiguous(),
        val_blocks[: config.val_blocks, 1:].contiguous(),
    )
    return train_dataset, val_dataset, int(payload["vocab_size"])


def _build_shared_config(config: NanoStyleCompareConfig) -> RealTextConfig:
    return RealTextConfig(
        seed=config.seed,
        sequence_length=config.sequence_length,
        train_steps=config.train_steps,
        eval_interval=config.eval_interval,
        batch_size=config.batch_size,
        eval_batch_size=config.eval_batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=config.device,
        use_amp=config.use_amp,
        pin_memory=config.pin_memory,
        use_fused_adamw=config.use_fused_adamw,
        tensor_batching=config.tensor_batching,
        cache_dataset_on_device=config.cache_dataset_on_device,
        recurrent_embedding_dim=config.recurrent_embedding_dim,
        recurrent_hidden_dim=config.recurrent_hidden_dim,
        recurrent_memory_dim=config.recurrent_memory_dim,
        gpt_d_model=config.gpt_d_model,
        gpt_heads=config.gpt_heads,
        gpt_layers=config.gpt_layers,
        gpt_ff_dim=config.gpt_ff_dim,
    )


def _estimate_target_bytes(dataset: TokenBlockDataset, *, tokenizer) -> int:
    total_bytes = 0
    for row in dataset.targets:
        text = tokenizer.decode(row.tolist(), clean_up_tokenization_spaces=False)
        total_bytes += len(text.encode("utf-8"))
    return max(total_bytes, 1)


def _precision_mode(config: NanoStyleCompareConfig) -> str:
    if config.device != "cuda":
        return "fp32"
    return "amp_fp16" if config.use_amp else "fp32"


def _gpu_name(device: str) -> str | None:
    if device != "cuda" or not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name(0)


def _peak_vram_mb(device: str) -> float | None:
    if device != "cuda" or not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def _train_with_stats(
    model: torch.nn.Module,
    train_dataset: TokenBlockDataset,
    val_dataset: TokenBlockDataset,
    *,
    tokenizer,
    config: RealTextConfig,
    compute_val_bpb: bool,
) -> dict[str, Any]:
    device = torch.device(config.device)
    model.to(device)
    optimizer = _build_optimizer(model, config)
    scaler = torch.amp.GradScaler(device="cuda", enabled=config.use_amp and device.type == "cuda")
    use_amp = config.use_amp and device.type == "cuda"
    parameter_list = [parameter for parameter in model.parameters() if parameter.requires_grad]

    if config.tensor_batching:
        train_source = _dataset_tensors(
            train_dataset,
            device=device,
            cache_on_device=config.cache_dataset_on_device,
            pin_memory=config.pin_memory,
        )
        val_source = _dataset_tensors(
            val_dataset,
            device=device,
            cache_on_device=config.cache_dataset_on_device,
            pin_memory=config.pin_memory,
        )
    else:
        pin_memory = config.pin_memory and device.type == "cuda"
        train_source = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=pin_memory,
            num_workers=0,
            persistent_workers=False,
        )
        val_source = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=0,
            persistent_workers=False,
        )

    train_iterator = (
        _iter_tensor_batches(
            train_source[0],
            train_source[1],
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            device=device,
            non_blocking=config.pin_memory and device.type == "cuda",
        )
        if config.tensor_batching
        else cycle(train_source)
    )

    initial_val_loss = evaluate_loss(model, val_source, device=device, use_amp=use_amp, config=config)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    step_times: list[float] = []
    eval_history: list[dict[str, float]] = [
        {
            "step": 0.0,
            "sequences_seen": 0.0,
            "tokens_seen": 0.0,
            "train_loss": float("nan"),
            "val_loss": initial_val_loss,
        }
    ]
    tokens_seen = 0
    sequences_seen = 0
    start = time.perf_counter()

    for step in range(1, config.train_steps + 1):
        if config.tensor_batching:
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = _iter_tensor_batches(
                    train_source[0],
                    train_source[1],
                    batch_size=config.batch_size,
                    shuffle=True,
                    drop_last=True,
                    device=device,
                    non_blocking=config.pin_memory and device.type == "cuda",
                )
                batch = next(train_iterator)
        else:
            batch = next(train_iterator)
            batch = _move_batch(batch, device)

        step_start = time.perf_counter()
        model.train()
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(batch["input_ids"])
            loss, token_count = _loss_and_tokens(logits, batch["targets"])
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameter_list, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_times.append(time.perf_counter() - step_start)

        tokens_seen += token_count
        sequences_seen += batch["input_ids"].size(0)

        if step % config.eval_interval == 0 or step == config.train_steps:
            val_loss = evaluate_loss(model, val_source, device=device, use_amp=use_amp, config=config)
            eval_history.append(
                {
                    "step": float(step),
                    "sequences_seen": float(sequences_seen),
                    "tokens_seen": float(tokens_seen),
                    "train_loss": float(loss.item()),
                    "val_loss": val_loss,
                }
            )

    total_training_time = time.perf_counter() - start
    final_val_loss = float(eval_history[-1]["val_loss"])
    target_bytes = _estimate_target_bytes(val_dataset, tokenizer=tokenizer) if compute_val_bpb and tokenizer is not None else None
    total_val_tokens = int(val_dataset.targets.numel())
    total_val_bits = final_val_loss * total_val_tokens / math.log(2.0)
    return {
        "parameter_count": count_parameters(model),
        "precision_mode": "amp_fp16" if use_amp else "fp32",
        "history": eval_history,
        "initial_val_loss": float(initial_val_loss),
        "final_val_loss": final_val_loss,
        "val_bits_per_token": final_val_loss / math.log(2.0),
        "val_bpb": (total_val_bits / target_bytes) if target_bytes is not None else None,
        "total_training_time_seconds": total_training_time,
        "train_tokens_seen": tokens_seen,
        "train_tok_per_sec": tokens_seen / max(total_training_time, 1e-9),
        "step_time_mean_ms": statistics.fmean(step_times) * 1000.0,
        "step_time_median_ms": statistics.median(step_times) * 1000.0,
        "peak_vram_mb": _peak_vram_mb(config.device),
        "train_mfu_estimate": None,
    }


def run_nano_style_compare(config: NanoStyleCompareConfig) -> dict[str, Any]:
    set_global_seed(config.seed)
    train_dataset, val_dataset, vocab_size = _load_cached_datasets(config)
    tokenizer = None
    if config.compute_val_bpb:
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, use_fast=True, local_files_only=True)

    shared_config = _build_shared_config(config)
    models = build_models(shared_config, vocab_size=vocab_size)
    reports: dict[str, dict[str, Any]] = {}
    for model_name, model in models.items():
        reports[model_name] = _train_with_stats(
            model,
            train_dataset,
            val_dataset,
            tokenizer=tokenizer,
            config=shared_config,
            compute_val_bpb=config.compute_val_bpb,
        )

    return {
        "benchmark": "language_nano_style_compare",
        "config": {
            **asdict(config),
            "cache_path": str(config.cache_path),
        },
        "device": {
            "name": _gpu_name(config.device),
            "precision_mode": _precision_mode(config),
        },
        "dataset": {
            "cache_path": str(config.cache_path),
            "train_blocks": len(train_dataset),
            "val_blocks": len(val_dataset),
            "sequence_length": config.sequence_length,
        },
        "models": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Nanochat-style stats benchmark for recurrent vs GPT baselines.")
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--train-blocks", type=int, default=2048)
    parser.add_argument("--val-blocks", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, default=127)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--train-steps", type=int, default=64)
    parser.add_argument("--eval-interval", type=int, default=16)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = NanoStyleCompareConfig(
        cache_path=args.cache_path,
        train_blocks=args.train_blocks,
        val_blocks=args.val_blocks,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        train_steps=args.train_steps,
        eval_interval=args.eval_interval,
        seed=args.seed,
        device=args.device,
    )
    payload = run_nano_style_compare(config)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
