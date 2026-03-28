from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import traceback
from array import array
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch import nn
from transformers import AutoTokenizer

from arc_tactic3.language_candidate_learned_compressor import LearnedChunkWriterPartialLM
from arc_tactic3.language_candidate_dynamic_token_basis import DynamicTokenBasisAssociativeLM
from arc_tactic3.language_candidate_local_global_memory import LocalGlobalTokenPartialMemoryLM
from arc_tactic3.language_fastlearn_benchmark import count_parameters, set_global_seed
from arc_tactic3.language_nanochat_actual_compare import NanochatMiniLM
from arc_tactic3.language_realtext_microbench import (
    RealTextConfig,
    TokenBlockDataset,
    _build_optimizer,
    _build_scheduler,
    _build_train_batch_schedule,
    _dataset_tensors,
    _loss_and_tokens,
    _scheduled_batch_from_tensors,
)
from arc_tactic3.language_recurrent_memory_rewrites import ChunkedTokenMemoryPartialLM
from arc_tactic3.language_recurrent_nano_tricks import PartialUntiedAssociativeLM, _top_token_ids


@dataclass(frozen=True, slots=True)
class ModelDims:
    embedding_dim: int
    hidden_dim: int
    memory_dim: int
    partial_untied_tokens: int
    chunk_size: int = 8
    local_window: int = 32
    older_chunk_size: int = 8


@dataclass(frozen=True, slots=True)
class RealTextStage:
    name: str
    train_blocks: int
    val_blocks: int
    train_steps: int
    eval_interval: int
    batch_size: int
    eval_batch_size: int
    sequence_length: int
    cache_on_device: bool = False
    compute_val_bpb: bool = False


@dataclass(frozen=True, slots=True)
class SyntheticStage:
    train_steps: int = 64
    eval_interval: int = 16
    batch_size: int = 32
    eval_batch_size: int = 64
    sequence_length: int = 63


@dataclass(frozen=True, slots=True)
class BigRunStage:
    total_tokens: int = 100_000_000
    train_tokens: int = 98_000_000
    val_tokens: int = 2_000_000
    sequence_length: int = 127
    target_parameters: int = 20_000_000
    eval_interval: int = 512
    eval_batches: int = 64
    sample_every_evals: int = 2
    target_batch_size: int = 16
    max_batch_size: int = 32
    min_batch_size: int = 4


@dataclass(frozen=True, slots=True)
class OvernightSweepConfig:
    small_cache_path: Path
    big_cache_path: Path
    output_dir: Path
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer_name: str = "gpt2"
    seed_primary: int = 13
    seed_secondary: int = 29
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    dropout: float = 0.1
    use_amp: bool = torch.cuda.is_available()
    pin_memory: bool = torch.cuda.is_available()
    use_fused_adamw: bool = torch.cuda.is_available()
    optimizer_recipe: str = "default"
    warmup_steps: int = 0
    lr_schedule: str = "none"
    min_lr_scale: float = 1.0
    candidates: tuple[str, ...] = (
        "partial_untied",
        "chunk_mean_token_memory",
        "learned_chunk_writer",
        "local_global_partial_memory",
        "dynamic_token_basis",
    )
    cheap_stage: RealTextStage = field(
        default_factory=lambda: RealTextStage(
            name="cheap",
            train_blocks=1024,
            val_blocks=64,
            train_steps=16,
            eval_interval=8,
            batch_size=16,
            eval_batch_size=32,
            sequence_length=127,
            cache_on_device=False,
        )
    )
    medium_stage: RealTextStage = field(
        default_factory=lambda: RealTextStage(
            name="medium",
            train_blocks=2048,
            val_blocks=128,
            train_steps=64,
            eval_interval=16,
            batch_size=16,
            eval_batch_size=32,
            sequence_length=127,
            cache_on_device=True,
        )
    )
    long_stage: RealTextStage = field(
        default_factory=lambda: RealTextStage(
            name="long",
            train_blocks=4096,
            val_blocks=256,
            train_steps=192,
            eval_interval=64,
            batch_size=16,
            eval_batch_size=32,
            sequence_length=127,
            cache_on_device=False,
        )
    )
    synthetic_stage: SyntheticStage = field(default_factory=SyntheticStage)
    big_stage: BigRunStage = field(default_factory=BigRunStage)
    big_finalist_count: int = 1
    include_nanochat_reference: bool = False
    dry_run: bool = False


@dataclass(frozen=True, slots=True)
class SyntheticTaskBundle:
    name: str
    train_dataset: TokenBlockDataset
    val_dataset: TokenBlockDataset
    vocab_size: int
    answer_mask: torch.Tensor
    prompt_span: tuple[int, int]
    description: str


class RunLogger:
    def __init__(self, output_dir: Path, total_tasks: int) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.output_dir / "events.jsonl"
        self.state_path = self.output_dir / "state.json"
        self.summary_path = self.output_dir / "summary.json"
        self.sample_path = self.output_dir / "samples.log"
        self.state: dict[str, Any] = {
            "completed_tasks": [],
            "results": {},
            "finalists": [],
            "total_tasks": total_tasks,
        }
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.completed_tasks = len(self.state.get("completed_tasks", []))
        self.total_tasks = max(total_tasks, int(self.state.get("total_tasks", 0)), self.completed_tasks + 1)
        self.state["total_tasks"] = self.total_tasks

    def save_state(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")

    def event(self, kind: str, **payload: Any) -> None:
        event = {"time": time.time(), "kind": kind, **payload}
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        self.save_state()

    def start_task(self, task_id: str, label: str) -> None:
        print(f"\n[{self.completed_tasks + 1}/{self.total_tasks}] START {label}", flush=True)
        self.event("task_start", task_id=task_id, label=label)

    def finish_task(self, task_id: str, label: str, result: dict[str, Any]) -> None:
        if task_id not in self.state["completed_tasks"]:
            self.state["completed_tasks"].append(task_id)
            self.completed_tasks += 1
        self.state["results"][task_id] = result
        self.event("task_end", task_id=task_id, label=label, result_path=result.get("result_path"))
        self.summary_path.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[{self.completed_tasks}/{self.total_tasks}] DONE  {label}", flush=True)

    def skipped_task(self, task_id: str, label: str) -> None:
        print(f"\n[{self.completed_tasks}/{self.total_tasks}] SKIP  {label}", flush=True)
        self.event("task_skip", task_id=task_id, label=label)

    def progress(
        self,
        *,
        task_id: str,
        label: str,
        step: int,
        total_steps: int,
        train_loss: float,
        val_loss: float | None,
        tokens_seen: int,
        tok_per_sec: float,
        eta_seconds: float,
    ) -> None:
        bar_width = 28
        ratio = step / max(total_steps, 1)
        filled = min(bar_width, int(round(ratio * bar_width)))
        bar = "#" * filled + "-" * (bar_width - filled)
        val_text = f"{val_loss:.4f}" if val_loss is not None else "----"
        eta_text = _format_seconds(eta_seconds)
        print(
            f"\r[{self.completed_tasks + 1}/{self.total_tasks}] {label} "
            f"[{bar}] {step}/{total_steps} "
            f"train={train_loss:.4f} val={val_text} "
            f"tok={tokens_seen} tok/s={tok_per_sec:,.0f} eta={eta_text}",
            end="",
            flush=True,
        )
        if step == total_steps or (val_loss is not None and step % max(total_steps // 4, 1) == 0):
            print("", flush=True)
        self.event(
            "progress",
            task_id=task_id,
            label=label,
            step=step,
            total_steps=total_steps,
            train_loss=train_loss,
            val_loss=val_loss,
            tokens_seen=tokens_seen,
            tok_per_sec=tok_per_sec,
            eta_seconds=eta_seconds,
        )

    def sample(self, label: str, prompt: str, generated: str) -> None:
        text = f"\n[{label}] sample\nPROMPT: {prompt}\nGENERATED: {generated}\n"
        print(text, flush=True)
        with self.sample_path.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")
        self.event("sample", label=label, prompt=prompt, generated=generated)


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    return f"{minutes:02d}m{secs:02d}s"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _flat_tokens_to_blocks(tokens: torch.Tensor, *, block_size: int) -> torch.Tensor:
    usable_tokens = (int(tokens.numel()) // block_size) * block_size
    if usable_tokens <= 0:
        raise ValueError(f"Not enough tokens to form a block of size {block_size}.")
    return tokens[:usable_tokens].long().view(-1, block_size)


def _load_small_cache(cache_path: Path, *, sequence_length: int, train_blocks: int, val_blocks: int) -> tuple[TokenBlockDataset, TokenBlockDataset, int]:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    block_size = sequence_length + 1
    train_blocks_tensor = _flat_tokens_to_blocks(payload["train_tokens"], block_size=block_size)
    val_blocks_tensor = _flat_tokens_to_blocks(payload["val_tokens"], block_size=block_size)
    train_dataset = TokenBlockDataset(
        train_blocks_tensor[:train_blocks, :-1].contiguous(),
        train_blocks_tensor[:train_blocks, 1:].contiguous(),
    )
    val_dataset = TokenBlockDataset(
        val_blocks_tensor[:val_blocks, :-1].contiguous(),
        val_blocks_tensor[:val_blocks, 1:].contiguous(),
    )
    return train_dataset, val_dataset, int(payload["vocab_size"])


def _buffer_to_dataset(token_buffer: torch.Tensor, *, sequence_length: int) -> TokenBlockDataset:
    block_size = sequence_length + 1
    blocks = token_buffer.long().view(-1, block_size)
    return TokenBlockDataset(blocks[:, :-1].contiguous(), blocks[:, 1:].contiguous())


def _stream_token_budget(
    *,
    total_tokens: int,
    tokenizer_name: str,
    text_column: str,
    logger: RunLogger,
    cache_path: Path,
) -> tuple[torch.Tensor, int]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    tokenizer.model_max_length = int(1e9)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer must expose eos_token_id.")
    stream = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    buffer = array("I")
    tokens_seen = 0
    docs_seen = 0
    batch_texts: list[str] = []
    batch_size = 128
    update_every = 1_000_000
    next_update = update_every
    start = time.perf_counter()

    def _consume_pending(texts: list[str]) -> None:
        nonlocal tokens_seen, next_update
        encoded = tokenizer(texts, add_special_tokens=False)
        for token_ids in encoded["input_ids"]:
            if tokens_seen >= total_tokens:
                break
            if not token_ids:
                continue
            token_ids = list(token_ids) + [eos_id]
            remaining = total_tokens - tokens_seen
            if len(token_ids) > remaining:
                token_ids = token_ids[:remaining]
            buffer.extend(token_ids)
            tokens_seen += len(token_ids)
            if tokens_seen >= next_update or tokens_seen == total_tokens:
                elapsed = max(time.perf_counter() - start, 1e-9)
                logger.progress(
                    task_id="download_big_cache",
                    label="fineweb_100m_download",
                    step=tokens_seen,
                    total_steps=total_tokens,
                    train_loss=0.0,
                    val_loss=None,
                    tokens_seen=tokens_seen,
                    tok_per_sec=tokens_seen / elapsed,
                    eta_seconds=(total_tokens - tokens_seen) / max(tokens_seen / elapsed, 1e-9),
                )
                next_update += update_every

    for row in stream:
        text = row.get(text_column, "")
        if not text or not text.strip():
            continue
        docs_seen += 1
        batch_texts.append(text)
        if len(batch_texts) < batch_size:
            continue
        _consume_pending(batch_texts)
        batch_texts = []
        if tokens_seen >= total_tokens:
            break
    if batch_texts and tokens_seen < total_tokens:
        _consume_pending(batch_texts)
    if tokens_seen != total_tokens:
        raise RuntimeError(f"FineWeb stream ended early: expected {total_tokens}, got {tokens_seen}")
    token_tensor = torch.from_numpy(np.frombuffer(buffer, dtype=np.uint32).copy().astype(np.int32, copy=False))
    logger.event("download_complete", path=str(cache_path), tokens=tokens_seen, docs_seen=docs_seen)
    return token_tensor, tokenizer.vocab_size


def ensure_big_fineweb_cache(config: OvernightSweepConfig, logger: RunLogger) -> None:
    if config.big_cache_path.exists():
        logger.event("cache_exists", path=str(config.big_cache_path))
        return
    logger.start_task("download_big_cache", "Build 100M-token FineWeb cache")
    token_tensor, vocab_size = _stream_token_budget(
        total_tokens=config.big_stage.total_tokens,
        tokenizer_name=config.tokenizer_name,
        text_column="text",
        logger=logger,
        cache_path=config.big_cache_path,
    )
    train_tokens = token_tensor[: config.big_stage.train_tokens].clone()
    val_tokens = token_tensor[config.big_stage.train_tokens : config.big_stage.train_tokens + config.big_stage.val_tokens].clone()
    config.big_cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "train_tokens": train_tokens,
            "val_tokens": val_tokens,
            "vocab_size": vocab_size,
        },
        config.big_cache_path,
    )
    result = {
        "result_path": str(config.big_cache_path),
        "train_tokens": int(train_tokens.numel()),
        "val_tokens": int(val_tokens.numel()),
        "vocab_size": vocab_size,
    }
    logger.finish_task("download_big_cache", "Build 100M-token FineWeb cache", result)


def _load_big_cache(cache_path: Path, *, sequence_length: int) -> tuple[TokenBlockDataset, TokenBlockDataset, int]:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    return (
        _buffer_to_dataset(payload["train_tokens"], sequence_length=sequence_length),
        _buffer_to_dataset(payload["val_tokens"], sequence_length=sequence_length),
        int(payload["vocab_size"]),
    )


def _realtext_config(
    config: OvernightSweepConfig,
    stage: RealTextStage,
    *,
    seed: int,
    batch_size: int | None = None,
    eval_batch_size: int | None = None,
    train_steps: int | None = None,
    sequence_length: int | None = None,
) -> RealTextConfig:
    return RealTextConfig(
        seed=seed,
        sequence_length=stage.sequence_length if sequence_length is None else sequence_length,
        train_steps=stage.train_steps if train_steps is None else train_steps,
        eval_interval=stage.eval_interval,
        batch_size=stage.batch_size if batch_size is None else batch_size,
        eval_batch_size=stage.eval_batch_size if eval_batch_size is None else eval_batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=config.device,
        use_amp=config.use_amp,
        pin_memory=config.pin_memory,
        use_fused_adamw=config.use_fused_adamw,
        tensor_batching=False,
        cache_dataset_on_device=stage.cache_on_device,
        paired_train_batches=True,
        reseed_per_model=True,
        train_schedule_seed=seed,
        optimizer_recipe=config.optimizer_recipe,
        warmup_steps=config.warmup_steps,
        lr_schedule=config.lr_schedule,
        min_lr_scale=config.min_lr_scale,
    )


def _synthetic_config(
    config: OvernightSweepConfig,
    stage: SyntheticStage,
    *,
    seed: int,
) -> RealTextConfig:
    return RealTextConfig(
        seed=seed,
        sequence_length=stage.sequence_length,
        train_steps=stage.train_steps,
        eval_interval=stage.eval_interval,
        batch_size=stage.batch_size,
        eval_batch_size=stage.eval_batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=config.device,
        use_amp=config.use_amp,
        pin_memory=config.pin_memory,
        use_fused_adamw=config.use_fused_adamw,
        tensor_batching=False,
        cache_dataset_on_device=False,
        paired_train_batches=True,
        reseed_per_model=True,
        train_schedule_seed=seed,
        optimizer_recipe=config.optimizer_recipe,
        warmup_steps=config.warmup_steps,
        lr_schedule=config.lr_schedule,
        min_lr_scale=config.min_lr_scale,
    )


def _candidate_common(
    dims: ModelDims,
    *,
    vocab_size: int,
    sequence_length: int,
    dropout: float,
) -> dict[str, Any]:
    return {
        "vocab_size": vocab_size,
        "embedding_dim": dims.embedding_dim,
        "hidden_dim": dims.hidden_dim,
        "memory_dim": dims.memory_dim,
        "dropout": dropout,
        "max_length": sequence_length,
    }


def build_candidate_model(
    name: str,
    *,
    dims: ModelDims,
    vocab_size: int,
    sequence_length: int,
    train_dataset: TokenBlockDataset,
    dropout: float,
) -> nn.Module:
    common = _candidate_common(dims, vocab_size=vocab_size, sequence_length=sequence_length, dropout=dropout)
    partial_token_ids = _top_token_ids(
        train_dataset,
        count=min(dims.partial_untied_tokens, vocab_size),
        vocab_size=vocab_size,
    )
    if name == "partial_untied":
        return PartialUntiedAssociativeLM(untied_token_ids=partial_token_ids, **common)
    if name == "chunk_mean_token_memory":
        return ChunkedTokenMemoryPartialLM(untied_token_ids=partial_token_ids, chunk_size=dims.chunk_size, **common)
    if name == "learned_chunk_writer":
        return LearnedChunkWriterPartialLM(untied_token_ids=partial_token_ids, chunk_size=dims.chunk_size, **common)
    if name == "local_global_partial_memory":
        return LocalGlobalTokenPartialMemoryLM(
            untied_token_ids=partial_token_ids,
            local_window=dims.local_window,
            older_chunk_size=dims.older_chunk_size,
            **common,
        )
    if name == "dynamic_token_basis":
        return DynamicTokenBasisAssociativeLM(
            **common,
            basis_rank=48,
            routing_experts=4,
            routing_top_k=2,
        )
    raise ValueError(f"Unknown candidate {name}")


def build_nanochat_model(
    *,
    vocab_size: int,
    sequence_length: int,
    target_params: int,
) -> NanochatMiniLM:
    best: tuple[int, tuple[int, int, int]] | None = None
    for n_embd in range(64, 256 + 1, 16):
        for n_layer in range(4, 10):
            for n_head in (4, 8):
                if n_embd % n_head != 0:
                    continue
                model = NanochatMiniLM(
                    vocab_size=vocab_size,
                    sequence_length=sequence_length,
                    n_layer=n_layer,
                    n_head=n_head,
                    n_kv_head=n_head,
                    n_embd=n_embd,
                    window_pattern="SSSL",
                    softcap=15.0,
                    use_value_embeddings=True,
                    use_smear=True,
                    use_backout=True,
                )
                params = count_parameters(model)
                diff = abs(params - target_params)
                if best is None or diff < best[0]:
                    best = (diff, (n_embd, n_layer, n_head))
    if best is None:
        raise RuntimeError("Failed to size Nanochat challenger.")
    n_embd, n_layer, n_head = best[1]
    return NanochatMiniLM(
        vocab_size=vocab_size,
        sequence_length=sequence_length,
        n_layer=n_layer,
        n_head=n_head,
        n_kv_head=n_head,
        n_embd=n_embd,
        window_pattern="SSSL",
        softcap=15.0,
        use_value_embeddings=True,
        use_smear=True,
        use_backout=True,
    )


def _greedy_sample(
    model: nn.Module,
    prompt_ids: list[int],
    *,
    sequence_length: int,
    device: torch.device,
    max_new_tokens: int,
) -> list[int]:
    sequence = list(prompt_ids)
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            window = sequence[-sequence_length:]
            input_ids = torch.tensor(window, dtype=torch.long, device=device).unsqueeze(0)
            logits = model(input_ids)
            sequence.append(int(logits[0, -1].argmax().item()))
    return sequence


def _eval_limited(
    model: nn.Module,
    val_source: tuple[torch.Tensor, torch.Tensor],
    *,
    device: torch.device,
    use_amp: bool,
    batch_limit: int | None = None,
) -> float:
    model.eval()
    loss_sum = 0.0
    token_total = 0
    total = val_source[0].size(0)
    batch_size = min(32, total)
    batch_count = 0
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_indices = torch.arange(start, end, dtype=torch.long, device=val_source[0].device)
        batch = _scheduled_batch_from_tensors(
            val_source[0],
            val_source[1],
            batch_indices,
            device=device,
            non_blocking=device.type == "cuda",
        )
        with torch.no_grad():
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(batch["input_ids"])
                loss, tokens = _loss_and_tokens(logits, batch["targets"])
        loss_sum += float(loss.item()) * tokens
        token_total += tokens
        batch_count += 1
        if batch_limit is not None and batch_count >= batch_limit:
            break
    return loss_sum / max(token_total, 1)


def _iter_batch_schedule_stream(
    *,
    total_examples: int,
    batch_size: int,
    steps: int,
    seed: int,
) -> Any:
    usable_total = total_examples - (total_examples % batch_size)
    if usable_total <= 0:
        raise ValueError("Not enough examples to form a single batch.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    epoch_order = torch.randperm(total_examples, generator=generator)
    epoch_cursor = 0
    emitted = 0
    while emitted < steps:
        if epoch_cursor + batch_size > usable_total:
            epoch_order = torch.randperm(total_examples, generator=generator)
            epoch_cursor = 0
        batch_indices = epoch_order[epoch_cursor : epoch_cursor + batch_size]
        if batch_indices.numel() < batch_size:
            epoch_order = torch.randperm(total_examples, generator=generator)
            epoch_cursor = 0
            continue
        yield batch_indices.clone()
        epoch_cursor += batch_size
        emitted += 1


def train_with_progress(
    *,
    model: nn.Module,
    train_dataset: TokenBlockDataset,
    val_dataset: TokenBlockDataset,
    tokenizer,
    model_name: str,
    config: RealTextConfig,
    logger: RunLogger,
    task_id: str,
    label: str,
    sample_every_evals: int = 1,
    eval_batch_limit: int | None = None,
) -> dict[str, Any]:
    device = torch.device(config.device)
    set_global_seed(config.seed)
    model.to(device)
    optimizer = _build_optimizer(model, config, model_name=model_name)
    scheduler = _build_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler(device="cuda", enabled=config.use_amp and device.type == "cuda")
    use_amp = config.use_amp and device.type == "cuda"
    parameter_list = [parameter for parameter in model.parameters() if parameter.requires_grad]
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
    schedule_seed = config.seed if config.train_schedule_seed is None else config.train_schedule_seed
    if config.train_steps > 4096:
        batch_schedule = _iter_batch_schedule_stream(
            total_examples=len(train_dataset),
            batch_size=config.batch_size,
            steps=config.train_steps,
            seed=schedule_seed,
        )
    else:
        batch_schedule = _build_train_batch_schedule(
            len(train_dataset),
            batch_size=config.batch_size,
            steps=config.train_steps,
            seed=schedule_seed,
            drop_last=True,
        )
    initial_val_loss = _eval_limited(model, val_source, device=device, use_amp=use_amp, batch_limit=eval_batch_limit)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    history: list[dict[str, float]] = [
        {"step": 0.0, "tokens_seen": 0.0, "train_loss": float("nan"), "val_loss": float(initial_val_loss)}
    ]
    step_times: list[float] = []
    eval_count = 0
    tokens_seen = 0
    start = time.perf_counter()
    for step, batch_indices in enumerate(batch_schedule, start=1):
        batch = _scheduled_batch_from_tensors(
            train_source[0],
            train_source[1],
            batch_indices,
            device=device,
            non_blocking=config.pin_memory and device.type == "cuda",
        )
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
        if scheduler is not None:
            scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_duration = time.perf_counter() - step_start
        step_times.append(step_duration)
        tokens_seen += token_count
        elapsed = max(time.perf_counter() - start, 1e-9)
        tok_per_sec = tokens_seen / elapsed
        eta_seconds = (config.train_steps - step) * statistics.fmean(step_times)
        val_loss: float | None = None
        if step % config.eval_interval == 0 or step == config.train_steps:
            val_loss = _eval_limited(model, val_source, device=device, use_amp=use_amp, batch_limit=eval_batch_limit)
            history.append(
                {
                    "step": float(step),
                    "tokens_seen": float(tokens_seen),
                    "train_loss": float(loss.item()),
                    "val_loss": float(val_loss),
                }
            )
            eval_count += 1
            if tokenizer is not None and eval_count % max(sample_every_evals, 1) == 0:
                prompt_ids = val_dataset.input_ids[0, : min(24, val_dataset.input_ids.size(1))].tolist()
                generated_ids = _greedy_sample(
                    model,
                    prompt_ids,
                    sequence_length=config.sequence_length,
                    device=device,
                    max_new_tokens=32,
                )
                logger.sample(
                    label,
                    tokenizer.decode(prompt_ids),
                    tokenizer.decode(generated_ids),
                )
        logger.progress(
            task_id=task_id,
            label=label,
            step=step,
            total_steps=config.train_steps,
            train_loss=float(loss.item()),
            val_loss=val_loss,
            tokens_seen=tokens_seen,
            tok_per_sec=tok_per_sec,
            eta_seconds=eta_seconds,
        )
    total_time = max(time.perf_counter() - start, 1e-9)
    pure_train_time = max(sum(step_times), 1e-9)
    final_val_loss = float(history[-1]["val_loss"])
    peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.type == "cuda" else None
    return {
        "parameter_count": count_parameters(model),
        "initial_val_loss": float(initial_val_loss),
        "final_val_loss": final_val_loss,
        "train_tokens_seen": int(tokens_seen),
        "train_tok_per_sec": tokens_seen / total_time,
        "pure_train_tok_per_sec": tokens_seen / pure_train_time,
        "step_time_mean_ms": statistics.fmean(step_times) * 1000.0,
        "step_time_median_ms": statistics.median(step_times) * 1000.0,
        "peak_vram_mb": peak_vram_mb,
        "history": history,
        "total_training_time_seconds": total_time,
    }


def _compare_score(report: dict[str, Any], baseline: dict[str, Any]) -> float:
    loss_improvement = (baseline["final_val_loss"] - report["final_val_loss"]) / max(baseline["final_val_loss"], 1e-9)
    throughput_delta = (report["pure_train_tok_per_sec"] - baseline["pure_train_tok_per_sec"]) / max(
        baseline["pure_train_tok_per_sec"], 1e-9
    )
    if loss_improvement >= 0.005:
        return 1.0 + loss_improvement + 0.10 * max(throughput_delta, 0.0)
    if loss_improvement >= -0.002 and throughput_delta >= 0.0:
        return 0.25 + 0.10 * throughput_delta
    return loss_improvement - 0.05 * max(-throughput_delta, 0.0)


def run_realtext_pair_stage(
    *,
    config: OvernightSweepConfig,
    logger: RunLogger,
    stage: RealTextStage,
    candidate_names: list[str],
    seed: int,
    result_name: str,
) -> dict[str, Any]:
    task_id = f"realtext_{stage.name}_{seed}_{result_name}"
    result_path = config.output_dir / f"{task_id}.json"
    if result_path.exists():
        logger.skipped_task(task_id, result_name)
        return json.loads(result_path.read_text(encoding="utf-8"))
    logger.start_task(task_id, result_name)
    train_dataset, val_dataset, vocab_size = _load_small_cache(
        config.small_cache_path,
        sequence_length=stage.sequence_length,
        train_blocks=stage.train_blocks,
        val_blocks=stage.val_blocks,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, use_fast=True, local_files_only=True)
    stage_cfg = _realtext_config(config, stage, seed=seed)
    reports: dict[str, Any] = {}
    for candidate_name in candidate_names:
        dims = ModelDims(
            embedding_dim=144,
            hidden_dim=288,
            memory_dim=144,
            partial_untied_tokens=2048 if candidate_name == "local_global_partial_memory" else 1024,
            chunk_size=8,
            local_window=32,
            older_chunk_size=8,
        )
        model = build_candidate_model(
            candidate_name,
            dims=dims,
            vocab_size=vocab_size,
            sequence_length=stage.sequence_length,
            train_dataset=train_dataset,
            dropout=config.dropout,
        )
        reports[candidate_name] = train_with_progress(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            tokenizer=tokenizer,
            model_name=candidate_name,
            config=stage_cfg,
            logger=logger,
            task_id=task_id,
            label=f"{stage.name}:{candidate_name}:seed{seed}",
            sample_every_evals=1,
        )
    baseline = reports["partial_untied"]
    ranking = []
    for candidate_name, report in reports.items():
        score = 0.0 if candidate_name == "partial_untied" else _compare_score(report, baseline)
        ranking.append(
            {
                "candidate": candidate_name,
                "score": score,
                "final_val_loss": report["final_val_loss"],
                "pure_train_tok_per_sec": report["pure_train_tok_per_sec"],
            }
        )
    ranking.sort(key=lambda item: item["score"], reverse=True)
    payload = {
        "stage": stage.name,
        "seed": seed,
        "reports": reports,
        "ranking": ranking,
        "result_path": str(result_path),
    }
    _write_json(result_path, payload)
    logger.finish_task(task_id, result_name, {"result_path": str(result_path)})
    return payload


def _make_sequence_dataset(
    sequences: list[list[int]],
    *,
    sequence_length: int,
    answer_masks: list[list[int]] | None = None,
) -> tuple[TokenBlockDataset, torch.Tensor]:
    inputs = []
    targets = []
    masks = []
    for index, sequence in enumerate(sequences):
        if len(sequence) != sequence_length + 1:
            raise ValueError("Sequence length mismatch.")
        inputs.append(sequence[:-1])
        targets.append(sequence[1:])
        if answer_masks is None:
            masks.append([0] * sequence_length)
        else:
            masks.append(answer_masks[index][1:])
    return (
        TokenBlockDataset(torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)),
        torch.tensor(masks, dtype=torch.float32),
    )


def build_synthetic_tasks(sequence_length: int, seed: int) -> list[SyntheticTaskBundle]:
    rng = np.random.default_rng(seed)
    vocab_size = 256
    seq_plus = sequence_length + 1
    filler_low = 40
    filler_high = 120

    def _delayed_copy(num_sequences: int) -> tuple[TokenBlockDataset, torch.Tensor]:
        sequences: list[list[int]] = []
        masks: list[list[int]] = []
        for _ in range(num_sequences):
            key = rng.integers(121, 180, size=6).tolist()
            filler = rng.integers(filler_low, filler_high, size=seq_plus - 1 - 6 - 1 - 6).tolist()
            seq = [1] + key + [2] + filler + key
            mask = [0] * (len(seq) - 6) + [1] * 6
            sequences.append(seq)
            masks.append(mask)
        return _make_sequence_dataset(sequences, sequence_length=sequence_length, answer_masks=masks)

    def _kv_recall(num_sequences: int) -> tuple[TokenBlockDataset, torch.Tensor]:
        sequences = []
        masks = []
        for _ in range(num_sequences):
            keys = rng.integers(121, 170, size=4).tolist()
            values = rng.integers(171, 220, size=4).tolist()
            query_index = int(rng.integers(0, 4))
            header = [1]
            for key, value in zip(keys, values, strict=True):
                header.extend([key, value])
            filler_len = seq_plus - len(header) - 3
            filler = rng.integers(filler_low, filler_high, size=filler_len).tolist()
            seq = header + filler + [2, keys[query_index], values[query_index]]
            mask = [0] * (len(seq) - 1) + [1]
            sequences.append(seq)
            masks.append(mask)
        return _make_sequence_dataset(sequences, sequence_length=sequence_length, answer_masks=masks)

    def _rule_retention(num_sequences: int) -> tuple[TokenBlockDataset, torch.Tensor]:
        sequences = []
        masks = []
        alphabet = np.arange(121, 150)
        for _ in range(num_sequences):
            shift = int(rng.integers(1, 4))
            rule_token = 10 + shift
            query = rng.choice(alphabet, size=6, replace=True)
            answer = [int(alphabet[(np.where(alphabet == token)[0][0] + shift) % len(alphabet)]) for token in query]
            filler = rng.integers(filler_low, filler_high, size=seq_plus - 1 - 6 - 6).tolist()
            seq = [rule_token] + query.tolist() + filler + answer
            mask = [0] * (len(seq) - 6) + [1] * 6
            sequences.append(seq)
            masks.append(mask)
        return _make_sequence_dataset(sequences, sequence_length=sequence_length, answer_masks=masks)

    def _interference(num_sequences: int) -> tuple[TokenBlockDataset, torch.Tensor]:
        sequences = []
        masks = []
        for _ in range(num_sequences):
            base_key = int(rng.integers(121, 150))
            distractors = [base_key, base_key + 1, base_key + 2]
            answers = rng.integers(171, 220, size=3).tolist()
            header = [1]
            for key, value in zip(distractors, answers, strict=True):
                header.extend([key, value])
            filler = rng.integers(filler_low, filler_high, size=seq_plus - len(header) - 3).tolist()
            seq = header + filler + [2, distractors[-1]]
            seq.append(answers[-1])
            mask = [0] * (len(seq) - 1) + [1]
            sequences.append(seq)
            masks.append(mask)
        return _make_sequence_dataset(sequences, sequence_length=sequence_length, answer_masks=masks)

    tasks: list[SyntheticTaskBundle] = []
    for name, description, builder in (
        ("delayed_copy", "Recall a token span that appears early after a long filler.", _delayed_copy),
        ("kv_recall", "Store multiple key-value pairs and answer a late key query.", _kv_recall),
        ("rule_retention", "Remember an early transformation rule and apply it much later.", _rule_retention),
        ("interference", "Retrieve the correct late key-value pair despite similar distractors.", _interference),
    ):
        train_dataset, _ = builder(2048)
        val_dataset, answer_mask = builder(256)
        tasks.append(
            SyntheticTaskBundle(
                name=name,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                vocab_size=vocab_size,
                answer_mask=answer_mask,
                prompt_span=(0, sequence_length - 8),
                description=description,
            )
        )
    return tasks


def _evaluate_answer_metrics(
    model: nn.Module,
    bundle: SyntheticTaskBundle,
    *,
    device: torch.device,
    use_amp: bool,
    eval_batch_size: int,
) -> dict[str, float]:
    model.eval()
    token_hits = 0.0
    token_total = 0.0
    seq_hits = 0.0
    seq_total = 0.0
    input_ids = bundle.val_dataset.input_ids
    targets = bundle.val_dataset.targets
    masks = bundle.answer_mask
    for start in range(0, input_ids.size(0), eval_batch_size):
        end = min(start + eval_batch_size, input_ids.size(0))
        batch_inputs = input_ids[start:end].to(device)
        batch_targets = targets[start:end].to(device)
        batch_masks = masks[start:end].to(device)
        with torch.no_grad():
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(batch_inputs)
        preds = logits.argmax(dim=-1)
        hits = ((preds == batch_targets).float() * batch_masks).sum(dim=-1)
        needed = batch_masks.sum(dim=-1)
        token_hits += float(hits.sum().item())
        token_total += float(needed.sum().item())
        seq_hits += float((hits == needed).float().sum().item())
        seq_total += float((needed > 0).float().sum().item())
    return {
        "answer_token_accuracy": token_hits / max(token_total, 1.0),
        "answer_sequence_accuracy": seq_hits / max(seq_total, 1.0),
    }


def run_synthetic_pack(
    *,
    config: OvernightSweepConfig,
    logger: RunLogger,
    candidate_names: list[str],
) -> dict[str, Any]:
    task_id = "synthetic_pack"
    result_path = config.output_dir / f"{task_id}.json"
    if result_path.exists():
        logger.skipped_task(task_id, "synthetic memory pack")
        return json.loads(result_path.read_text(encoding="utf-8"))
    logger.start_task(task_id, "synthetic memory pack")
    tasks = build_synthetic_tasks(config.synthetic_stage.sequence_length, config.seed_primary)
    payload: dict[str, Any] = {"tasks": {}, "result_path": str(result_path)}
    for bundle in tasks:
        bundle_reports: dict[str, Any] = {}
        for candidate_name in candidate_names:
            dims = ModelDims(
                embedding_dim=144,
                hidden_dim=288,
                memory_dim=144,
                partial_untied_tokens=2048 if candidate_name == "local_global_partial_memory" else 1024,
            )
            stage_cfg = _synthetic_config(config, config.synthetic_stage, seed=config.seed_primary)
            model = build_candidate_model(
                candidate_name,
                dims=dims,
                vocab_size=bundle.vocab_size,
                sequence_length=config.synthetic_stage.sequence_length,
                train_dataset=bundle.train_dataset,
                dropout=config.dropout,
            )
            report = train_with_progress(
                model=model,
                train_dataset=bundle.train_dataset,
                val_dataset=bundle.val_dataset,
                tokenizer=None,
                model_name=f"{candidate_name}:{bundle.name}",
                config=stage_cfg,
                logger=logger,
                task_id=task_id,
                label=f"synthetic:{bundle.name}:{candidate_name}",
                sample_every_evals=999,
                eval_batch_limit=16,
            )
            metrics = _evaluate_answer_metrics(
                model,
                bundle,
                device=torch.device(config.device),
                use_amp=config.use_amp and config.device == "cuda",
                eval_batch_size=config.synthetic_stage.eval_batch_size,
            )
            bundle_reports[candidate_name] = {**report, **metrics}
        payload["tasks"][bundle.name] = {
            "description": bundle.description,
            "reports": bundle_reports,
        }
    _write_json(result_path, payload)
    logger.finish_task(task_id, "synthetic memory pack", {"result_path": str(result_path)})
    return payload


def run_sequence_length_sweep(
    *,
    config: OvernightSweepConfig,
    logger: RunLogger,
    champion_name: str,
    candidate_name: str,
) -> dict[str, Any]:
    task_id = "sequence_sweep"
    result_path = config.output_dir / f"{task_id}.json"
    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        existing_reports = payload.get("lengths", {}).get("64", {}).get("reports", {})
        if champion_name in existing_reports and candidate_name in existing_reports:
            logger.skipped_task(task_id, "sequence length sweep")
            return payload
    logger.start_task(task_id, "sequence length sweep")
    payload: dict[str, Any] = {
        "candidate_name": candidate_name,
        "champion_name": champion_name,
        "lengths": {},
        "result_path": str(result_path),
    }
    for sequence_length in (64, 96, 127):
        stage = RealTextStage(
            name=f"seq{sequence_length}",
            train_blocks=2048,
            val_blocks=128,
            train_steps=96,
            eval_interval=24,
            batch_size=16,
            eval_batch_size=32,
            sequence_length=sequence_length,
            cache_on_device=False,
        )
        result = run_realtext_pair_stage(
            config=config,
            logger=logger,
            stage=stage,
            candidate_names=[champion_name, candidate_name],
            seed=config.seed_primary,
            result_name=f"seq{sequence_length}_{champion_name}_vs_{candidate_name}",
        )
        payload["lengths"][str(sequence_length)] = result
    _write_json(result_path, payload)
    logger.finish_task(task_id, "sequence length sweep", {"result_path": str(result_path)})
    return payload


def _select_finalists(
    cheap: dict[str, Any],
    medium_payloads: list[dict[str, Any]],
    synthetic: dict[str, Any],
    long_hold: dict[str, Any] | None = None,
) -> list[str]:
    baseline_name = "partial_untied"
    candidate_scores: dict[str, float] = {baseline_name: 0.0}
    disqualified: set[str] = set()
    promoted_candidates: set[str] | None = None
    if medium_payloads:
        promoted_candidates = set(medium_payloads[0]["reports"].keys())
        promoted_candidates.discard(baseline_name)
        for payload in medium_payloads[1:]:
            promoted_candidates &= set(payload["reports"].keys())
            promoted_candidates.discard(baseline_name)
    for stage_payload in [cheap, *medium_payloads]:
        reports = stage_payload["reports"]
        baseline = reports[baseline_name]
        for candidate_name, report in reports.items():
            if candidate_name == baseline_name:
                continue
            if promoted_candidates is not None and candidate_name not in promoted_candidates:
                continue
            candidate_scores[candidate_name] = candidate_scores.get(candidate_name, 0.0) + _compare_score(report, baseline)
    for task_payload in synthetic["tasks"].values():
        reports = task_payload["reports"]
        baseline = reports[baseline_name]
        for candidate_name, report in reports.items():
            if candidate_name == baseline_name:
                continue
            if promoted_candidates is not None and candidate_name not in promoted_candidates:
                continue
            token_delta = report["answer_token_accuracy"] - baseline["answer_token_accuracy"]
            seq_delta = report["answer_sequence_accuracy"] - baseline["answer_sequence_accuracy"]
            candidate_scores[candidate_name] = candidate_scores.get(candidate_name, 0.0) + token_delta + seq_delta
    if long_hold is not None:
        reports = long_hold["reports"]
        baseline = reports[baseline_name]
        for candidate_name, report in reports.items():
            if candidate_name == baseline_name:
                continue
            if promoted_candidates is not None and candidate_name not in promoted_candidates:
                continue
            if report["final_val_loss"] >= baseline["final_val_loss"]:
                disqualified.add(candidate_name)
                continue
            candidate_scores[candidate_name] = candidate_scores.get(candidate_name, 0.0) + 2.5 * _compare_score(report, baseline)
    ranking = sorted(
        (item for item in candidate_scores.items() if item[0] != baseline_name and item[0] not in disqualified),
        key=lambda item: item[1],
        reverse=True,
    )
    return [name for name, _ in ranking[:2]]


def _sequence_sweep_gate(seq_sweep: dict[str, Any]) -> dict[str, Any]:
    champion_name = seq_sweep["champion_name"]
    candidate_name = seq_sweep["candidate_name"]
    deltas: list[float] = []
    wins = 0
    for length_payload in seq_sweep["lengths"].values():
        reports = length_payload["reports"]
        champion_loss = float(reports[champion_name]["final_val_loss"])
        candidate_loss = float(reports[candidate_name]["final_val_loss"])
        relative_delta = (champion_loss - candidate_loss) / max(champion_loss, 1e-9)
        deltas.append(relative_delta)
        if candidate_loss < champion_loss:
            wins += 1
    mean_relative_delta = float(statistics.fmean(deltas)) if deltas else 0.0
    return {
        "candidate_name": candidate_name,
        "champion_name": champion_name,
        "wins": wins,
        "loss_deltas": deltas,
        "mean_relative_delta": mean_relative_delta,
        "passed": wins >= 2 and mean_relative_delta >= 0.0,
    }


def _search_dims_for_target(
    candidate_name: str,
    *,
    target_params: int,
    vocab_size: int,
    sequence_length: int,
    train_dataset: TokenBlockDataset,
    dropout: float,
) -> ModelDims:
    best_dims: ModelDims | None = None
    best_diff: int | None = None
    for embedding_dim in range(160, 352 + 1, 16):
        hidden_dim = embedding_dim * 2
        memory_dim = embedding_dim
        dims = ModelDims(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            memory_dim=memory_dim,
            partial_untied_tokens=2048 if candidate_name == "local_global_partial_memory" else 1024,
        )
        model = build_candidate_model(
            candidate_name,
            dims=dims,
            vocab_size=vocab_size,
            sequence_length=sequence_length,
            train_dataset=train_dataset,
            dropout=dropout,
        )
        params = count_parameters(model)
        diff = abs(params - target_params)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_dims = dims
    if best_dims is None:
        raise RuntimeError("Failed to size recurrent candidate.")
    return best_dims


def _find_safe_batch_size(
    *,
    model: nn.Module,
    device: str,
    sequence_length: int,
    vocab_size: int,
    target_batch_size: int,
    max_batch_size: int,
    min_batch_size: int,
) -> int:
    device_obj = torch.device(device)
    batch_size = min(target_batch_size, max_batch_size)
    while batch_size >= min_batch_size:
        try:
            if device_obj.type == "cuda":
                torch.cuda.empty_cache()
            model.to(device_obj)
            input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length), device=device_obj)
            targets = torch.randint(0, vocab_size, (batch_size, sequence_length), device=device_obj)
            logits = model(input_ids)
            loss, _ = _loss_and_tokens(logits, targets)
            loss.backward()
            if device_obj.type == "cuda":
                torch.cuda.synchronize()
            for parameter in model.parameters():
                parameter.grad = None
            return batch_size
        except torch.OutOfMemoryError:
            batch_size //= 2
            if device_obj.type == "cuda":
                torch.cuda.empty_cache()
    raise RuntimeError("Unable to find a safe batch size.")


def run_big_finalist(
    *,
    config: OvernightSweepConfig,
    logger: RunLogger,
    finalist_name: str,
) -> dict[str, Any]:
    task_id = f"big_{finalist_name}"
    result_path = config.output_dir / f"{task_id}.json"
    if result_path.exists():
        logger.skipped_task(task_id, f"big run {finalist_name}")
        return json.loads(result_path.read_text(encoding="utf-8"))
    ensure_big_fineweb_cache(config, logger)
    logger.start_task(task_id, f"100M-token big run {finalist_name}")
    train_dataset, val_dataset, vocab_size = _load_big_cache(config.big_cache_path, sequence_length=config.big_stage.sequence_length)
    dims = _search_dims_for_target(
        finalist_name,
        target_params=config.big_stage.target_parameters,
        vocab_size=vocab_size,
        sequence_length=config.big_stage.sequence_length,
        train_dataset=train_dataset,
        dropout=config.dropout,
    )
    model = build_candidate_model(
        finalist_name,
        dims=dims,
        vocab_size=vocab_size,
        sequence_length=config.big_stage.sequence_length,
        train_dataset=train_dataset,
        dropout=config.dropout,
    )
    safe_batch_size = _find_safe_batch_size(
        model=model,
        device=config.device,
        sequence_length=config.big_stage.sequence_length,
        vocab_size=vocab_size,
        target_batch_size=config.big_stage.target_batch_size,
        max_batch_size=config.big_stage.max_batch_size,
        min_batch_size=config.big_stage.min_batch_size,
    )
    train_steps = math.ceil(config.big_stage.train_tokens / (safe_batch_size * config.big_stage.sequence_length))
    stage = RealTextStage(
        name="big",
        train_blocks=len(train_dataset),
        val_blocks=len(val_dataset),
        train_steps=train_steps,
        eval_interval=config.big_stage.eval_interval,
        batch_size=safe_batch_size,
        eval_batch_size=safe_batch_size,
        sequence_length=config.big_stage.sequence_length,
        cache_on_device=False,
        compute_val_bpb=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, use_fast=True)
    stage_cfg = _realtext_config(
        config,
        stage,
        seed=config.seed_primary,
        batch_size=safe_batch_size,
        eval_batch_size=safe_batch_size,
        train_steps=train_steps,
        sequence_length=config.big_stage.sequence_length,
    )
    report = train_with_progress(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        model_name=f"{finalist_name}_20m",
        config=stage_cfg,
        logger=logger,
        task_id=task_id,
        label=f"big:{finalist_name}",
        sample_every_evals=config.big_stage.sample_every_evals,
        eval_batch_limit=config.big_stage.eval_batches,
    )
    payload = {
        "candidate": finalist_name,
        "dims": asdict(dims),
        "safe_batch_size": safe_batch_size,
        "train_steps": train_steps,
        "report": report,
        "result_path": str(result_path),
    }
    _write_json(result_path, payload)
    logger.finish_task(task_id, f"100M-token big run {finalist_name}", {"result_path": str(result_path)})
    return payload


def build_task_count(config: OvernightSweepConfig) -> int:
    return 9 + config.big_finalist_count + (0 if config.big_cache_path.exists() else 1)


def run_overnight(config: OvernightSweepConfig) -> dict[str, Any]:
    logger = RunLogger(config.output_dir, total_tasks=build_task_count(config))
    summary: dict[str, Any] = {
        "config": {
            **asdict(config),
            "small_cache_path": str(config.small_cache_path),
            "big_cache_path": str(config.big_cache_path),
            "output_dir": str(config.output_dir),
        }
    }
    try:
        cheap = run_realtext_pair_stage(
            config=config,
            logger=logger,
            stage=config.cheap_stage,
            candidate_names=list(config.candidates),
            seed=config.seed_primary,
            result_name="cheap_screen",
        )
        summary["cheap"] = cheap
        medium_candidates = ["partial_untied"] + [item["candidate"] for item in cheap["ranking"][:2] if item["candidate"] != "partial_untied"]
        medium = run_realtext_pair_stage(
            config=config,
            logger=logger,
            stage=config.medium_stage,
            candidate_names=medium_candidates,
            seed=config.seed_primary,
            result_name="medium_hold",
        )
        summary["medium"] = medium
        medium_secondary = run_realtext_pair_stage(
            config=config,
            logger=logger,
            stage=config.medium_stage,
            candidate_names=medium_candidates,
            seed=config.seed_secondary,
            result_name="medium_hold_seed29",
        )
        summary["medium_secondary"] = medium_secondary
        long_candidates = ["partial_untied"] + [item["candidate"] for item in medium["ranking"][:1] if item["candidate"] != "partial_untied"]
        long_hold = run_realtext_pair_stage(
            config=config,
            logger=logger,
            stage=config.long_stage,
            candidate_names=long_candidates,
            seed=config.seed_primary,
            result_name="long_hold",
        )
        summary["long_hold"] = long_hold
        synthetic = run_synthetic_pack(
            config=config,
            logger=logger,
            candidate_names=["partial_untied"] + [item["candidate"] for item in medium["ranking"][:2] if item["candidate"] != "partial_untied"],
        )
        summary["synthetic"] = synthetic
        finalists = _select_finalists(cheap, [medium, medium_secondary], synthetic, long_hold)
        if not finalists:
            finalists = [item["candidate"] for item in medium["ranking"] if item["candidate"] != "partial_untied"][:1]
        logger.state["finalists"] = finalists[: config.big_finalist_count]
        logger.save_state()
        seq_sweep = run_sequence_length_sweep(
            config=config,
            logger=logger,
            champion_name="partial_untied",
            candidate_name=finalists[0],
        )
        summary["sequence_sweep"] = seq_sweep
        sequence_gate = _sequence_sweep_gate(seq_sweep)
        summary["sequence_gate"] = sequence_gate
        if not sequence_gate["passed"]:
            summary["big_runs"] = []
            summary["big_run_skipped_reason"] = (
                f"{sequence_gate['candidate_name']} failed the sequence sweep gate against "
                f"{sequence_gate['champion_name']}"
            )
            logger.event(
                "big_run_skipped",
                candidate_name=sequence_gate["candidate_name"],
                champion_name=sequence_gate["champion_name"],
                mean_relative_delta=sequence_gate["mean_relative_delta"],
                wins=sequence_gate["wins"],
            )
            logger.summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
            return summary
        big_results = []
        for finalist_name in finalists[: config.big_finalist_count]:
            big_results.append(run_big_finalist(config=config, logger=logger, finalist_name=finalist_name))
        summary["big_runs"] = big_results
    except Exception as exc:
        error_payload = {
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        logger.event("error", **error_payload)
        (config.output_dir / "error.json").write_text(json.dumps(error_payload, indent=2), encoding="utf-8")
        summary["error"] = error_payload
        logger.summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        raise
    logger.summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args() -> OvernightSweepConfig:
    parser = argparse.ArgumentParser(description="Run an overnight recurrent memory benchmark sweep with visible progress.")
    parser.add_argument("--small-cache-path", type=Path, default=Path("E:/DEVNEW/arc_tactic3/benchmark_runs/fineweb_edu_first20m_gpt2tokens_cache.pt"))
    parser.add_argument("--big-cache-path", type=Path, default=Path("E:/DEVNEW/arc_tactic3/benchmark_runs/fineweb_edu_first100m_gpt2tokens_cache.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("E:/DEVNEW/arc_tactic3/benchmark_runs") / f"overnight_recurrent_sweep_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--big-finalist-count", type=int, default=1)
    parser.add_argument("--include-nanochat-reference", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return OvernightSweepConfig(
        small_cache_path=args.small_cache_path,
        big_cache_path=args.big_cache_path,
        output_dir=args.output_dir,
        device=args.device,
        big_finalist_count=max(1, args.big_finalist_count),
        include_nanochat_reference=args.include_nanochat_reference,
        dry_run=args.dry_run,
    )


def main() -> None:
    config = parse_args()
    if config.dry_run:
        print(json.dumps({"task_count": build_task_count(config), "config": asdict(config)}, indent=2, default=str))
        return
    summary = run_overnight(config)
    print("\nOvernight sweep complete.")
    print(json.dumps({"output_dir": str(config.output_dir), "big_runs": summary.get("big_runs", [])}, indent=2))


if __name__ == "__main__":
    main()
