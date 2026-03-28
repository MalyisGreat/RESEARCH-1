from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from datasets import DownloadConfig, load_dataset
from transformers import AutoTokenizer

from arc_tactic3.language_fastlearn_benchmark import count_parameters, set_global_seed
from arc_tactic3.language_nanochat_actual_compare import _peak_vram_mb
from arc_tactic3.language_partial_untied_watch import (
    _append_jsonl,
    _bar,
    _format_eta,
    _load_json_if_exists,
    _write_json,
)
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
from arc_tactic3.language_recurrent_nano_tricks import PartialUntiedAssociativeLM, _top_token_ids


_DEFAULT_PROMPTS = (
    "The capital of France is",
    "Machine learning is a field of study that",
    "Python is a programming language that",
    "In a surprising result, the experiment showed that",
)


@dataclass(frozen=True, slots=True)
class ClusterHardwareProfile:
    name: str
    device_name: str
    total_memory_gb: float
    amp_dtype: str
    batch_size: int
    eval_batch_size: int
    tokenization_batch_size: int
    eval_interval: int
    log_interval: int
    checkpoint_interval: int
    sample_interval: int
    cache_dataset_on_device: bool
    use_compile: bool


@dataclass(frozen=True, slots=True)
class PartialUntiedClusterConfig:
    output_dir: Path
    cache_path: Path | None = None
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    split: str = "train"
    text_column: str = "text"
    tokenizer_name: str = "gpt2"
    total_tokens: int = 100_000_000
    train_tokens: int = 98_000_000
    val_tokens: int = 2_000_000
    sequence_length: int = 127
    seed: int = 13
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    local_files_only: bool = False
    use_amp: bool = torch.cuda.is_available()
    amp_dtype: str = "auto"
    pin_memory: bool = torch.cuda.is_available()
    use_fused_adamw: bool = torch.cuda.is_available()
    cache_dataset_on_device: bool = True
    use_compile: bool = False
    tokenization_batch_size: int = 0
    batch_size: int = 0
    eval_batch_size: int = 0
    eval_interval: int = 0
    log_interval: int = 0
    checkpoint_interval: int = 0
    sample_interval: int = 0
    embedding_dim: int = 320
    hidden_dim: int = 640
    memory_dim: int = 320
    partial_token_count: int = 1024
    dropout: float = 0.1
    initial_eval: bool = True
    train_schedule_seed: int | None = None
    sample_prompts: tuple[str, ...] = _DEFAULT_PROMPTS
    sample_max_new_tokens: int = 64
    save_best_checkpoint: bool = True
    save_latest_checkpoint: bool = True
    save_final_checkpoint: bool = True
    hf_xet_high_performance: bool = True
    tokenizer_parallelism: bool = True


def _hardware_profile_from_name(device_name: str, total_memory_gb: float) -> ClusterHardwareProfile:
    upper = device_name.upper()
    if "H100" in upper:
        return ClusterHardwareProfile(
            name="h100",
            device_name=device_name,
            total_memory_gb=total_memory_gb,
            amp_dtype="bf16",
            batch_size=128,
            eval_batch_size=192,
            tokenization_batch_size=4096,
            eval_interval=256,
            log_interval=32,
            checkpoint_interval=512,
            sample_interval=256,
            cache_dataset_on_device=True,
            use_compile=False,
        )
    if "A100" in upper and total_memory_gb >= 70.0:
        return ClusterHardwareProfile(
            name="a100_80g",
            device_name=device_name,
            total_memory_gb=total_memory_gb,
            amp_dtype="bf16",
            batch_size=96,
            eval_batch_size=160,
            tokenization_batch_size=3072,
            eval_interval=256,
            log_interval=32,
            checkpoint_interval=512,
            sample_interval=256,
            cache_dataset_on_device=True,
            use_compile=False,
        )
    if "A100" in upper:
        return ClusterHardwareProfile(
            name="a100_40g",
            device_name=device_name,
            total_memory_gb=total_memory_gb,
            amp_dtype="bf16",
            batch_size=64,
            eval_batch_size=128,
            tokenization_batch_size=2048,
            eval_interval=256,
            log_interval=32,
            checkpoint_interval=512,
            sample_interval=256,
            cache_dataset_on_device=True,
            use_compile=False,
        )
    return ClusterHardwareProfile(
        name="generic_cuda" if device_name.upper() != "CPU" else "cpu",
        device_name=device_name,
        total_memory_gb=total_memory_gb,
        amp_dtype="fp16" if device_name.upper() != "CPU" else "fp32",
        batch_size=32 if device_name.upper() != "CPU" else 8,
        eval_batch_size=64 if device_name.upper() != "CPU" else 16,
        tokenization_batch_size=1024,
        eval_interval=256,
        log_interval=32,
        checkpoint_interval=512,
        sample_interval=256,
        cache_dataset_on_device=device_name.upper() != "CPU",
        use_compile=False,
    )


def detect_cluster_hardware_profile(device: str | None = None) -> ClusterHardwareProfile:
    device_name = "cpu"
    total_memory_gb = 0.0
    if torch.cuda.is_available() and (device is None or device.startswith("cuda")):
        device_index = 0
        if device is not None and ":" in device:
            device_index = int(device.split(":", 1)[1])
        props = torch.cuda.get_device_properties(device_index)
        device_name = props.name
        total_memory_gb = props.total_memory / (1024**3)
    return _hardware_profile_from_name(device_name, total_memory_gb)


def resolve_cluster_config(config: PartialUntiedClusterConfig) -> tuple[PartialUntiedClusterConfig, ClusterHardwareProfile]:
    profile = detect_cluster_hardware_profile(config.device)
    amp_dtype = config.amp_dtype
    if amp_dtype == "auto":
        amp_dtype = profile.amp_dtype
    use_amp = config.use_amp and config.device.startswith("cuda") and amp_dtype != "fp32"
    resolved = replace(
        config,
        batch_size=config.batch_size if config.batch_size > 0 else profile.batch_size,
        eval_batch_size=config.eval_batch_size if config.eval_batch_size > 0 else profile.eval_batch_size,
        tokenization_batch_size=config.tokenization_batch_size if config.tokenization_batch_size > 0 else profile.tokenization_batch_size,
        eval_interval=config.eval_interval if config.eval_interval > 0 else profile.eval_interval,
        log_interval=config.log_interval if config.log_interval > 0 else profile.log_interval,
        checkpoint_interval=config.checkpoint_interval if config.checkpoint_interval > 0 else profile.checkpoint_interval,
        sample_interval=config.sample_interval if config.sample_interval > 0 else profile.sample_interval,
        use_amp=use_amp,
        pin_memory=config.pin_memory and config.device.startswith("cuda"),
        use_fused_adamw=config.use_fused_adamw and config.device.startswith("cuda"),
        cache_dataset_on_device=config.cache_dataset_on_device and config.device.startswith("cuda"),
        use_compile=config.use_compile or profile.use_compile,
        amp_dtype=amp_dtype,
    )
    return resolved, profile


def _default_cache_path(config: PartialUntiedClusterConfig) -> Path:
    safe_name = f"finewebedu_train{config.train_tokens}_val{config.val_tokens}_seq{config.sequence_length}_{config.tokenizer_name}.pt"
    return config.output_dir / "cache" / safe_name


def _validate_token_budget(config: PartialUntiedClusterConfig) -> None:
    if config.train_tokens + config.val_tokens != config.total_tokens:
        raise ValueError("train_tokens + val_tokens must equal total_tokens.")
    block_size = config.sequence_length + 1
    if config.train_tokens % block_size != 0 or config.val_tokens % block_size != 0:
        raise ValueError("train_tokens and val_tokens must each be divisible by sequence_length + 1.")


def _configure_runtime_env(config: PartialUntiedClusterConfig) -> None:
    if config.hf_xet_high_performance:
        os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    if config.tokenizer_parallelism:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    if torch.cuda.is_available() and config.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def _config_payload(config: PartialUntiedClusterConfig, *, profile: ClusterHardwareProfile) -> dict[str, Any]:
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir)
    payload["cache_path"] = str(config.cache_path) if config.cache_path is not None else None
    payload["hardware_profile"] = asdict(profile)
    return payload


def _fill_token_buffer(
    texts: Iterable[str],
    *,
    tokenizer,
    total_tokens: int,
    batch_size: int,
    print_prefix: str | None,
) -> torch.Tensor:
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError(f"Tokenizer {tokenizer.name_or_path} does not expose eos_token_id.")
    buffer = torch.empty(total_tokens, dtype=torch.int32)
    cursor = 0
    pending: list[str] = []
    started = time.perf_counter()
    last_report = started

    def _consume(batch_texts: Sequence[str]) -> None:
        nonlocal cursor
        encoded = tokenizer(batch_texts, add_special_tokens=False, truncation=False)
        for token_ids in encoded["input_ids"]:
            if cursor >= total_tokens:
                break
            if not token_ids:
                continue
            ids = list(token_ids) + [eos_id]
            remaining = total_tokens - cursor
            if len(ids) > remaining:
                ids = ids[:remaining]
            chunk = torch.tensor(ids, dtype=torch.int32)
            buffer[cursor : cursor + chunk.numel()] = chunk
            cursor += chunk.numel()

    for text in texts:
        if cursor >= total_tokens:
            break
        if not text or not text.strip():
            continue
        pending.append(text)
        if len(pending) < batch_size:
            continue
        _consume(pending)
        pending = []
        now = time.perf_counter()
        if print_prefix is not None and now - last_report >= 5.0:
            tok_per_sec = cursor / max(now - started, 1e-9)
            print(
                f"{print_prefix} cache_build {_bar(cursor / total_tokens)} "
                f"{cursor:,}/{total_tokens:,} tokens tok/s={tok_per_sec:,.0f}",
                flush=True,
            )
            last_report = now
    if pending and cursor < total_tokens:
        _consume(pending)
    if cursor != total_tokens:
        raise RuntimeError(f"Token stream ended early: expected {total_tokens} tokens, got {cursor}.")
    return buffer


def _buffer_to_dataset(token_buffer: torch.Tensor, *, sequence_length: int) -> TokenBlockDataset:
    block_size = sequence_length + 1
    blocks = token_buffer.long().view(-1, block_size)
    return TokenBlockDataset(blocks[:, :-1].contiguous(), blocks[:, 1:].contiguous())


def ensure_fineweb_cache(
    config: PartialUntiedClusterConfig,
    *,
    print_progress: bool = True,
) -> tuple[TokenBlockDataset, TokenBlockDataset, int, Path]:
    _validate_token_budget(config)
    cache_path = config.cache_path if config.cache_path is not None else _default_cache_path(config)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        train_tokens = payload["train_tokens"]
        val_tokens = payload["val_tokens"]
        if train_tokens.numel() < config.train_tokens or val_tokens.numel() < config.val_tokens:
            raise ValueError(
                f"Cache {cache_path} is too small for requested token budget: "
                f"train {train_tokens.numel()} < {config.train_tokens} or val {val_tokens.numel()} < {config.val_tokens}."
            )
        train_dataset = _buffer_to_dataset(train_tokens[: config.train_tokens], sequence_length=config.sequence_length)
        val_dataset = _buffer_to_dataset(val_tokens[: config.val_tokens], sequence_length=config.sequence_length)
        return train_dataset, val_dataset, int(payload["vocab_size"]), cache_path

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name,
        use_fast=True,
        local_files_only=config.local_files_only,
    )
    tokenizer.model_max_length = int(1e9)
    download_config = DownloadConfig(max_retries=20, resume_download=True)
    stream = load_dataset(
        config.dataset_name,
        split=config.split,
        streaming=True,
        download_config=download_config,
    )
    token_buffer = _fill_token_buffer(
        (row[config.text_column] for row in stream),
        tokenizer=tokenizer,
        total_tokens=config.total_tokens,
        batch_size=config.tokenization_batch_size,
        print_prefix="cluster_partial_untied" if print_progress else None,
    )
    train_tokens = token_buffer[: config.train_tokens].clone()
    val_tokens = token_buffer[config.train_tokens : config.train_tokens + config.val_tokens].clone()
    torch.save(
        {
            "dataset_name": config.dataset_name,
            "split": config.split,
            "tokenizer_name": config.tokenizer_name,
            "sequence_length": config.sequence_length,
            "train_tokens": train_tokens,
            "val_tokens": val_tokens,
            "vocab_size": tokenizer.vocab_size,
            "total_tokens": config.total_tokens,
        },
        cache_path,
    )
    train_dataset = _buffer_to_dataset(train_tokens, sequence_length=config.sequence_length)
    val_dataset = _buffer_to_dataset(val_tokens, sequence_length=config.sequence_length)
    return train_dataset, val_dataset, tokenizer.vocab_size, cache_path


def _make_realtext_config(config: PartialUntiedClusterConfig, *, train_steps: int) -> RealTextConfig:
    return RealTextConfig(
        seed=config.seed,
        sequence_length=config.sequence_length,
        train_steps=train_steps,
        eval_interval=config.eval_interval,
        batch_size=config.batch_size,
        eval_batch_size=config.eval_batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=config.device,
        use_amp=config.use_amp and config.device.startswith("cuda"),
        pin_memory=config.pin_memory,
        use_fused_adamw=config.use_fused_adamw,
        tensor_batching=True,
        cache_dataset_on_device=config.cache_dataset_on_device,
        paired_train_batches=True,
        reseed_per_model=True,
        train_schedule_seed=config.train_schedule_seed,
        dropout=config.dropout,
        initial_eval=False,
    )


def _autocast_context(config: PartialUntiedClusterConfig):
    if not config.use_amp or not config.device.startswith("cuda"):
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if config.amp_dtype == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


def _evaluate_loss(
    model: torch.nn.Module,
    val_source: tuple[torch.Tensor, torch.Tensor],
    *,
    device: torch.device,
    config: PartialUntiedClusterConfig,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    input_ids, targets = val_source
    batch_size = config.eval_batch_size
    with torch.no_grad():
        for start in range(0, input_ids.size(0), batch_size):
            batch_input = input_ids[start : start + batch_size].to(device, non_blocking=config.pin_memory and device.type == "cuda")
            batch_targets = targets[start : start + batch_size].to(device, non_blocking=config.pin_memory and device.type == "cuda")
            with _autocast_context(config):
                logits = model(batch_input)
                loss, token_count = _loss_and_tokens(logits, batch_targets)
            total_loss += float(loss.item()) * token_count
            total_tokens += token_count
    return total_loss / max(total_tokens, 1)


def _greedy_generate(
    model: torch.nn.Module,
    prompt_ids: list[int],
    *,
    sequence_length: int,
    device: torch.device,
    max_new_tokens: int,
    config: PartialUntiedClusterConfig,
) -> list[int]:
    sequence = list(prompt_ids)
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            window = sequence[-sequence_length:]
            input_ids = torch.tensor(window, dtype=torch.long, device=device).unsqueeze(0)
            with _autocast_context(config):
                logits = model(input_ids)
            sequence.append(int(logits[0, -1].argmax().item()))
    return sequence


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler | None,
    config: PartialUntiedClusterConfig,
    step: int,
    tokens_seen: int,
    latest_val_loss: float,
    best_val_loss: float,
    cache_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "step": step,
            "tokens_seen": tokens_seen,
            "latest_val_loss": latest_val_loss,
            "best_val_loss": best_val_loss,
            "config": asdict(config),
            "cache_path": str(cache_path),
        },
        path,
    )


def _load_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    if scaler is not None and payload.get("scaler_state") is not None:
        scaler.load_state_dict(payload["scaler_state"])
    return payload


def watch_cluster_run(output_dir: Path, *, refresh_seconds: float = 5.0, once: bool = False) -> None:
    state_path = output_dir / "state.json"
    final_path = output_dir / "final.json"
    metrics_path = output_dir / "metrics.jsonl"
    samples_path = output_dir / "samples.jsonl"
    last_seen_step = -1
    last_seen_eval = -1
    last_seen_sample = -1
    while True:
        state = _load_json_if_exists(state_path)
        final_payload = _load_json_if_exists(final_path)
        if state is None:
            print(f"waiting for state file: {state_path}", flush=True)
            if once:
                return
            time.sleep(refresh_seconds)
            continue

        progress = float(state.get("progress", 0.0))
        step = int(state.get("step", 0))
        train_steps = int(state.get("train_steps", 0))
        tokens_seen = int(state.get("tokens_seen", 0))
        target_tokens = int(state.get("target_tokens", 0))
        train_tok_per_sec = float(state.get("train_tok_per_sec", float("nan")))
        pure_train_tok_per_sec = float(state.get("pure_train_tok_per_sec", float("nan")))
        train_loss = state.get("latest_train_loss")
        val_loss = state.get("latest_val_loss")
        eta_seconds = state.get("eta_seconds")
        peak_vram_mb = state.get("peak_vram_mb")
        status = str(state.get("status", "running"))
        if step != last_seen_step or once or status == "completed":
            train_loss_text = "----" if train_loss is None or not math.isfinite(float(train_loss)) else f"{float(train_loss):.4f}"
            val_loss_text = "----" if val_loss is None or not math.isfinite(float(val_loss)) else f"{float(val_loss):.4f}"
            print(
                f"cluster_partial_untied {_bar(progress)} {progress * 100:5.1f}% "
                f"step={step:,}/{train_steps:,} tok={tokens_seen:,}/{target_tokens:,} "
                f"train={train_loss_text} val={val_loss_text} "
                f"tok/s={train_tok_per_sec:,.0f} pure_tok/s={pure_train_tok_per_sec:,.0f} "
                f"eta={_format_eta(float(eta_seconds) if eta_seconds is not None else None)} "
                f"vram={float(peak_vram_mb):.0f}MB status={status}",
                flush=True,
            )
            last_seen_step = step
        if metrics_path.exists():
            with metrics_path.open("rb") as handle:
                try:
                    handle.seek(-4096, 2)
                except OSError:
                    handle.seek(0)
                tail_text = handle.read().decode("utf-8", errors="replace")
            for line in reversed(tail_text.splitlines()):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if payload.get("kind") == "eval":
                    eval_step = int(payload.get("step", 0))
                    if eval_step != last_seen_eval or once or status == "completed":
                        print(
                            f"latest eval: step={eval_step:,} tokens={int(payload.get('tokens_seen', 0)):,} "
                            f"val={float(payload.get('val_loss', float('nan'))):.4f}",
                            flush=True,
                        )
                        last_seen_eval = eval_step
                    break
        if samples_path.exists():
            with samples_path.open("rb") as handle:
                try:
                    handle.seek(-4096, 2)
                except OSError:
                    handle.seek(0)
                tail_text = handle.read().decode("utf-8", errors="replace")
            for line in reversed(tail_text.splitlines()):
                if not line.strip():
                    continue
                payload = json.loads(line)
                sample_step = int(payload.get("step", 0))
                if sample_step != last_seen_sample or once or status == "completed":
                    sample = payload["samples"][0]
                    print(
                        f"latest sample: step={sample_step:,} prompt={sample['prompt']!r} generated={sample['generated']!r}",
                        flush=True,
                    )
                    last_seen_sample = sample_step
                break
        if final_payload is not None and status == "completed":
            report = final_payload.get("report", {})
            print(
                f"completed final_val={float(report.get('final_val_loss', float('nan'))):.4f} "
                f"tokens={int(report.get('train_tokens_seen', 0)):,} "
                f"train_tok/s={float(report.get('train_tok_per_sec', float('nan'))):,.0f} "
                f"pure_tok/s={float(report.get('pure_train_tok_per_sec', float('nan'))):,.0f}",
                flush=True,
            )
            return
        if once:
            return
        time.sleep(refresh_seconds)


def run_partial_untied_cluster(
    config: PartialUntiedClusterConfig,
    *,
    print_progress: bool = True,
    print_samples: bool = True,
) -> dict[str, Any]:
    _configure_runtime_env(config)
    resolved_config, profile = resolve_cluster_config(config)
    train_dataset, val_dataset, vocab_size, resolved_cache_path = ensure_fineweb_cache(
        resolved_config,
        print_progress=print_progress,
    )
    tokens_per_step = resolved_config.sequence_length * resolved_config.batch_size
    train_steps = math.ceil(resolved_config.train_tokens / tokens_per_step)
    actual_train_tokens = train_steps * tokens_per_step
    real_config = _make_realtext_config(resolved_config, train_steps=train_steps)
    state_path = resolved_config.output_dir / "state.json"
    metrics_path = resolved_config.output_dir / "metrics.jsonl"
    samples_path = resolved_config.output_dir / "samples.jsonl"
    latest_ckpt = resolved_config.output_dir / "checkpoints" / "latest.pt"
    best_ckpt = resolved_config.output_dir / "checkpoints" / "best.pt"
    final_ckpt = resolved_config.output_dir / "checkpoints" / "final.pt"
    final_path = resolved_config.output_dir / "final.json"
    resolved_config.output_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(resolved_config.seed)
    partial_token_ids = _top_token_ids(train_dataset, count=resolved_config.partial_token_count, vocab_size=vocab_size)
    model = PartialUntiedAssociativeLM(
        vocab_size=vocab_size,
        embedding_dim=resolved_config.embedding_dim,
        hidden_dim=resolved_config.hidden_dim,
        memory_dim=resolved_config.memory_dim,
        dropout=resolved_config.dropout,
        max_length=resolved_config.sequence_length,
        untied_token_ids=partial_token_ids,
    )
    device = torch.device(real_config.device)
    model.to(device)
    if resolved_config.use_compile and hasattr(torch, "compile"):
        model = torch.compile(model, mode="max-autotune")
    optimizer = _build_optimizer(model, real_config, model_name="partial_untied_cluster")
    scheduler = _build_scheduler(optimizer, real_config)
    use_scaler = resolved_config.use_amp and resolved_config.amp_dtype == "fp16" and device.type == "cuda"
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_scaler) if device.type == "cuda" else None
    parameter_list = [parameter for parameter in model.parameters() if parameter.requires_grad]

    train_source = _dataset_tensors(
        train_dataset,
        device=device,
        cache_on_device=real_config.cache_dataset_on_device,
        pin_memory=real_config.pin_memory,
    )
    val_source = _dataset_tensors(
        val_dataset,
        device=device,
        cache_on_device=real_config.cache_dataset_on_device,
        pin_memory=real_config.pin_memory,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_config.tokenizer_name,
        use_fast=True,
        local_files_only=resolved_config.local_files_only,
    )
    tokenizer.model_max_length = int(1e9)

    schedule_seed = resolved_config.seed if resolved_config.train_schedule_seed is None else resolved_config.train_schedule_seed
    batch_schedule = _build_train_batch_schedule(
        len(train_dataset),
        batch_size=real_config.batch_size,
        steps=real_config.train_steps,
        seed=schedule_seed,
        drop_last=True,
    )

    history: list[dict[str, float]] = []
    step_times: list[float] = []
    tokens_seen = 0
    sequences_seen = 0
    start_step = 1
    best_val_loss = float("inf")
    latest_val_loss = float("nan")
    initial_val_loss = float("nan")

    if latest_ckpt.exists():
        checkpoint_payload = _load_checkpoint(
            latest_ckpt,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        start_step = int(checkpoint_payload["step"]) + 1
        tokens_seen = int(checkpoint_payload["tokens_seen"])
        latest_val_loss = float(checkpoint_payload.get("latest_val_loss", float("nan")))
        best_val_loss = float(checkpoint_payload.get("best_val_loss", float("inf")))
        if print_progress:
            print(f"resumed from checkpoint {latest_ckpt} at step={start_step-1:,} tokens={tokens_seen:,}", flush=True)

    if math.isnan(latest_val_loss) and resolved_config.initial_eval:
        initial_val_loss = _evaluate_loss(model, val_source, device=device, config=resolved_config)
        latest_val_loss = initial_val_loss
        best_val_loss = initial_val_loss
        history.append(
            {
                "step": 0.0,
                "tokens_seen": 0.0,
                "sequences_seen": 0.0,
                "train_loss": float("nan"),
                "val_loss": float(initial_val_loss),
            }
        )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    prompt_ids = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in resolved_config.sample_prompts]
    for step_index, batch_indices in enumerate(batch_schedule[start_step - 1 :], start=start_step):
        batch = _scheduled_batch_from_tensors(
            train_source[0],
            train_source[1],
            batch_indices,
            device=device,
            non_blocking=real_config.pin_memory and device.type == "cuda",
        )
        step_start = time.perf_counter()
        model.train()
        with _autocast_context(resolved_config):
            logits = model(batch["input_ids"])
            loss, token_count = _loss_and_tokens(logits, batch["targets"])
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(parameter_list, max_norm=1.0)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_duration = time.perf_counter() - step_start
        step_times.append(step_duration)
        tokens_seen += token_count
        sequences_seen += batch["input_ids"].size(0)
        elapsed = time.perf_counter() - start
        progress = step_index / real_config.train_steps
        train_tok_per_sec = tokens_seen / max(elapsed, 1e-9)
        pure_train_tok_per_sec = tokens_seen / max(sum(step_times), 1e-9)
        remaining_tokens = max(actual_train_tokens - tokens_seen, 0)
        eta_seconds = remaining_tokens / max(train_tok_per_sec, 1e-9)
        state_payload = {
            "benchmark": "language_partial_untied_cluster",
            "status": "running",
            "step": step_index,
            "train_steps": real_config.train_steps,
            "progress": progress,
            "tokens_seen": tokens_seen,
            "target_tokens": actual_train_tokens,
            "train_tok_per_sec": train_tok_per_sec,
            "pure_train_tok_per_sec": pure_train_tok_per_sec,
            "eta_seconds": eta_seconds,
            "latest_train_loss": float(loss.item()),
            "latest_val_loss": latest_val_loss,
            "best_val_loss": best_val_loss,
            "peak_vram_mb": _peak_vram_mb(real_config.device),
            "parameter_count": count_parameters(model),
            "hardware_profile": profile.name,
            "cache_path": str(resolved_cache_path),
        }
        if step_index % resolved_config.log_interval == 0 or step_index == real_config.train_steps:
            _write_json(state_path, state_payload)
            _append_jsonl(metrics_path, {"kind": "train", **state_payload, "timestamp": time.time()})
            if print_progress:
                print(
                    f"cluster_partial_untied {_bar(progress)} {step_index}/{real_config.train_steps} "
                    f"train={loss.item():.4f} val={latest_val_loss:.4f} tok={tokens_seen:,}/{actual_train_tokens:,} "
                    f"tok/s={train_tok_per_sec:,.0f} eta={eta_seconds/60:.1f}m",
                    flush=True,
                )

        should_eval = step_index % real_config.eval_interval == 0 or step_index == real_config.train_steps
        should_ckpt = step_index % resolved_config.checkpoint_interval == 0 or should_eval
        should_sample = step_index % resolved_config.sample_interval == 0 or step_index == real_config.train_steps

        if should_eval:
            latest_val_loss = _evaluate_loss(model, val_source, device=device, config=resolved_config)
            prior_best = best_val_loss
            best_val_loss = min(best_val_loss, latest_val_loss)
            history.append(
                {
                    "step": float(step_index),
                    "tokens_seen": float(tokens_seen),
                    "sequences_seen": float(sequences_seen),
                    "train_loss": float(loss.item()),
                    "val_loss": float(latest_val_loss),
                }
            )
            eval_payload = {
                "kind": "eval",
                "step": step_index,
                "tokens_seen": tokens_seen,
                "val_loss": latest_val_loss,
                "train_loss": float(loss.item()),
                "train_tok_per_sec": train_tok_per_sec,
                "pure_train_tok_per_sec": pure_train_tok_per_sec,
                "peak_vram_mb": _peak_vram_mb(real_config.device),
                "timestamp": time.time(),
            }
            _write_json(state_path, {**state_payload, "latest_val_loss": latest_val_loss, "best_val_loss": best_val_loss})
            _append_jsonl(metrics_path, eval_payload)
            if print_progress:
                print(
                    f"eval step={step_index:,} tokens={tokens_seen:,} val={latest_val_loss:.4f} "
                    f"train_tok/s={train_tok_per_sec:,.0f} pure_tok/s={pure_train_tok_per_sec:,.0f}",
                    flush=True,
                )
            if resolved_config.save_best_checkpoint and latest_val_loss <= prior_best:
                _save_checkpoint(
                    best_ckpt,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    config=resolved_config,
                    step=step_index,
                    tokens_seen=tokens_seen,
                    latest_val_loss=latest_val_loss,
                    best_val_loss=best_val_loss,
                    cache_path=resolved_cache_path,
                )

        if should_sample:
            samples = []
            for prompt, ids in zip(resolved_config.sample_prompts, prompt_ids, strict=True):
                generated_ids = _greedy_generate(
                    model,
                    ids,
                    sequence_length=resolved_config.sequence_length,
                    device=device,
                    max_new_tokens=resolved_config.sample_max_new_tokens,
                    config=resolved_config,
                )
                samples.append({"prompt": prompt, "generated": tokenizer.decode(generated_ids)})
            _append_jsonl(
                samples_path,
                {
                    "kind": "sample",
                    "step": step_index,
                    "tokens_seen": tokens_seen,
                    "samples": samples,
                    "timestamp": time.time(),
                },
            )
            if print_samples:
                print(f"[sample step={step_index:,} tokens={tokens_seen:,}]", flush=True)
                for sample in samples:
                    print(f"PROMPT: {sample['prompt']}", flush=True)
                    print(f"GENERATED: {sample['generated']}", flush=True)

        if should_ckpt and resolved_config.save_latest_checkpoint:
            _save_checkpoint(
                latest_ckpt,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=resolved_config,
                step=step_index,
                tokens_seen=tokens_seen,
                latest_val_loss=latest_val_loss,
                best_val_loss=best_val_loss,
                cache_path=resolved_cache_path,
            )

    total_time = time.perf_counter() - start
    pure_train_time = sum(step_times)
    report = {
        "parameter_count": count_parameters(model),
        "history": history,
        "initial_val_loss": initial_val_loss,
        "final_val_loss": latest_val_loss,
        "best_val_loss": best_val_loss,
        "train_tokens_seen": tokens_seen,
        "train_tok_per_sec": tokens_seen / max(total_time, 1e-9),
        "pure_train_tok_per_sec": tokens_seen / max(pure_train_time, 1e-9),
        "step_time_mean_ms": statistics.fmean(step_times) * 1000.0,
        "step_time_median_ms": statistics.median(step_times) * 1000.0,
        "pure_train_time_seconds": pure_train_time,
        "eval_overhead_seconds": max(total_time - pure_train_time, 0.0),
        "peak_vram_mb": _peak_vram_mb(real_config.device),
        "total_training_time_seconds": total_time,
    }
    if resolved_config.save_final_checkpoint:
        _save_checkpoint(
            final_ckpt,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=resolved_config,
            step=real_config.train_steps,
            tokens_seen=tokens_seen,
            latest_val_loss=latest_val_loss,
            best_val_loss=best_val_loss,
            cache_path=resolved_cache_path,
        )
    final_payload = {
        "benchmark": "language_partial_untied_cluster",
        "config": {
            **_config_payload(resolved_config, profile=profile),
            "resolved_train_steps": train_steps,
            "tokens_per_step": tokens_per_step,
            "actual_train_tokens": actual_train_tokens,
            "resolved_cache_path": str(resolved_cache_path),
        },
        "report": report,
    }
    _write_json(final_path, final_payload)
    _write_json(
        state_path,
        {
            "status": "completed",
            "step": train_steps,
            "train_steps": train_steps,
            "progress": 1.0,
            "tokens_seen": tokens_seen,
            "target_tokens": actual_train_tokens,
            "latest_val_loss": latest_val_loss,
            "best_val_loss": best_val_loss,
            "train_tok_per_sec": report["train_tok_per_sec"],
            "pure_train_tok_per_sec": report["pure_train_tok_per_sec"],
            "peak_vram_mb": report["peak_vram_mb"],
            "final_path": str(final_path),
            "parameter_count": report["parameter_count"],
            "hardware_profile": profile.name,
        },
    )
    if print_progress:
        print(
            f"completed cluster_partial_untied final_val={latest_val_loss:.4f} best_val={best_val_loss:.4f} "
            f"tokens={tokens_seen:,} train_tok/s={report['train_tok_per_sec']:,.0f} "
            f"pure_tok/s={report['pure_train_tok_per_sec']:,.0f}",
            flush=True,
        )
    return final_payload


def _parse_prompts(prompt_file: Path | None) -> tuple[str, ...]:
    if prompt_file is None:
        return _DEFAULT_PROMPTS
    prompts = [line.strip() for line in prompt_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"Prompt file {prompt_file} did not contain any non-empty prompts.")
    return tuple(prompts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run cluster-grade partial_untied training with automatic H100/A100 defaults, FineWeb-Edu streaming cache build, checkpoints, and fixed-prompt samples."
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-path", type=Path)
    parser.add_argument("--target-watch-dir", type=Path)
    parser.add_argument("--watch-refresh", type=float, default=5.0)
    parser.add_argument("--watch-once", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--total-tokens", type=int, default=100_000_000)
    parser.add_argument("--train-tokens", type=int, default=98_000_000)
    parser.add_argument("--val-tokens", type=int, default=2_000_000)
    parser.add_argument("--sequence-length", type=int, default=127)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=0)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--sample-interval", type=int, default=0)
    parser.add_argument("--embedding-dim", type=int, default=320)
    parser.add_argument("--hidden-dim", type=int, default=640)
    parser.add_argument("--memory-dim", type=int, default=320)
    parser.add_argument("--partial-token-count", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--amp-dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--tokenization-batch-size", type=int, default=0)
    parser.add_argument("--sample-max-new-tokens", type=int, default=64)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if args.target_watch_dir is not None:
        watch_cluster_run(args.target_watch_dir, refresh_seconds=args.watch_refresh, once=args.watch_once)
        return

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path("runs") / f"partial_untied_cluster_{timestamp}"

    config = PartialUntiedClusterConfig(
        output_dir=output_dir,
        cache_path=args.cache_path,
        seed=args.seed,
        device=args.device,
        total_tokens=args.total_tokens,
        train_tokens=args.train_tokens,
        val_tokens=args.val_tokens,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        sample_interval=args.sample_interval,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        memory_dim=args.memory_dim,
        partial_token_count=args.partial_token_count,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        amp_dtype=args.amp_dtype,
        tokenization_batch_size=args.tokenization_batch_size,
        sample_max_new_tokens=args.sample_max_new_tokens,
        sample_prompts=_parse_prompts(args.prompt_file),
        use_compile=args.compile,
        local_files_only=args.local_files_only,
    )
    resolved, profile = resolve_cluster_config(config)
    if args.print_config:
        print(json.dumps(_config_payload(resolved, profile=profile), indent=2))
        return
    payload = run_partial_untied_cluster(resolved, print_progress=True, print_samples=True)
    print(json.dumps({"final_val_loss": payload["report"]["final_val_loss"], "final_path": str(output_dir / "final.json")}, indent=2))


if __name__ == "__main__":
    main()
