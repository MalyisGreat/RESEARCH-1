from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from arc_tactic3.language_fastlearn_benchmark import count_parameters, set_global_seed
from arc_tactic3.language_nanochat_actual_compare import NanochatMiniLM, _peak_vram_mb
from arc_tactic3.language_partial_untied_document_compare import DocumentResetBatcher, DocumentStreamBatcher
from arc_tactic3.language_partial_untied_watch import _append_jsonl, _load_json_if_exists, _write_json
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
from arc_tactic3.language_recurrent_nano_tricks import (
    FactorizedUntiedHeadAssociativeLM,
    PartialUntiedAssociativeLM,
    UntiedHeadAssociativeLM,
    _top_token_ids,
)


@dataclass(frozen=True, slots=True)
class DocAwareHeadToHeadConfig:
    cache_path: Path
    output_dir: Path
    target_tokens: int = 100_000_000
    seed: int = 13
    sequence_length: int = 127
    score_target_loss: float = 5.303466700017452
    model_filter: str = "both"
    eval_interval: int = 512
    log_interval: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = torch.cuda.is_available()
    use_fused_adamw: bool = torch.cuda.is_available()
    dropout: float = 0.1
    tokenizer_name: str = "gpt2"
    sample_prompt: str | None = "i went to the mall because"
    sample_every_tokens: int = 10_000_000
    sample_generation_tokens: int = 40

    partial_batch_size: int = 16
    partial_eval_batch_size: int = 16
    partial_learning_rate: float = 2e-3
    partial_weight_decay: float = 1e-4
    partial_optimizer_recipe: str = "default"
    partial_warmup_steps: int = 0
    partial_lr_schedule: str = "none"
    partial_min_lr_scale: float = 1.0
    partial_embedding_dim: int = 320
    partial_hidden_dim: int = 640
    partial_memory_dim: int = 320
    partial_token_count: int = 1024

    untied_embedding_dim: int = 128
    untied_hidden_dim: int = 1408
    untied_memory_dim: int = 128

    factorized_embedding_dim: int = 128
    factorized_hidden_dim: int = 896
    factorized_memory_dim: int = 128
    factorized_untied_rank: int = 192

    nano_batch_size: int = 16
    nano_eval_batch_size: int = 16
    nano_learning_rate: float = 2e-3
    nano_weight_decay: float = 1e-4
    nano_optimizer_recipe: str = "default"
    nano_warmup_steps: int = 0
    nano_lr_schedule: str = "none"
    nano_min_lr_scale: float = 1.0
    nano_n_layer: int = 8
    nano_n_head: int = 4
    nano_n_kv_head: int = 4
    nano_n_embd: int = 64
    nano_window_pattern: str = "SSSL"
    nano_softcap: float = 15.0
    nano_use_value_embeddings: bool = True
    nano_use_smear: bool = True
    nano_use_backout: bool = True


def _bar(progress: float, width: int = 28) -> str:
    progress = min(max(progress, 0.0), 1.0)
    filled = int(progress * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _format_eta(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0.0:
        return "?"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{secs:02d}s"


def watch_docaware_run(output_dir: Path, *, refresh_seconds: float = 5.0, once: bool = False) -> None:
    state_path = output_dir / "state.json"
    final_path = output_dir / "summary.json"
    last_seen_step = -1
    while True:
        state = _load_json_if_exists(state_path)
        final_payload = _load_json_if_exists(final_path)
        if state is None:
            print(f"waiting for state file: {state_path}", flush=True)
            if once:
                return
            time.sleep(refresh_seconds)
            continue

        current_model = str(state.get("current_model", "?"))
        step = int(state.get("step", 0))
        train_steps = int(state.get("train_steps", 0))
        progress = float(state.get("progress", 0.0))
        tokens_seen = int(state.get("tokens_seen", 0))
        target_tokens = int(state.get("target_tokens", 0))
        train_loss = state.get("latest_train_loss")
        val_loss = state.get("latest_val_loss")
        train_tok_per_sec = float(state.get("train_tok_per_sec", float("nan")))
        pure_train_tok_per_sec = float(state.get("pure_train_tok_per_sec", float("nan")))
        peak_vram_mb = state.get("peak_vram_mb")
        status = str(state.get("status", "running"))
        eta_seconds = state.get("eta_seconds")
        if step != last_seen_step or once or status == "completed":
            train_loss_text = "----" if train_loss is None or not math.isfinite(float(train_loss)) else f"{float(train_loss):.4f}"
            val_loss_text = "----" if val_loss is None or not math.isfinite(float(val_loss)) else f"{float(val_loss):.4f}"
            vram_text = "----" if peak_vram_mb is None else f"{float(peak_vram_mb):.0f}"
            print(
                f"{current_model} {_bar(progress)} {progress * 100:5.1f}% "
                f"step={step:,}/{train_steps:,} tok={tokens_seen:,}/{target_tokens:,} "
                f"train={train_loss_text} val={val_loss_text} tok/s={train_tok_per_sec:,.0f} "
                f"pure_tok/s={pure_train_tok_per_sec:,.0f} eta={_format_eta(float(eta_seconds) if eta_seconds is not None else None)} "
                f"vram={vram_text}MB status={status}",
                flush=True,
            )
            last_seen_step = step
        if final_payload is not None and status == "completed":
            print(json.dumps(final_payload, indent=2), flush=True)
            return
        if once:
            return
        time.sleep(refresh_seconds)


def _load_doc_cache(cache_path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    required = {"train_tokens", "val_tokens", "train_doc_offsets", "val_doc_offsets", "vocab_size"}
    missing = sorted(required - set(payload.keys()))
    if missing:
        raise ValueError(f"Document cache is missing required keys: {missing}")
    return (
        payload["train_tokens"].long(),
        payload["train_doc_offsets"].long(),
        payload["val_tokens"].long(),
        payload["val_doc_offsets"].long(),
        int(payload["vocab_size"]),
    )


def _block_dataset_from_tokens(tokens: torch.Tensor, *, sequence_length: int) -> TokenBlockDataset:
    block_size = sequence_length + 1
    usable = (tokens.numel() // block_size) * block_size
    blocks = tokens[:usable].view(-1, block_size)
    return TokenBlockDataset(blocks[:, :-1].contiguous(), blocks[:, 1:].contiguous())


def _make_realtext_config(
    *,
    seed: int,
    sequence_length: int,
    train_steps: int,
    batch_size: int,
    eval_batch_size: int,
    eval_interval: int,
    learning_rate: float,
    weight_decay: float,
    optimizer_recipe: str,
    warmup_steps: int,
    lr_schedule: str,
    min_lr_scale: float,
    device: str,
    use_amp: bool,
    use_fused_adamw: bool,
    dropout: float,
) -> RealTextConfig:
    return RealTextConfig(
        seed=seed,
        sequence_length=sequence_length,
        train_steps=train_steps,
        eval_interval=eval_interval,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optimizer_recipe=optimizer_recipe,
        warmup_steps=warmup_steps,
        lr_schedule=lr_schedule,
        min_lr_scale=min_lr_scale,
        device=device,
        use_amp=use_amp,
        pin_memory=torch.cuda.is_available(),
        use_fused_adamw=use_fused_adamw,
        tensor_batching=True,
        cache_dataset_on_device=True,
        paired_train_batches=True,
        reseed_per_model=True,
        train_schedule_seed=seed,
        dropout=dropout,
        initial_eval=False,
    )


def _forward_logits(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    output = model(input_ids)
    if isinstance(output, tuple):
        return output[0]
    return output


def _greedy_generate_sample(
    model: torch.nn.Module,
    *,
    prompt_ids: list[int],
    sequence_length: int,
    device: torch.device,
    total_steps: int,
) -> list[int]:
    sequence = prompt_ids[:]
    model.eval()
    with torch.no_grad():
        for _ in range(total_steps):
            window = sequence[-sequence_length:]
            input_ids = torch.tensor(window, dtype=torch.long, device=device).unsqueeze(0)
            logits = _forward_logits(model, input_ids)
            sequence.append(int(logits[0, -1].argmax().item()))
    return sequence


def _evaluate_reset_loss(
    model: torch.nn.Module,
    *,
    tokens: torch.Tensor,
    doc_offsets: torch.Tensor,
    sequence_length: int,
    batch_size: int,
    steps: int,
    seed: int,
    device: torch.device,
    use_amp: bool,
) -> float:
    batcher = DocumentResetBatcher(
        tokens=tokens,
        doc_offsets=doc_offsets,
        sequence_length=sequence_length,
        batch_size=batch_size,
        steps=steps,
        seed=seed,
    )
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for _ in range(steps):
            batch = batcher.next_batch(device=device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = _forward_logits(model, batch["input_ids"])
                loss, token_count = _loss_and_tokens(logits, batch["targets"])
            total_loss += float(loss.item()) * token_count
            total_tokens += token_count
    return total_loss / max(total_tokens, 1)


def _build_partial_model(config: DocAwareHeadToHeadConfig, *, vocab_size: int, partial_token_ids: torch.Tensor) -> torch.nn.Module:
    return PartialUntiedAssociativeLM(
        vocab_size=vocab_size,
        embedding_dim=config.partial_embedding_dim,
        hidden_dim=config.partial_hidden_dim,
        memory_dim=config.partial_memory_dim,
        dropout=config.dropout,
        max_length=config.sequence_length,
        untied_token_ids=partial_token_ids,
    )


def _build_nanochat_model(config: DocAwareHeadToHeadConfig, *, vocab_size: int) -> torch.nn.Module:
    return NanochatMiniLM(
        vocab_size=vocab_size,
        sequence_length=config.sequence_length,
        n_layer=config.nano_n_layer,
        n_head=config.nano_n_head,
        n_kv_head=config.nano_n_kv_head,
        n_embd=config.nano_n_embd,
        window_pattern=config.nano_window_pattern,
        softcap=config.nano_softcap,
        use_value_embeddings=config.nano_use_value_embeddings,
        use_smear=config.nano_use_smear,
        use_backout=config.nano_use_backout,
    )


def _build_untied_model(config: DocAwareHeadToHeadConfig, *, vocab_size: int) -> torch.nn.Module:
    return UntiedHeadAssociativeLM(
        vocab_size=vocab_size,
        embedding_dim=config.untied_embedding_dim,
        hidden_dim=config.untied_hidden_dim,
        memory_dim=config.untied_memory_dim,
        dropout=config.dropout,
        max_length=config.sequence_length,
    )


def _build_factorized_model(config: DocAwareHeadToHeadConfig, *, vocab_size: int) -> torch.nn.Module:
    return FactorizedUntiedHeadAssociativeLM(
        vocab_size=vocab_size,
        embedding_dim=config.factorized_embedding_dim,
        hidden_dim=config.factorized_hidden_dim,
        memory_dim=config.factorized_memory_dim,
        dropout=config.dropout,
        max_length=config.sequence_length,
        untied_rank=config.factorized_untied_rank,
    )


def _train_model(
    *,
    model_name: str,
    config: DocAwareHeadToHeadConfig,
    train_tokens: torch.Tensor,
    train_doc_offsets: torch.Tensor,
    val_tokens: torch.Tensor,
    val_doc_offsets: torch.Tensor,
    vocab_size: int,
    partial_token_ids: torch.Tensor,
    state_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    device = torch.device(config.device)
    if model_name == "partial_untied_20m":
        model = _build_partial_model(config, vocab_size=vocab_size, partial_token_ids=partial_token_ids)
        batch_size = config.partial_batch_size
        eval_batch_size = config.partial_eval_batch_size
        learning_rate = config.partial_learning_rate
        weight_decay = config.partial_weight_decay
        optimizer_recipe = config.partial_optimizer_recipe
        warmup_steps = config.partial_warmup_steps
        lr_schedule = config.partial_lr_schedule
        min_lr_scale = config.partial_min_lr_scale
        optimizer_model_name = "partial_untied"
        train_seed_offset = 17
        eval_seed_offset = 18
        train_mode = "doc_reset"
    elif model_name == "untied_20m":
        model = _build_untied_model(config, vocab_size=vocab_size)
        batch_size = config.partial_batch_size
        eval_batch_size = config.partial_eval_batch_size
        learning_rate = config.partial_learning_rate
        weight_decay = config.partial_weight_decay
        optimizer_recipe = config.partial_optimizer_recipe
        warmup_steps = config.partial_warmup_steps
        lr_schedule = config.partial_lr_schedule
        min_lr_scale = config.partial_min_lr_scale
        optimizer_model_name = "partial_untied"
        train_seed_offset = 19
        eval_seed_offset = 20
        train_mode = "doc_reset"
    elif model_name == "factorized_20m":
        model = _build_factorized_model(config, vocab_size=vocab_size)
        batch_size = config.partial_batch_size
        eval_batch_size = config.partial_eval_batch_size
        learning_rate = config.partial_learning_rate
        weight_decay = config.partial_weight_decay
        optimizer_recipe = config.partial_optimizer_recipe
        warmup_steps = config.partial_warmup_steps
        lr_schedule = config.partial_lr_schedule
        min_lr_scale = config.partial_min_lr_scale
        optimizer_model_name = "partial_untied"
        train_seed_offset = 21
        eval_seed_offset = 22
        train_mode = "doc_reset"
    elif model_name == "nanochat_20m":
        model = _build_nanochat_model(config, vocab_size=vocab_size)
        batch_size = config.nano_batch_size
        eval_batch_size = config.nano_eval_batch_size
        learning_rate = config.nano_learning_rate
        weight_decay = config.nano_weight_decay
        optimizer_recipe = config.nano_optimizer_recipe
        warmup_steps = config.nano_warmup_steps
        lr_schedule = config.nano_lr_schedule
        min_lr_scale = config.nano_min_lr_scale
        optimizer_model_name = "nanochat_small"
        train_seed_offset = 29
        eval_seed_offset = 30
        train_mode = "flat_reset"
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    model.to(device)
    parameter_count = count_parameters(model)
    train_steps = math.ceil(config.target_tokens / (config.sequence_length * batch_size))
    actual_target_tokens = train_steps * config.sequence_length * batch_size
    real_config = _make_realtext_config(
        seed=config.seed,
        sequence_length=config.sequence_length,
        train_steps=train_steps,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        eval_interval=config.eval_interval,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optimizer_recipe=optimizer_recipe,
        warmup_steps=warmup_steps,
        lr_schedule=lr_schedule,
        min_lr_scale=min_lr_scale,
        device=config.device,
        use_amp=config.use_amp,
        use_fused_adamw=config.use_fused_adamw,
        dropout=config.dropout,
    )
    optimizer = _build_optimizer(model, real_config, model_name=optimizer_model_name)
    scheduler = _build_scheduler(optimizer, real_config)
    scaler = torch.amp.GradScaler(device="cuda", enabled=real_config.use_amp and device.type == "cuda")
    use_amp = real_config.use_amp and device.type == "cuda"
    parameter_list = [parameter for parameter in model.parameters() if parameter.requires_grad]

    train_batcher = None
    train_source = None
    batch_schedule = None
    if train_mode == "doc_reset":
        train_batcher = DocumentResetBatcher(
            tokens=train_tokens,
            doc_offsets=train_doc_offsets,
            sequence_length=config.sequence_length,
            batch_size=batch_size,
            steps=train_steps,
            seed=config.seed + train_seed_offset,
        )
    else:
        train_dataset = _block_dataset_from_tokens(train_tokens, sequence_length=config.sequence_length)
        train_source = _dataset_tensors(
            train_dataset,
            device=device,
            cache_on_device=True,
            pin_memory=torch.cuda.is_available(),
        )
        batch_schedule = _build_train_batch_schedule(
            len(train_dataset),
            batch_size=batch_size,
            steps=train_steps,
            seed=config.seed + train_seed_offset,
            drop_last=True,
        )
    eval_steps = 64
    history: list[dict[str, float]] = []
    step_times: list[float] = []
    tokens_seen = 0
    best_val_loss = float("inf")
    tokens_to_target_loss = None
    seconds_to_target_loss = None
    samples: list[dict[str, Any]] = []
    tokenizer = None
    prompt_ids: list[int] | None = None
    next_sample_tokens: int | None = None
    if config.sample_prompt:
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, use_fast=True)
        tokenizer.model_max_length = int(1e9)
        prompt_ids = tokenizer(config.sample_prompt, add_special_tokens=False)["input_ids"]
        if not prompt_ids:
            raise ValueError("sample_prompt tokenized to an empty prompt.")
        next_sample_tokens = config.sample_every_tokens
    initial_val_loss = _evaluate_reset_loss(
        model,
        tokens=val_tokens,
        doc_offsets=val_doc_offsets,
        sequence_length=config.sequence_length,
        batch_size=eval_batch_size,
        steps=eval_steps,
        seed=config.seed + eval_seed_offset,
        device=device,
        use_amp=use_amp,
    )
    latest_val_loss = initial_val_loss
    history.append({"step": 0.0, "tokens_seen": 0.0, "train_loss": float("nan"), "val_loss": float(initial_val_loss)})
    if initial_val_loss <= config.score_target_loss:
        tokens_to_target_loss = 0
        seconds_to_target_loss = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    run_dir = config.output_dir / model_name
    metrics_path = run_dir / "metrics.jsonl"
    final_path = run_dir / "final.json"
    start = time.perf_counter()

    for step in range(1, train_steps + 1):
        if train_mode == "doc_reset":
            assert train_batcher is not None
            batch = train_batcher.next_batch(device=device)
        else:
            assert train_source is not None and batch_schedule is not None
            batch = _scheduled_batch_from_tensors(
                train_source[0],
                train_source[1],
                batch_schedule[step - 1],
                device=device,
                non_blocking=torch.cuda.is_available() and device.type == "cuda",
            )
        step_start = time.perf_counter()
        model.train()
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = _forward_logits(model, batch["input_ids"])
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
        step_times.append(time.perf_counter() - step_start)
        tokens_seen += token_count

        pure_train_time = sum(step_times)
        elapsed = time.perf_counter() - start
        train_tok_per_sec = tokens_seen / max(elapsed, 1e-9)
        pure_train_tok_per_sec = tokens_seen / max(pure_train_time, 1e-9)
        progress = step / train_steps
        eta_seconds = (train_steps - step) * (elapsed / max(step, 1))
        state_payload = {
            "status": "running",
            "current_model": model_name,
            "step": step,
            "train_steps": train_steps,
            "progress": progress,
            "tokens_seen": tokens_seen,
            "target_tokens": actual_target_tokens,
            "latest_train_loss": float(loss.item()),
            "latest_val_loss": latest_val_loss,
            "train_tok_per_sec": train_tok_per_sec,
            "pure_train_tok_per_sec": pure_train_tok_per_sec,
            "eta_seconds": eta_seconds,
            "peak_vram_mb": _peak_vram_mb(config.device),
        }
        _write_json(state_path, state_payload)

        if step % config.log_interval == 0 or step == train_steps:
            _append_jsonl(metrics_path, {**state_payload, "kind": "train", "timestamp": time.time()})

        if step % config.eval_interval == 0 or step == train_steps:
            latest_val_loss = _evaluate_reset_loss(
                model,
                tokens=val_tokens,
                doc_offsets=val_doc_offsets,
                sequence_length=config.sequence_length,
                batch_size=eval_batch_size,
                steps=eval_steps,
                seed=config.seed + eval_seed_offset,
                device=device,
                use_amp=use_amp,
            )
            best_val_loss = min(best_val_loss, latest_val_loss)
            if latest_val_loss <= config.score_target_loss and tokens_to_target_loss is None:
                tokens_to_target_loss = tokens_seen
                seconds_to_target_loss = elapsed
            history.append(
                {
                    "step": float(step),
                    "tokens_seen": float(tokens_seen),
                    "train_loss": float(loss.item()),
                    "val_loss": float(latest_val_loss),
                }
            )
            _append_jsonl(
                metrics_path,
                {
                    "kind": "eval",
                    "timestamp": time.time(),
                    "current_model": model_name,
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "val_loss": latest_val_loss,
                    "train_loss": float(loss.item()),
                    "train_tok_per_sec": train_tok_per_sec,
                    "pure_train_tok_per_sec": pure_train_tok_per_sec,
                    "peak_vram_mb": _peak_vram_mb(config.device),
                },
            )
            print(
                f"{model_name} {_bar(progress)} {step:,}/{train_steps:,} "
                f"train={loss.item():.4f} val={latest_val_loss:.4f} tok={tokens_seen:,}/{actual_target_tokens:,} "
                f"tok/s={train_tok_per_sec:,.0f} eta={_format_eta(eta_seconds)}",
                flush=True,
            )
            if prompt_ids is not None and tokenizer is not None and next_sample_tokens is not None and (
                tokens_seen >= next_sample_tokens or step == train_steps
            ):
                generated_ids = _greedy_generate_sample(
                    model,
                    prompt_ids=prompt_ids,
                    sequence_length=config.sequence_length,
                    device=device,
                    total_steps=config.sample_generation_tokens,
                )
                sample_text = tokenizer.decode(generated_ids)
                sample_entry = {
                    "step": float(step),
                    "tokens_seen": float(tokens_seen),
                    "target_sample_tokens": float(next_sample_tokens),
                    "prompt": config.sample_prompt,
                    "generated": sample_text,
                }
                samples.append(sample_entry)
                print(
                    f"[{model_name}] sample tok={tokens_seen:,}\n"
                    f"PROMPT: {config.sample_prompt}\n"
                    f"GENERATED: {sample_text}",
                    flush=True,
                )
                next_sample_tokens += config.sample_every_tokens

    total_time = time.perf_counter() - start
    pure_train_time = sum(step_times)
    report = {
        "parameter_count": parameter_count,
        "initial_val_loss": initial_val_loss,
        "final_val_loss": latest_val_loss,
        "best_val_loss": best_val_loss,
        "train_tokens_seen": tokens_seen,
        "train_steps": train_steps,
        "train_tok_per_sec": tokens_seen / max(total_time, 1e-9),
        "pure_train_tok_per_sec": tokens_seen / max(pure_train_time, 1e-9),
        "step_time_mean_ms": statistics.fmean(step_times) * 1000.0,
        "step_time_median_ms": statistics.median(step_times) * 1000.0,
        "peak_vram_mb": _peak_vram_mb(config.device),
        "total_training_time_seconds": total_time,
        "tokens_to_target_loss": tokens_to_target_loss,
        "seconds_to_target_loss": seconds_to_target_loss,
        "train_mode": train_mode,
        "samples": samples,
        "history": history,
    }
    _write_json(
        final_path,
        {
            "benchmark": "language_docaware_head_to_head",
            "model_name": model_name,
            "config": {
                **asdict(config),
                "cache_path": str(config.cache_path),
                "output_dir": str(config.output_dir),
            },
            "report": report,
        },
    )
    return report


def run_docaware_head_to_head(config: DocAwareHeadToHeadConfig) -> dict[str, Any]:
    set_global_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    train_tokens, train_doc_offsets, val_tokens, val_doc_offsets, vocab_size = _load_doc_cache(config.cache_path)
    block_size = config.sequence_length + 1
    train_blocks = train_tokens[: (train_tokens.numel() // block_size) * block_size].view(-1, block_size)
    partial_targets = train_blocks[:, 1:].contiguous()
    partial_dataset = type("PartialDataset", (), {"targets": partial_targets})()
    partial_token_ids = _top_token_ids(partial_dataset, count=config.partial_token_count, vocab_size=vocab_size)

    state_path = config.output_dir / "state.json"
    summary_path = config.output_dir / "summary.json"
    summary: dict[str, Any] = {
        "benchmark": "language_docaware_head_to_head",
        "config": {
            **asdict(config),
            "cache_path": str(config.cache_path),
            "output_dir": str(config.output_dir),
        },
        "models": {},
        "status": "running",
    }
    _write_json(summary_path, summary)

    models_to_run = []
    if config.model_filter in {"both", "partial_untied_20m"}:
        models_to_run.append("partial_untied_20m")
    if config.model_filter in {"both", "untied_20m", "promoted_recurrent_20m"}:
        models_to_run.append("untied_20m")
    if config.model_filter in {"both", "factorized_20m", "promoted_recurrent_20m", "partial_factorized_20m"}:
        if config.model_filter == "partial_factorized_20m" and "partial_untied_20m" not in models_to_run:
            models_to_run.append("partial_untied_20m")
        models_to_run.append("factorized_20m")
    if config.model_filter in {"both", "nanochat_20m"}:
        models_to_run.append("nanochat_20m")

    for model_name in models_to_run:
        report = _train_model(
            model_name=model_name,
            config=config,
            train_tokens=train_tokens,
            train_doc_offsets=train_doc_offsets,
            val_tokens=val_tokens,
            val_doc_offsets=val_doc_offsets,
            vocab_size=vocab_size,
            partial_token_ids=partial_token_ids,
            state_path=state_path,
            summary_path=summary_path,
        )
        summary["models"][model_name] = report
        _write_json(summary_path, summary)

    target = config.score_target_loss
    scoreboard: dict[str, Any] = {}
    for model_name, report in summary["models"].items():
        scoreboard[model_name] = {
            "final_val_loss": report["final_val_loss"],
            "best_val_loss": report["best_val_loss"],
            "tokens_to_target_loss": report["tokens_to_target_loss"],
            "seconds_to_target_loss": report["seconds_to_target_loss"],
            "train_tok_per_sec": report["train_tok_per_sec"],
            "parameter_count": report["parameter_count"],
            "hit_target_loss": report["tokens_to_target_loss"] is not None,
            "target_loss": target,
        }
    summary["scoreboard"] = scoreboard
    summary["status"] = "completed"
    _write_json(summary_path, summary)
    _write_json(
        state_path,
        {
            "status": "completed",
            "current_model": models_to_run[-1] if models_to_run else None,
            "step": 0,
            "train_steps": 0,
            "progress": 1.0,
            "tokens_seen": 0,
            "target_tokens": config.target_tokens,
            "latest_train_loss": None,
            "latest_val_loss": None,
            "train_tok_per_sec": None,
            "pure_train_tok_per_sec": None,
            "eta_seconds": 0.0,
            "peak_vram_mb": None,
            "final_path": str(summary_path),
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a doc-aware 20M-parameter benchmark between recurrent candidates and Nanochat.")
    parser.add_argument("--cache-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--target-tokens", type=int, default=100_000_000)
    parser.add_argument("--score-target-loss", type=float, default=5.303466700017452)
    parser.add_argument("--sequence-length", type=int, default=127)
    parser.add_argument(
        "--model",
        choices=(
            "both",
            "partial_untied_20m",
            "untied_20m",
            "factorized_20m",
            "partial_factorized_20m",
            "promoted_recurrent_20m",
            "nanochat_20m",
        ),
        default="both",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-interval", type=int, default=512)
    parser.add_argument("--log-interval", type=int, default=256)
    parser.add_argument("--sample-prompt", type=str, default="i went to the mall because")
    parser.add_argument("--sample-every-tokens", type=int, default=10_000_000)
    parser.add_argument("--sample-generation-tokens", type=int, default=40)
    parser.add_argument("--partial-learning-rate", type=float, default=2e-3)
    parser.add_argument("--partial-weight-decay", type=float, default=1e-4)
    parser.add_argument("--partial-batch-size", type=int, default=16)
    parser.add_argument("--partial-eval-batch-size", type=int, default=16)
    parser.add_argument("--partial-embedding-dim", type=int, default=320)
    parser.add_argument("--partial-hidden-dim", type=int, default=640)
    parser.add_argument("--partial-memory-dim", type=int, default=320)
    parser.add_argument("--partial-token-count", type=int, default=1024)
    parser.add_argument("--partial-optimizer-recipe", type=str, default="default")
    parser.add_argument("--partial-warmup-steps", type=int, default=0)
    parser.add_argument("--partial-lr-schedule", choices=("none", "cosine", "linear"), default="none")
    parser.add_argument("--partial-min-lr-scale", type=float, default=1.0)
    parser.add_argument("--untied-embedding-dim", type=int, default=128)
    parser.add_argument("--untied-hidden-dim", type=int, default=1408)
    parser.add_argument("--untied-memory-dim", type=int, default=128)
    parser.add_argument("--factorized-embedding-dim", type=int, default=128)
    parser.add_argument("--factorized-hidden-dim", type=int, default=896)
    parser.add_argument("--factorized-memory-dim", type=int, default=128)
    parser.add_argument("--factorized-untied-rank", type=int, default=192)
    parser.add_argument("--nano-optimizer-recipe", type=str, default="default")
    parser.add_argument("--nano-warmup-steps", type=int, default=0)
    parser.add_argument("--nano-lr-schedule", choices=("none", "cosine", "linear"), default="none")
    parser.add_argument("--nano-min-lr-scale", type=float, default=1.0)
    parser.add_argument("--nano-batch-size", type=int, default=16)
    parser.add_argument("--nano-eval-batch-size", type=int, default=16)
    parser.add_argument("--watch-dir", type=Path)
    parser.add_argument("--watch-refresh", type=float, default=5.0)
    parser.add_argument("--watch-once", action="store_true")
    args = parser.parse_args()

    if args.watch_dir is not None:
        watch_docaware_run(args.watch_dir, refresh_seconds=args.watch_refresh, once=args.watch_once)
        return

    if args.cache_path is None or args.output_dir is None:
        parser.error("--cache-path and --output-dir are required unless --watch-dir is used")

    config = DocAwareHeadToHeadConfig(
        cache_path=args.cache_path,
        output_dir=args.output_dir,
        target_tokens=args.target_tokens,
        sequence_length=args.sequence_length,
        score_target_loss=args.score_target_loss,
        model_filter=args.model,
        seed=args.seed,
        device=args.device,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        sample_prompt=args.sample_prompt,
        sample_every_tokens=args.sample_every_tokens,
        sample_generation_tokens=args.sample_generation_tokens,
        partial_learning_rate=args.partial_learning_rate,
        partial_weight_decay=args.partial_weight_decay,
        partial_batch_size=args.partial_batch_size,
        partial_eval_batch_size=args.partial_eval_batch_size,
        partial_optimizer_recipe=args.partial_optimizer_recipe,
        partial_warmup_steps=args.partial_warmup_steps,
        partial_lr_schedule=args.partial_lr_schedule,
        partial_min_lr_scale=args.partial_min_lr_scale,
        partial_embedding_dim=args.partial_embedding_dim,
        partial_hidden_dim=args.partial_hidden_dim,
        partial_memory_dim=args.partial_memory_dim,
        partial_token_count=args.partial_token_count,
        untied_embedding_dim=args.untied_embedding_dim,
        untied_hidden_dim=args.untied_hidden_dim,
        untied_memory_dim=args.untied_memory_dim,
        factorized_embedding_dim=args.factorized_embedding_dim,
        factorized_hidden_dim=args.factorized_hidden_dim,
        factorized_memory_dim=args.factorized_memory_dim,
        factorized_untied_rank=args.factorized_untied_rank,
        nano_batch_size=args.nano_batch_size,
        nano_eval_batch_size=args.nano_eval_batch_size,
        nano_optimizer_recipe=args.nano_optimizer_recipe,
        nano_warmup_steps=args.nano_warmup_steps,
        nano_lr_schedule=args.nano_lr_schedule,
        nano_min_lr_scale=args.nano_min_lr_scale,
    )
    payload = run_docaware_head_to_head(config)
    print(json.dumps(payload["scoreboard"], indent=2))


if __name__ == "__main__":
    main()
