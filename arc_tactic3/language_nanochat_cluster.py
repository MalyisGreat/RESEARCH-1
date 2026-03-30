from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from arc_tactic3.language_fastlearn_benchmark import count_parameters, set_global_seed
from arc_tactic3.language_nanochat_actual_compare import NanochatMiniLM, _peak_vram_mb
from arc_tactic3.language_partial_untied_cluster import (
    ClusterHardwareProfile,
    _DEFAULT_PROMPTS,
    _MAX_CACHE_ON_DEVICE_BYTES,
    _autocast_context,
    _batch_from_flat_tokens,
    _compute_resume_aware_rates,
    _evaluate_loss,
    _hardware_profile_from_override,
    _iter_train_batch_indices,
    _prepare_token_buffer,
    _sample_generate,
    _save_checkpoint,
    _token_block_count,
    detect_cluster_hardware_profile,
    ensure_fineweb_cache,
)
from arc_tactic3.language_partial_untied_watch import (
    _append_jsonl,
    _bar,
    _format_eta,
    _load_json_if_exists,
    _write_json,
)
from arc_tactic3.language_realtext_microbench import (
    RealTextConfig,
    _build_optimizer,
    _build_scheduler,
    _loss_and_tokens,
)

_GPT2_VOCAB_SIZE = 50_257
_NANOCHAT_PAD_VOCAB_SIZE_TO = 64


@dataclass(frozen=True, slots=True)
class NanochatPreset:
    train_tokens: int
    val_tokens: int
    batch_size: int
    eval_batch_size: int
    eval_interval: int
    log_interval: int
    checkpoint_interval: int
    sample_interval: int
    n_layer: int
    n_head: int
    n_kv_head: int
    n_embd: int
    learning_rate: float
    warmup_steps: int
    lr_schedule: str
    min_lr_scale: float
    cache_dataset_on_device: bool
    tokenization_batch_size: int
    sample_temperature: float
    sample_top_p: float
    sample_top_k: int
    sample_repetition_penalty: float


_NAMED_PRESETS: dict[str, NanochatPreset] = {
    "h100_100m_1b": NanochatPreset(
        train_tokens=980_000_000,
        val_tokens=20_000_000,
        batch_size=192,
        eval_batch_size=256,
        eval_interval=2048,
        log_interval=64,
        checkpoint_interval=4096,
        sample_interval=4096,
        n_layer=8,
        n_head=8,
        n_kv_head=2,
        n_embd=512,
        learning_rate=6e-4,
        warmup_steps=1024,
        lr_schedule="cosine",
        min_lr_scale=0.1,
        cache_dataset_on_device=True,
        tokenization_batch_size=8192,
        sample_temperature=0.9,
        sample_top_p=0.95,
        sample_top_k=50,
        sample_repetition_penalty=1.05,
    ),
}


@dataclass(frozen=True, slots=True)
class NanochatClusterConfig:
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
    warmup_steps: int = 0
    lr_schedule: str = "none"
    min_lr_scale: float = 1.0
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
    initial_eval: bool = True
    train_schedule_seed: int | None = None
    sample_prompts: tuple[str, ...] = _DEFAULT_PROMPTS
    sample_max_new_tokens: int = 64
    sample_temperature: float = 0.9
    sample_top_p: float = 0.95
    sample_top_k: int = 50
    sample_repetition_penalty: float = 1.05
    save_best_checkpoint: bool = True
    save_latest_checkpoint: bool = True
    save_final_checkpoint: bool = True
    hf_xet_high_performance: bool = True
    tokenizer_parallelism: bool = True
    hardware_profile_override: str | None = None
    nano_n_layer: int = 4
    nano_n_head: int = 4
    nano_n_kv_head: int = 4
    nano_n_embd: int = 40
    nano_window_pattern: str = "SSSL"
    nano_softcap: float = 15.0
    nano_use_value_embeddings: bool = True
    nano_use_smear: bool = True
    nano_use_backout: bool = True


def _nano_padded_vocab_size(vocab_size: int) -> int:
    return ((vocab_size + _NANOCHAT_PAD_VOCAB_SIZE_TO - 1) // _NANOCHAT_PAD_VOCAB_SIZE_TO) * _NANOCHAT_PAD_VOCAB_SIZE_TO


def _nano_value_embedding_layers(layer_count: int) -> int:
    return (layer_count + 1) // 2


def _estimate_nanochat_parameter_count(config: NanochatClusterConfig, *, vocab_size: int = _GPT2_VOCAB_SIZE) -> int:
    padded_vocab_size = _nano_padded_vocab_size(vocab_size)
    n_layer = config.nano_n_layer
    n_head = config.nano_n_head
    n_kv_head = config.nano_n_kv_head
    n_embd = config.nano_n_embd
    kv_dim = (n_embd * n_kv_head) // n_head
    ve_layers = _nano_value_embedding_layers(n_layer) if config.nano_use_value_embeddings else 0
    ve_gate_channels = min(12, n_embd)
    smear_gate_channels = min(24, n_embd)
    embedding_params = padded_vocab_size * n_embd
    block_params = n_layer * (10 * n_embd * n_embd + 2 * n_embd * kv_dim)
    value_embed_params = ve_layers * padded_vocab_size * kv_dim
    ve_gate_params = ve_layers * ve_gate_channels * n_kv_head
    misc_params = (2 * n_layer) + smear_gate_channels + 1
    return embedding_params + block_params + embedding_params + value_embed_params + ve_gate_params + misc_params


def _token_cache_bytes(config: NanochatClusterConfig) -> int:
    return config.total_tokens * torch.tensor([], dtype=torch.int32).element_size()


def _apply_named_preset(config: NanochatClusterConfig, preset_name: str | None) -> NanochatClusterConfig:
    if preset_name is None:
        return config
    preset = _NAMED_PRESETS[preset_name]
    return replace(
        config,
        total_tokens=preset.train_tokens + preset.val_tokens,
        train_tokens=preset.train_tokens,
        val_tokens=preset.val_tokens,
        batch_size=preset.batch_size,
        eval_batch_size=preset.eval_batch_size,
        eval_interval=preset.eval_interval,
        log_interval=preset.log_interval,
        checkpoint_interval=preset.checkpoint_interval,
        sample_interval=preset.sample_interval,
        learning_rate=preset.learning_rate,
        warmup_steps=preset.warmup_steps,
        lr_schedule=preset.lr_schedule,
        min_lr_scale=preset.min_lr_scale,
        cache_dataset_on_device=preset.cache_dataset_on_device,
        tokenization_batch_size=preset.tokenization_batch_size,
        sample_temperature=preset.sample_temperature,
        sample_top_p=preset.sample_top_p,
        sample_top_k=preset.sample_top_k,
        sample_repetition_penalty=preset.sample_repetition_penalty,
        nano_n_layer=preset.n_layer,
        nano_n_head=preset.n_head,
        nano_n_kv_head=preset.n_kv_head,
        nano_n_embd=preset.n_embd,
    )


def resolve_cluster_config(config: NanochatClusterConfig) -> tuple[NanochatClusterConfig, ClusterHardwareProfile]:
    profile = (
        _hardware_profile_from_override(config.hardware_profile_override)
        if config.hardware_profile_override is not None
        else detect_cluster_hardware_profile(config.device)
    )
    simulated_cuda = (
        config.hardware_profile_override is not None
        and config.hardware_profile_override != "cpu"
        and config.device.startswith("cuda")
        and not torch.cuda.is_available()
    )
    cuda_like = config.device.startswith("cuda") and (
        torch.cuda.is_available() or (config.hardware_profile_override is not None and config.hardware_profile_override != "cpu")
    )
    amp_requested = config.use_amp or simulated_cuda
    pin_memory_requested = config.pin_memory or simulated_cuda
    fused_adamw_requested = config.use_fused_adamw or simulated_cuda
    amp_dtype = config.amp_dtype
    if amp_dtype == "auto":
        amp_dtype = profile.amp_dtype
    use_amp = amp_requested and cuda_like and amp_dtype != "fp32"
    cache_on_device = config.cache_dataset_on_device and cuda_like
    if cache_on_device and _token_cache_bytes(config) > _MAX_CACHE_ON_DEVICE_BYTES:
        cache_on_device = False
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
        pin_memory=pin_memory_requested and cuda_like,
        use_fused_adamw=fused_adamw_requested and cuda_like,
        cache_dataset_on_device=cache_on_device,
        use_compile=config.use_compile or profile.use_compile,
        amp_dtype=amp_dtype,
    )
    return resolved, profile


def _configure_runtime_env(config: NanochatClusterConfig) -> None:
    import os

    if config.hf_xet_high_performance:
        os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    if config.tokenizer_parallelism:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    if torch.cuda.is_available() and config.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def _config_payload(config: NanochatClusterConfig, *, profile: ClusterHardwareProfile) -> dict[str, Any]:
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir)
    payload["cache_path"] = str(config.cache_path) if config.cache_path is not None else None
    payload["hardware_profile"] = asdict(profile)
    tokens_per_step = config.sequence_length * config.batch_size
    resolved_train_steps = math.ceil(config.train_tokens / max(tokens_per_step, 1)) if tokens_per_step > 0 else 0
    payload["estimated_parameter_count"] = _estimate_nanochat_parameter_count(config)
    payload["token_cache_bytes"] = _token_cache_bytes(config)
    payload["token_cache_gib"] = round(_token_cache_bytes(config) / (1024**3), 3)
    payload["tokens_per_step"] = tokens_per_step
    payload["resolved_train_steps"] = resolved_train_steps
    payload["actual_train_tokens"] = resolved_train_steps * tokens_per_step
    return payload


def _make_realtext_config(config: NanochatClusterConfig, *, train_steps: int) -> RealTextConfig:
    return RealTextConfig(
        seed=config.seed,
        sequence_length=config.sequence_length,
        train_steps=train_steps,
        eval_interval=config.eval_interval,
        batch_size=config.batch_size,
        eval_batch_size=config.eval_batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        lr_schedule=config.lr_schedule,
        min_lr_scale=config.min_lr_scale,
        device=config.device,
        use_amp=config.use_amp and config.device.startswith("cuda"),
        pin_memory=config.pin_memory,
        use_fused_adamw=config.use_fused_adamw,
        tensor_batching=True,
        cache_dataset_on_device=config.cache_dataset_on_device,
        paired_train_batches=True,
        reseed_per_model=True,
        train_schedule_seed=config.train_schedule_seed,
        initial_eval=False,
    )


def watch_nanochat_cluster_run(output_dir: Path, *, refresh_seconds: float = 5.0, once: bool = False) -> None:
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
                f"nanochat_cluster {_bar(progress)} {progress * 100:5.1f}% "
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


def run_nanochat_cluster(
    config: NanochatClusterConfig,
    *,
    print_progress: bool = True,
    print_samples: bool = True,
) -> dict[str, Any]:
    _configure_runtime_env(config)
    resolved_config, profile = resolve_cluster_config(config)
    train_tokens, val_tokens, vocab_size, resolved_cache_path = ensure_fineweb_cache(
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
    nonfinite_ckpt = resolved_config.output_dir / "checkpoints" / "latest_nonfinite.pt"
    final_path = resolved_config.output_dir / "final.json"
    resolved_config.output_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(resolved_config.seed)
    model = NanochatMiniLM(
        vocab_size=vocab_size,
        sequence_length=resolved_config.sequence_length,
        n_layer=resolved_config.nano_n_layer,
        n_head=resolved_config.nano_n_head,
        n_kv_head=resolved_config.nano_n_kv_head,
        n_embd=resolved_config.nano_n_embd,
        window_pattern=resolved_config.nano_window_pattern,
        softcap=resolved_config.nano_softcap,
        use_value_embeddings=resolved_config.nano_use_value_embeddings,
        use_smear=resolved_config.nano_use_smear,
        use_backout=resolved_config.nano_use_backout,
    )
    device = torch.device(real_config.device)
    model.to(device)
    if resolved_config.use_compile and hasattr(torch, "compile"):
        model = torch.compile(model, mode="max-autotune")
    optimizer = _build_optimizer(model, real_config, model_name="nanochat_cluster")
    scheduler = _build_scheduler(optimizer, real_config)
    use_scaler = resolved_config.use_amp and resolved_config.amp_dtype == "fp16" and device.type == "cuda"
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_scaler) if device.type == "cuda" else None
    parameter_list = [parameter for parameter in model.parameters() if parameter.requires_grad]

    train_source = _prepare_token_buffer(
        train_tokens,
        device=device,
        cache_on_device=real_config.cache_dataset_on_device,
        pin_memory=real_config.pin_memory,
    )
    val_source = _prepare_token_buffer(
        val_tokens,
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
    train_examples = _token_block_count(train_source, sequence_length=resolved_config.sequence_length)

    history: list[dict[str, float]] = []
    step_times: list[float] = []
    tokens_seen = 0
    sequences_seen = 0
    start_step = 1
    best_val_loss = float("inf")
    latest_val_loss = float("nan")
    initial_val_loss = float("nan")
    resume_tokens_seen = 0
    resume_wall_time_seconds = 0.0
    resume_pure_train_time_seconds = 0.0

    if latest_ckpt.exists():
        checkpoint_payload = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint_payload["model_state"])
        optimizer.load_state_dict(checkpoint_payload["optimizer_state"])
        if scheduler is not None and checkpoint_payload.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint_payload["scheduler_state"])
        if scaler is not None and checkpoint_payload.get("scaler_state") is not None:
            scaler.load_state_dict(checkpoint_payload["scaler_state"])
        if scheduler is not None and checkpoint_payload.get("scheduler_state") is None:
            completed_step = int(checkpoint_payload.get("step", 0))
            if completed_step > 0:
                scheduler.step(completed_step - 1)
        start_step = int(checkpoint_payload["step"]) + 1
        tokens_seen = int(checkpoint_payload["tokens_seen"])
        resume_tokens_seen = tokens_seen
        latest_val_loss = float(checkpoint_payload.get("latest_val_loss", float("nan")))
        best_val_loss = float(checkpoint_payload.get("best_val_loss", float("inf")))
        resume_wall_time_seconds = float(checkpoint_payload.get("wall_time_seconds", 0.0) or 0.0)
        resume_pure_train_time_seconds = float(checkpoint_payload.get("pure_train_time_seconds", 0.0) or 0.0)
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
    for step_index, batch_indices in _iter_train_batch_indices(
        train_examples,
        batch_size=real_config.batch_size,
        steps=real_config.train_steps,
        seed=schedule_seed,
        start_step=start_step,
        drop_last=True,
    ):
        batch = _batch_from_flat_tokens(
            train_source,
            sequence_length=resolved_config.sequence_length,
            batch_indices=batch_indices,
            device=device,
            non_blocking=real_config.pin_memory and device.type == "cuda",
        )
        step_start = time.perf_counter()
        model.train()
        with _autocast_context(resolved_config):
            logits = model(batch["input_ids"])
            loss, token_count = _loss_and_tokens(logits, batch["targets"])
        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            elapsed_since_resume = time.perf_counter() - start
            train_tok_per_sec, pure_train_tok_per_sec, wall_time_seconds, pure_train_time_seconds = _compute_resume_aware_rates(
                tokens_seen=tokens_seen,
                resume_tokens_seen=resume_tokens_seen,
                elapsed_since_resume=elapsed_since_resume,
                step_times=step_times,
                resume_wall_time_seconds=resume_wall_time_seconds,
                resume_pure_train_time_seconds=resume_pure_train_time_seconds,
            )
            progress = (step_index - 1) / real_config.train_steps
            failure_payload = {
                "benchmark": "language_nanochat_cluster",
                "status": "failed_nonfinite_loss",
                "failure_step": step_index,
                "step": step_index - 1,
                "train_steps": real_config.train_steps,
                "progress": progress,
                "tokens_seen": tokens_seen,
                "target_tokens": actual_train_tokens,
                "train_tok_per_sec": train_tok_per_sec,
                "pure_train_tok_per_sec": pure_train_tok_per_sec,
                "eta_seconds": float("inf"),
                "latest_train_loss": loss_value,
                "latest_val_loss": latest_val_loss,
                "best_val_loss": best_val_loss,
                "peak_vram_mb": _peak_vram_mb(real_config.device),
                "parameter_count": count_parameters(model),
                "hardware_profile": profile.name,
                "cache_path": str(resolved_cache_path),
                "wall_time_seconds": wall_time_seconds,
                "pure_train_time_seconds": pure_train_time_seconds,
            }
            _write_json(state_path, failure_payload)
            _append_jsonl(metrics_path, {"kind": "failure", **failure_payload, "timestamp": time.time()})
            _save_checkpoint(
                nonfinite_ckpt,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=resolved_config,
                step=step_index - 1,
                tokens_seen=tokens_seen,
                latest_val_loss=latest_val_loss,
                best_val_loss=best_val_loss,
                cache_path=resolved_cache_path,
                wall_time_seconds=wall_time_seconds,
                pure_train_time_seconds=pure_train_time_seconds,
            )
            raise RuntimeError(
                f"Non-finite training loss at step {step_index:,}: {loss_value!r}. "
                f"Preserved the last good training state in {nonfinite_ckpt}."
            )
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameter_list, max_norm=1.0)
        grad_norm_value = float(grad_norm)
        if not math.isfinite(grad_norm_value):
            optimizer.zero_grad(set_to_none=True)
            elapsed_since_resume = time.perf_counter() - start
            train_tok_per_sec, pure_train_tok_per_sec, wall_time_seconds, pure_train_time_seconds = _compute_resume_aware_rates(
                tokens_seen=tokens_seen,
                resume_tokens_seen=resume_tokens_seen,
                elapsed_since_resume=elapsed_since_resume,
                step_times=step_times,
                resume_wall_time_seconds=resume_wall_time_seconds,
                resume_pure_train_time_seconds=resume_pure_train_time_seconds,
            )
            progress = (step_index - 1) / real_config.train_steps
            failure_payload = {
                "benchmark": "language_nanochat_cluster",
                "status": "failed_nonfinite_grad_norm",
                "failure_step": step_index,
                "step": step_index - 1,
                "train_steps": real_config.train_steps,
                "progress": progress,
                "tokens_seen": tokens_seen,
                "target_tokens": actual_train_tokens,
                "train_tok_per_sec": train_tok_per_sec,
                "pure_train_tok_per_sec": pure_train_tok_per_sec,
                "eta_seconds": float("inf"),
                "latest_train_loss": loss_value,
                "latest_val_loss": latest_val_loss,
                "best_val_loss": best_val_loss,
                "peak_vram_mb": _peak_vram_mb(real_config.device),
                "parameter_count": count_parameters(model),
                "hardware_profile": profile.name,
                "cache_path": str(resolved_cache_path),
                "wall_time_seconds": wall_time_seconds,
                "pure_train_time_seconds": pure_train_time_seconds,
                "latest_grad_norm": grad_norm_value,
            }
            _write_json(state_path, failure_payload)
            _append_jsonl(metrics_path, {"kind": "failure", **failure_payload, "timestamp": time.time()})
            _save_checkpoint(
                nonfinite_ckpt,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=resolved_config,
                step=step_index - 1,
                tokens_seen=tokens_seen,
                latest_val_loss=latest_val_loss,
                best_val_loss=best_val_loss,
                cache_path=resolved_cache_path,
                wall_time_seconds=wall_time_seconds,
                pure_train_time_seconds=pure_train_time_seconds,
            )
            raise RuntimeError(
                f"Non-finite grad norm at step {step_index:,}: {grad_norm_value!r}. "
                f"Preserved the last good training state in {nonfinite_ckpt}."
            )
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
        elapsed_since_resume = time.perf_counter() - start
        progress = step_index / real_config.train_steps
        train_tok_per_sec, pure_train_tok_per_sec, wall_time_seconds, pure_train_time_seconds = _compute_resume_aware_rates(
            tokens_seen=tokens_seen,
            resume_tokens_seen=resume_tokens_seen,
            elapsed_since_resume=elapsed_since_resume,
            step_times=step_times,
            resume_wall_time_seconds=resume_wall_time_seconds,
            resume_pure_train_time_seconds=resume_pure_train_time_seconds,
        )
        remaining_tokens = max(actual_train_tokens - tokens_seen, 0)
        eta_seconds = remaining_tokens / max(train_tok_per_sec, 1e-9)
        state_payload = {
            "benchmark": "language_nanochat_cluster",
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
            "wall_time_seconds": wall_time_seconds,
            "pure_train_time_seconds": pure_train_time_seconds,
        }
        if step_index % resolved_config.log_interval == 0 or step_index == real_config.train_steps:
            _write_json(state_path, state_payload)
            _append_jsonl(metrics_path, {"kind": "train", **state_payload, "timestamp": time.time()})
            if print_progress:
                print(
                    f"nanochat_cluster {_bar(progress)} {step_index}/{real_config.train_steps} "
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
            _append_jsonl(metrics_path, eval_payload)
            _write_json(state_path, {**state_payload, **eval_payload, "status": "running", "best_val_loss": best_val_loss})
            if print_progress:
                status = "best" if latest_val_loss <= prior_best else "eval"
                print(
                    f"[{status}] step={step_index:,} tok={tokens_seen:,} "
                    f"val={latest_val_loss:.4f} best={best_val_loss:.4f}",
                    flush=True,
                )
            if latest_val_loss <= prior_best and resolved_config.save_best_checkpoint:
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
                    wall_time_seconds=wall_time_seconds,
                    pure_train_time_seconds=pure_train_time_seconds,
                )

        if should_sample:
            samples = []
            for prompt, ids in zip(resolved_config.sample_prompts, prompt_ids, strict=True):
                generated_ids = _sample_generate(
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
                wall_time_seconds=wall_time_seconds,
                pure_train_time_seconds=pure_train_time_seconds,
            )

    elapsed_since_resume = time.perf_counter() - start
    _, _, total_time, pure_train_time = _compute_resume_aware_rates(
        tokens_seen=tokens_seen,
        resume_tokens_seen=resume_tokens_seen,
        elapsed_since_resume=elapsed_since_resume,
        step_times=step_times,
        resume_wall_time_seconds=resume_wall_time_seconds,
        resume_pure_train_time_seconds=resume_pure_train_time_seconds,
    )
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
            wall_time_seconds=total_time,
            pure_train_time_seconds=pure_train_time,
        )
    final_payload = {
        "benchmark": "language_nanochat_cluster",
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
            f"completed nanochat_cluster final_val={latest_val_loss:.4f} best_val={best_val_loss:.4f} "
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
        description="Run cluster-grade nanochat training with automatic H100/A100 defaults, FineWeb-Edu streaming cache build, checkpoints, and sampled prompt completions."
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-path", type=Path)
    parser.add_argument("--target-watch-dir", type=Path)
    parser.add_argument("--watch-refresh", type=float, default=5.0)
    parser.add_argument("--watch-once", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--preset", choices=tuple(_NAMED_PRESETS), default=None)
    parser.add_argument("--hardware-profile", choices=("h100", "a100_80g", "a100_40g", "generic_cuda", "cpu"), default=None)
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
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--lr-schedule", choices=("none", "cosine", "linear"), default="none")
    parser.add_argument("--min-lr-scale", type=float, default=1.0)
    parser.add_argument("--amp-dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--tokenization-batch-size", type=int, default=0)
    parser.add_argument("--cache-dataset-on-device", dest="cache_dataset_on_device", action="store_true")
    parser.add_argument("--no-cache-dataset-on-device", dest="cache_dataset_on_device", action="store_false")
    parser.add_argument("--sample-max-new-tokens", type=int, default=64)
    parser.add_argument("--sample-temperature", type=float, default=0.9)
    parser.add_argument("--sample-top-p", type=float, default=0.95)
    parser.add_argument("--sample-top-k", type=int, default=50)
    parser.add_argument("--sample-repetition-penalty", type=float, default=1.05)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--nano-n-layer", type=int, default=4)
    parser.add_argument("--nano-n-head", type=int, default=4)
    parser.add_argument("--nano-n-kv-head", type=int, default=4)
    parser.add_argument("--nano-n-embd", type=int, default=40)
    parser.add_argument("--nano-window-pattern", type=str, default="SSSL")
    parser.add_argument("--nano-softcap", type=float, default=15.0)
    parser.add_argument("--nano-use-value-embeddings", dest="nano_use_value_embeddings", action="store_true")
    parser.add_argument("--nano-no-value-embeddings", dest="nano_use_value_embeddings", action="store_false")
    parser.add_argument("--nano-use-smear", dest="nano_use_smear", action="store_true")
    parser.add_argument("--nano-no-smear", dest="nano_use_smear", action="store_false")
    parser.add_argument("--nano-use-backout", dest="nano_use_backout", action="store_true")
    parser.add_argument("--nano-no-backout", dest="nano_use_backout", action="store_false")
    parser.set_defaults(
        cache_dataset_on_device=True,
        nano_use_value_embeddings=True,
        nano_use_smear=True,
        nano_use_backout=True,
    )
    args = parser.parse_args()

    if args.target_watch_dir is not None:
        watch_nanochat_cluster_run(args.target_watch_dir, refresh_seconds=args.watch_refresh, once=args.watch_once)
        return

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path("runs") / f"nanochat_cluster_{timestamp}"

    config = NanochatClusterConfig(
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
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_schedule=args.lr_schedule,
        min_lr_scale=args.min_lr_scale,
        amp_dtype=args.amp_dtype,
        tokenization_batch_size=args.tokenization_batch_size,
        cache_dataset_on_device=args.cache_dataset_on_device,
        sample_max_new_tokens=args.sample_max_new_tokens,
        sample_temperature=args.sample_temperature,
        sample_top_p=args.sample_top_p,
        sample_top_k=args.sample_top_k,
        sample_repetition_penalty=args.sample_repetition_penalty,
        sample_prompts=_parse_prompts(args.prompt_file),
        use_compile=args.compile,
        local_files_only=args.local_files_only,
        hardware_profile_override=args.hardware_profile,
        nano_n_layer=args.nano_n_layer,
        nano_n_head=args.nano_n_head,
        nano_n_kv_head=args.nano_n_kv_head,
        nano_n_embd=args.nano_n_embd,
        nano_window_pattern=args.nano_window_pattern,
        nano_softcap=args.nano_softcap,
        nano_use_value_embeddings=args.nano_use_value_embeddings,
        nano_use_smear=args.nano_use_smear,
        nano_use_backout=args.nano_use_backout,
    )
    config = _apply_named_preset(config, args.preset)
    resolved, profile = resolve_cluster_config(config)
    if args.print_config:
        print(json.dumps(_config_payload(resolved, profile=profile), indent=2))
        return
    payload = run_nanochat_cluster(resolved, print_progress=True, print_samples=True)
    print(json.dumps({"final_val_loss": payload["report"]["final_val_loss"], "final_path": str(output_dir / "final.json")}, indent=2))


if __name__ == "__main__":
    main()
