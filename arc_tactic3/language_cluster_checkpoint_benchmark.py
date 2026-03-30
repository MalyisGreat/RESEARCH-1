from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from arc_tactic3.language_fastlearn_benchmark import count_parameters
from arc_tactic3.language_nano_style_compare import _estimate_target_bytes
from arc_tactic3.language_nanochat_actual_compare import NanochatMiniLM, _peak_vram_mb
from arc_tactic3.language_nanochat_cluster import (
    NanochatClusterConfig,
    _make_realtext_config as _make_nano_realtext_config,
)
from arc_tactic3.language_partial_untied_cluster import (
    PartialUntiedClusterConfig,
    _DEFAULT_PROMPTS,
    _autocast_context,
    _batch_from_flat_tokens,
    _evaluate_loss,
    _iter_train_batch_indices,
    _make_realtext_config as _make_partial_realtext_config,
    _prepare_token_buffer,
    _sample_generate,
    _top_token_ids_from_token_buffer,
)
from arc_tactic3.language_realtext_microbench import RealTextConfig, TokenBlockDataset, _build_optimizer, _build_scheduler, _loss_and_tokens
from arc_tactic3.language_recurrent_nano_tricks import PartialUntiedAssociativeLM


@dataclass(frozen=True, slots=True)
class ClusterCheckpointBenchmarkConfig:
    cache_path: Path | None = None
    output_path: Path | None = None
    partial_run_dir: Path | None = None
    partial_checkpoint_path: Path | None = None
    partial_checkpoint_name: str = "best.pt"
    nanochat_run_dir: Path | None = None
    nanochat_checkpoint_path: Path | None = None
    nanochat_checkpoint_name: str = "best.pt"
    model_filter: str = "both"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compute_val_bpb: bool = False
    probe_steps: int = 128
    sample_max_new_tokens: int = 64
    tokenizer_name: str = "gpt2"
    local_files_only: bool = False


def _coerce_config(dataclass_type: type[Any], raw: dict[str, Any]) -> Any:
    valid = {field.name for field in fields(dataclass_type)}
    payload = {key: value for key, value in raw.items() if key in valid}
    if "output_dir" in payload:
        payload["output_dir"] = Path(payload["output_dir"])
    if payload.get("cache_path") is not None:
        payload["cache_path"] = Path(payload["cache_path"])
    return dataclass_type(**payload)


def _resolve_checkpoint_path(*, run_dir: Path | None, checkpoint_path: Path | None, checkpoint_name: str) -> Path | None:
    if checkpoint_path is not None:
        return checkpoint_path
    if run_dir is None:
        return None
    return run_dir / "checkpoints" / checkpoint_name


def _load_cache(cache_path: Path) -> tuple[torch.Tensor, torch.Tensor, int]:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    return payload["train_tokens"].long(), payload["val_tokens"].long(), int(payload["vocab_size"])


def _flat_tokens_to_dataset(token_buffer: torch.Tensor, *, sequence_length: int) -> TokenBlockDataset:
    block_size = sequence_length + 1
    usable = (token_buffer.numel() // block_size) * block_size
    blocks = token_buffer[:usable].view(-1, block_size)
    return TokenBlockDataset(blocks[:, :-1].contiguous(), blocks[:, 1:].contiguous())


def _default_output_path() -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return Path("artifacts") / "benchmark_runs" / "language" / f"language_cluster_checkpoint_benchmark_{timestamp}.json"


def _load_partial_bundle(
    checkpoint_path: Path,
    *,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    vocab_size: int,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint_payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = _coerce_config(PartialUntiedClusterConfig, checkpoint_payload["config"])
    config = replace(config, device=str(device))
    tokens_per_step = config.sequence_length * config.batch_size
    train_steps = math.ceil(config.train_tokens / max(tokens_per_step, 1))
    real_config = _make_partial_realtext_config(config, train_steps=train_steps)
    partial_token_ids = _top_token_ids_from_token_buffer(
        train_tokens[: config.train_tokens],
        count=config.partial_token_count,
        vocab_size=vocab_size,
    )
    model = PartialUntiedAssociativeLM(
        vocab_size=vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        memory_dim=config.memory_dim,
        dropout=config.dropout,
        max_length=config.sequence_length,
        untied_token_ids=partial_token_ids,
    )
    model.to(device)
    optimizer = _build_optimizer(model, real_config, model_name="partial_untied_cluster")
    scheduler = _build_scheduler(optimizer, real_config)
    use_scaler = config.use_amp and config.amp_dtype == "fp16" and device.type == "cuda"
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_scaler) if device.type == "cuda" else None
    model.load_state_dict(checkpoint_payload["model_state"])
    optimizer.load_state_dict(checkpoint_payload["optimizer_state"])
    if scheduler is not None and checkpoint_payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint_payload["scheduler_state"])
    elif scheduler is not None and int(checkpoint_payload.get("step", 0)) > 0:
        scheduler.step(int(checkpoint_payload["step"]) - 1)
    if scaler is not None and checkpoint_payload.get("scaler_state") is not None:
        scaler.load_state_dict(checkpoint_payload["scaler_state"])
    return {
        "name": "partial_untied",
        "checkpoint_payload": checkpoint_payload,
        "checkpoint_path": checkpoint_path,
        "config": config,
        "real_config": real_config,
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "scaler": scaler,
        "train_source": _prepare_token_buffer(
            train_tokens[: config.train_tokens],
            device=device,
            cache_on_device=real_config.cache_dataset_on_device,
            pin_memory=real_config.pin_memory,
        ),
        "val_source": _prepare_token_buffer(
            val_tokens[: config.val_tokens],
            device=device,
            cache_on_device=real_config.cache_dataset_on_device,
            pin_memory=real_config.pin_memory,
        ),
    }


def _load_nanochat_bundle(
    checkpoint_path: Path,
    *,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    vocab_size: int,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint_payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = _coerce_config(NanochatClusterConfig, checkpoint_payload["config"])
    config = replace(config, device=str(device))
    tokens_per_step = config.sequence_length * config.batch_size
    train_steps = math.ceil(config.train_tokens / max(tokens_per_step, 1))
    real_config = _make_nano_realtext_config(config, train_steps=train_steps)
    model = NanochatMiniLM(
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
    model.to(device)
    optimizer = _build_optimizer(model, real_config, model_name="nanochat_cluster")
    scheduler = _build_scheduler(optimizer, real_config)
    use_scaler = config.use_amp and config.amp_dtype == "fp16" and device.type == "cuda"
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_scaler) if device.type == "cuda" else None
    model.load_state_dict(checkpoint_payload["model_state"])
    optimizer.load_state_dict(checkpoint_payload["optimizer_state"])
    if scheduler is not None and checkpoint_payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint_payload["scheduler_state"])
    elif scheduler is not None and int(checkpoint_payload.get("step", 0)) > 0:
        scheduler.step(int(checkpoint_payload["step"]) - 1)
    if scaler is not None and checkpoint_payload.get("scaler_state") is not None:
        scaler.load_state_dict(checkpoint_payload["scaler_state"])
    return {
        "name": "nanochat",
        "checkpoint_payload": checkpoint_payload,
        "checkpoint_path": checkpoint_path,
        "config": config,
        "real_config": real_config,
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "scaler": scaler,
        "train_source": _prepare_token_buffer(
            train_tokens[: config.train_tokens],
            device=device,
            cache_on_device=real_config.cache_dataset_on_device,
            pin_memory=real_config.pin_memory,
        ),
        "val_source": _prepare_token_buffer(
            val_tokens[: config.val_tokens],
            device=device,
            cache_on_device=real_config.cache_dataset_on_device,
            pin_memory=real_config.pin_memory,
        ),
    }


def _sample_texts(
    *,
    model: torch.nn.Module,
    config: PartialUntiedClusterConfig | NanochatClusterConfig,
    tokenizer,
    device: torch.device,
    max_new_tokens: int,
) -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []
    for prompt in config.sample_prompts:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        generated_ids = _sample_generate(
            model,
            prompt_ids,
            sequence_length=config.sequence_length,
            device=device,
            max_new_tokens=max_new_tokens,
            config=config,
        )
        samples.append({"prompt": prompt, "generated": tokenizer.decode(generated_ids)})
    return samples


def _run_train_probe(bundle: dict[str, Any], *, probe_steps: int) -> dict[str, Any] | None:
    if probe_steps <= 0:
        return None
    config = bundle["config"]
    real_config: RealTextConfig = bundle["real_config"]
    model = bundle["model"]
    optimizer = bundle["optimizer"]
    scheduler = bundle["scheduler"]
    scaler = bundle["scaler"]
    train_source = bundle["train_source"]
    device = torch.device(real_config.device)
    train_examples = train_source.numel() // (config.sequence_length + 1)
    schedule_seed = config.seed if config.train_schedule_seed is None else config.train_schedule_seed
    start_step = int(bundle["checkpoint_payload"]["step"]) + 1
    parameter_list = [parameter for parameter in model.parameters() if parameter.requires_grad]
    step_times: list[float] = []
    tokens_seen = 0
    last_loss = float("nan")
    failure: str | None = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    completed_steps = 0
    for offset, (step_index, batch_indices) in enumerate(
        _iter_train_batch_indices(
            train_examples,
            batch_size=real_config.batch_size,
            steps=real_config.train_steps,
            seed=schedule_seed,
            start_step=start_step,
            drop_last=True,
        ),
        start=1,
    ):
        if offset > probe_steps:
            break
        batch = _batch_from_flat_tokens(
            train_source,
            sequence_length=config.sequence_length,
            batch_indices=batch_indices,
            device=device,
            non_blocking=real_config.pin_memory and device.type == "cuda",
        )
        step_start = time.perf_counter()
        model.train()
        with _autocast_context(config):
            logits = model(batch["input_ids"])
            loss, token_count = _loss_and_tokens(logits, batch["targets"])
        last_loss = float(loss.item())
        if not math.isfinite(last_loss):
            failure = f"nonfinite_loss_step_{step_index}"
            break
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
            failure = f"nonfinite_grad_norm_step_{step_index}"
            break
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_times.append(time.perf_counter() - step_start)
        tokens_seen += token_count
        completed_steps += 1
    elapsed = time.perf_counter() - start
    return {
        "completed_steps": completed_steps,
        "target_steps": probe_steps,
        "final_train_loss": last_loss,
        "failure": failure,
        "train_tokens_seen": tokens_seen,
        "train_tok_per_sec": tokens_seen / max(elapsed, 1e-9),
        "pure_train_tok_per_sec": tokens_seen / max(sum(step_times), 1e-9),
        "step_time_mean_ms": statistics.fmean(step_times) * 1000.0 if step_times else None,
        "step_time_median_ms": statistics.median(step_times) * 1000.0 if step_times else None,
        "peak_vram_mb": _peak_vram_mb(real_config.device),
        "total_training_time_seconds": elapsed,
    }


def _benchmark_bundle(
    bundle: dict[str, Any],
    *,
    tokenizer,
    device: torch.device,
    compute_val_bpb: bool,
    sample_max_new_tokens: int,
    probe_steps: int,
) -> dict[str, Any]:
    checkpoint_payload = bundle["checkpoint_payload"]
    config = bundle["config"]
    model = bundle["model"]
    val_source = bundle["val_source"]
    eval_loss = _evaluate_loss(model, val_source, device=device, config=config)
    val_bpb = None
    if compute_val_bpb:
        val_dataset = _flat_tokens_to_dataset(val_source.cpu(), sequence_length=config.sequence_length)
        target_bytes = _estimate_target_bytes(val_dataset, tokenizer=tokenizer)
        total_val_bits = eval_loss * int(val_dataset.targets.numel()) / math.log(2.0)
        val_bpb = total_val_bits / target_bytes
    samples = _sample_texts(
        model=model,
        config=config,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=sample_max_new_tokens,
    )
    probe_report = _run_train_probe(bundle, probe_steps=probe_steps)
    return {
        "checkpoint_path": str(bundle["checkpoint_path"]),
        "checkpoint_step": int(checkpoint_payload["step"]),
        "checkpoint_tokens_seen": int(checkpoint_payload["tokens_seen"]),
        "checkpoint_latest_val_loss": float(checkpoint_payload.get("latest_val_loss", float("nan"))),
        "checkpoint_best_val_loss": float(checkpoint_payload.get("best_val_loss", float("nan"))),
        "parameter_count": count_parameters(model),
        "eval_val_loss": eval_loss,
        "val_bits_per_token": eval_loss / math.log(2.0),
        "val_bpb": val_bpb,
        "sample_max_new_tokens": sample_max_new_tokens,
        "samples": samples,
        "train_probe": probe_report,
        "peak_vram_mb_after_eval": _peak_vram_mb(str(device)),
        "config": {
            "sequence_length": config.sequence_length,
            "batch_size": config.batch_size,
            "eval_batch_size": config.eval_batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "warmup_steps": getattr(config, "warmup_steps", 0),
            "lr_schedule": getattr(config, "lr_schedule", "none"),
            "min_lr_scale": getattr(config, "min_lr_scale", 1.0),
        },
    }


def run_cluster_checkpoint_benchmark(config: ClusterCheckpointBenchmarkConfig) -> dict[str, Any]:
    partial_checkpoint = _resolve_checkpoint_path(
        run_dir=config.partial_run_dir,
        checkpoint_path=config.partial_checkpoint_path,
        checkpoint_name=config.partial_checkpoint_name,
    )
    nanochat_checkpoint = _resolve_checkpoint_path(
        run_dir=config.nanochat_run_dir,
        checkpoint_path=config.nanochat_checkpoint_path,
        checkpoint_name=config.nanochat_checkpoint_name,
    )
    checkpoint_paths = [path for path in (partial_checkpoint, nanochat_checkpoint) if path is not None]
    if not checkpoint_paths:
        raise ValueError("At least one checkpoint path or run directory must be provided.")
    for checkpoint_path in checkpoint_paths:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    if config.cache_path is None:
        first_payload = torch.load(checkpoint_paths[0], map_location="cpu", weights_only=False)
        raw_cache_path = first_payload.get("cache_path")
        if raw_cache_path is None:
            raise ValueError("Cache path was not provided and could not be inferred from the checkpoint.")
        cache_path = Path(raw_cache_path)
    else:
        cache_path = config.cache_path
    train_tokens, val_tokens, vocab_size = _load_cache(cache_path)
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name,
        use_fast=True,
        local_files_only=config.local_files_only,
    )
    tokenizer.model_max_length = int(1e9)
    device = torch.device(config.device)

    results: dict[str, Any] = {}
    if partial_checkpoint is not None and config.model_filter in {"both", "partial"}:
        partial_bundle = _load_partial_bundle(
            partial_checkpoint,
            train_tokens=train_tokens,
            val_tokens=val_tokens,
            vocab_size=vocab_size,
            device=device,
        )
        results["partial_untied"] = _benchmark_bundle(
            partial_bundle,
            tokenizer=tokenizer,
            device=device,
            compute_val_bpb=config.compute_val_bpb,
            sample_max_new_tokens=config.sample_max_new_tokens,
            probe_steps=config.probe_steps,
        )
    if nanochat_checkpoint is not None and config.model_filter in {"both", "nanochat"}:
        nano_bundle = _load_nanochat_bundle(
            nanochat_checkpoint,
            train_tokens=train_tokens,
            val_tokens=val_tokens,
            vocab_size=vocab_size,
            device=device,
        )
        results["nanochat"] = _benchmark_bundle(
            nano_bundle,
            tokenizer=tokenizer,
            device=device,
            compute_val_bpb=config.compute_val_bpb,
            sample_max_new_tokens=config.sample_max_new_tokens,
            probe_steps=config.probe_steps,
        )
    return {
        "benchmark": "language_cluster_checkpoint_benchmark",
        "config": {
            **asdict(config),
            "cache_path": str(cache_path),
            "output_path": str(config.output_path) if config.output_path is not None else None,
            "partial_run_dir": str(config.partial_run_dir) if config.partial_run_dir is not None else None,
            "partial_checkpoint_path": str(partial_checkpoint) if partial_checkpoint is not None else None,
            "nanochat_run_dir": str(config.nanochat_run_dir) if config.nanochat_run_dir is not None else None,
            "nanochat_checkpoint_path": str(nanochat_checkpoint) if nanochat_checkpoint is not None else None,
        },
        "results": results,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark trained cluster checkpoints for partial_untied and nanochat.")
    parser.add_argument("--cache-path", type=Path)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--partial-run-dir", type=Path)
    parser.add_argument("--partial-checkpoint-path", type=Path)
    parser.add_argument("--partial-checkpoint-name", type=str, default="best.pt")
    parser.add_argument("--nanochat-run-dir", type=Path)
    parser.add_argument("--nanochat-checkpoint-path", type=Path)
    parser.add_argument("--nanochat-checkpoint-name", type=str, default="best.pt")
    parser.add_argument("--model-filter", choices=("both", "partial", "nanochat"), default="both")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compute-val-bpb", action="store_true")
    parser.add_argument("--probe-steps", type=int, default=128)
    parser.add_argument("--sample-max-new-tokens", type=int, default=64)
    parser.add_argument("--tokenizer-name", type=str, default="gpt2")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    config = ClusterCheckpointBenchmarkConfig(
        cache_path=args.cache_path,
        output_path=args.output_path,
        partial_run_dir=args.partial_run_dir,
        partial_checkpoint_path=args.partial_checkpoint_path,
        partial_checkpoint_name=args.partial_checkpoint_name,
        nanochat_run_dir=args.nanochat_run_dir,
        nanochat_checkpoint_path=args.nanochat_checkpoint_path,
        nanochat_checkpoint_name=args.nanochat_checkpoint_name,
        model_filter=args.model_filter,
        device=args.device,
        compute_val_bpb=args.compute_val_bpb,
        probe_steps=args.probe_steps,
        sample_max_new_tokens=args.sample_max_new_tokens,
        tokenizer_name=args.tokenizer_name,
        local_files_only=args.local_files_only,
    )
    payload = run_cluster_checkpoint_benchmark(config)
    output_path = config.output_path or _default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), "models": sorted(payload["results"])}, indent=2))


if __name__ == "__main__":
    main()
