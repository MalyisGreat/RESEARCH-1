from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch


def _load_base_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("standalone_longseq_anchor_train", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load base trainer from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class FreshContinueConfig:
    base_script: Path
    cache_path: Path
    validation_cache_path: Path
    output_dir: Path
    run_name: str
    resume_checkpoint: Path
    target_train_steps: int
    fresh_train_steps: int
    train_token_offset: int
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    split: str = "train"
    text_column: str = "text"
    tokenizer_name: str = "gpt2"
    sequence_length: int = 10_160
    val_blocks: int = 32
    batch_size: int = 1
    eval_interval: int = 9_843
    checkpoint_interval: int = 9_843
    milestone_checkpoint_interval: int = 98_430
    seed: int = 13
    embedding_dim: int = 1_831
    block_type: str = "relu_square"
    conv_layers: int = 1
    conv_kernel_size: int = 7
    conv_rank: int = 531
    memory_rank: int = 64
    attention_heads: int = 4
    landmark_stride: int = 128
    learning_rate: float = 6e-5
    min_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    amp_dtype: str = "fp16"
    sampled_vocab_size: int = 16_384
    token_stride: int = 16
    token_chunk_size: int = 1_024
    full_eval_token_chunk_size: int = 1_024
    tokenization_batch_size: int = 2_048


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def progress_bar(frac: float, width: int = 28) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def fill_token_buffer(
    rows: Iterable[tuple[int, str]],
    *,
    tokenizer,
    total_tokens: int,
    skip_tokens: int,
    initial_skipped_tokens: int,
    progress_path: Path,
    batch_size: int,
) -> torch.Tensor:
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError(f"Tokenizer {tokenizer.name_or_path} does not expose eos_token_id.")
    buffer = torch.empty(total_tokens, dtype=torch.int32)
    cursor = 0
    skipped = initial_skipped_tokens
    pending: list[tuple[int, str]] = []
    started = time.perf_counter()
    last_report = started
    last_progress = started

    def save_progress(rows_consumed: int, *, force: bool = False) -> None:
        nonlocal last_progress
        if skipped >= skip_tokens:
            return
        now = time.perf_counter()
        if not force and now - last_progress < 5.0:
            return
        write_json(
            progress_path,
            {
                "rows_consumed": rows_consumed,
                "skipped_tokens": skipped,
                "skip_tokens": skip_tokens,
                "total_tokens": total_tokens,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
        last_progress = now

    def consume(batch_rows: Sequence[tuple[int, str]]) -> None:
        nonlocal cursor, skipped
        encoded = tokenizer([text for _, text in batch_rows], add_special_tokens=False, truncation=False)
        for (rows_consumed, _), token_ids in zip(batch_rows, encoded["input_ids"]):
            if cursor >= total_tokens:
                break
            if not token_ids:
                save_progress(rows_consumed)
                continue
            ids = list(token_ids) + [eos_id]
            if skipped < skip_tokens:
                remaining_skip = skip_tokens - skipped
                if len(ids) <= remaining_skip:
                    skipped += len(ids)
                    save_progress(rows_consumed, force=True)
                    continue
                ids = ids[remaining_skip:]
                skipped = skip_tokens
            remaining = total_tokens - cursor
            if len(ids) > remaining:
                ids = ids[:remaining]
            chunk = torch.tensor(ids, dtype=torch.int32)
            buffer[cursor : cursor + chunk.numel()] = chunk
            cursor += chunk.numel()

    for rows_consumed, text in rows:
        if cursor >= total_tokens:
            break
        if not text or not text.strip():
            save_progress(rows_consumed)
            continue
        pending.append((rows_consumed, text))
        if len(pending) < batch_size:
            continue
        consume(pending)
        pending = []
        now = time.perf_counter()
        if now - last_report >= 5.0:
            processed = skipped + cursor
            tok_per_sec = processed / max(now - started, 1e-9)
            print(
                f"fresh_cache_build {progress_bar(cursor / total_tokens)} "
                f"{cursor:,}/{total_tokens:,} tokens skipped={skipped:,}/{skip_tokens:,} tok/s={tok_per_sec:,.0f}",
                flush=True,
            )
            last_report = now
    if pending and cursor < total_tokens:
        consume(pending)
    if cursor != total_tokens:
        raise RuntimeError(f"Token stream ended early: expected {total_tokens} tokens, got {cursor}.")
    if progress_path.exists():
        progress_path.unlink()
    return buffer


def ensure_fresh_cache(config: FreshContinueConfig) -> None:
    block_size = config.sequence_length + 1
    train_tokens_needed = config.fresh_train_steps * block_size
    val_tokens_needed = config.val_blocks * block_size
    if config.cache_path.exists():
        payload = torch.load(config.cache_path, map_location="cpu", weights_only=False)
        train_tokens = payload["train_tokens"]
        val_tokens = payload["val_tokens"]
        if train_tokens.numel() >= train_tokens_needed and val_tokens.numel() >= val_tokens_needed:
            print(f"FRESH_CACHE_EXISTS {config.cache_path} train={train_tokens.numel()} val={val_tokens.numel()}", flush=True)
            return
        raise ValueError(
            f"Existing cache is too small: train {train_tokens.numel()} < {train_tokens_needed} "
            f"or val {val_tokens.numel()} < {val_tokens_needed}"
        )

    from datasets import DownloadConfig, load_dataset
    from transformers import AutoTokenizer

    config.cache_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, use_fast=True)
    tokenizer.model_max_length = int(1e9)
    download_config = DownloadConfig(max_retries=20, resume_download=True)
    stream = load_dataset(config.dataset_name, split=config.split, streaming=True, download_config=download_config)
    progress_path = config.cache_path.with_suffix(config.cache_path.suffix + ".skip_progress.json")
    resume_rows = 0
    resume_skipped = 0
    progress = read_json(progress_path)
    if (
        progress
        and int(progress.get("skip_tokens", -1)) == config.train_token_offset
        and int(progress.get("total_tokens", -1)) == train_tokens_needed
    ):
        resume_rows = max(0, int(progress.get("rows_consumed", 0)))
        resume_skipped = max(0, int(progress.get("skipped_tokens", 0)))
    if resume_rows:
        print(
            f"FRESH_CACHE_RESUME rows={resume_rows:,} skipped={resume_skipped:,}/{config.train_token_offset:,}",
            flush=True,
        )
        stream = stream.skip(resume_rows)

    def text_rows() -> Iterable[tuple[int, str]]:
        rows_consumed = resume_rows
        for row in stream:
            rows_consumed += 1
            yield rows_consumed, row[config.text_column]

    print(
        f"FRESH_CACHE_BUILD_START path={config.cache_path} train_tokens={train_tokens_needed} "
        f"offset={config.train_token_offset}",
        flush=True,
    )
    train_tokens = fill_token_buffer(
        text_rows(),
        tokenizer=tokenizer,
        total_tokens=train_tokens_needed,
        skip_tokens=config.train_token_offset,
        initial_skipped_tokens=resume_skipped,
        progress_path=progress_path,
        batch_size=config.tokenization_batch_size,
    )
    old_payload = torch.load(config.validation_cache_path, map_location="cpu", weights_only=False)
    val_tokens = old_payload["val_tokens"][:val_tokens_needed].clone()
    torch.save(
        {
            "dataset_name": config.dataset_name,
            "split": config.split,
            "tokenizer_name": config.tokenizer_name,
            "sequence_length": config.sequence_length,
            "train_token_offset": config.train_token_offset,
            "validation_cache_path": str(config.validation_cache_path),
            "train_tokens": train_tokens,
            "val_tokens": val_tokens,
            "vocab_size": int(old_payload.get("vocab_size", 50_257)),
            "total_tokens": train_tokens_needed + val_tokens_needed,
        },
        config.cache_path,
    )
    print(f"FRESH_CACHE_BUILD_DONE {config.cache_path} bytes={config.cache_path.stat().st_size}", flush=True)


def continuation_lr(config: FreshContinueConfig, *, start_step: int, step: int) -> float:
    if step < start_step:
        return config.learning_rate
    denom = max(config.target_train_steps - start_step + 1, 1)
    progress = min(max((step - start_step) / denom, 0.0), 1.0)
    cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())
    return config.min_learning_rate + (config.learning_rate - config.min_learning_rate) * cosine


def train_fresh_continuation(config: FreshContinueConfig) -> None:
    base = _load_base_module(config.base_script)
    train_config = base.TrainConfig(
        cache_path=config.cache_path,
        output_dir=config.output_dir,
        run_name=config.run_name,
        sequence_length=config.sequence_length,
        batch_size=config.batch_size,
        train_steps=config.target_train_steps,
        eval_interval=config.eval_interval,
        val_blocks=config.val_blocks,
        checkpoint_interval=config.checkpoint_interval,
        milestone_checkpoint_interval=config.milestone_checkpoint_interval,
        seed=config.seed,
        embedding_dim=config.embedding_dim,
        block_type=config.block_type,
        conv_layers=config.conv_layers,
        conv_kernel_size=config.conv_kernel_size,
        conv_rank=config.conv_rank,
        memory_rank=config.memory_rank,
        attention_heads=config.attention_heads,
        landmark_stride=config.landmark_stride,
        learning_rate=config.learning_rate,
        min_learning_rate=config.min_learning_rate,
        warmup_steps=0,
        weight_decay=config.weight_decay,
        amp_dtype=config.amp_dtype,
        sampled_vocab_size=config.sampled_vocab_size,
        token_stride=config.token_stride,
        token_chunk_size=config.token_chunk_size,
        full_eval_token_chunk_size=config.full_eval_token_chunk_size,
        resume_checkpoint=config.resume_checkpoint,
    )
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if config.amp_dtype == "bf16" else torch.float16
    autocast_kwargs = {"device_type": "cuda", "dtype": dtype, "enabled": device.type == "cuda"}
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"
    result_path = output_dir / "result.json"
    checkpoint_path = output_dir / "checkpoint.pt"
    base.write_json_atomic(state_path, {"status": "loading_fresh_cache", "run_name": config.run_name})
    train_inputs, train_targets, val_inputs, val_targets, vocab_size = base.load_cache(train_config)
    if vocab_size != train_config.vocab_size:
        raise RuntimeError(f"cache vocab size {vocab_size} != configured vocab size {train_config.vocab_size}")
    fixed_candidate_ids = base.top_token_ids(train_targets, count=config.sampled_vocab_size, vocab_size=vocab_size)

    model = base.CausalConvFactorizedLM(train_config).to(device)
    parameter_count = base.count_parameters(model)
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, fused=device.type == "cuda"
        )
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler(device="cuda", enabled=device.type == "cuda" and config.amp_dtype == "fp16")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    checkpoint = torch.load(config.resume_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scaler.load_state_dict(checkpoint.get("scaler_state", {}))
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    cuda_rng_state_all = checkpoint.get("cuda_rng_state_all")
    if cuda_rng_state_all is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng_state_all])
    history = list(checkpoint.get("history", []))
    step_times = list(checkpoint.get("step_times", []))
    tokens_seen = int(checkpoint.get("tokens_seen", 0))
    start_step = int(checkpoint.get("step", 0)) + 1
    remaining_steps = max(config.target_train_steps - start_step + 1, 0)
    schedule = base.build_batch_schedule(
        len(train_inputs),
        batch_size=config.batch_size,
        steps=remaining_steps,
        seed=config.seed + start_step,
    )
    print(f"RESUMED_FRESH step={start_step - 1} tokens={tokens_seen} fresh_steps={remaining_steps}", flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    base.write_json_atomic(
        state_path,
        {
            "status": "running",
            "run_name": config.run_name,
            "step": start_step - 1,
            "train_steps": config.target_train_steps,
            "tokens_seen": tokens_seen,
            "parameter_count": parameter_count,
            "fresh_cache_path": str(config.cache_path),
            "resume_checkpoint": str(config.resume_checkpoint),
        },
    )
    print(
        f"START_FRESH_CONTINUE run={config.run_name} device={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'} "
        f"params={parameter_count:,} seq={config.sequence_length} start={start_step} target={config.target_train_steps}",
        flush=True,
    )

    latest_val_loss = float("nan")
    latest_train_loss = float("nan")
    candidate_sizes: list[int] = []
    for offset, batch_indices in enumerate(schedule):
        step = start_step + offset
        batch_inputs = train_inputs.index_select(0, batch_indices).to(device, non_blocking=True)
        batch_targets = train_targets.index_select(0, batch_indices).to(device, non_blocking=True)
        if batch_inputs.dtype != torch.long:
            batch_inputs = batch_inputs.long()
        if batch_targets.dtype != torch.long:
            batch_targets = batch_targets.long()
        current_lr = continuation_lr(config, start_step=start_step, step=step)
        base.set_optimizer_lr(optimizer, current_lr)
        step_start = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(**autocast_kwargs):
            loss, token_count, candidate_size = base.anchor_loss(
                model,
                batch_inputs,
                batch_targets,
                fixed_candidate_ids=fixed_candidate_ids,
                config=train_config,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_duration = time.perf_counter() - step_start
        step_times.append(step_duration)
        tokens_seen += token_count
        candidate_sizes.append(candidate_size)
        latest_train_loss = float(loss.detach().item())
        pure_time = sum(step_times)
        pure_tps = tokens_seen / max(pure_time, 1e-9)
        if step == start_step or step % 100 == 0:
            base.write_json_atomic(
                state_path,
                {
                    "status": "running",
                    "run_name": config.run_name,
                    "step": step,
                    "train_steps": config.target_train_steps,
                    "tokens_seen": tokens_seen,
                    "latest_train_loss": latest_train_loss,
                    "latest_val_loss": latest_val_loss,
                    "latest_learning_rate": current_lr,
                    "pure_train_tok_per_sec": pure_tps,
                    "peak_vram_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else None,
                    "checkpoint_path": str(checkpoint_path),
                },
            )
            if step == start_step or step % 500 == 0:
                print(
                    f"TRAIN step={step}/{config.target_train_steps} tokens={tokens_seen} "
                    f"loss={latest_train_loss:.4f} lr={current_lr:.6g} pure_tok_s={pure_tps:.0f}",
                    flush=True,
                )
        should_eval = step % config.eval_interval == 0 or step == config.target_train_steps
        if should_eval:
            eval_start = time.perf_counter()
            latest_val_loss = base.evaluate_full_loss(
                model, val_inputs, val_targets, config=train_config, device=device, autocast_kwargs=autocast_kwargs
            )
            eval_seconds = time.perf_counter() - eval_start
            history.append(
                {
                    "step": float(step),
                    "tokens_seen": float(tokens_seen),
                    "train_loss": latest_train_loss,
                    "val_loss": latest_val_loss,
                    "learning_rate": current_lr,
                }
            )
            print(
                f"EVAL step={step}/{config.target_train_steps} tokens={tokens_seen} "
                f"train={latest_train_loss:.4f} val={latest_val_loss:.4f} eval_s={eval_seconds:.1f}",
                flush=True,
            )
            base.write_json_atomic(
                state_path,
                {
                    "status": "running",
                    "run_name": config.run_name,
                    "step": step,
                    "train_steps": config.target_train_steps,
                    "tokens_seen": tokens_seen,
                    "latest_train_loss": latest_train_loss,
                    "latest_val_loss": latest_val_loss,
                    "latest_learning_rate": current_lr,
                    "pure_train_tok_per_sec": pure_tps,
                    "peak_vram_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else None,
                    "history": history,
                    "checkpoint_path": str(checkpoint_path),
                },
            )
        should_checkpoint = (
            (config.checkpoint_interval > 0 and step % config.checkpoint_interval == 0)
            or should_eval
            or step == config.target_train_steps
        )
        if should_checkpoint:
            base.save_checkpoint(
                checkpoint_path,
                config=train_config,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                step=step,
                tokens_seen=tokens_seen,
                history=history,
                step_times=step_times,
            )
            if config.milestone_checkpoint_interval > 0 and step % config.milestone_checkpoint_interval == 0:
                milestone = output_dir / f"checkpoint.step{step}_tokens{tokens_seen}.pt"
                base.save_checkpoint(
                    milestone,
                    config=train_config,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=step,
                    tokens_seen=tokens_seen,
                    history=history,
                    step_times=step_times,
                )
                print(f"MILESTONE {milestone}", flush=True)

    pure_time = sum(step_times)
    config_payload = {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}
    result = {
        "benchmark": "standalone_longseq_anchor_fresh_continue",
        "config": config_payload,
        "report": {
            "parameter_count": parameter_count,
            "train_tokens_seen": tokens_seen,
            "final_train_loss": latest_train_loss,
            "final_val_loss": latest_val_loss,
            "pure_train_tok_per_sec": tokens_seen / max(pure_time, 1e-9),
            "step_time_mean_ms": statistics.fmean(step_times) * 1000.0,
            "step_time_median_ms": statistics.median(step_times) * 1000.0,
            "peak_vram_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else None,
            "candidate_size_mean": statistics.fmean(candidate_sizes) if candidate_sizes else None,
            "history": history,
            "checkpoint_path": str(checkpoint_path),
        },
    }
    base.write_json_atomic(result_path, result)
    base.write_json_atomic(
        state_path,
        {
            "status": "completed",
            "run_name": config.run_name,
            "step": config.target_train_steps,
            "train_steps": config.target_train_steps,
            "tokens_seen": tokens_seen,
            "final_train_loss": latest_train_loss,
            "final_val_loss": latest_val_loss,
            "pure_train_tok_per_sec": result["report"]["pure_train_tok_per_sec"],
            "peak_vram_mb": result["report"]["peak_vram_mb"],
            "checkpoint_path": str(checkpoint_path),
            "result_path": str(result_path),
        },
    )
    print("RESULT " + json.dumps(result["report"], sort_keys=True), flush=True)


def parse_args() -> FreshContinueConfig:
    parser = argparse.ArgumentParser(description="Continue longseq anchor training on a fresh post-offset token cache.")
    parser.add_argument("--base-script", type=Path, required=True)
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--validation-cache-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--resume-checkpoint", type=Path, required=True)
    parser.add_argument("--target-train-steps", type=int, required=True)
    parser.add_argument("--fresh-train-steps", type=int, required=True)
    parser.add_argument("--train-token-offset", type=int, required=True)
    parser.add_argument("--eval-interval", type=int, default=9_843)
    parser.add_argument("--checkpoint-interval", type=int, default=9_843)
    parser.add_argument("--milestone-checkpoint-interval", type=int, default=98_430)
    parser.add_argument("--embedding-dim", type=int, default=1_831)
    parser.add_argument(
        "--block-type",
        choices=(
            "relu_square",
            "gated",
            "memory",
            "landmark_attention",
            "multi_scale",
            "adaptive_multi_scale",
            "token_gated_multi_scale",
            "multi_scale_memory",
            "multi_scale_lowrank_conv_memory",
            "token_gated_lowrank_conv_memory",
            "dilated_multi_scale",
        ),
        default="relu_square",
    )
    parser.add_argument("--conv-layers", type=int, default=1)
    parser.add_argument("--conv-kernel-size", type=int, default=7)
    parser.add_argument("--conv-rank", type=int, default=531)
    parser.add_argument("--memory-rank", type=int, default=64)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--landmark-stride", type=int, default=128)
    parser.add_argument("--sampled-vocab-size", type=int, default=16_384)
    parser.add_argument("--token-stride", type=int, default=16)
    parser.add_argument("--token-chunk-size", type=int, default=1_024)
    parser.add_argument("--full-eval-token-chunk-size", type=int, default=1_024)
    parser.add_argument("--learning-rate", type=float, default=6e-5)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--tokenization-batch-size", type=int, default=2_048)
    args = parser.parse_args()
    return FreshContinueConfig(
        base_script=args.base_script,
        cache_path=args.cache_path,
        validation_cache_path=args.validation_cache_path,
        output_dir=args.output_dir,
        run_name=args.run_name,
        resume_checkpoint=args.resume_checkpoint,
        target_train_steps=args.target_train_steps,
        fresh_train_steps=args.fresh_train_steps,
        train_token_offset=args.train_token_offset,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        milestone_checkpoint_interval=args.milestone_checkpoint_interval,
        embedding_dim=args.embedding_dim,
        block_type=args.block_type,
        conv_layers=args.conv_layers,
        conv_kernel_size=args.conv_kernel_size,
        conv_rank=args.conv_rank,
        memory_rank=args.memory_rank,
        attention_heads=args.attention_heads,
        landmark_stride=args.landmark_stride,
        sampled_vocab_size=args.sampled_vocab_size,
        token_stride=args.token_stride,
        token_chunk_size=args.token_chunk_size,
        full_eval_token_chunk_size=args.full_eval_token_chunk_size,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        tokenization_batch_size=args.tokenization_batch_size,
    )


def main() -> None:
    config = parse_args()
    ensure_fresh_cache(config)
    train_fresh_continuation(config)


if __name__ == "__main__":
    main()
