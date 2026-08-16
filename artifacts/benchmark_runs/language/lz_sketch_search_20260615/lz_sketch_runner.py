from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class LZConfig:
    orders: tuple[int, ...]
    slots: int
    tables: int
    init_logit: float = -2.0


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    lz: LZConfig | None
    steps: int
    seed: int
    warm_start_checkpoint: Path | None = None


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_orders(value: str) -> tuple[int, ...]:
    if not value or value.lower() in {"none", "baseline"}:
        return ()
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def shifted_tokens(input_ids: torch.Tensor, shift: int) -> torch.Tensor:
    if shift <= 0:
        return input_ids
    pad = input_ids.new_zeros(input_ids.size(0), shift)
    return torch.cat((pad, input_ids[:, :-shift]), dim=1)


def rolling_context_hash(input_ids: torch.Tensor, order: int) -> torch.Tensor:
    if order == 0:
        return torch.zeros_like(input_ids, dtype=torch.long)
    modulus = 2_147_483_647
    hashed = torch.zeros_like(input_ids, dtype=torch.long)
    tokens = input_ids.long()
    for index in range(order):
        token = shifted_tokens(tokens, order - 1 - index)
        hashed = (hashed * 1_000_003 + token + 97 * (index + 1)) % modulus
    return hashed


def pair_hash(context_hash: torch.Tensor, token_ids: torch.Tensor, table: int, slots: int) -> torch.Tensor:
    modulus = 2_147_483_647
    a = 1_103_515_245 + 97_531 * table
    b = 12_345 + 31_337 * table
    return ((context_hash.long() * a + token_ids.long() * 1_000_003 + b) % modulus) % slots


class LZSketchRuntime:
    def __init__(self, input_ids: torch.Tensor, targets: torch.Tensor, lz: LZConfig, dtype: torch.dtype) -> None:
        self.input_ids = input_ids
        self.targets = targets
        self.lz = lz
        self.dtype = dtype
        self.contexts: list[torch.Tensor] = []
        self.prefixes: list[list[torch.Tensor]] = []
        if not lz.orders:
            return
        with torch.no_grad():
            for order in lz.orders:
                context = rolling_context_hash(input_ids, order)
                self.contexts.append(context)
                per_table: list[torch.Tensor] = []
                for table in range(lz.tables):
                    write_slot = pair_hash(context, targets, table=table, slots=lz.slots)
                    writes = torch.zeros(
                        input_ids.size(0),
                        input_ids.size(1),
                        lz.slots,
                        device=input_ids.device,
                        dtype=dtype,
                    )
                    writes.scatter_add_(2, write_slot.unsqueeze(-1), torch.ones_like(write_slot, dtype=dtype).unsqueeze(-1))
                    per_table.append(writes.cumsum(dim=1) - writes)
                self.prefixes.append(per_table)

    @property
    def enabled(self) -> bool:
        return bool(self.lz.orders)

    def bonus_for(
        self,
        model: "LZSketchLM",
        hidden_chunk: torch.Tensor,
        positions: torch.Tensor,
        candidate_ids: torch.Tensor,
        logits_dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if not self.enabled:
            return None
        if model.lz_gate is None or model.lz_scale is None:
            return None

        gate = torch.sigmoid(model.lz_gate(hidden_chunk))
        total_bonus = hidden_chunk.new_zeros(hidden_chunk.size(0), hidden_chunk.size(1), candidate_ids.numel())
        candidate_view = candidate_ids.view(1, 1, -1)

        for order_index, _order in enumerate(self.lz.orders):
            table_sum = None
            for table_index in range(self.lz.tables):
                with torch.no_grad():
                    context = self.contexts[order_index].index_select(1, positions)
                    query_context = context.unsqueeze(-1).expand(-1, -1, candidate_ids.numel())
                    query_tokens = candidate_view.expand(hidden_chunk.size(0), hidden_chunk.size(1), -1)
                    query_slot = pair_hash(query_context, query_tokens, table=table_index, slots=self.lz.slots)
                    prefix = self.prefixes[order_index][table_index].index_select(1, positions)
                    counts = prefix.gather(2, query_slot)
                    raw_bonus = torch.log1p(counts.float()).to(dtype=logits_dtype)
                table_sum = raw_bonus if table_sum is None else table_sum + raw_bonus
            raw = table_sum / max(self.lz.tables, 1)
            scale = torch.sigmoid(model.lz_scale[order_index]).to(dtype=logits_dtype)
            total_bonus = total_bonus + gate[:, :, order_index : order_index + 1] * scale * raw
        return total_bonus


class LZSketchLM(nn.Module):
    def __init__(self, trainer: Any, config: Any, lz: LZConfig | None) -> None:
        super().__init__()
        self.trunk = trainer.CausalConvFactorizedLM(config)
        self.lz = lz
        if lz is not None and lz.orders:
            self.lz_gate = nn.Linear(config.conv_rank, len(lz.orders))
            self.lz_scale = nn.Parameter(torch.full((len(lz.orders),), float(lz.init_logit)))
            nn.init.zeros_(self.lz_gate.weight)
            nn.init.zeros_(self.lz_gate.bias)
        else:
            self.lz_gate = None
            self.lz_scale = None

    @property
    def factor_up(self) -> nn.Linear:
        return self.trunk.factor_up

    def features(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.trunk.features(input_ids)

    def factor_down_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.trunk.factor_down(self.trunk.features(input_ids))


def lz_descriptor(lz: LZConfig | None) -> dict[str, Any]:
    if lz is None or not lz.orders:
        return {"enabled": False, "orders": [], "slots": 0, "tables": 0, "init_logit": None}
    return {
        "enabled": True,
        "orders": list(lz.orders),
        "slots": lz.slots,
        "tables": lz.tables,
        "init_logit": lz.init_logit,
    }


def logits_for_candidates(
    model: LZSketchLM,
    hidden_chunk: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    runtime: LZSketchRuntime | None,
    positions: torch.Tensor,
    candidate_ids: torch.Tensor,
) -> torch.Tensor:
    logits = F.linear(hidden_chunk, weight, bias)
    if runtime is not None and runtime.enabled:
        bonus = runtime.bonus_for(model, hidden_chunk, positions, candidate_ids, logits.dtype)
        if bonus is not None:
            logits = logits + bonus
    return logits


def cross_entropy_sum_lz(
    model: LZSketchLM,
    hidden: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    runtime: LZSketchRuntime | None,
    positions: torch.Tensor,
    candidate_ids: torch.Tensor,
    *,
    token_chunk_size: int,
) -> torch.Tensor:
    if targets.dtype != torch.long:
        targets = targets.long()
    loss_sum = None
    chunk = max(1, int(token_chunk_size))
    for start in range(0, hidden.size(1), chunk):
        end = min(start + chunk, hidden.size(1))
        pos = positions[start:end]
        logits = logits_for_candidates(
            model,
            hidden[:, start:end, :],
            weight,
            bias,
            runtime,
            pos,
            candidate_ids,
        )
        chunk_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets[:, start:end].reshape(-1), reduction="sum")
        loss_sum = chunk_loss if loss_sum is None else loss_sum + chunk_loss
    if loss_sum is None:
        raise RuntimeError("empty LZ cross entropy")
    return loss_sum


def anchor_loss_lz(
    trainer: Any,
    model: LZSketchLM,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    *,
    fixed_candidate_ids: torch.Tensor,
    config: Any,
    lz: LZConfig | None,
) -> tuple[torch.Tensor, int, int]:
    candidate_ids = trainer.candidate_ids_with_targets(fixed_candidate_ids, targets)
    candidate_map = torch.full((config.vocab_size,), -1, dtype=torch.long, device=targets.device)
    candidate_map[candidate_ids] = torch.arange(candidate_ids.numel(), dtype=torch.long, device=targets.device)
    reduced_targets = candidate_map[targets.reshape(-1)].view_as(targets)
    if bool((reduced_targets < 0).any()):
        raise RuntimeError("candidate set missed a target token")

    hidden = model.factor_down_features(input_ids)
    runtime = None
    if lz is not None and lz.orders:
        runtime = LZSketchRuntime(input_ids, targets, lz, dtype=hidden.dtype)

    positions = torch.arange(hidden.size(1), device=hidden.device)
    candidate_weight = model.factor_up.weight.index_select(0, candidate_ids)
    candidate_bias = model.factor_up.bias.index_select(0, candidate_ids) if model.factor_up.bias is not None else None
    sampled_sum = cross_entropy_sum_lz(
        model,
        hidden,
        reduced_targets,
        candidate_weight,
        candidate_bias,
        runtime,
        positions,
        candidate_ids,
        token_chunk_size=config.token_chunk_size,
    )
    sampled_loss = sampled_sum / targets.numel()

    anchor_positions = torch.arange(0, hidden.size(1), config.token_stride, device=hidden.device)
    anchor_hidden = hidden.index_select(1, anchor_positions)
    anchor_targets = targets.index_select(1, anchor_positions)
    all_ids = torch.arange(config.vocab_size, device=hidden.device)
    anchor_sum = cross_entropy_sum_lz(
        model,
        anchor_hidden,
        anchor_targets,
        model.factor_up.weight,
        model.factor_up.bias,
        runtime,
        anchor_positions,
        all_ids,
        token_chunk_size=min(config.token_chunk_size, 128),
    )
    full_anchor_loss = anchor_sum / anchor_targets.numel()
    return 0.5 * (sampled_loss + full_anchor_loss), int(targets.numel()), int(candidate_ids.numel())


@torch.inference_mode()
def evaluate_full_loss_lz(
    model: LZSketchLM,
    val_inputs: torch.Tensor,
    val_targets: torch.Tensor,
    *,
    config: Any,
    lz: LZConfig | None,
    device: torch.device,
    autocast_kwargs: dict[str, Any],
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    all_ids = torch.arange(config.vocab_size, device=device)
    for row in range(val_inputs.size(0)):
        batch_inputs = val_inputs[row : row + 1].to(device, non_blocking=True).long()
        batch_targets = val_targets[row : row + 1].to(device, non_blocking=True).long()
        with torch.autocast(**autocast_kwargs):
            hidden = model.factor_down_features(batch_inputs)
            runtime = None
            if lz is not None and lz.orders:
                runtime = LZSketchRuntime(batch_inputs, batch_targets, lz, dtype=hidden.dtype)
            for start in range(0, hidden.size(1), config.full_eval_token_chunk_size):
                end = min(start + config.full_eval_token_chunk_size, hidden.size(1))
                positions = torch.arange(start, end, device=device)
                logits = logits_for_candidates(
                    model,
                    hidden[:, start:end, :],
                    model.factor_up.weight,
                    model.factor_up.bias,
                    runtime,
                    positions,
                    all_ids,
                )
                chunk_targets = batch_targets[:, start:end]
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), chunk_targets.reshape(-1))
                token_count = int(chunk_targets.numel())
                total_loss += float(loss.item()) * token_count
                total_tokens += token_count
    return total_loss / max(total_tokens, 1)


def load_compatible_state(model: nn.Module, checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    source_state = checkpoint.get("model_state", checkpoint)
    target_state = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in source_state.items():
        target_key = f"trunk.{key}" if not key.startswith("trunk.") else key
        if target_key in target_state and tuple(target_state[target_key].shape) == tuple(value.shape):
            compatible[target_key] = value
        else:
            skipped.append(key)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return {
        "checkpoint": str(checkpoint_path),
        "source_step": int(checkpoint.get("step", -1)) if isinstance(checkpoint, dict) else -1,
        "source_tokens_seen": int(checkpoint.get("tokens_seen", -1)) if isinstance(checkpoint, dict) else -1,
        "loaded_tensors": len(compatible),
        "skipped_tensors": skipped[:20],
        "missing_tensors": list(missing)[:20],
        "unexpected_tensors": list(unexpected)[:20],
    }


def make_train_config(trainer: Any, args: argparse.Namespace, spec: ExperimentSpec):
    return trainer.TrainConfig(
        cache_path=args.cache_path,
        output_dir=args.output_root / spec.name,
        run_name=spec.name,
        sequence_length=args.sequence_length,
        seed=spec.seed,
        train_steps=spec.steps,
        eval_interval=spec.steps,
        checkpoint_interval=0,
        milestone_checkpoint_interval=0,
        val_blocks=args.val_blocks,
        embedding_dim=args.embedding_dim,
        block_type="multi_scale_lowrank_conv_memory",
        conv_layers=args.conv_layers,
        conv_kernel_size=args.conv_kernel_size,
        conv_rank=args.conv_rank,
        memory_rank=args.memory_rank,
        landmark_stride=128,
        sampled_vocab_size=args.sampled_vocab_size,
        token_stride=args.token_stride,
        token_chunk_size=args.token_chunk_size,
        full_eval_token_chunk_size=args.full_eval_token_chunk_size,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_steps=args.warmup_steps,
    )


def train_one(trainer: Any, args: argparse.Namespace, spec: ExperimentSpec) -> dict[str, Any]:
    config = make_train_config(trainer, args, spec)
    lz = spec.lz
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"
    result_path = output_dir / "result.json"

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and "3080" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"refusing to run on non-3080 device: {torch.cuda.get_device_name(0)}")
    amp_dtype = torch.bfloat16 if config.amp_dtype == "bf16" else torch.float16
    autocast_kwargs = {"device_type": "cuda", "dtype": amp_dtype, "enabled": device.type == "cuda"}

    trainer.write_json_atomic(
        state_path,
        {"status": "loading_cache", "run_name": config.run_name, "lz": lz_descriptor(lz), "train_steps": config.train_steps},
    )
    train_inputs, train_targets, val_inputs, val_targets, vocab_size = trainer.load_cache(config)
    if vocab_size != config.vocab_size:
        raise RuntimeError(f"cache vocab size {vocab_size} != configured vocab size {config.vocab_size}")
    fixed_candidate_ids = trainer.top_token_ids(train_targets, count=config.sampled_vocab_size, vocab_size=config.vocab_size)
    schedule = trainer.build_batch_schedule(len(train_inputs), batch_size=config.batch_size, steps=config.train_steps, seed=config.seed)

    model = LZSketchLM(trainer, config, lz).to(device)
    warm_start_report = None
    if spec.warm_start_checkpoint is not None:
        warm_start_report = load_compatible_state(model, spec.warm_start_checkpoint, device)
        print("LZ_WARM_START " + json.dumps(warm_start_report, sort_keys=True), flush=True)

    parameter_count = trainer.count_parameters(model)
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, fused=device.type == "cuda")
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler(device="cuda", enabled=device.type == "cuda" and config.amp_dtype == "fp16")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    history: list[dict[str, float]] = []
    step_times: list[float] = []
    tokens_seen = 0
    latest_train_loss = float("nan")
    latest_val_loss = float("nan")
    candidate_sizes: list[int] = []

    trainer.write_json_atomic(
        state_path,
        {
            "status": "running",
            "run_name": config.run_name,
            "lz": lz_descriptor(lz),
            "step": 0,
            "train_steps": config.train_steps,
            "parameter_count": parameter_count,
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "warm_start": warm_start_report,
        },
    )
    print(
        f"START run={config.run_name} device={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'} "
        f"params={parameter_count:,} seq={config.sequence_length} steps={config.train_steps} lz={json.dumps(lz_descriptor(lz), sort_keys=True)}",
        flush=True,
    )

    for step in range(1, config.train_steps + 1):
        batch_indices = schedule[step - 1]
        batch_inputs = train_inputs.index_select(0, batch_indices).to(device, non_blocking=True).long()
        batch_targets = train_targets.index_select(0, batch_indices).to(device, non_blocking=True).long()
        current_lr = trainer.scheduled_learning_rate(config, step)
        trainer.set_optimizer_lr(optimizer, current_lr)
        step_start = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(**autocast_kwargs):
            loss, token_count, candidate_size = anchor_loss_lz(
                trainer,
                model,
                batch_inputs,
                batch_targets,
                fixed_candidate_ids=fixed_candidate_ids,
                config=config,
                lz=lz,
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
        pure_tps = tokens_seen / max(sum(step_times), 1e-9)
        if step == 1 or step % max(1, args.log_interval) == 0:
            print(
                f"TRAIN step={step}/{config.train_steps} tokens={tokens_seen} loss={latest_train_loss:.4f} "
                f"lr={current_lr:.6g} pure_tok_s={pure_tps:.0f}",
                flush=True,
            )
        if step == config.train_steps:
            eval_start = time.perf_counter()
            latest_val_loss = evaluate_full_loss_lz(
                model,
                val_inputs,
                val_targets,
                config=config,
                lz=lz,
                device=device,
                autocast_kwargs=autocast_kwargs,
            )
            eval_s = time.perf_counter() - eval_start
            history.append(
                {
                    "step": float(step),
                    "tokens_seen": float(tokens_seen),
                    "train_loss": latest_train_loss,
                    "val_loss": latest_val_loss,
                    "learning_rate": float(current_lr),
                }
            )
            print(
                f"EVAL step={step}/{config.train_steps} tokens={tokens_seen} train={latest_train_loss:.4f} "
                f"val={latest_val_loss:.4f} eval_s={eval_s:.1f}",
                flush=True,
            )
            trainer.write_json_atomic(
                state_path,
                {
                    "status": "running",
                    "run_name": config.run_name,
                    "lz": lz_descriptor(lz),
                    "step": step,
                    "train_steps": config.train_steps,
                    "tokens_seen": tokens_seen,
                    "latest_train_loss": latest_train_loss,
                    "latest_val_loss": latest_val_loss,
                    "parameter_count": parameter_count,
                },
            )

    report = {
        "parameter_count": parameter_count,
        "train_tokens_seen": tokens_seen,
        "final_train_loss": latest_train_loss,
        "final_val_loss": latest_val_loss,
        "pure_train_tok_per_sec": tokens_seen / max(sum(step_times), 1e-9),
        "step_time_mean_ms": statistics.fmean(step_times) * 1000.0 if step_times else None,
        "step_time_median_ms": statistics.median(step_times) * 1000.0 if step_times else None,
        "peak_vram_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else None,
        "candidate_size_mean": statistics.fmean(candidate_sizes) if candidate_sizes else None,
        "history": history,
    }
    result = {
        "benchmark": "lz_sketch_search_20260615",
        "config": {**asdict(config), "cache_path": str(config.cache_path), "output_dir": str(config.output_dir), "resume_checkpoint": None},
        "lz": lz_descriptor(lz),
        "warm_start": warm_start_report,
        "report": report,
    }
    trainer.write_json_atomic(result_path, result)
    trainer.write_json_atomic(state_path, {**result, "status": "completed"})
    print("LZ_RESULT " + json.dumps({"name": spec.name, **lz_descriptor(lz), **report}, sort_keys=True), flush=True)
    return {"name": spec.name, **lz_descriptor(lz), **report, "warm_start_checkpoint": str(spec.warm_start_checkpoint) if spec.warm_start_checkpoint else ""}


def build_suite(args: argparse.Namespace) -> list[ExperimentSpec]:
    main = LZConfig((0, 1, 2, 4), 2048, 2, args.lz_init_logit)
    if args.suite == "smoke":
        return [
            ExperimentSpec("smoke_baseline", None, args.smoke_steps, args.seed),
            ExperimentSpec("smoke_lz_o0_1_2_4", main, args.smoke_steps, args.seed),
        ]
    specs: list[ExperimentSpec] = [
        ExperimentSpec("scratch_baseline", None, args.steps, args.seed),
        ExperimentSpec("scratch_lz_o0", LZConfig((0,), 2048, 2, args.lz_init_logit), args.steps, args.seed),
        ExperimentSpec("scratch_lz_o0_1", LZConfig((0, 1), 2048, 2, args.lz_init_logit), args.steps, args.seed),
        ExperimentSpec("scratch_lz_o0_1_2_4", main, args.steps, args.seed),
        ExperimentSpec("scratch_lz_o0_1_2_4_8", LZConfig((0, 1, 2, 4, 8), 2048, 2, args.lz_init_logit), args.steps, args.seed),
        ExperimentSpec("scratch_lz_4096", LZConfig((0, 1, 2, 4), 4096, 2, args.lz_init_logit), args.steps, args.seed),
        ExperimentSpec("scratch_lz_3hash", LZConfig((0, 1, 2, 4), 2048, 3, args.lz_init_logit), args.steps, args.seed),
    ]
    if args.warm_start_checkpoint:
        checkpoint = args.warm_start_checkpoint
        specs.extend(
            [
                ExperimentSpec("warm_baseline", None, args.warm_steps, args.seed, checkpoint),
                ExperimentSpec("warm_lz_o0_1_2_4", main, args.warm_steps, args.seed, checkpoint),
            ]
        )
    if args.suite == "grid":
        return specs[:7]
    return specs


def write_summary(output_root: Path, rows: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "summary.json"
    csv_path = output_root / "summary.csv"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = [
        "name",
        "enabled",
        "orders",
        "slots",
        "tables",
        "warm_start_checkpoint",
        "parameter_count",
        "train_tokens_seen",
        "final_train_loss",
        "final_val_loss",
        "pure_train_tok_per_sec",
        "peak_vram_mb",
        "step_time_mean_ms",
        "candidate_size_mean",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row["orders"] = ",".join(str(item) for item in row.get("orders", []))
            writer.writerow(row)
    print(f"LZ_SUMMARY_JSON={json_path}", flush=True)
    print(f"LZ_SUMMARY_CSV={csv_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Short LZ-sketch head screens for the longseq conv-memory trainer.")
    parser.add_argument("--trainer-path", type=Path, required=True)
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--suite", choices=("smoke", "grid", "all_short"), default="all_short")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--warm-steps", type=int, default=64)
    parser.add_argument("--smoke-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--val-blocks", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=10_160)
    parser.add_argument("--embedding-dim", type=int, default=896)
    parser.add_argument("--conv-layers", type=int, default=2)
    parser.add_argument("--conv-kernel-size", type=int, default=7)
    parser.add_argument("--conv-rank", type=int, default=320)
    parser.add_argument("--memory-rank", type=int, default=64)
    parser.add_argument("--sampled-vocab-size", type=int, default=32_768)
    parser.add_argument("--token-stride", type=int, default=4)
    parser.add_argument("--token-chunk-size", type=int, default=128)
    parser.add_argument("--full-eval-token-chunk-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=64)
    parser.add_argument("--lz-init-logit", type=float, default=-2.0)
    parser.add_argument("--warm-start-checkpoint", type=Path, default=None)
    parser.add_argument("--log-interval", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trainer = load_module(args.trainer_path, "longseq_trainer_for_lz_sketch")
    args.output_root.mkdir(parents=True, exist_ok=True)
    specs = build_suite(args)
    print("LZ_SUITE " + json.dumps({"suite": args.suite, "count": len(specs), "output_root": str(args.output_root)}, sort_keys=True), flush=True)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        started = time.perf_counter()
        print("LZ_EXPERIMENT_START " + json.dumps({"name": spec.name, "lz": lz_descriptor(spec.lz), "steps": spec.steps, "warm_start_checkpoint": str(spec.warm_start_checkpoint) if spec.warm_start_checkpoint else None}, sort_keys=True), flush=True)
        try:
            row = train_one(trainer, args, spec)
            row["elapsed_s"] = time.perf_counter() - started
            row["status"] = "completed"
        except Exception as exc:
            row = {
                "name": spec.name,
                **lz_descriptor(spec.lz),
                "warm_start_checkpoint": str(spec.warm_start_checkpoint) if spec.warm_start_checkpoint else "",
                "status": "failed",
                "failure": repr(exc),
                "elapsed_s": time.perf_counter() - started,
            }
            print("LZ_EXPERIMENT_FAILED " + json.dumps(row, sort_keys=True), flush=True)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        rows.append(row)
        write_summary(args.output_root, rows)
        if row.get("status") == "failed" and args.suite == "smoke":
            return 1
    write_summary(args.output_root, rows)
    return 0 if all(row.get("status") == "completed" for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
