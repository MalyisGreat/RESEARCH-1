from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


ROOT = Path(__file__).resolve().parent
TRAINER_PATH = ROOT / "standalone_longseq_anchor_train.py"
OUTPUT_ROOT = ROOT / "objective_loss_sweep_2080_20260614"
CACHE_PATH = (
    ROOT
    / "longseq_anchor16_40m_600m_20260602"
    / "cache"
    / "finewebedu_train600088338_val325152_seq10160_gpt2.pt"
)


def load_trainer() -> Any:
    spec = importlib.util.spec_from_file_location("longseq_trainer_objective_sweep", TRAINER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import trainer from {TRAINER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


trainer = load_trainer()


@dataclass(frozen=True)
class VariantSpec:
    name: str
    random_anchor: bool = False
    hard_negatives: bool = False
    random_tail: bool = False
    mtp: bool = False
    token_order: bool = False
    copy_bucket: bool = False
    prefix_inventory: bool = False
    rdrop: bool = False
    unlikelihood: bool = False
    rtd: bool = False


VARIANTS: list[VariantSpec] = [
    VariantSpec("baseline"),
    VariantSpec("random_anchor", random_anchor=True),
    VariantSpec("hardneg_random_anchor", random_anchor=True, hard_negatives=True, random_tail=True),
    VariantSpec("mtp_aux", random_anchor=True, mtp=True),
    VariantSpec("token_order_aux", random_anchor=True, token_order=True),
    VariantSpec("copy_bucket_aux", random_anchor=True, copy_bucket=True),
    VariantSpec("prefix_inventory_aux", random_anchor=True, prefix_inventory=True),
    VariantSpec("rdrop_anchor_aux", random_anchor=True, rdrop=True),
    VariantSpec("unlikelihood_aux", random_anchor=True, unlikelihood=True),
    VariantSpec("causal_rtd_aux", random_anchor=True, rtd=True),
    VariantSpec(
        "objective_v2_mtp_copy_hardneg",
        random_anchor=True,
        hard_negatives=True,
        random_tail=True,
        mtp=True,
        copy_bucket=True,
    ),
    VariantSpec(
        "objective_v2_plus_inventory",
        random_anchor=True,
        hard_negatives=True,
        random_tail=True,
        mtp=True,
        copy_bucket=True,
        prefix_inventory=True,
    ),
]


def make_config(run_name: str, train_steps: int, eval_interval: int) -> Any:
    return trainer.TrainConfig(
        cache_path=CACHE_PATH,
        output_dir=OUTPUT_ROOT / run_name,
        run_name=run_name,
        sequence_length=10160,
        seed=13,
        train_steps=train_steps,
        eval_interval=eval_interval,
        checkpoint_interval=0,
        milestone_checkpoint_interval=0,
        val_blocks=32,
        embedding_dim=640,
        block_type="multi_scale_lowrank_conv_memory",
        conv_layers=2,
        conv_kernel_size=7,
        conv_rank=224,
        memory_rank=64,
        landmark_stride=128,
        sampled_vocab_size=32768,
        token_stride=4,
        token_chunk_size=1024,
        full_eval_token_chunk_size=1024,
        learning_rate=6e-4,
        min_learning_rate=1e-5,
        warmup_steps=500,
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def build_candidate_ids(
    *,
    fixed_candidate_ids: torch.Tensor,
    targets: torch.Tensor,
    vocab_size: int,
    extra_targets: list[torch.Tensor] | None = None,
    hard_ids: torch.Tensor | None = None,
    random_tail_count: int = 0,
) -> torch.Tensor:
    pieces = [fixed_candidate_ids, targets.reshape(-1)]
    if extra_targets:
        pieces.extend(t.reshape(-1) for t in extra_targets)
    if hard_ids is not None and hard_ids.numel() > 0:
        pieces.append(hard_ids.reshape(-1))
    if random_tail_count > 0:
        pieces.append(torch.randint(0, vocab_size, (random_tail_count,), device=targets.device, dtype=torch.long))
    return torch.cat(pieces).unique(sorted=True)


def sampled_ce(
    model: nn.Module,
    hidden: torch.Tensor,
    targets: torch.Tensor,
    candidate_ids: torch.Tensor,
    *,
    vocab_size: int,
    token_chunk_size: int,
) -> torch.Tensor:
    candidate_map = torch.full((vocab_size,), -1, dtype=torch.long, device=targets.device)
    candidate_map[candidate_ids] = torch.arange(candidate_ids.numel(), dtype=torch.long, device=targets.device)
    reduced_targets = candidate_map[targets.reshape(-1)]
    if bool((reduced_targets < 0).any()):
        raise RuntimeError("candidate set missed a target token")
    reduced_targets = reduced_targets.view_as(targets)
    candidate_weight = model.factor_up.weight.index_select(0, candidate_ids)
    candidate_bias = model.factor_up.bias.index_select(0, candidate_ids) if model.factor_up.bias is not None else None
    loss_sum = trainer.linear_cross_entropy_sum_chunked(
        hidden,
        reduced_targets,
        candidate_weight,
        candidate_bias,
        token_chunk_size=token_chunk_size,
    )
    return loss_sum / targets.numel()


def full_anchor_ce(
    model: nn.Module,
    hidden: torch.Tensor,
    targets: torch.Tensor,
    *,
    token_stride: int,
    token_chunk_size: int,
    offset: int,
) -> torch.Tensor:
    anchor_hidden = hidden[:, offset::token_stride, :]
    anchor_targets = targets[:, offset::token_stride]
    loss_sum = trainer.linear_cross_entropy_sum_chunked(
        anchor_hidden,
        anchor_targets,
        model.factor_up.weight,
        model.factor_up.bias,
        token_chunk_size=token_chunk_size,
    )
    return loss_sum / anchor_targets.numel()


def select_hard_ids(
    model: nn.Module,
    hidden: torch.Tensor,
    targets: torch.Tensor,
    *,
    offset: int,
    token_stride: int,
    limit: int = 2048,
    per_position_topk: int = 4,
) -> torch.Tensor:
    with torch.no_grad():
        anchor_hidden = hidden[:, offset::token_stride, :].detach()
        anchor_targets = targets[:, offset::token_stride].reshape(-1)
        found: list[torch.Tensor] = []
        flat_hidden = anchor_hidden.reshape(-1, anchor_hidden.size(-1))
        for start in range(0, flat_hidden.size(0), 256):
            end = min(start + 256, flat_hidden.size(0))
            logits = model.factor_up(flat_hidden[start:end])
            top_ids = torch.topk(logits.float(), k=per_position_topk, dim=-1).indices.reshape(-1)
            found.append(top_ids)
        if not found:
            return torch.empty(0, device=targets.device, dtype=torch.long)
        ids = torch.cat(found)
        ids = ids.unique(sorted=True)
        if ids.numel() > limit:
            ids = ids[torch.randperm(ids.numel(), device=ids.device)[:limit]]
        return ids


def score_tokens(model: nn.Module, hidden: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    weight = model.factor_up.weight[token_ids]
    score = (hidden * weight).sum(dim=-1)
    if model.factor_up.bias is not None:
        score = score + model.factor_up.bias[token_ids]
    return score


def token_order_loss(model: nn.Module, hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    weighted_terms: list[torch.Tensor] = []
    for near, far, weight in ((1, 2, 0.03), (1, 4, 0.03), (2, 8, 0.02)):
        near_idx = near - 1
        far_idx = far - 1
        usable = targets.size(1) - far_idx
        if usable <= 0:
            continue
        h = hidden[:, :usable, :]
        near_tokens = targets[:, near_idx : near_idx + usable]
        far_tokens = targets[:, far_idx : far_idx + usable]
        near_score = score_tokens(model, h, near_tokens)
        far_score = score_tokens(model, h, far_tokens)
        weighted_terms.append(weight * F.softplus(-(near_score - far_score)).mean())
    return sum(weighted_terms) if weighted_terms else hidden.sum() * 0.0


def copy_bucket_labels(input_ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    labels = torch.empty_like(targets)
    for row in range(targets.size(0)):
        last_seen: dict[int, int] = {}
        input_row = input_ids[row].detach().cpu().tolist()
        target_row = targets[row].detach().cpu().tolist()
        output: list[int] = []
        for index, token in enumerate(target_row):
            last_seen[input_row[index]] = index
            previous = last_seen.get(token)
            if previous is None:
                bucket = 0
            else:
                distance = index - previous + 1
                if distance <= 16:
                    bucket = 1
                elif distance <= 64:
                    bucket = 2
                elif distance <= 256:
                    bucket = 3
                elif distance <= 1024:
                    bucket = 4
                else:
                    bucket = 5
            output.append(bucket)
        labels[row] = torch.tensor(output, dtype=torch.long, device=targets.device)
    return labels


def inventory_labels(
    input_ids: torch.Tensor,
    inventory_map: torch.Tensor,
    *,
    inventory_size: int,
    anchors: torch.Tensor,
    window: int,
) -> torch.Tensor:
    mapped = inventory_map[input_ids].clamp_min(inventory_size)
    one_hot = F.one_hot(mapped, num_classes=inventory_size + 1)[..., :inventory_size].to(torch.float32)
    prefix = one_hot.cumsum(dim=1)
    current = prefix.index_select(1, anchors)
    starts = (anchors - window).clamp_min(0)
    previous = torch.zeros_like(current)
    valid = starts > 0
    if bool(valid.any()):
        previous[:, valid, :] = prefix.index_select(1, starts[valid] - 1)
    return (current - previous).gt(0).to(torch.float32)


def unlikelihood_loss(model: nn.Module, hidden: torch.Tensor, input_ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    negative_ids = input_ids
    mask = negative_ids.ne(targets)
    if not bool(mask.any()):
        return hidden.sum() * 0.0
    scores = score_tokens(model, hidden, negative_ids)
    return F.softplus(scores[mask].float()).mean() * 0.01


class AuxModules(nn.Module):
    def __init__(self, spec: VariantSpec, rank_dim: int, inventory_size: int) -> None:
        super().__init__()
        self.mtp = nn.ModuleDict()
        if spec.mtp:
            for horizon in (1, 2, 4):
                layer = nn.Linear(rank_dim, rank_dim)
                nn.init.eye_(layer.weight)
                nn.init.zeros_(layer.bias)
                self.mtp[str(horizon)] = layer
        self.copy_head = nn.Linear(rank_dim, 6) if spec.copy_bucket else None
        self.inventory_512 = nn.Linear(rank_dim, inventory_size) if spec.prefix_inventory else None
        self.inventory_2048 = nn.Linear(rank_dim, inventory_size) if spec.prefix_inventory else None
        self.rtd_head = nn.Linear(rank_dim, 1) if spec.rtd else None


def objective_loss(
    *,
    spec: VariantSpec,
    model: nn.Module,
    aux: AuxModules,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    fixed_candidate_ids: torch.Tensor,
    inventory_map: torch.Tensor,
    config: Any,
    step: int,
    autocast_kwargs: dict[str, Any],
) -> tuple[torch.Tensor, int, int, dict[str, float]]:
    features = model.features(input_ids)
    hidden = model.factor_down(features)
    offset = random.randrange(config.token_stride) if spec.random_anchor else 0
    extra_targets: list[torch.Tensor] = []
    if spec.mtp:
        for horizon in (1, 2, 4):
            extra_targets.append(targets[:, horizon:])
    hard_ids = (
        select_hard_ids(model, hidden, targets, offset=offset, token_stride=config.token_stride)
        if spec.hard_negatives
        else None
    )
    candidate_ids = build_candidate_ids(
        fixed_candidate_ids=fixed_candidate_ids,
        targets=targets,
        extra_targets=extra_targets,
        hard_ids=hard_ids,
        random_tail_count=1024 if spec.random_tail else 0,
        vocab_size=config.vocab_size,
    )
    sampled = sampled_ce(
        model,
        hidden,
        targets,
        candidate_ids,
        vocab_size=config.vocab_size,
        token_chunk_size=config.token_chunk_size,
    )
    anchor = full_anchor_ce(
        model,
        hidden,
        targets,
        token_stride=config.token_stride,
        token_chunk_size=config.token_chunk_size,
        offset=offset,
    )
    loss = 0.5 * (sampled + anchor)
    parts = {"sampled": float(sampled.detach().item()), "anchor": float(anchor.detach().item())}

    if spec.mtp:
        mtp_loss = hidden.sum() * 0.0
        for horizon, weight in ((1, 0.10), (2, 0.07), (4, 0.04)):
            h = aux.mtp[str(horizon)](hidden[:, :-horizon, :])
            mtp_loss = mtp_loss + weight * sampled_ce(
                model,
                h,
                targets[:, horizon:],
                candidate_ids,
                vocab_size=config.vocab_size,
                token_chunk_size=config.token_chunk_size,
            )
        loss = loss + mtp_loss
        parts["mtp"] = float(mtp_loss.detach().item())

    if spec.token_order:
        top = token_order_loss(model, hidden, targets)
        loss = loss + top
        parts["token_order"] = float(top.detach().item())

    if spec.copy_bucket and aux.copy_head is not None:
        labels = copy_bucket_labels(input_ids, targets)
        logits = aux.copy_head(hidden)
        copy_loss = F.cross_entropy(logits.reshape(-1, 6), labels.reshape(-1)) * 0.05
        loss = loss + copy_loss
        parts["copy_bucket"] = float(copy_loss.detach().item())

    if spec.prefix_inventory and aux.inventory_512 is not None and aux.inventory_2048 is not None:
        anchors = torch.arange(127, hidden.size(1), 128, device=hidden.device)
        inv_hidden = hidden.index_select(1, anchors)
        labels512 = inventory_labels(input_ids, inventory_map, inventory_size=512, anchors=anchors, window=512)
        labels2048 = inventory_labels(input_ids, inventory_map, inventory_size=512, anchors=anchors, window=2048)
        inv512 = F.binary_cross_entropy_with_logits(aux.inventory_512(inv_hidden).float(), labels512) * 0.015
        inv2048 = F.binary_cross_entropy_with_logits(aux.inventory_2048(inv_hidden).float(), labels2048) * 0.015
        inv_loss = inv512 + inv2048
        loss = loss + inv_loss
        parts["prefix_inventory"] = float(inv_loss.detach().item())

    if spec.rdrop and step % 16 == 0:
        hidden2 = model.factor_down(model.features(input_ids))
        anchor1 = hidden[:, offset::config.token_stride, :]
        anchor2 = hidden2[:, offset::config.token_stride, :]
        candidate_weight = model.factor_up.weight.index_select(0, candidate_ids)
        candidate_bias = model.factor_up.bias.index_select(0, candidate_ids) if model.factor_up.bias is not None else None
        logits1 = F.linear(anchor1.reshape(-1, anchor1.size(-1)), candidate_weight, candidate_bias).float()
        logits2 = F.linear(anchor2.reshape(-1, anchor2.size(-1)), candidate_weight, candidate_bias).float()
        logp1 = F.log_softmax(logits1, dim=-1)
        logp2 = F.log_softmax(logits2, dim=-1)
        p1 = logp1.exp()
        p2 = logp2.exp()
        kl = (F.kl_div(logp1, p2, reduction="batchmean") + F.kl_div(logp2, p1, reduction="batchmean")) * 0.025
        loss = loss + kl
        parts["rdrop"] = float(kl.detach().item())

    if spec.unlikelihood:
        ul = unlikelihood_loss(model, hidden, input_ids, targets)
        loss = loss + ul
        parts["unlikelihood"] = float(ul.detach().item())

    if spec.rtd and step % 8 == 0 and aux.rtd_head is not None:
        corrupt_mask = torch.rand_like(input_ids.float()) < 0.15
        corrupt_ids = torch.randint(0, config.vocab_size, input_ids.shape, device=input_ids.device, dtype=torch.long)
        corrupt_input = torch.where(corrupt_mask, corrupt_ids, input_ids)
        corrupt_hidden = model.factor_down(model.features(corrupt_input))
        rtd_logits = aux.rtd_head(corrupt_hidden).squeeze(-1).float()
        rtd = F.binary_cross_entropy_with_logits(rtd_logits, corrupt_mask.float()) * 0.03
        loss = loss + rtd
        parts["causal_rtd"] = float(rtd.detach().item())

    return loss, int(targets.numel()), int(candidate_ids.numel()), parts


def run_variant(spec: VariantSpec, train_steps: int, eval_interval: int) -> dict[str, Any]:
    run_name = f"objloss_2080_50m_{spec.name}"
    config = make_config(run_name, train_steps, eval_interval)
    run_dir = config.output_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "state.json"
    result_path = run_dir / "result.json"

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if config.amp_dtype == "bf16" else torch.float16
    autocast_kwargs = {"device_type": "cuda", "dtype": dtype, "enabled": device.type == "cuda"}

    write_json(state_path, {"status": "loading_cache", "run_name": run_name, "variant": asdict(spec)})
    train_inputs, train_targets, val_inputs, val_targets, vocab_size = trainer.load_cache(config)
    if vocab_size != config.vocab_size:
        raise RuntimeError(f"cache vocab size {vocab_size} != configured vocab size {config.vocab_size}")
    fixed_candidate_ids = trainer.top_token_ids(train_targets, count=config.sampled_vocab_size, vocab_size=config.vocab_size).to(device)
    inventory_ids = trainer.top_token_ids(train_targets, count=512, vocab_size=config.vocab_size).to(device)
    inventory_map = torch.full((config.vocab_size,), 512, dtype=torch.long, device=device)
    inventory_map[inventory_ids] = torch.arange(512, device=device, dtype=torch.long)
    schedule = trainer.build_batch_schedule(len(train_inputs), batch_size=config.batch_size, steps=config.train_steps, seed=config.seed)

    model = trainer.CausalConvFactorizedLM(config).to(device)
    aux = AuxModules(spec, rank_dim=config.conv_rank, inventory_size=512).to(device)
    parameter_count = int(trainer.count_parameters(model) + trainer.count_parameters(aux))
    parameters = [p for p in list(model.parameters()) + list(aux.parameters()) if p.requires_grad]
    try:
        optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay, fused=device.type == "cuda")
    except TypeError:
        optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler(device="cuda", enabled=device.type == "cuda" and config.amp_dtype == "fp16")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history: list[dict[str, float]] = []
    step_times: list[float] = []
    candidate_sizes: list[int] = []
    latest_train_loss = float("nan")
    latest_val_loss = float("nan")
    tokens_seen = 0
    objective_parts_last: dict[str, float] = {}
    print(
        f"START_OBJECTIVE run={run_name} device={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'} "
        f"params={parameter_count:,} steps={config.train_steps} variant={spec.name}",
        flush=True,
    )
    write_json(
        state_path,
        {
            "status": "running",
            "run_name": run_name,
            "variant": asdict(spec),
            "train_steps": config.train_steps,
            "parameter_count": parameter_count,
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        },
    )

    for step in range(1, config.train_steps + 1):
        batch_indices = schedule[step - 1]
        batch_inputs = train_inputs.index_select(0, batch_indices).to(device, non_blocking=True).long()
        batch_targets = train_targets.index_select(0, batch_indices).to(device, non_blocking=True).long()
        current_lr = trainer.scheduled_learning_rate(config, step)
        trainer.set_optimizer_lr(optimizer, current_lr)
        step_start = time.perf_counter()
        model.train()
        aux.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(**autocast_kwargs):
            loss, token_count, candidate_size, parts = objective_loss(
                spec=spec,
                model=model,
                aux=aux,
                input_ids=batch_inputs,
                targets=batch_targets,
                fixed_candidate_ids=fixed_candidate_ids,
                inventory_map=inventory_map,
                config=config,
                step=step,
                autocast_kwargs=autocast_kwargs,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        duration = time.perf_counter() - step_start
        step_times.append(duration)
        tokens_seen += token_count
        candidate_sizes.append(candidate_size)
        latest_train_loss = float(loss.detach().item())
        objective_parts_last = parts
        pure_tps = tokens_seen / max(sum(step_times), 1e-9)

        if step == 1 or step % 100 == 0:
            write_json(
                state_path,
                {
                    "status": "running",
                    "run_name": run_name,
                    "variant": asdict(spec),
                    "step": step,
                    "train_steps": config.train_steps,
                    "tokens_seen": tokens_seen,
                    "latest_train_loss": latest_train_loss,
                    "latest_val_loss": latest_val_loss,
                    "latest_learning_rate": current_lr,
                    "pure_train_tok_per_sec": pure_tps,
                    "objective_parts_last": objective_parts_last,
                    "peak_vram_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else None,
                },
            )
            if step == 1 or step % 500 == 0:
                print(
                    f"TRAIN_OBJECTIVE run={run_name} step={step}/{config.train_steps} tokens={tokens_seen} "
                    f"loss={latest_train_loss:.4f} lr={current_lr:.6g} pure_tok_s={pure_tps:.0f}",
                    flush=True,
                )

        should_eval = step % config.eval_interval == 0 or step == config.train_steps
        if should_eval:
            model.eval()
            aux.eval()
            eval_start = time.perf_counter()
            latest_val_loss = trainer.evaluate_full_loss(
                model,
                val_inputs,
                val_targets,
                config=config,
                device=device,
                autocast_kwargs=autocast_kwargs,
            )
            eval_seconds = time.perf_counter() - eval_start
            row = {
                "step": float(step),
                "tokens_seen": float(tokens_seen),
                "train_loss": latest_train_loss,
                "val_loss": latest_val_loss,
                "learning_rate": current_lr,
            }
            history.append(row)
            print(
                f"EVAL_OBJECTIVE run={run_name} step={step}/{config.train_steps} tokens={tokens_seen} "
                f"train={latest_train_loss:.4f} val={latest_val_loss:.4f} eval_s={eval_seconds:.1f}",
                flush=True,
            )
            write_json(
                state_path,
                {
                    "status": "running",
                    "run_name": run_name,
                    "variant": asdict(spec),
                    "step": step,
                    "train_steps": config.train_steps,
                    "tokens_seen": tokens_seen,
                    "latest_train_loss": latest_train_loss,
                    "latest_val_loss": latest_val_loss,
                    "latest_learning_rate": current_lr,
                    "pure_train_tok_per_sec": pure_tps,
                    "objective_parts_last": objective_parts_last,
                    "history": history,
                    "peak_vram_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else None,
                },
            )

    pure_time = sum(step_times)
    result = {
        "benchmark": "objective_loss_sweep_2080_20260614",
        "variant": asdict(spec),
        "config": {
            **asdict(config),
            "cache_path": str(config.cache_path),
            "output_dir": str(config.output_dir),
            "resume_checkpoint": None,
        },
        "report": {
            "parameter_count": parameter_count,
            "base_model_parameter_count": int(trainer.count_parameters(model)),
            "aux_parameter_count": int(trainer.count_parameters(aux)),
            "train_tokens_seen": tokens_seen,
            "final_train_loss": latest_train_loss,
            "final_val_loss": latest_val_loss,
            "pure_train_tok_per_sec": tokens_seen / max(pure_time, 1e-9),
            "step_time_mean_ms": statistics.fmean(step_times) * 1000.0,
            "step_time_median_ms": statistics.median(step_times) * 1000.0,
            "peak_vram_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else None,
            "candidate_size_mean": statistics.fmean(candidate_sizes) if candidate_sizes else None,
            "history": history,
            "objective_parts_last": objective_parts_last,
        },
    }
    write_json(result_path, result)
    write_json(
        state_path,
        {
            "status": "completed",
            "run_name": run_name,
            "variant": asdict(spec),
            "step": config.train_steps,
            "train_steps": config.train_steps,
            "tokens_seen": tokens_seen,
            "latest_train_loss": latest_train_loss,
            "latest_val_loss": latest_val_loss,
            "result_path": str(result_path),
        },
    )
    print("RESULT_OBJECTIVE " + json.dumps(result["report"], sort_keys=True), flush=True)
    del model, aux, optimizer, scaler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def selected_variants() -> list[VariantSpec]:
    wanted = os.environ.get("OBJECTIVE_SWEEP_VARIANTS", "").strip()
    if not wanted:
        return VARIANTS
    names = [name.strip() for name in wanted.split(",") if name.strip()]
    mapping = {variant.name: variant for variant in VARIANTS}
    unknown = [name for name in names if name not in mapping]
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")
    return [mapping[name] for name in names]


def main() -> None:
    if not CACHE_PATH.exists():
        raise FileNotFoundError(CACHE_PATH)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    train_steps = int(os.environ.get("OBJECTIVE_SWEEP_STEPS", "4922"))
    eval_interval = int(os.environ.get("OBJECTIVE_SWEEP_EVAL_INTERVAL", str(max(1, train_steps // 2))))
    all_results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    write_json(
        OUTPUT_ROOT / "sweep_config.json",
        {
            "cache_path": str(CACHE_PATH),
            "train_steps": train_steps,
            "tokens_per_run": train_steps * 10160,
            "eval_interval": eval_interval,
            "variants": [asdict(variant) for variant in selected_variants()],
            "targets_from_prior_wave10": {
                "60m_val_loss": 5.162740405025132,
                "100m_val_loss": 5.045041977952471,
            },
        },
    )
    for variant in selected_variants():
        try:
            result = run_variant(variant, train_steps=train_steps, eval_interval=eval_interval)
            all_results.append(result)
        except Exception as exc:  # keep the sweep moving if one objective is broken
            failure = {"variant": variant.name, "error": repr(exc)}
            failures.append(failure)
            print("FAILED_OBJECTIVE " + json.dumps(failure, sort_keys=True), flush=True)
            write_json(OUTPUT_ROOT / f"failed_{variant.name}.json", failure)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        rows = []
        baseline_val = None
        for result in all_results:
            if result["variant"]["name"] == "baseline":
                baseline_val = result["report"]["final_val_loss"]
                break
        for result in all_results:
            report = result["report"]
            row = {
                "variant": result["variant"]["name"],
                "final_train_loss": report["final_train_loss"],
                "final_val_loss": report["final_val_loss"],
                "delta_vs_exact_baseline": (
                    None if baseline_val is None else report["final_val_loss"] - baseline_val
                ),
                "train_tokens_seen": report["train_tokens_seen"],
                "tok_per_sec": report["pure_train_tok_per_sec"],
                "peak_vram_mb": report["peak_vram_mb"],
                "parameter_count": report["parameter_count"],
                "aux_parameter_count": report["aux_parameter_count"],
                "candidate_size_mean": report["candidate_size_mean"],
            }
            rows.append(row)
        rows.sort(key=lambda item: item["final_val_loss"])
        write_json(OUTPUT_ROOT / "summary.json", {"rows": rows, "failures": failures})
        with (OUTPUT_ROOT / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
            if rows:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)


if __name__ == "__main__":
    main()
