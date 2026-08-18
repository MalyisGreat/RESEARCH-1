from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from statistics import NormalDist
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


HERE = Path(__file__).resolve().parent
LANGUAGE_DIR = HERE.parent
sys.path.insert(0, str(LANGUAGE_DIR))

import h100_wave10_fullvocab_train as h100
import standalone_longseq_anchor_train as base


P_MEAN = -1.2
P_STD = 1.2
SIGMA_MIN = 0.002
SIGMA_MAX = 80.0
SIGMA_DATA = 0.5


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def block_sigma_boundaries(num_blocks: int) -> list[float]:
    normal = NormalDist()
    cdf_min = normal.cdf((math.log(SIGMA_MIN) - P_MEAN) / P_STD)
    cdf_max = normal.cdf((math.log(SIGMA_MAX) - P_MEAN) / P_STD)
    ascending = []
    for index in range(num_blocks + 1):
        probability = cdf_min + (cdf_max - cdf_min) * index / num_blocks
        ascending.append(math.exp(P_MEAN + P_STD * normal.inv_cdf(probability)))
    return list(reversed(ascending))


class FourierNoiseConditioner(nn.Module):
    def __init__(self, dim: int, features: int = 64) -> None:
        super().__init__()
        if features % 2:
            raise ValueError("Fourier feature count must be even")
        frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), features // 2))
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.net = nn.Sequential(nn.Linear(features, dim), nn.SiLU(), nn.Linear(dim, dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, c_noise: torch.Tensor) -> torch.Tensor:
        angles = c_noise[:, None] * self.frequencies[None, :]
        return self.net(torch.cat((angles.sin(), angles.cos()), dim=-1))


class DiffusionBlocksLM(nn.Module):
    """Block-local denoisers adapted from DiffusionBlocks to the causal Wave/delta mixer."""

    def __init__(self, model: nn.Module, *, overlap: float = 0.1) -> None:
        super().__init__()
        self.model = model
        self.overlap = overlap
        self.num_blocks = len(model.blocks)
        self.noise_conditioners = nn.ModuleList(
            FourierNoiseConditioner(model.embedding.embedding_dim) for _ in range(self.num_blocks)
        )
        self.sigma_boundaries = block_sigma_boundaries(self.num_blocks)

    def normalized_embedding(self, token_ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.model.embedding(token_ids), p=2, dim=-1)

    def sigma_interval(self, block_index: int) -> tuple[float, float]:
        high = self.sigma_boundaries[block_index]
        low = self.sigma_boundaries[block_index + 1]
        if self.overlap:
            log_range = math.log(high) - math.log(low)
            high = min(SIGMA_MAX, math.exp(math.log(high) + self.overlap * log_range))
            low = max(SIGMA_MIN, math.exp(math.log(low) - self.overlap * log_range))
        return low, high

    def sample_sigmas(self, block_index: int, batch: int, *, device: torch.device) -> torch.Tensor:
        low, high = self.sigma_interval(block_index)
        normal = NormalDist()
        cdf_low = normal.cdf((math.log(low) - P_MEAN) / P_STD)
        cdf_high = normal.cdf((math.log(high) - P_MEAN) / P_STD)
        uniform = torch.rand(batch, device=device, dtype=torch.float32)
        probabilities = cdf_low + uniform * (cdf_high - cdf_low)
        gaussian = math.sqrt(2.0) * torch.erfinv(2.0 * probabilities - 1.0)
        return torch.exp(P_MEAN + P_STD * gaussian)

    def denoise(
        self,
        input_ids: torch.Tensor,
        noisy_target: torch.Tensor,
        sigma: torch.Tensor,
        block_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sigma3 = sigma[:, None, None]
        c_skip = SIGMA_DATA**2 / (sigma3.square() + SIGMA_DATA**2)
        c_out = sigma3 * SIGMA_DATA / torch.sqrt(sigma3.square() + SIGMA_DATA**2)
        c_in = torch.rsqrt(sigma3.square() + SIGMA_DATA**2)
        c_noise = 0.25 * sigma.log()

        clean_context = self.normalized_embedding(input_ids)
        time_condition = self.noise_conditioners[block_index](c_noise)[:, None, :]
        states = clean_context + c_in * noisy_target + time_condition
        states = self.model.blocks[block_index](states)
        prediction = F.normalize(self.model.final_norm(states), p=2, dim=-1)
        denoised = c_skip * noisy_target + c_out * prediction
        hidden = self.model.factor_down(denoised)
        return denoised, hidden

    def training_hidden(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        block_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        clean_target = self.normalized_embedding(targets)
        sigma = self.sample_sigmas(block_index, input_ids.size(0), device=input_ids.device).to(clean_target)
        noisy_target = clean_target + sigma[:, None, None] * torch.randn_like(clean_target)
        _, hidden = self.denoise(input_ids, noisy_target, sigma, block_index)
        weights = (sigma.square() + SIGMA_DATA**2) / (sigma * SIGMA_DATA).square()
        # A constant per-block rescaling preserves the optimum and keeps one LR stable across noise ranges.
        weights = weights / weights.mean().clamp_min(1e-8)
        return hidden, weights[:, None].expand_as(targets), sigma

    @torch.inference_mode()
    def diffusion_hidden(
        self,
        input_ids: torch.Tensor,
        *,
        generator: torch.Generator,
        known_targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shape = (*input_ids.shape, self.model.embedding.embedding_dim)
        noisy = torch.randn(shape, device=input_ids.device, dtype=self.model.embedding.weight.dtype, generator=generator)
        noisy = noisy * math.sqrt(1.0 + self.sigma_boundaries[0] ** 2)
        known_clean = self.normalized_embedding(known_targets) if known_targets is not None else None

        for block_index in range(self.num_blocks):
            sigma_value = self.sigma_boundaries[block_index]
            next_sigma_value = self.sigma_boundaries[block_index + 1]
            if known_clean is not None:
                noisy[:, :-1] = known_clean[:, :-1]
            sigma = noisy.new_full((input_ids.size(0),), sigma_value)
            denoised, _ = self.denoise(input_ids, noisy, sigma, block_index)
            derivative = (noisy - denoised) / sigma_value
            noisy = noisy + (next_sigma_value - sigma_value) * derivative

        if known_clean is not None:
            noisy[:, :-1] = known_clean[:, :-1]
        final_sigma = noisy.new_full((input_ids.size(0),), self.sigma_boundaries[-1])
        _, hidden = self.denoise(input_ids, noisy, final_sigma, self.num_blocks - 1)
        return hidden


def make_config(args: argparse.Namespace, output_dir: Path) -> base.TrainConfig:
    return base.TrainConfig(
        cache_path=args.cache,
        output_dir=output_dir,
        run_name=f"{args.mode}_{args.run_name}",
        vocab_size=50_257,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        train_steps=args.steps if args.mode == "baseline" else args.steps * args.num_blocks,
        eval_interval=args.steps,
        val_blocks=args.val_blocks,
        checkpoint_interval=10**9,
        milestone_checkpoint_interval=10**9,
        seed=args.seed,
        embedding_dim=args.dim,
        block_type="multi_scale_lowrank_conv_memory",
        conv_layers=args.num_blocks,
        conv_rank=args.rank,
        conv_kernel_size=7,
        memory_rank=args.memory_rank,
        attention_heads=4,
        landmark_stride=128,
        sampled_vocab_size=args.sampled_vocab,
        token_stride=args.token_stride,
        token_chunk_size=args.token_chunk,
        full_eval_token_chunk_size=args.token_chunk,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=1e-4,
        amp_dtype="bf16",
        resume_checkpoint=None,
    )


def load_token_cache(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    return payload["train_tokens"], payload["val_tokens"]


def fixed_candidate_ids(path: Path, count: int, device: torch.device) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    values = payload["candidate_ids"] if isinstance(payload, dict) else payload
    return values[:count].long().to(device)


def crop_batch(tokens: torch.Tensor, starts: list[int], sequence_length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    blocks = torch.stack([tokens[start : start + sequence_length + 1] for start in starts])
    return blocks[:, :-1].to(device, non_blocking=True), blocks[:, 1:].to(device, non_blocking=True)


def make_starts(tokens: torch.Tensor, *, steps: int, batch: int, sequence_length: int, seed: int) -> list[list[int]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    maximum = tokens.numel() - sequence_length - 1
    values = torch.randint(0, maximum, (steps, batch), generator=generator)
    return [[int(value) for value in row] for row in values]


def candidate_loss(
    model: nn.Module,
    hidden: torch.Tensor,
    targets: torch.Tensor,
    fixed_ids: torch.Tensor,
    *,
    token_stride: int,
    token_chunk: int,
    token_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    candidate_ids = torch.unique(torch.cat((fixed_ids, targets.reshape(-1))))
    candidate_map = torch.full((model.factor_up.out_features,), -1, dtype=torch.long, device=targets.device)
    candidate_map[candidate_ids] = torch.arange(candidate_ids.numel(), device=targets.device)
    reduced_targets = candidate_map[targets]
    if token_weights is None:
        token_weights = torch.ones_like(targets, dtype=hidden.dtype)

    def weighted_sum(
        selected_hidden: torch.Tensor,
        selected_targets: torch.Tensor,
        selected_weights: torch.Tensor,
        output_weight: torch.Tensor,
        output_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        flat_hidden = selected_hidden.reshape(-1, selected_hidden.size(-1))
        flat_targets = selected_targets.reshape(-1)
        flat_weights = selected_weights.reshape(-1).float()
        total = flat_hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, flat_hidden.size(0), token_chunk):
            stop = min(start + token_chunk, flat_hidden.size(0))
            logits = F.linear(flat_hidden[start:stop], output_weight, output_bias).float()
            losses = F.cross_entropy(logits, flat_targets[start:stop], reduction="none")
            total = total + (losses * flat_weights[start:stop]).sum()
        return total / flat_weights.sum().clamp_min(1.0)

    sampled = weighted_sum(
        hidden,
        reduced_targets,
        token_weights,
        model.factor_up.weight.index_select(0, candidate_ids),
        model.factor_up.bias.index_select(0, candidate_ids) if model.factor_up.bias is not None else None,
    )
    full = weighted_sum(
        hidden[:, ::token_stride],
        targets[:, ::token_stride],
        token_weights[:, ::token_stride],
        model.factor_up.weight,
        model.factor_up.bias,
    )
    return 0.5 * (sampled + full), int(candidate_ids.numel())


@torch.inference_mode()
def full_vocab_loss(model: nn.Module, hidden: torch.Tensor, targets: torch.Tensor, chunk: int) -> float:
    flat_hidden = hidden.reshape(-1, hidden.size(-1))
    flat_targets = targets.reshape(-1)
    total = 0.0
    for start in range(0, flat_hidden.size(0), chunk):
        stop = min(start + chunk, flat_hidden.size(0))
        selected = flat_hidden[start:stop]
        if selected.device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = F.linear(selected, model.factor_up.weight, model.factor_up.bias).float()
        else:
            logits = F.linear(selected.to(model.factor_up.weight.dtype), model.factor_up.weight, model.factor_up.bias)
        total += float(F.cross_entropy(logits, flat_targets[start:stop], reduction="sum").item())
    return total / flat_targets.numel()


def learning_rate(step: int, total_steps: int, args: argparse.Namespace) -> float:
    local_step = step if args.mode == "baseline" else step // args.num_blocks
    if local_step < args.warmup_steps:
        return args.learning_rate * (local_step + 1) / args.warmup_steps
    progress = min(1.0, (local_step - args.warmup_steps) / max(1, args.steps - args.warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.min_learning_rate + (args.learning_rate - args.min_learning_rate) * cosine


def build_model(config: base.TrainConfig, architecture: str, device: torch.device) -> nn.Module:
    model = base.CausalConvFactorizedLM(config)
    model = h100.apply_architecture(model, architecture)
    return model.to(device)


def validate(
    model: nn.Module,
    dblock: DiffusionBlocksLM | None,
    val_tokens: torch.Tensor,
    config: base.TrainConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    starts = [index * (config.sequence_length + 1) for index in range(config.val_blocks)]
    losses = []
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 999)
    model.eval()
    for start in starts:
        inputs, targets = crop_batch(val_tokens, [start], config.sequence_length, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if dblock is None:
                hidden = model.factor_down(model.features(inputs))
            else:
                # Evaluated slots must start from noise. Clamping them to their
                # clean targets would leak the answer into validation.
                hidden = dblock.diffusion_hidden(inputs, generator=generator, known_targets=None)
        losses.append(full_vocab_loss(model, hidden, targets, args.token_chunk))
    model.train()
    return {"val_full_vocab_loss": statistics.mean(losses), "val_full_vocab_stdev": statistics.pstdev(losses)}


@torch.inference_mode()
def sample_text(
    model: nn.Module,
    dblock: DiffusionBlocksLM | None,
    *,
    device: torch.device,
    seed: int,
    max_new_tokens: int,
) -> list[dict[str, str]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    prompts = ("The history of computing began", "Question: Why does ice float on water?\nAnswer:")
    rows = []
    for prompt_index, prompt in enumerate(prompts):
        ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + 10_000 + prompt_index)
        for _ in range(max_new_tokens):
            context = ids[:, -512:]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                if dblock is None:
                    hidden = model.factor_down(model.features(context))[:, -1]
                else:
                    known = torch.cat((context[:, 1:], context[:, -1:]), dim=1)
                    hidden = dblock.diffusion_hidden(context, generator=generator, known_targets=known)[:, -1]
                logits = model.factor_up(hidden).float().squeeze(0)
            top_values, top_indices = torch.topk(logits / 0.8, 50)
            sampled = torch.multinomial(torch.softmax(top_values, dim=-1), 1, generator=generator)
            next_token = top_indices[sampled].view(1, 1)
            ids = torch.cat((ids, next_token), dim=1)
        completion = tokenizer.decode(ids[0, -max_new_tokens:].tolist(), skip_special_tokens=True)
        rows.append({"prompt": prompt, "completion": completion})
    return rows


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_root / f"{args.mode}_{args.run_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    config = make_config(args, output_dir)
    train_tokens, val_tokens = load_token_cache(args.cache)
    starts = make_starts(
        train_tokens,
        steps=args.steps,
        batch=args.batch_size,
        sequence_length=args.sequence_length,
        seed=args.seed,
    )
    candidates = fixed_candidate_ids(args.candidate_ids, args.sampled_vocab, device)
    model = build_model(config, args.architecture, device)
    dblock = DiffusionBlocksLM(model, overlap=args.overlap).to(device) if args.mode == "dblock" else None
    trainable = dblock if dblock is not None else model
    optimizer = torch.optim.AdamW(trainable.parameters(), lr=args.learning_rate, weight_decay=config.weight_decay, fused=device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    total_steps = args.steps if dblock is None else args.steps * args.num_blocks
    block_order = []
    if dblock is not None:
        order_generator = random.Random(args.seed + 17)
        for _ in range(args.steps):
            group = list(range(args.num_blocks))
            order_generator.shuffle(group)
            block_order.extend(group)
    block_counts = [0] * args.num_blocks
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
    start_wall = time.perf_counter()
    timed_tokens = 0
    timed_seconds = 0.0
    last_loss = math.nan
    finite_gradients = True
    print(
        f"START mode={args.mode} params={parameter_count(trainable):,} base_params={parameter_count(model):,} "
        f"steps={total_steps} seq={args.sequence_length} batch={args.batch_size} architecture={args.architecture}",
        flush=True,
    )

    for step in range(total_steps):
        if dblock is None:
            schedule_index = step
            block_index = -1
        else:
            block_index = block_order[step]
            schedule_index = block_counts[block_index]
            block_counts[block_index] += 1
        inputs, targets = crop_batch(train_tokens, starts[schedule_index], args.sequence_length, device)
        lr = learning_rate(step, total_steps, args)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        tick = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if dblock is None:
                hidden = model.factor_down(model.features(inputs))
                token_weights = None
                sigma_mean = math.nan
            else:
                hidden, token_weights, sigma = dblock.training_hidden(inputs, targets, block_index)
                sigma_mean = float(sigma.float().mean().item())
            loss, candidate_count = candidate_loss(
                model,
                hidden,
                targets,
                candidates,
                token_stride=args.token_stride,
                token_chunk=args.token_chunk,
                token_weights=token_weights,
            )
        scaler.scale(loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            finite_gradients = False
            raise FloatingPointError(f"non-finite gradient at step {step + 1}")
        scaler.step(optimizer)
        scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - tick
        if step >= args.timing_warmup:
            timed_tokens += targets.numel()
            timed_seconds += elapsed
        last_loss = float(loss.detach().item())
        if step == 0 or (step + 1) % args.log_interval == 0 or step + 1 == total_steps:
            row = {
                "step": step + 1,
                "mode": args.mode,
                "block_index": block_index,
                "tokens_presented": (step + 1) * targets.numel(),
                "equivalent_full_network_tokens": (step + 1) * targets.numel() / (args.num_blocks if dblock else 1),
                "loss": last_loss,
                "grad_norm": float(grad_norm.item()),
                "learning_rate": lr,
                "sigma_mean": sigma_mean,
                "candidate_count": candidate_count,
                "raw_tok_per_sec": timed_tokens / timed_seconds if timed_seconds else math.nan,
                "effective_full_network_tok_per_sec": timed_tokens / timed_seconds / (args.num_blocks if dblock else 1)
                if timed_seconds
                else math.nan,
                "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0,
                "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20 if device.type == "cuda" else 0.0,
            }
            write_jsonl(metrics_path, row)
            print("TRAIN " + json.dumps(row, sort_keys=True), flush=True)

    validation = validate(model, dblock, val_tokens, config, args, device)
    samples = (
        sample_text(
            model,
            dblock,
            device=device,
            seed=args.seed,
            max_new_tokens=args.sample_tokens,
        )
        if args.sample_tokens > 0
        else []
    )
    wall_seconds = time.perf_counter() - start_wall
    result = {
        "mode": args.mode,
        "architecture": args.architecture,
        "config": asdict(config),
        "diffusion_blocks": {
            "enabled": dblock is not None,
            "num_blocks": args.num_blocks,
            "overlap": args.overlap,
            "sigma_boundaries": dblock.sigma_boundaries if dblock is not None else None,
            "p_mean": P_MEAN,
            "p_std": P_STD,
            "sigma_data": SIGMA_DATA,
            "weight_normalization": "per-batch mean; constant rescaling within a block",
        },
        "parameter_count": parameter_count(trainable),
        "base_parameter_count": parameter_count(model),
        "total_steps": total_steps,
        "tokens_presented": total_steps * args.batch_size * args.sequence_length,
        "equivalent_full_network_tokens": total_steps * args.batch_size * args.sequence_length / (args.num_blocks if dblock else 1),
        "final_train_loss": last_loss,
        "finite_gradients": finite_gradients,
        "raw_tok_per_sec": timed_tokens / timed_seconds,
        "effective_full_network_tok_per_sec": timed_tokens / timed_seconds / (args.num_blocks if dblock else 1),
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20 if device.type == "cuda" else 0.0,
        "wall_seconds_including_eval": wall_seconds,
        **validation,
        "samples": samples,
        "limitations": [
            "DiffusionBlocks validation uses an all-position deterministic-noise four-step decode; it is not exact autoregressive likelihood.",
            "All block parameters remain resident on one GPU in this probe, so measured savings are activation/gradient savings rather than the paper's multi-device or one-resident-block theoretical bound.",
            "The shared embedding and factorized vocabulary head receive updates on every block-local step, matching the official public implementation pattern.",
        ],
    }
    write_json(output_dir / "result.json", result)
    print("RESULT " + json.dumps(result, default=str, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "dblock"), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--candidate-ids", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", default="probe")
    parser.add_argument("--architecture", choices=("wave", "delta_gain", "delta_router"), default="delta_router")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=256, help="Updates per block; DiffusionBlocks runs steps*num_blocks.")
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--dim", type=int, default=640)
    parser.add_argument("--rank", type=int, default=512)
    parser.add_argument("--memory-rank", type=int, default=128)
    parser.add_argument("--sampled-vocab", type=int, default=8192)
    parser.add_argument("--token-stride", type=int, default=32)
    parser.add_argument("--token-chunk", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=32)
    parser.add_argument("--overlap", type=float, default=0.1)
    parser.add_argument("--val-blocks", type=int, default=2)
    parser.add_argument("--sample-tokens", type=int, default=24)
    parser.add_argument("--timing-warmup", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
