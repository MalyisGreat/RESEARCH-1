from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from arc_tactic3.language_fastlearn_benchmark import count_parameters, set_global_seed
from arc_tactic3.language_nanochat_actual_compare import _load_cached_datasets
from arc_tactic3.language_realtext_microbench import (
    RealTextConfig,
    TokenBlockDataset,
    _build_train_batch_schedule,
    _dataset_tensors,
    _loss_and_tokens,
    _scheduled_batch_from_tensors,
    evaluate_loss,
)
from arc_tactic3.language_recurrent_nano_tricks import PartialUntiedAssociativeLM, _top_token_ids


def _orthogonalize_newton_schulz(update: torch.Tensor, *, steps: int, eps: float) -> torch.Tensor:
    if update.ndim != 2:
        raise ValueError("Muon updates are only defined for 2D tensors.")
    original_dtype = update.dtype
    x = update.float()
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.t()
    x = x / x.norm().clamp_min(eps)
    a = 3.4445
    b = -4.7750
    c = 2.0315
    for _ in range(steps):
        xx_t = x @ x.t()
        x = a * x + (b * xx_t + c * (xx_t @ xx_t)) @ x
    if transposed:
        x = x.t()
    return x.to(dtype=original_dtype)


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[nn.Parameter],
        *,
        lr: float,
        weight_decay: float,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        eps: float = 1e-7,
    ) -> None:
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "eps": eps,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.ndim != 2:
                    raise ValueError("Muon received a non-2D parameter.")
                grad = parameter.grad
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(parameter)
                buffer = state["momentum_buffer"]
                buffer.mul_(momentum).add_(grad)
                update = grad.add(buffer, alpha=momentum) if nesterov else buffer
                update = _orthogonalize_newton_schulz(update, steps=ns_steps, eps=eps)
                scale = math.sqrt(max(1.0, parameter.size(0) / max(parameter.size(1), 1)))
                parameter.mul_(1.0 - lr * weight_decay)
                parameter.add_(update, alpha=-lr * scale)
        return loss


class SplitMuonAdamW:
    def __init__(
        self,
        *,
        muon_params: list[nn.Parameter],
        adamw_params: list[nn.Parameter],
        muon_lr: float,
        adamw_lr: float,
        weight_decay: float,
        use_fused_adamw: bool,
        device: torch.device,
        momentum: float,
        ns_steps: int,
    ) -> None:
        self.muon = Muon(
            muon_params,
            lr=muon_lr,
            weight_decay=weight_decay,
            momentum=momentum,
            ns_steps=ns_steps,
        )
        adamw_kwargs: dict[str, Any] = {"lr": adamw_lr, "weight_decay": weight_decay}
        if use_fused_adamw and device.type == "cuda":
            adamw_kwargs["fused"] = True
        self.adamw = torch.optim.AdamW(adamw_params, **adamw_kwargs)
        self.param_groups = self.muon.param_groups + self.adamw.param_groups

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self.muon.step()
        self.adamw.step()


@dataclass(frozen=True, slots=True)
class PartialUntiedMuonProbeConfig:
    cache_path: Path
    train_blocks: int = 1024
    val_blocks: int = 128
    sequence_length: int = 63
    batch_size: int = 16
    eval_batch_size: int = 32
    train_steps: int = 64
    eval_interval: int = 16
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    seed: int = 13
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = False
    use_fused_adamw: bool = torch.cuda.is_available()
    cache_dataset_on_device: bool = torch.cuda.is_available()
    pin_memory: bool = torch.cuda.is_available()
    recurrent_embedding_dim: int = 144
    recurrent_hidden_dim: int = 288
    recurrent_memory_dim: int = 144
    dropout: float = 0.1
    partial_untied_tokens: int = 512
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    muon_lr_multiplier: float = 1.0


def make_synthetic_cache(
    path: Path,
    *,
    seed: int,
    vocab_size: int,
    sequence_length: int,
    train_blocks: int,
    val_blocks: int,
) -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    block_size = sequence_length + 1

    def _make(blocks: int) -> torch.Tensor:
        tokens = torch.empty((blocks, block_size), dtype=torch.long)
        tokens[:, 0] = torch.randint(0, vocab_size, (blocks,), generator=generator)
        offsets = (torch.arange(blocks) % 37).long()
        for position in range(1, block_size):
            jump = torch.randint(1, 23, (blocks,), generator=generator)
            random_tokens = torch.randint(0, vocab_size, (blocks,), generator=generator)
            random_mask = torch.rand(blocks, generator=generator) < 0.08
            patterned = (tokens[:, position - 1] + jump + offsets + position) % vocab_size
            tokens[:, position] = torch.where(random_mask, random_tokens, patterned)
        return tokens.reshape(-1)

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "train_tokens": _make(train_blocks),
            "val_tokens": _make(val_blocks),
            "vocab_size": vocab_size,
        },
        path,
    )


def _peak_vram_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)


def _shared_realtext_config(config: PartialUntiedMuonProbeConfig) -> RealTextConfig:
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
        tensor_batching=True,
        cache_dataset_on_device=config.cache_dataset_on_device,
        paired_train_batches=True,
        reseed_per_model=True,
        initial_eval=True,
    )


def _build_model(
    config: PartialUntiedMuonProbeConfig,
    *,
    vocab_size: int,
    partial_token_ids: torch.Tensor,
) -> PartialUntiedAssociativeLM:
    return PartialUntiedAssociativeLM(
        vocab_size=vocab_size,
        embedding_dim=config.recurrent_embedding_dim,
        hidden_dim=config.recurrent_hidden_dim,
        memory_dim=config.recurrent_memory_dim,
        dropout=config.dropout,
        max_length=config.sequence_length,
        untied_token_ids=partial_token_ids,
    )


def _split_muon_params(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter], dict[str, int]]:
    muon_params: list[nn.Parameter] = []
    adamw_params: list[nn.Parameter] = []
    muon_count = 0
    adamw_count = 0
    muon_names: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        use_muon = (
            parameter.ndim == 2
            and "embedding" not in name
            and "partial_head" not in name
            and "output_bias" not in name
        )
        if use_muon:
            muon_params.append(parameter)
            muon_count += parameter.numel()
            muon_names.append(name)
        else:
            adamw_params.append(parameter)
            adamw_count += parameter.numel()
    return muon_params, adamw_params, {
        "muon_parameter_count": muon_count,
        "adamw_parameter_count": adamw_count,
        "muon_tensor_count": len(muon_params),
        "adamw_tensor_count": len(adamw_params),
        "muon_parameter_names": muon_names,
    }


def _build_optimizer(
    model: nn.Module,
    *,
    optimizer_name: str,
    config: PartialUntiedMuonProbeConfig,
    device: torch.device,
) -> tuple[Any, dict[str, Any]]:
    if optimizer_name == "adamw":
        kwargs: dict[str, Any] = {"lr": config.learning_rate, "weight_decay": config.weight_decay}
        if config.use_fused_adamw and device.type == "cuda":
            kwargs["fused"] = True
        return torch.optim.AdamW(model.parameters(), **kwargs), {
            "optimizer": "adamw",
            "fused": bool(kwargs.get("fused", False)),
            "adamw_parameter_count": count_parameters(model),
            "muon_parameter_count": 0,
        }
    if optimizer_name == "muon_adamw":
        muon_params, adamw_params, split_report = _split_muon_params(model)
        return SplitMuonAdamW(
            muon_params=muon_params,
            adamw_params=adamw_params,
            muon_lr=config.learning_rate * config.muon_lr_multiplier,
            adamw_lr=config.learning_rate,
            weight_decay=config.weight_decay,
            use_fused_adamw=config.use_fused_adamw,
            device=device,
            momentum=config.muon_momentum,
            ns_steps=config.muon_ns_steps,
        ), {
            "optimizer": "muon_adamw",
            "muon_lr": config.learning_rate * config.muon_lr_multiplier,
            "adamw_lr": config.learning_rate,
            **split_report,
        }
    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def _train_one(
    model: nn.Module,
    train_dataset: TokenBlockDataset,
    val_dataset: TokenBlockDataset,
    *,
    config: PartialUntiedMuonProbeConfig,
    batch_schedule: list[torch.Tensor],
    optimizer_name: str,
) -> dict[str, Any]:
    device = torch.device(config.device)
    model.to(device)
    optimizer, optimizer_report = _build_optimizer(model, optimizer_name=optimizer_name, config=config, device=device)
    use_amp = config.use_amp and device.type == "cuda"
    parameter_list = [parameter for parameter in model.parameters() if parameter.requires_grad]
    real_config = _shared_realtext_config(config)
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

    initial_val_loss = evaluate_loss(model, val_source, device=device, use_amp=use_amp, config=real_config)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history: list[dict[str, float]] = [
        {
            "step": 0.0,
            "tokens_seen": 0.0,
            "train_loss": float("nan"),
            "val_loss": float(initial_val_loss),
        }
    ]
    step_times: list[float] = []
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
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameter_list, max_norm=1.0)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_times.append(time.perf_counter() - step_start)
        tokens_seen += token_count
        if step % config.eval_interval == 0 or step == config.train_steps:
            val_loss = evaluate_loss(model, val_source, device=device, use_amp=use_amp, config=real_config)
            history.append(
                {
                    "step": float(step),
                    "tokens_seen": float(tokens_seen),
                    "train_loss": float(loss.detach().item()),
                    "val_loss": float(val_loss),
                }
            )
    total_time = time.perf_counter() - start
    pure_train_time = sum(step_times)
    return {
        "parameter_count": count_parameters(model),
        "optimizer_report": optimizer_report,
        "initial_val_loss": float(initial_val_loss),
        "final_val_loss": float(history[-1]["val_loss"]),
        "train_tokens_seen": tokens_seen,
        "train_tok_per_sec": tokens_seen / max(total_time, 1e-9),
        "pure_train_tok_per_sec": tokens_seen / max(pure_train_time, 1e-9),
        "step_time_mean_ms": statistics.fmean(step_times) * 1000.0,
        "step_time_median_ms": statistics.median(step_times) * 1000.0,
        "total_training_time_seconds": total_time,
        "pure_train_time_seconds": pure_train_time,
        "eval_overhead_seconds": max(total_time - pure_train_time, 0.0),
        "peak_vram_mb": _peak_vram_mb(device),
        "history": history,
    }


def run_partial_untied_muon_probe(config: PartialUntiedMuonProbeConfig) -> dict[str, Any]:
    set_global_seed(config.seed)
    train_dataset, val_dataset, vocab_size = _load_cached_datasets(config)
    partial_token_ids = _top_token_ids(
        train_dataset,
        count=config.partial_untied_tokens,
        vocab_size=vocab_size,
    )
    schedule = _build_train_batch_schedule(
        len(train_dataset),
        batch_size=config.batch_size,
        steps=config.train_steps,
        seed=config.seed,
        drop_last=True,
    )
    reports: dict[str, dict[str, Any]] = {}
    for optimizer_name in ("adamw", "muon_adamw"):
        set_global_seed(config.seed)
        model = _build_model(config, vocab_size=vocab_size, partial_token_ids=partial_token_ids)
        reports[optimizer_name] = _train_one(
            model,
            train_dataset,
            val_dataset,
            config=config,
            batch_schedule=schedule,
            optimizer_name=optimizer_name,
        )
    adamw_loss = reports["adamw"]["final_val_loss"]
    muon_loss = reports["muon_adamw"]["final_val_loss"]
    adamw_speed = reports["adamw"]["pure_train_tok_per_sec"]
    muon_speed = reports["muon_adamw"]["pure_train_tok_per_sec"]
    return {
        "benchmark": "language_partial_untied_muon_probe",
        "config": {**asdict(config), "cache_path": str(config.cache_path)},
        "fairness": {
            "same_model_init": True,
            "same_dataset": True,
            "same_batch_schedule": True,
            "same_token_budget": True,
            "probe_only": True,
            "amp_disabled_for_optimizer_comparison": not config.use_amp,
        },
        "results": reports,
        "comparison": {
            "muon_minus_adamw_final_val_loss": muon_loss - adamw_loss,
            "muon_speed_ratio_vs_adamw": muon_speed / max(adamw_speed, 1e-9),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Muon+AdamW against AdamW on partial_untied.")
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--make-synthetic-cache", action="store_true")
    parser.add_argument("--synthetic-vocab-size", type=int, default=2048)
    parser.add_argument("--train-blocks", type=int, default=1024)
    parser.add_argument("--val-blocks", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=63)
    parser.add_argument("--train-steps", type=int, default=64)
    parser.add_argument("--eval-interval", type=int, default=16)
    parser.add_argument("--partial-untied-tokens", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--muon-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.make_synthetic_cache:
        make_synthetic_cache(
            args.cache_path,
            seed=13,
            vocab_size=args.synthetic_vocab_size,
            sequence_length=args.sequence_length,
            train_blocks=args.train_blocks,
            val_blocks=args.val_blocks,
        )

    config = PartialUntiedMuonProbeConfig(
        cache_path=args.cache_path,
        train_blocks=args.train_blocks,
        val_blocks=args.val_blocks,
        sequence_length=args.sequence_length,
        train_steps=args.train_steps,
        eval_interval=args.eval_interval,
        partial_untied_tokens=args.partial_untied_tokens,
        learning_rate=args.learning_rate,
        muon_lr_multiplier=args.muon_lr_multiplier,
        device=args.device,
        use_amp=args.use_amp,
    )
    payload = run_partial_untied_muon_probe(config)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
