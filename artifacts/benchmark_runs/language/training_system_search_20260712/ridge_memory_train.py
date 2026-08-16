from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

import rotating_anchor_collapsed_train as experiment


class CausalDiagonalRidgeMemory(nn.Module):
    """Parallel diagonal-RLS associative memory with a fixed recurrent state."""

    def __init__(self, dim: int, rank: int) -> None:
        super().__init__()
        self.rank = rank
        self.query = nn.Linear(dim, rank, bias=False)
        self.key = nn.Linear(dim, rank, bias=False)
        self.value = nn.Linear(dim, rank, bias=False)
        self.output = nn.Linear(rank, dim, bias=False)
        self.gate = nn.Linear(dim, 1)
        self.log_regularizer = nn.Parameter(torch.tensor(0.0))
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = F.normalize(self.query(x), dim=-1)
        key = F.normalize(self.key(x), dim=-1)
        value = self.value(x)
        writes = key.unsqueeze(-1) * value.unsqueeze(-2)
        state = writes.cumsum(dim=1) - writes
        key_energy = key.square()
        precision = key_energy.cumsum(dim=1) - key_energy
        regularizer = F.softplus(self.log_regularizer).to(precision.dtype) + 1e-3
        coefficients = state / (precision.unsqueeze(-1) + regularizer)
        retrieved = torch.einsum("bnr,bnrv->bnv", query, coefficients)
        gate = torch.sigmoid(self.gate(x))
        return self.output(F.silu(retrieved)) * gate


class RidgeMemoryBlock(nn.Module):
    def __init__(self, base: nn.Module, rank: int) -> None:
        super().__init__()
        self.base = base
        del self.base.memory_down
        del self.base.memory_depthwise
        del self.base.memory_up
        dim = base.mix.in_features
        self.ridge_memory = CausalDiagonalRidgeMemory(dim, rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base
        conv_input = base.conv_norm(x).transpose(1, 2)
        conv_output = base.collapsed_depthwise(conv_input)
        x = x + base.dropout(base.mix(F.relu(conv_output).square()))
        normalized = base.memory_norm(x)
        x = x + base.dropout(self.ridge_memory(normalized))
        hidden = F.relu(base.ffn_in(base.ffn_norm(x))).square()
        return x + base.dropout(base.ffn_out(hidden))


ExistingDropIn = experiment.trainer.CausalMultiScaleLowRankConvMemoryBlock


class RidgeMemoryDropIn(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        collapsed = ExistingDropIn(**kwargs)
        rank = int(os.environ.get("RIDGE_MEMORY_RANK", "16"))
        self.block = RidgeMemoryBlock(collapsed.block, rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def benchmark(iterations: int) -> dict[str, object]:
    device = torch.device("cuda")
    kwargs = dict(
        dim=512,
        expansion=2,
        kernel_size=7,
        dilation=1,
        dropout=0.0,
        memory_rank=64,
        memory_kernel_size=128,
    )
    torch.manual_seed(13)
    models = {
        "collapsed_wave10": ExistingDropIn(**kwargs).to(device),
        "ridge_memory": RidgeMemoryDropIn(**kwargs).to(device),
    }
    x = torch.randn(1, 10_160, 512, device=device)
    results = {}
    for name, model in models.items():
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]

        def step() -> tuple[float, bool]:
            model.zero_grad(set_to_none=True)
            local_x = x.detach().requires_grad_(True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(local_x)
                loss = output.float().square().mean()
            loss.backward()
            torch.cuda.synchronize()
            finite = bool(torch.isfinite(output).all() and torch.isfinite(local_x.grad).all())
            return time.perf_counter() - started, finite

        for _ in range(3):
            step()
        samples = [step() for _ in range(iterations)]
        times = [sample[0] for sample in samples]
        results[name] = {
            "parameters": sum(parameter.numel() for parameter in parameters),
            "mean_step_ms": statistics.fmean(times) * 1_000,
            "median_step_ms": statistics.median(times) * 1_000,
            "all_finite": all(sample[1] for sample in samples),
            "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20,
        }
        torch.cuda.reset_peak_memory_stats()
    results["speedup"] = results["collapsed_wave10"]["mean_step_ms"] / results["ridge_memory"]["mean_step_ms"]
    return results


def main() -> None:
    if "--probe-output" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--probe-output", type=Path, required=True)
        parser.add_argument("--iterations", type=int, default=10)
        args = parser.parse_args()
        payload = {"device": torch.cuda.get_device_name(0), "benchmark": benchmark(args.iterations)}
        rendered = json.dumps(payload, indent=2)
        args.probe_output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered, flush=True)
        return
    experiment.trainer.CausalMultiScaleLowRankConvMemoryBlock = RidgeMemoryDropIn
    experiment.trainer.train(experiment.trainer.parse_args())


if __name__ == "__main__":
    main()
