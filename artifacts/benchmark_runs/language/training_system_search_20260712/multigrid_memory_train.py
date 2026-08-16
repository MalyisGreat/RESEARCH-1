from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

import rotating_anchor_collapsed_train as experiment


class CausalMultigridMemoryBlock(nn.Module):
    def __init__(self, base: nn.Module, block_sizes: tuple[int, ...] = (8, 32, 128, 512)) -> None:
        super().__init__()
        self.base = base
        del self.base.memory_depthwise
        dim = base.mix.in_features
        rank = base.memory_down.out_features
        self.block_sizes = block_sizes
        self.level_gate = nn.Linear(dim, len(block_sizes))
        self.level_scale = nn.Parameter(torch.ones(len(block_sizes), rank))
        nn.init.zeros_(self.level_gate.weight)
        nn.init.zeros_(self.level_gate.bias)

    @staticmethod
    def previous_block_summary(values: torch.Tensor, block_size: int, length: int) -> torch.Tensor:
        pooled = F.avg_pool1d(values, kernel_size=block_size, stride=block_size)
        shifted = F.pad(pooled, (1, 0))
        expanded = shifted.repeat_interleave(block_size, dim=-1)
        if expanded.size(-1) < length:
            expanded = F.pad(expanded, (0, length - expanded.size(-1)))
        return expanded[..., :length].transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base
        conv_input = base.conv_norm(x).transpose(1, 2)
        conv_output = base.collapsed_depthwise(conv_input)
        x = x + base.dropout(base.mix(F.relu(conv_output).square()))

        normalized = base.memory_norm(x)
        memory_values = base.memory_down(normalized).transpose(1, 2)
        summaries = torch.stack(
            [self.previous_block_summary(memory_values, size, x.size(1)) for size in self.block_sizes],
            dim=-2,
        )
        scales = self.level_scale.to(dtype=summaries.dtype).view(1, 1, len(self.block_sizes), -1)
        gates = torch.softmax(self.level_gate(normalized), dim=-1).unsqueeze(-1)
        memory_output = (summaries * scales * gates).sum(dim=-2)
        x = x + base.dropout(base.memory_up(F.silu(memory_output)))

        hidden = F.relu(base.ffn_in(base.ffn_norm(x))).square()
        return x + base.dropout(base.ffn_out(hidden))


ExistingDropIn = experiment.trainer.CausalMultiScaleLowRankConvMemoryBlock


class MultigridDropIn(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        collapsed = ExistingDropIn(**kwargs)
        self.block = CausalMultigridMemoryBlock(collapsed.block)

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
        "multigrid": MultigridDropIn(**kwargs).to(device),
    }
    x = torch.randn(1, 10_160, 512, device=device)
    results = {}
    for name, model in models.items():
        model.train()
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
        }
    results["speedup"] = results["collapsed_wave10"]["mean_step_ms"] / results["multigrid"]["mean_step_ms"]
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
    experiment.trainer.CausalMultiScaleLowRankConvMemoryBlock = MultigridDropIn
    experiment.trainer.train(experiment.trainer.parse_args())


if __name__ == "__main__":
    main()
