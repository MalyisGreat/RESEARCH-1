from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

import rotating_anchor_collapsed_train as experiment


class CausalFenwickSemanticMemory(nn.Module):
    """Semantic attention over a dyadic decomposition of every causal prefix."""

    def __init__(self, dim: int, rank: int, minimum_scale: int = 8) -> None:
        super().__init__()
        self.rank = rank
        self.minimum_scale = minimum_scale
        self.query = nn.Linear(dim, rank, bias=False)
        self.key = nn.Linear(dim, rank, bias=False)
        self.value = nn.Linear(dim, rank, bias=False)
        self.output = nn.Linear(rank, dim, bias=False)
        self.gate = nn.Linear(dim, 1)
        self.scale_bias = nn.Parameter(torch.zeros(16))
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        query = F.normalize(self.query(x), dim=-1)
        keys = F.normalize(self.key(x), dim=-1)
        values = self.value(x)
        key_prefix = F.pad(keys.cumsum(dim=1), (0, 0, 1, 0))
        value_prefix = F.pad(values.cumsum(dim=1), (0, 0, 1, 0))
        prefix_lengths = torch.arange(1, length + 1, device=x.device)
        max_power = int(math.floor(math.log2(length)))
        scales = [1 << power for power in range(int(math.log2(self.minimum_scale)), max_power + 1)]
        key_blocks = []
        value_blocks = []
        active_blocks = []
        for scale in scales:
            active = torch.bitwise_and(prefix_lengths, scale) != 0
            end = prefix_lengths - torch.remainder(prefix_lengths, scale)
            start = (end - scale).clamp(min=0)
            key_summary = (key_prefix.index_select(1, end) - key_prefix.index_select(1, start)) / scale
            value_summary = (value_prefix.index_select(1, end) - value_prefix.index_select(1, start)) / scale
            key_blocks.append(key_summary)
            value_blocks.append(value_summary)
            active_blocks.append(active)
        block_keys = torch.stack(key_blocks, dim=2)
        block_values = torch.stack(value_blocks, dim=2)
        active = torch.stack(active_blocks, dim=1).view(1, length, len(scales))
        scores = (query.unsqueeze(2) * block_keys).sum(dim=-1) / math.sqrt(self.rank)
        scores = scores + self.scale_bias[: len(scales)].view(1, 1, -1)
        scores = scores.masked_fill(~active, -30.0)
        weights = torch.softmax(scores, dim=-1) * active.to(scores.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        retrieved = (weights.unsqueeze(-1) * block_values).sum(dim=2)
        has_memory = active.any(dim=-1, keepdim=True).to(retrieved.dtype)
        gate = torch.sigmoid(self.gate(x))
        return self.output(F.silu(retrieved)) * gate * has_memory


class FenwickMemoryBlock(nn.Module):
    def __init__(self, base: nn.Module, rank: int) -> None:
        super().__init__()
        self.base = base
        del self.base.memory_down
        del self.base.memory_depthwise
        del self.base.memory_up
        self.memory = CausalFenwickSemanticMemory(base.mix.in_features, rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base
        conv_input = base.conv_norm(x).transpose(1, 2)
        conv_output = base.collapsed_depthwise(conv_input)
        x = x + base.dropout(base.mix(F.relu(conv_output).square()))
        x = x + base.dropout(self.memory(base.memory_norm(x)))
        hidden = F.relu(base.ffn_in(base.ffn_norm(x))).square()
        return x + base.dropout(base.ffn_out(hidden))


ExistingDropIn = experiment.trainer.CausalMultiScaleLowRankConvMemoryBlock


class FenwickMemoryDropIn(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        collapsed = ExistingDropIn(**kwargs)
        rank = int(os.environ.get("FENWICK_MEMORY_RANK", "16"))
        self.block = FenwickMemoryBlock(collapsed.block, rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def benchmark(iterations: int) -> dict[str, object]:
    device = torch.device("cuda")
    kwargs = dict(dim=512, expansion=2, kernel_size=7, dilation=1, dropout=0.0, memory_rank=64, memory_kernel_size=128)
    torch.manual_seed(13)
    models = {"collapsed_wave10": ExistingDropIn(**kwargs).to(device), "fenwick_memory": FenwickMemoryDropIn(**kwargs).to(device)}
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
    results["speedup"] = results["collapsed_wave10"]["mean_step_ms"] / results["fenwick_memory"]["mean_step_ms"]
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
    experiment.trainer.CausalMultiScaleLowRankConvMemoryBlock = FenwickMemoryDropIn
    experiment.trainer.train(experiment.trainer.parse_args())


if __name__ == "__main__":
    main()
