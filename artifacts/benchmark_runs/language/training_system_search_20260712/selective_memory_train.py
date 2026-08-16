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

import sorted_induction_train as experiment


trainer = experiment.experiment.experiment.trainer
ExistingDropIn = trainer.CausalMultiScaleLowRankConvMemoryBlock


class SelectiveMemoryBlock(nn.Module):
    """Wave10 with content-conditioned write and read control in memory rank space."""

    def __init__(self, base: nn.Module, dim: int, memory_rank: int) -> None:
        super().__init__()
        self.base = base
        self.gate_mode = os.environ.get("SELECTIVE_MEMORY_GATE_MODE", "rankwise")
        if self.gate_mode == "rankwise":
            # Keep every downstream baseline parameter bit-identical under a matched seed.
            rng_state = torch.random.get_rng_state()
            self.write_gate = nn.Linear(dim, memory_rank)
            self.read_gate = nn.Linear(dim, memory_rank)
            torch.random.set_rng_state(rng_state)
        elif self.gate_mode == "self":
            self.write_scale = nn.Parameter(torch.zeros(memory_rank))
            self.read_scale = nn.Parameter(torch.zeros(memory_rank))
        else:
            raise ValueError("SELECTIVE_MEMORY_GATE_MODE must be rankwise or self")
        initial_probability = float(os.environ.get("SELECTIVE_MEMORY_INITIAL_GATE", "0.9"))
        if not 0.0 < initial_probability < 1.0:
            raise ValueError("SELECTIVE_MEMORY_INITIAL_GATE must be between zero and one")
        self.gate_normalizer = initial_probability
        initial_bias = math.log(initial_probability / (1.0 - initial_probability))
        if self.gate_mode == "rankwise":
            nn.init.zeros_(self.write_gate.weight)
            nn.init.zeros_(self.read_gate.weight)
            nn.init.constant_(self.write_gate.bias, initial_bias)
            nn.init.constant_(self.read_gate.bias, initial_bias)
        else:
            self.write_bias = nn.Parameter(torch.full((memory_rank,), initial_bias))
            self.read_bias = nn.Parameter(torch.full((memory_rank,), initial_bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base
        conv_input = base.conv_norm(x).transpose(1, 2)
        conv_output = base.collapsed_depthwise(conv_input)
        x = x + base.dropout(base.mix(F.relu(conv_output).square()))

        normalized = base.memory_norm(x)
        memory_projection = base.memory_down(normalized)
        if self.gate_mode == "rankwise":
            write = torch.sigmoid(self.write_gate(normalized)) / self.gate_normalizer
        else:
            write = torch.sigmoid(
                memory_projection * self.write_scale + self.write_bias
            ) / self.gate_normalizer
        memory_input = (memory_projection * write).transpose(1, 2)
        memory_output = base.memory_depthwise(
            F.pad(memory_input, (base.memory_left_padding, 0))
        ).transpose(1, 2)
        if self.gate_mode == "rankwise":
            read = torch.sigmoid(self.read_gate(normalized)) / self.gate_normalizer
        else:
            read = torch.sigmoid(memory_output * self.read_scale + self.read_bias) / self.gate_normalizer
        x = x + base.dropout(base.memory_up(F.silu(memory_output) * read))

        hidden = F.relu(base.ffn_in(base.ffn_norm(x))).square()
        return x + base.dropout(base.ffn_out(hidden))


class SelectiveMemoryDropIn(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        collapsed = ExistingDropIn(**kwargs)
        target_dilation = int(os.environ.get("SELECTIVE_MEMORY_ONLY_DILATION", "0"))
        if target_dilation and kwargs["dilation"] != target_dilation:
            self.block = collapsed.block
        else:
            self.block = SelectiveMemoryBlock(collapsed.block, kwargs["dim"], kwargs["memory_rank"])

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
        "selective_memory": SelectiveMemoryDropIn(**kwargs).to(device),
    }
    sample = torch.randn(1, 10_160, 512, device=device)
    results: dict[str, object] = {}
    for name, model in models.items():
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]

        def step() -> tuple[float, bool]:
            model.zero_grad(set_to_none=True)
            local_sample = sample.detach().requires_grad_(True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(local_sample)
                loss = output.float().square().mean()
            loss.backward()
            torch.cuda.synchronize()
            finite = bool(torch.isfinite(output).all() and torch.isfinite(local_sample.grad).all())
            return time.perf_counter() - started, finite

        for _ in range(3):
            step()
        samples = [step() for _ in range(iterations)]
        times = [sample_result[0] for sample_result in samples]
        results[name] = {
            "parameters": sum(parameter.numel() for parameter in parameters),
            "mean_step_ms": statistics.fmean(times) * 1_000,
            "median_step_ms": statistics.median(times) * 1_000,
            "all_finite": all(sample_result[1] for sample_result in samples),
            "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20,
        }
        torch.cuda.reset_peak_memory_stats()
        del model
        torch.cuda.empty_cache()
    baseline_ms = float(results["collapsed_wave10"]["mean_step_ms"])
    candidate_ms = float(results["selective_memory"]["mean_step_ms"])
    results["baseline_over_candidate_speed"] = baseline_ms / candidate_ms
    return results


def main() -> None:
    if "--probe-output" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--probe-output", type=Path, required=True)
        parser.add_argument("--iterations", type=int, default=10)
        args = parser.parse_args()
        payload = {"device": torch.cuda.get_device_name(0), "benchmark": benchmark(args.iterations)}
        args.probe_output.parent.mkdir(parents=True, exist_ok=True)
        rendered = json.dumps(payload, indent=2)
        args.probe_output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered, flush=True)
        return
    trainer.CausalMultiScaleLowRankConvMemoryBlock = SelectiveMemoryDropIn
    trainer.train(trainer.parse_args())


if __name__ == "__main__":
    main()
