from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

import collapsed_token_recall_train as experiment


class SpectralFFN(nn.Module):
    """Two learned circulant channel maps with a squared-ReLU between them."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        frequencies = dim // 2 + 1
        scale = dim**-0.5
        self.dim = dim
        self.gain_in = nn.Parameter(torch.randn(frequencies, 2) * scale)
        self.gain_out = nn.Parameter(torch.randn(frequencies, 2) * scale)
        self.diagonal_in = nn.Parameter(torch.ones(dim))
        self.diagonal_out = nn.Parameter(torch.ones(dim))

    def transform(self, x: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(x.float(), dim=-1, norm="ortho")
        complex_gain = torch.view_as_complex(gain.contiguous())
        return torch.fft.irfft(spectrum * complex_gain, n=self.dim, dim=-1, norm="ortho").to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.transform(x * self.diagonal_in, self.gain_in)
        hidden = F.relu(hidden).square()
        return self.transform(hidden, self.gain_out) * self.diagonal_out


class SpectralCollapsedBlock(nn.Module):
    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base
        dim = base.mix.in_features
        self.spectral_ffn = SpectralFFN(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base
        conv_input = base.conv_norm(x).transpose(1, 2)
        conv_output = base.collapsed_depthwise(conv_input)
        x = x + base.dropout(base.mix(F.relu(conv_output).square()))
        memory_input = base.memory_down(base.memory_norm(x)).transpose(1, 2)
        memory_output = base.memory_depthwise(F.pad(memory_input, (base.memory_left_padding, 0))).transpose(1, 2)
        x = x + base.dropout(base.memory_up(F.silu(memory_output)))
        return x + base.dropout(self.spectral_ffn(base.ffn_norm(x)))


def benchmark(model: nn.Module, x: torch.Tensor, iterations: int) -> dict[str, float | bool | int]:
    model.train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]

    def step() -> tuple[float, bool]:
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(x)
            loss = output.float().square().mean()
        loss.backward()
        torch.cuda.synchronize()
        finite = bool(torch.isfinite(output).all() and all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters))
        return time.perf_counter() - started, finite

    for _ in range(3):
        step()
    samples = [step() for _ in range(iterations)]
    times = [sample[0] for sample in samples]
    return {
        "parameters": sum(parameter.numel() for parameter in parameters),
        "mean_step_ms": statistics.fmean(times) * 1_000,
        "median_step_ms": statistics.median(times) * 1_000,
        "all_finite": all(sample[1] for sample in samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(13)
    original = experiment.CollapsedWave10DropIn(
        dim=512,
        expansion=2,
        kernel_size=7,
        dilation=1,
        dropout=0.0,
        memory_rank=64,
        memory_kernel_size=128,
    ).block.to(device)
    spectral = SpectralCollapsedBlock(
        experiment.CollapsedWave10DropIn(
            dim=512,
            expansion=2,
            kernel_size=7,
            dilation=1,
            dropout=0.0,
            memory_rank=64,
            memory_kernel_size=128,
        ).block
    ).to(device)
    x = torch.randn(1, 10_160, 512, device=device, requires_grad=True)
    payload = {
        "device": torch.cuda.get_device_name(0),
        "dense": benchmark(original, x, args.iterations),
        "spectral": benchmark(spectral, x, args.iterations),
    }
    payload["speedup"] = payload["dense"]["mean_step_ms"] / payload["spectral"]["mean_step_ms"]
    rendered = json.dumps(payload, indent=2)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered, flush=True)


if __name__ == "__main__":
    main()
