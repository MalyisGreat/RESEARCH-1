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


class CausalBlockLocalAttention(nn.Module):
    def __init__(self, dim: int, attention_dim: int, heads: int, block_size: int, dropout: float) -> None:
        super().__init__()
        if attention_dim % heads:
            raise ValueError("attention_dim must be divisible by heads")
        self.attention_dim = attention_dim
        self.heads = heads
        self.head_dim = attention_dim // heads
        self.block_size = block_size
        self.dropout = dropout
        self.qkv = nn.Linear(dim, 3 * attention_dim, bias=False)
        self.output = nn.Linear(attention_dim, dim, bias=False)
        self.gate = nn.Linear(dim, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        block = self.block_size
        block_count = (length + block - 1) // block
        padded_length = block_count * block
        if padded_length != length:
            x_padded = F.pad(x, (0, 0, 0, padded_length - length))
        else:
            x_padded = x
        qkv = self.qkv(x_padded).view(batch, block_count, block, 3, self.heads, self.head_dim)
        query, key, value = qkv.unbind(dim=3)
        zero_block = torch.zeros_like(key[:, :1])
        context_key = torch.cat((torch.cat((zero_block, key[:, :-1]), dim=1), key), dim=2)
        context_value = torch.cat((torch.cat((zero_block, value[:, :-1]), dim=1), value), dim=2)
        query = query.permute(0, 1, 3, 2, 4).reshape(batch * block_count, self.heads, block, self.head_dim)
        context_key = context_key.permute(0, 1, 3, 2, 4).reshape(
            batch * block_count, self.heads, 2 * block, self.head_dim
        )
        context_value = context_value.permute(0, 1, 3, 2, 4).reshape(
            batch * block_count, self.heads, 2 * block, self.head_dim
        )
        block_starts = torch.arange(block_count, device=x.device) * block
        query_positions = block_starts[:, None] + torch.arange(block, device=x.device)[None, :]
        key_positions = block_starts[:, None] - block + torch.arange(2 * block, device=x.device)[None, :]
        mask = (key_positions[:, None, :] >= 0) & (key_positions[:, None, :] < length)
        mask = mask & (key_positions[:, None, :] <= query_positions[:, :, None])
        mask = mask.repeat(batch, 1, 1).unsqueeze(1)
        attended = F.scaled_dot_product_attention(
            query,
            context_key,
            context_value,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.view(batch, block_count, self.heads, block, self.head_dim)
        attended = attended.permute(0, 1, 3, 2, 4).reshape(batch, padded_length, self.attention_dim)[:, :length]
        return self.output(attended) * torch.sigmoid(self.gate(x))


class LocalAttentionMemoryBlock(nn.Module):
    def __init__(self, base: nn.Module, attention_dim: int, heads: int, block_size: int) -> None:
        super().__init__()
        self.base = base
        self.keep_original_memory = os.environ.get("LOCAL_ATTENTION_KEEP_MEMORY", "0") == "1"
        if not self.keep_original_memory:
            del self.base.memory_down
            del self.base.memory_depthwise
            del self.base.memory_up
        self.attention = CausalBlockLocalAttention(
            base.mix.in_features, attention_dim, heads, block_size, base.dropout.p
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base
        conv_input = base.conv_norm(x).transpose(1, 2)
        conv_output = base.collapsed_depthwise(conv_input)
        x = x + base.dropout(base.mix(F.relu(conv_output).square()))
        if self.keep_original_memory:
            memory_input = base.memory_down(base.memory_norm(x)).transpose(1, 2)
            memory_output = base.memory_depthwise(
                F.pad(memory_input, (base.memory_left_padding, 0))
            ).transpose(1, 2)
            x = x + base.dropout(base.memory_up(F.silu(memory_output)))
        x = x + base.dropout(self.attention(base.memory_norm(x)))
        hidden = F.relu(base.ffn_in(base.ffn_norm(x))).square()
        return x + base.dropout(base.ffn_out(hidden))


ExistingDropIn = experiment.trainer.CausalMultiScaleLowRankConvMemoryBlock


class LocalAttentionDropIn(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        collapsed = ExistingDropIn(**kwargs)
        target_dilation = int(os.environ.get("LOCAL_ATTENTION_ONLY_DILATION", "0"))
        if target_dilation and kwargs["dilation"] != target_dilation:
            self.block = collapsed.block
            return
        attention_dim = int(os.environ.get("LOCAL_ATTENTION_DIM", "64"))
        heads = int(os.environ.get("LOCAL_ATTENTION_HEADS", "4"))
        block_size = int(os.environ.get("LOCAL_ATTENTION_BLOCK", "64"))
        self.block = LocalAttentionMemoryBlock(collapsed.block, attention_dim, heads, block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def benchmark(iterations: int) -> dict[str, object]:
    device = torch.device("cuda")
    kwargs = dict(dim=512, expansion=2, kernel_size=7, dilation=1, dropout=0.0, memory_rank=64, memory_kernel_size=128)
    torch.manual_seed(13)
    models = {"collapsed_wave10": ExistingDropIn(**kwargs).to(device), "local_attention": LocalAttentionDropIn(**kwargs).to(device)}
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
    results["speedup"] = results["collapsed_wave10"]["mean_step_ms"] / results["local_attention"]["mean_step_ms"]
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
    experiment.trainer.CausalMultiScaleLowRankConvMemoryBlock = LocalAttentionDropIn
    experiment.trainer.train(experiment.trainer.parse_args())


if __name__ == "__main__":
    main()
