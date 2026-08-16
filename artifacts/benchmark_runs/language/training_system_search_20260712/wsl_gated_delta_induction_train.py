from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F
from torch import nn

import fla.ops.utils.cache as fla_cache


def use_smallest_triton_config(self, key) -> None:
    def score(config) -> int:
        volume = math.prod(
            value for value in config.kwargs.values() if isinstance(value, int) and value > 0
        ) if config.kwargs else 1
        return volume * config.num_warps * config.num_stages

    self.cache[key.autotune_key] = min(self.configs, key=score)


fla_cache.FLA_CACHE_MODE = fla_cache.FlaCacheMode.ALWAYS
fla_cache.CachedAutotuner.maybe_load_cached_config = use_smallest_triton_config

from fla.ops.gated_delta_rule import chunk_gated_delta_rule

import sorted_induction_train as induction


trainer = induction.experiment.experiment.trainer
ExistingDropIn = trainer.CausalMultiScaleLowRankConvMemoryBlock


class GatedDeltaMemory(nn.Module):
    def __init__(self, dim: int, heads: int = 2, key_dim: int = 16, value_dim: int = 16) -> None:
        super().__init__()
        self.heads = heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.query = nn.Linear(dim, heads * key_dim, bias=False)
        self.key = nn.Linear(dim, heads * key_dim, bias=False)
        self.value = nn.Linear(dim, heads * value_dim, bias=False)
        self.decay = nn.Linear(dim, heads)
        self.write = nn.Linear(dim, heads)
        self.output_gate = nn.Linear(dim, heads * value_dim)
        self.output = nn.Linear(heads * value_dim, dim, bias=False)
        nn.init.constant_(self.decay.bias, 2.0)
        nn.init.zeros_(self.write.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        query = self.query(x).view(batch, length, self.heads, self.key_dim)
        key = self.key(x).view(batch, length, self.heads, self.key_dim)
        value = self.value(x).view(batch, length, self.heads, self.value_dim)
        decay = F.logsigmoid(self.decay(x))
        write = self.write(x)
        retrieved, _ = chunk_gated_delta_rule(
            query,
            key,
            value,
            decay,
            write,
            use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
        )
        retrieved = retrieved.reshape(batch, length, self.heads * self.value_dim)
        retrieved = retrieved * F.silu(self.output_gate(x))
        return self.output(retrieved)


class GatedDeltaBlock(nn.Module):
    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base
        del self.base.memory_down
        del self.base.memory_depthwise
        del self.base.memory_up
        self.memory = GatedDeltaMemory(base.mix.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base
        conv_input = base.conv_norm(x).transpose(1, 2)
        conv_output = base.collapsed_depthwise(conv_input)
        x = x + base.dropout(base.mix(F.relu(conv_output).square()))
        x = x + base.dropout(self.memory(base.memory_norm(x)))
        hidden = F.relu(base.ffn_in(base.ffn_norm(x))).square()
        return x + base.dropout(base.ffn_out(hidden))


class HybridGatedDeltaDropIn(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        collapsed = ExistingDropIn(**kwargs)
        target_dilation = int(os.environ.get("GATED_DELTA_DILATION", "2"))
        self.block = GatedDeltaBlock(collapsed.block) if kwargs["dilation"] == target_dilation else collapsed.block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


trainer.CausalMultiScaleLowRankConvMemoryBlock = HybridGatedDeltaDropIn
trainer.CausalConvFactorizedLM = induction.SortedInductionModel


if __name__ == "__main__":
    trainer.train(trainer.parse_args())
