from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import torch
from torch import nn


LANGUAGE_DIR = Path(__file__).resolve().parents[1]
if str(LANGUAGE_DIR) not in sys.path:
    sys.path.insert(0, str(LANGUAGE_DIR))

import standalone_longseq_anchor_train as trainer


def replace_layer_norms(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.LayerNorm):
            setattr(module, name, nn.RMSNorm(child.normalized_shape, eps=1e-6))
        else:
            replace_layer_norms(child)


def lecun_initialize(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=1.0 / math.sqrt(module.in_features), a=-3.0, b=3.0)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=1.0 / math.sqrt(module.embedding_dim), a=-3.0, b=3.0)
    elif isinstance(module, nn.Conv1d):
        fan_in = module.in_channels * module.kernel_size[0] / module.groups
        nn.init.trunc_normal_(module.weight, std=1.0 / math.sqrt(fan_in), a=-3.0, b=3.0)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class StableHRMTextWaveLM(nn.Module):
    """Local HRM-Text stability recipe with subquadratic H and Wave L cores."""

    def __init__(self, config: trainer.TrainConfig) -> None:
        super().__init__()
        self.high_cycles = int(os.environ.get("HRM_HIGH_CYCLES", "2"))
        self.low_cycles = int(os.environ.get("HRM_LOW_CYCLES", "3"))
        self.bp_min_steps = int(os.environ.get("HRM_BP_MIN_STEPS", "2"))
        self.bp_max_steps = int(os.environ.get("HRM_BP_MAX_STEPS", "5"))
        self.bp_warmup_ratio = float(os.environ.get("HRM_BP_WARMUP_RATIO", "0.2"))
        self.total_train_steps = int(os.environ.get("HRM_TOTAL_TRAIN_STEPS", str(config.train_steps)))
        if self.high_cycles < 1 or self.low_cycles < 1:
            raise ValueError("HRM cycles must be positive")
        if not 2 <= self.bp_min_steps <= self.bp_max_steps:
            raise ValueError("expected 2 <= bp_min_steps <= bp_max_steps")

        common = {
            "dim": config.embedding_dim,
            "expansion": 2,
            "kernel_size": config.conv_kernel_size,
            "dropout": 0.1,
        }
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.high_core = trainer.CausalLandmarkAttentionConvMixerBlock(
            **common,
            dilation=1,
            attention_heads=config.attention_heads,
            landmark_stride=config.landmark_stride,
        )
        self.low_core = trainer.CausalMultiScaleLowRankConvMemoryBlock(
            **common,
            dilation=2,
            memory_rank=config.memory_rank,
            memory_kernel_size=config.conv_kernel_size,
        )
        self.low_state_mode = os.environ.get("HRM_LOW_STATE_MODE", "independent")
        if self.low_state_mode not in {"independent", "embedded"}:
            raise ValueError("HRM_LOW_STATE_MODE must be independent or embedded")
        self.register_buffer("low_initial_state", torch.empty(config.embedding_dim), persistent=True)
        self.final_norm = nn.RMSNorm(config.embedding_dim, eps=1e-6)
        self.head_fc = nn.Linear(config.embedding_dim, 4 * config.embedding_dim)
        self.head_proj = nn.Linear(4 * config.embedding_dim, config.embedding_dim)
        self.factor_down = nn.Linear(config.embedding_dim, config.conv_rank, bias=False)
        self.factor_up = nn.Linear(config.conv_rank, config.vocab_size, bias=True)

        replace_layer_norms(self.high_core)
        replace_layer_norms(self.low_core)
        if os.environ.get("HRM_LECUN_INIT", "1") == "1":
            self.apply(lecun_initialize)
        low_init_std = float(os.environ.get("HRM_LOW_INIT_STD", "1.0"))
        nn.init.trunc_normal_(self.low_initial_state, std=low_init_std, a=-3.0, b=3.0)
        self.register_buffer("training_forward_count", torch.zeros((), dtype=torch.long), persistent=False)

    def current_bp_steps(self) -> int:
        warmup_steps = max(1, int(self.total_train_steps * self.bp_warmup_ratio))
        progress = min(1.0, int(self.training_forward_count.item()) / warmup_steps)
        return self.bp_min_steps + int(progress * (self.bp_max_steps - self.bp_min_steps))

    def features(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.training_forward_count.add_(1)
        bp_steps = self.current_bp_steps() if self.training else self.bp_max_steps
        high_bp_steps = min(self.high_cycles, bp_steps - 1)
        low_bp_steps = bp_steps - high_bp_steps

        embedded = self.embedding(input_ids)
        high = embedded
        if self.low_state_mode == "embedded":
            low = embedded
        else:
            low = self.low_initial_state.to(dtype=embedded.dtype).view(1, 1, -1).expand_as(embedded)
        total_low = self.high_cycles * self.low_cycles
        low_index = 0
        for high_index in range(self.high_cycles):
            for _ in range(self.low_cycles):
                track = torch.is_grad_enabled() and low_index >= total_low - low_bp_steps
                with torch.set_grad_enabled(track):
                    low = self.low_core(low + high)
                low_index += 1
            track = torch.is_grad_enabled() and high_index >= self.high_cycles - high_bp_steps
            with torch.set_grad_enabled(track):
                high = self.high_core(high + low)

        states = high
        head_features = torch.relu(self.head_fc(self.final_norm(states))).square()
        return self.head_proj(head_features)


trainer.CausalConvFactorizedLM = StableHRMTextWaveLM


if __name__ == "__main__":
    trainer.train(trainer.parse_args())
