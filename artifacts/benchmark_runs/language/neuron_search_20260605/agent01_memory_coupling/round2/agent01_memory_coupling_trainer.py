from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class TrainConfig:
    cache_path: Path
    output_dir: Path
    run_name: str
    vocab_size: int = 50_257
    sequence_length: int = 10_160
    batch_size: int = 1
    train_steps: int = 196_851
    eval_interval: int = 9_843
    val_blocks: int = 32
    checkpoint_interval: int = 9_843
    milestone_checkpoint_interval: int = 59_058
    seed: int = 13
    embedding_dim: int = 1_831
    block_type: str = "relu_square"
    conv_layers: int = 1
    conv_rank: int = 531
    conv_kernel_size: int = 7
    memory_rank: int = 64
    attention_heads: int = 4
    landmark_stride: int = 128
    sampled_vocab_size: int = 16_384
    token_stride: int = 16
    token_chunk_size: int = 1_024
    full_eval_token_chunk_size: int = 1_024
    learning_rate: float = 6e-4
    min_learning_rate: float = 6e-5
    warmup_steps: int = 2_000
    weight_decay: float = 1e-4
    amp_dtype: str = "fp16"
    resume_checkpoint: Path | None = None


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


class CausalConvMixerBlock(nn.Module):
    def __init__(self, *, dim: int, expansion: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv_norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(dim, dim, kernel_size=kernel_size, dilation=dilation, groups=dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn_in = nn.Linear(dim, expansion * dim)
        self.ffn_out = nn.Linear(expansion * dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_input = self.conv_norm(x).transpose(1, 2)
        conv_output = self.depthwise(F.pad(conv_input, (self.left_padding, 0))).transpose(1, 2)
        x = x + self.dropout(F.relu(conv_output).square())
        hidden = F.relu(self.ffn_in(self.ffn_norm(x))).square()
        return x + self.dropout(self.ffn_out(hidden))


class CausalGatedConvMixerBlock(nn.Module):
    def __init__(self, *, dim: int, expansion: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv_norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(dim, 2 * dim, kernel_size=kernel_size, dilation=dilation, groups=dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn_in = nn.Linear(dim, 2 * expansion * dim)
        self.ffn_out = nn.Linear(expansion * dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_input = self.conv_norm(x).transpose(1, 2)
        conv_output = self.depthwise(F.pad(conv_input, (self.left_padding, 0))).transpose(1, 2)
        conv_value, conv_gate = conv_output.chunk(2, dim=-1)
        x = x + self.dropout(F.silu(conv_value) * torch.sigmoid(conv_gate))
        hidden, gate = self.ffn_in(self.ffn_norm(x)).chunk(2, dim=-1)
        return x + self.dropout(self.ffn_out(F.silu(hidden) * torch.sigmoid(gate)))


class CausalMemoryConvMixerBlock(nn.Module):
    def __init__(
        self, *, dim: int, expansion: int, kernel_size: int, dilation: int, dropout: float, memory_rank: int
    ) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv_norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(dim, dim, kernel_size=kernel_size, dilation=dilation, groups=dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.memory_down = nn.Linear(dim, memory_rank, bias=False)
        self.memory_up = nn.Linear(memory_rank, dim, bias=False)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn_in = nn.Linear(dim, expansion * dim)
        self.ffn_out = nn.Linear(expansion * dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_input = self.conv_norm(x).transpose(1, 2)
        conv_output = self.depthwise(F.pad(conv_input, (self.left_padding, 0))).transpose(1, 2)
        x = x + self.dropout(F.relu(conv_output).square())

        memory_source = self.memory_norm(x)
        positions = torch.arange(1, x.size(1) + 1, device=x.device, dtype=memory_source.dtype).view(1, -1, 1)
        causal_mean = memory_source.cumsum(dim=1) / positions
        x = x + self.dropout(self.memory_up(F.silu(self.memory_down(causal_mean))))

        hidden = F.relu(self.ffn_in(self.ffn_norm(x))).square()
        return x + self.dropout(self.ffn_out(hidden))


class CausalLandmarkAttentionConvMixerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        expansion: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        attention_heads: int,
        landmark_stride: int,
    ) -> None:
        super().__init__()
        if dim % attention_heads != 0:
            raise ValueError(f"embedding dim {dim} must divide attention heads {attention_heads}")
        self.left_padding = (kernel_size - 1) * dilation
        self.landmark_stride = max(1, landmark_stride)
        self.attention_heads = attention_heads
        self.head_dim = dim // attention_heads
        self.conv_norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(dim, dim, kernel_size=kernel_size, dilation=dilation, groups=dim)
        self.attn_norm = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn_in = nn.Linear(dim, expansion * dim)
        self.ffn_out = nn.Linear(expansion * dim, dim)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = tensor.shape
        return tensor.view(batch, tokens, self.attention_heads, dim // self.attention_heads).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_input = self.conv_norm(x).transpose(1, 2)
        conv_output = self.depthwise(F.pad(conv_input, (self.left_padding, 0))).transpose(1, 2)
        x = x + self.dropout(F.relu(conv_output).square())

        attn_source = self.attn_norm(x)
        token_count = attn_source.size(1)
        prefix_positions = torch.arange(1, token_count + 1, device=x.device, dtype=attn_source.dtype).view(1, -1, 1)
        prefix_mean = attn_source.cumsum(dim=1) / prefix_positions
        landmark_positions = torch.arange(0, token_count, self.landmark_stride, device=x.device)
        if landmark_positions[-1] != token_count - 1:
            landmark_positions = torch.cat((landmark_positions, landmark_positions.new_tensor([token_count - 1])))
        landmarks = prefix_mean.index_select(1, landmark_positions)

        queries = self.split_heads(self.query(attn_source))
        keys = self.split_heads(self.key(landmarks))
        values = self.split_heads(self.value(landmarks))
        logits = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        token_positions = torch.arange(token_count, device=x.device).view(1, 1, token_count, 1)
        causal_mask = landmark_positions.view(1, 1, 1, -1) <= token_positions
        logits = logits.masked_fill(~causal_mask, torch.finfo(logits.dtype).min)
        attended = torch.matmul(torch.softmax(logits, dim=-1), values)
        attended = attended.transpose(1, 2).contiguous().view_as(x)
        x = x + self.dropout(self.attn_out(attended))

        hidden = F.relu(self.ffn_in(self.ffn_norm(x))).square()
        return x + self.dropout(self.ffn_out(hidden))


class CausalMultiScaleConvMixerBlock(nn.Module):
    def __init__(self, *, dim: int, expansion: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        kernels = tuple(dict.fromkeys((3, kernel_size, 2 * kernel_size + 1)))
        self.left_paddings = [(kernel - 1) * dilation for kernel in kernels]
        self.conv_norm = nn.LayerNorm(dim)
        self.depthwise_layers = nn.ModuleList(
            [nn.Conv1d(dim, dim, kernel_size=kernel, dilation=dilation, groups=dim) for kernel in kernels]
        )
        self.mix = nn.Linear(dim, dim, bias=False)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn_in = nn.Linear(dim, expansion * dim)
        self.ffn_out = nn.Linear(expansion * dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_input = self.conv_norm(x).transpose(1, 2)
        outputs = [
            layer(F.pad(conv_input, (left_padding, 0))).transpose(1, 2)
            for layer, left_padding in zip(self.depthwise_layers, self.left_paddings)
        ]
        conv_output = torch.stack(outputs, dim=0).mean(dim=0)
        x = x + self.dropout(self.mix(F.relu(conv_output).square()))
        hidden = F.relu(self.ffn_in(self.ffn_norm(x))).square()
        return x + self.dropout(self.ffn_out(hidden))


class CausalAdaptiveMultiScaleConvMixerBlock(nn.Module):
    def __init__(self, *, dim: int, expansion: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        kernels = tuple(dict.fromkeys((3, kernel_size, 2 * kernel_size + 1)))
        self.left_paddings = [(kernel - 1) * dilation for kernel in kernels]
        self.conv_norm = nn.LayerNorm(dim)
        self.depthwise_layers = nn.ModuleList(
            [nn.Conv1d(dim, dim, kernel_size=kernel, dilation=dilation, groups=dim) for kernel in kernels]
        )
        self.branch_logits = nn.Parameter(torch.zeros(len(kernels), dim))
        self.mix = nn.Linear(dim, dim, bias=False)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn_in = nn.Linear(dim, expansion * dim)
        self.ffn_out = nn.Linear(expansion * dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_input = self.conv_norm(x).transpose(1, 2)
        outputs = [
            layer(F.pad(conv_input, (left_padding, 0))).transpose(1, 2)
            for layer, left_padding in zip(self.depthwise_layers, self.left_paddings)
        ]
        weights = torch.softmax(self.branch_logits, dim=0).view(len(outputs), 1, 1, -1)
        conv_output = (torch.stack(outputs, dim=0) * weights).sum(dim=0)
        x = x + self.dropout(self.mix(F.relu(conv_output).square()))
        hidden = F.relu(self.ffn_in(self.ffn_norm(x))).square()
        return x + self.dropout(self.ffn_out(hidden))


class CausalTokenGatedMultiScaleConvMixerBlock(nn.Module):
    def __init__(self, *, dim: int, expansion: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        kernels = tuple(dict.fromkeys((3, kernel_size, 2 * kernel_size + 1)))
        self.left_paddings = [(kernel - 1) * dilation for kernel in kernels]
        self.conv_norm = nn.LayerNorm(dim)
        self.depthwise_layers = nn.ModuleList(
            [nn.Conv1d(dim, dim, kernel_size=kernel, dilation=dilation, groups=dim) for kernel in kernels]
        )
        self.gate_norm = nn.LayerNorm(dim)
        self.branch_gate = nn.Linear(dim, len(kernels))
        self.mix = nn.Linear(dim, dim, bias=False)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn_in = nn.Linear(dim, expansion * dim)
        self.ffn_out = nn.Linear(expansion * dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_input = self.conv_norm(x).transpose(1, 2)
        outputs = [
            layer(F.pad(conv_input, (left_padding, 0))).transpose(1, 2)
            for layer, left_padding in zip(self.depthwise_layers, self.left_paddings)
        ]
        branch_outputs = torch.stack(outputs, dim=2)
        branch_weights = torch.softmax(self.branch_gate(self.gate_norm(x)), dim=-1).unsqueeze(-1)
        conv_output = (branch_outputs * branch_weights).sum(dim=2)
        x = x + self.dropout(self.mix(F.relu(conv_output).square()))
        hidden = F.relu(self.ffn_in(self.ffn_norm(x))).square()
        return x + self.dropout(self.ffn_out(hidden))


class CausalTokenGatedLowRankConvMemoryBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        expansion: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        memory_rank: int,
        memory_kernel_size: int,
    ) -> None:
        super().__init__()
        kernels = tuple(dict.fromkeys((3, kernel_size, 2 * kernel_size + 1)))
        self.left_paddings = [(kernel - 1) * dilation for kernel in kernels]
        self.memory_left_padding = max(1, memory_kernel_size) - 1
        self.conv_norm = nn.LayerNorm(dim)
        self.depthwise_layers = nn.ModuleList(
            [nn.Conv1d(dim, dim, kernel_size=kernel, dilation=dilation, groups=dim) for kernel in kernels]
        )
        self.gate_norm = nn.LayerNorm(dim)
        self.branch_gate = nn.Linear(dim, len(kernels))
        self.mix = nn.Linear(dim, dim, bias=False)
        self.memory_norm = nn.LayerNorm(dim)
        self.memory_down = nn.Linear(dim, memory_rank, bias=False)
        self.memory_depthwise = nn.Conv1d(
            memory_rank,
            memory_rank,
            kernel_size=max(1, memory_kernel_size),
            groups=memory_rank,
            bias=False,
        )
        self.memory_up = nn.Linear(memory_rank, dim, bias=False)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn_in = nn.Linear(dim, expansion * dim)
        self.ffn_out = nn.Linear(expansion * dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_input = self.conv_norm(x).transpose(1, 2)
        outputs = [
            layer(F.pad(conv_input, (left_padding, 0))).transpose(1, 2)
            for layer, left_padding in zip(self.depthwise_layers, self.left_paddings)
        ]
        branch_outputs = torch.stack(outputs, dim=2)
        branch_weights = torch.softmax(self.branch_gate(self.gate_norm(x)), dim=-1).unsqueeze(-1)
        conv_output = (branch_outputs * branch_weights).sum(dim=2)
        x = x + self.dropout(self.mix(F.relu(conv_output).square()))

        memory_input = self.memory_down(self.memory_norm(x)).transpose(1, 2)
        memory_output = self.memory_depthwise(F.pad(memory_input, (self.memory_left_padding, 0))).transpose(1, 2)
        x = x + self.dropout(self.memory_up(F.silu(memory_output)))

        hidden = F.relu(self.ffn_in(self.ffn_norm(x))).square()
        return x + self.dropout(self.ffn_out(hidden))


class CausalMultiScaleMemoryConvMixerBlock(nn.Module):
    def __init__(
        self, *, dim: int, expansion: int, kernel_size: int, dilation: int, dropout: float, memory_rank: int
    ) -> None:
        super().__init__()
        kernels = tuple(dict.fromkeys((3, kernel_size, 2 * kernel_size + 1)))
        self.left_paddings = [(kernel - 1) * dilation for kernel in kernels]
        self.conv_norm = nn.LayerNorm(dim)
        self.depthwise_layers = nn.ModuleList(
            [nn.Conv1d(dim, dim, kernel_size=kernel, dilation=dilation, groups=dim) for kernel in kernels]
        )
        self.mix = nn.Linear(dim, dim, bias=False)
        self.memory_norm = nn.LayerNorm(dim)
        self.memory_down = nn.Linear(dim, memory_rank, bias=False)
        self.memory_up = nn.Linear(memory_rank, dim, bias=False)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn_in = nn.Linear(dim, expansion * dim)
        self.ffn_out = nn.Linear(expansion * dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_input = self.conv_norm(x).transpose(1, 2)
        outputs = [
            layer(F.pad(conv_input, (left_padding, 0))).transpose(1, 2)
            for layer, left_padding in zip(self.depthwise_layers, self.left_paddings)
        ]
        conv_output = torch.stack(outputs, dim=0).mean(dim=0)
        x = x + self.dropout(self.mix(F.relu(conv_output).square()))

        memory_source = self.memory_norm(x)
        positions = torch.arange(1, x.size(1) + 1, device=x.device, dtype=memory_source.dtype).view(1, -1, 1)
        causal_mean = memory_source.cumsum(dim=1) / positions
        x = x + self.dropout(self.memory_up(F.silu(self.memory_down(causal_mean))))

        hidden = F.relu(self.ffn_in(self.ffn_norm(x))).square()
        return x + self.dropout(self.ffn_out(hidden))


class CausalMultiScaleLowRankConvMemoryBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        expansion: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        memory_rank: int,
        memory_kernel_size: int,
    ) -> None:
        super().__init__()
        kernels = tuple(dict.fromkeys((3, kernel_size, 2 * kernel_size + 1)))
        self.left_paddings = [(kernel - 1) * dilation for kernel in kernels]
        self.memory_left_padding = max(1, memory_kernel_size) - 1
        self.conv_norm = nn.LayerNorm(dim)
        self.depthwise_layers = nn.ModuleList(
            [nn.Conv1d(dim, dim, kernel_size=kernel, dilation=dilation, groups=dim) for kernel in kernels]
        )
        self.mix = nn.Linear(dim, dim, bias=False)
        self.memory_norm = nn.LayerNorm(dim)
        self.memory_down = nn.Linear(dim, memory_rank, bias=False)
        self.memory_depthwise = nn.Conv1d(
            memory_rank,
            memory_rank,
            kernel_size=max(1, memory_kernel_size),
            groups=memory_rank,
            bias=False,
        )
        self.memory_up = nn.Linear(memory_rank, dim, bias=False)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn_in = nn.Linear(dim, expansion * dim)
        self.ffn_out = nn.Linear(expansion * dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_input = self.conv_norm(x).transpose(1, 2)
        outputs = [
            layer(F.pad(conv_input, (left_padding, 0))).transpose(1, 2)
            for layer, left_padding in zip(self.depthwise_layers, self.left_paddings)
        ]
        conv_output = torch.stack(outputs, dim=0).mean(dim=0)
        x = x + self.dropout(self.mix(F.relu(conv_output).square()))

        memory_input = self.memory_down(self.memory_norm(x)).transpose(1, 2)
        memory_output = self.memory_depthwise(F.pad(memory_input, (self.memory_left_padding, 0))).transpose(1, 2)
        x = x + self.dropout(self.memory_up(F.silu(memory_output)))

        hidden = F.relu(self.ffn_in(self.ffn_norm(x))).square()
        return x + self.dropout(self.ffn_out(hidden))


class CausalMultiScaleLowRankConvMemorySwiGLUBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        expansion: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        memory_rank: int,
        memory_kernel_size: int,
    ) -> None:
        super().__init__()
        kernels = tuple(dict.fromkeys((3, kernel_size, 2 * kernel_size + 1)))
        self.left_paddings = [(kernel - 1) * dilation for kernel in kernels]
        self.memory_left_padding = max(1, memory_kernel_size) - 1
        self.conv_norm = nn.LayerNorm(dim)
        self.depthwise_layers = nn.ModuleList(
            [nn.Conv1d(dim, dim, kernel_size=kernel, dilation=dilation, groups=dim) for kernel in kernels]
        )
        self.mix = nn.Linear(dim, dim, bias=False)
        self.memory_norm = nn.LayerNorm(dim)
        self.memory_down = nn.Linear(dim, memory_rank, bias=False)
        self.memory_depthwise = nn.Conv1d(
            memory_rank,
            memory_rank,
            kernel_size=max(1, memory_kernel_size),
            groups=memory_rank,
            bias=False,
        )
        self.memory_up = nn.Linear(memory_rank, dim, bias=False)
        self.ffn_norm = nn.LayerNorm(dim)
        hidden_dim = max(1, (2 * expansion * dim) // 3)
        self.ffn_value = nn.Linear(dim, hidden_dim)
        self.ffn_gate = nn.Linear(dim, hidden_dim)
        self.ffn_out = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_input = self.conv_norm(x).transpose(1, 2)
        outputs = [
            layer(F.pad(conv_input, (left_padding, 0))).transpose(1, 2)
            for layer, left_padding in zip(self.depthwise_layers, self.left_paddings)
        ]
        conv_output = torch.stack(outputs, dim=0).mean(dim=0)
        x = x + self.dropout(self.mix(F.relu(conv_output).square()))

        memory_input = self.memory_down(self.memory_norm(x)).transpose(1, 2)
        memory_output = self.memory_depthwise(F.pad(memory_input, (self.memory_left_padding, 0))).transpose(1, 2)
        x = x + self.dropout(self.memory_up(F.silu(memory_output)))

        ffn_input = self.ffn_norm(x)
        hidden = F.silu(self.ffn_value(ffn_input)) * self.ffn_gate(ffn_input)
        return x + self.dropout(self.ffn_out(hidden))


class CausalMultiScaleLowRankConvMemorySiLUSquareBlock(CausalMultiScaleLowRankConvMemoryBlock):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_input = self.conv_norm(x).transpose(1, 2)
        outputs = [
            layer(F.pad(conv_input, (left_padding, 0))).transpose(1, 2)
            for layer, left_padding in zip(self.depthwise_layers, self.left_paddings)
        ]
        conv_output = torch.stack(outputs, dim=0).mean(dim=0)
        x = x + self.dropout(self.mix(F.relu(conv_output).square()))

        memory_input = self.memory_down(self.memory_norm(x)).transpose(1, 2)
        memory_output = self.memory_depthwise(F.pad(memory_input, (self.memory_left_padding, 0))).transpose(1, 2)
        x = x + self.dropout(self.memory_up(F.silu(memory_output)))

        hidden = F.silu(self.ffn_in(self.ffn_norm(x))).square()
        return x + self.dropout(self.ffn_out(hidden))


class CausalMemoryThresholdBasisLowRankBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        expansion: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        memory_rank: int,
        memory_kernel_size: int,
    ) -> None:
        super().__init__()
        kernels = tuple(dict.fromkeys((3, kernel_size, 2 * kernel_size + 1)))
        self.left_paddings = [(kernel - 1) * dilation for kernel in kernels]
        self.memory_left_padding = max(1, memory_kernel_size) - 1
        hidden_dim = expansion * dim
        self.conv_norm = nn.LayerNorm(dim)
        self.depthwise_layers = nn.ModuleList(
            [nn.Conv1d(dim, dim, kernel_size=kernel, dilation=dilation, groups=dim) for kernel in kernels]
        )
        self.mix = nn.Linear(dim, dim, bias=False)
        self.memory_norm = nn.LayerNorm(dim)
        self.memory_down = nn.Linear(dim, memory_rank, bias=False)
        self.memory_depthwise = nn.Conv1d(
            memory_rank,
            memory_rank,
            kernel_size=max(1, memory_kernel_size),
            groups=memory_rank,
            bias=False,
        )
        self.memory_up = nn.Linear(memory_rank, dim, bias=False)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn_in = nn.Linear(dim, hidden_dim)
        self.memory_threshold = nn.Linear(memory_rank, hidden_dim)
        self.memory_basis = nn.Linear(memory_rank, hidden_dim, bias=False)
        self.ffn_out = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_input = self.conv_norm(x).transpose(1, 2)
        outputs = [
            layer(F.pad(conv_input, (left_padding, 0))).transpose(1, 2)
            for layer, left_padding in zip(self.depthwise_layers, self.left_paddings)
        ]
        conv_output = torch.stack(outputs, dim=0).mean(dim=0)
        x = x + self.dropout(self.mix(F.relu(conv_output).square()))

        memory_input = self.memory_down(self.memory_norm(x)).transpose(1, 2)
        memory_output = self.memory_depthwise(F.pad(memory_input, (self.memory_left_padding, 0))).transpose(1, 2)
        memory_features = F.silu(memory_output)
        x = x + self.dropout(self.memory_up(memory_features))

        ffn_input = self.ffn_norm(x)
        preact = self.ffn_in(ffn_input)
        threshold = 0.25 * torch.tanh(self.memory_threshold(memory_features))
        basis_shift = 0.1 * self.memory_basis(memory_features)
        hidden = F.relu(preact - threshold + basis_shift).square()
        return x + self.dropout(self.ffn_out(hidden))


class CausalDilatedMultiScaleConvMixerBlock(nn.Module):
    def __init__(self, *, dim: int, expansion: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        branch_specs = tuple(
            dict.fromkeys(
                (
                    (3, dilation),
                    (kernel_size, dilation),
                    (kernel_size, 2 * dilation),
                    (kernel_size, 4 * dilation),
                )
            )
        )
        self.left_paddings = [(kernel - 1) * branch_dilation for kernel, branch_dilation in branch_specs]
        self.conv_norm = nn.LayerNorm(dim)
        self.depthwise_layers = nn.ModuleList(
            [
                nn.Conv1d(dim, dim, kernel_size=kernel, dilation=branch_dilation, groups=dim)
                for kernel, branch_dilation in branch_specs
            ]
        )
        self.mix = nn.Linear(dim, dim, bias=False)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn_in = nn.Linear(dim, expansion * dim)
        self.ffn_out = nn.Linear(expansion * dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_input = self.conv_norm(x).transpose(1, 2)
        outputs = [
            layer(F.pad(conv_input, (left_padding, 0))).transpose(1, 2)
            for layer, left_padding in zip(self.depthwise_layers, self.left_paddings)
        ]
        conv_output = torch.stack(outputs, dim=0).mean(dim=0)
        x = x + self.dropout(self.mix(F.relu(conv_output).square()))
        hidden = F.relu(self.ffn_in(self.ffn_norm(x))).square()
        return x + self.dropout(self.ffn_out(hidden))


def make_mixer_block(config: TrainConfig, layer_index: int) -> nn.Module:
    kwargs = {
        "dim": config.embedding_dim,
        "expansion": 2,
        "kernel_size": config.conv_kernel_size,
        "dilation": 2 ** (layer_index % 6),
        "dropout": 0.1,
    }
    if config.block_type == "relu_square":
        return CausalConvMixerBlock(**kwargs)
    if config.block_type == "gated":
        return CausalGatedConvMixerBlock(**kwargs)
    if config.block_type == "memory":
        return CausalMemoryConvMixerBlock(**kwargs, memory_rank=config.memory_rank)
    if config.block_type == "landmark_attention":
        return CausalLandmarkAttentionConvMixerBlock(
            **kwargs,
            attention_heads=config.attention_heads,
            landmark_stride=config.landmark_stride,
        )
    if config.block_type == "multi_scale":
        return CausalMultiScaleConvMixerBlock(**kwargs)
    if config.block_type == "adaptive_multi_scale":
        return CausalAdaptiveMultiScaleConvMixerBlock(**kwargs)
    if config.block_type == "token_gated_multi_scale":
        return CausalTokenGatedMultiScaleConvMixerBlock(**kwargs)
    if config.block_type == "multi_scale_memory":
        return CausalMultiScaleMemoryConvMixerBlock(**kwargs, memory_rank=config.memory_rank)
    if config.block_type == "multi_scale_lowrank_conv_memory":
        return CausalMultiScaleLowRankConvMemoryBlock(
            **kwargs,
            memory_rank=config.memory_rank,
            memory_kernel_size=config.landmark_stride,
        )
    if config.block_type == "multi_scale_lowrank_conv_memory_swiglu":
        return CausalMultiScaleLowRankConvMemorySwiGLUBlock(
            **kwargs,
            memory_rank=config.memory_rank,
            memory_kernel_size=config.landmark_stride,
        )
    if config.block_type == "multi_scale_lowrank_conv_memory_silu_square":
        return CausalMultiScaleLowRankConvMemorySiLUSquareBlock(
            **kwargs,
            memory_rank=config.memory_rank,
            memory_kernel_size=config.landmark_stride,
        )
    if config.block_type == "memory_threshold_basis_lowrank":
        return CausalMemoryThresholdBasisLowRankBlock(
            **kwargs,
            memory_rank=config.memory_rank,
            memory_kernel_size=config.landmark_stride,
        )
    if config.block_type == "token_gated_lowrank_conv_memory":
        return CausalTokenGatedLowRankConvMemoryBlock(
            **kwargs,
            memory_rank=config.memory_rank,
            memory_kernel_size=config.landmark_stride,
        )
    if config.block_type == "dilated_multi_scale":
        return CausalDilatedMultiScaleConvMixerBlock(**kwargs)
    raise ValueError(f"unknown block_type {config.block_type!r}")


class CausalConvFactorizedLM(nn.Module):
    def __init__(self, config: TrainConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.blocks = nn.ModuleList(
            [
                make_mixer_block(config, layer_index)
                for layer_index in range(config.conv_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        self.head_fc = nn.Linear(config.embedding_dim, 4 * config.embedding_dim)
        self.head_proj = nn.Linear(4 * config.embedding_dim, config.embedding_dim)
        self.factor_down = nn.Linear(config.embedding_dim, config.conv_rank, bias=False)
        self.factor_up = nn.Linear(config.conv_rank, config.vocab_size, bias=True)

    def features(self, input_ids: torch.Tensor) -> torch.Tensor:
        states = self.embedding(input_ids)
        for block in self.blocks:
            states = block(states)
        head_features = F.relu(self.head_fc(self.final_norm(states))).square()
        return self.head_proj(head_features)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def scheduled_learning_rate(config: TrainConfig, step: int) -> float:
    if step <= 0:
        return 0.0 if config.warmup_steps > 0 else config.learning_rate
    if config.warmup_steps > 0 and step <= config.warmup_steps:
        return config.learning_rate * (step / config.warmup_steps)
    decay_steps = max(config.train_steps - config.warmup_steps, 1)
    progress = min(max((step - config.warmup_steps) / decay_steps, 0.0), 1.0)
    decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + (config.learning_rate - config.min_learning_rate) * decay


def set_optimizer_lr(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def load_cache(config: TrainConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    payload = torch.load(config.cache_path, map_location="cpu", weights_only=False)
    train_tokens = payload["train_tokens"]
    val_tokens = payload["val_tokens"]
    vocab_size = int(payload.get("vocab_size", config.vocab_size))
    block_size = config.sequence_length + 1
    train_blocks = train_tokens[: (train_tokens.numel() // block_size) * block_size].view(-1, block_size)
    val_blocks = val_tokens[: (val_tokens.numel() // block_size) * block_size].view(-1, block_size)
    if train_blocks.size(0) <= 0:
        raise RuntimeError("cache does not contain any train blocks")
    if val_blocks.size(0) < config.val_blocks:
        raise RuntimeError(f"cache has {val_blocks.size(0)} val blocks, need {config.val_blocks}")
    return (
        train_blocks[:, :-1],
        train_blocks[:, 1:],
        val_blocks[: config.val_blocks, :-1].contiguous(),
        val_blocks[: config.val_blocks, 1:].contiguous(),
        vocab_size,
    )


def top_token_ids(targets: torch.Tensor, *, count: int, vocab_size: int) -> torch.Tensor:
    histogram = torch.zeros(vocab_size, dtype=torch.long)
    rows_per_chunk = max(1, 8_000_000 // max(int(targets.size(-1)), 1))
    for start in range(0, targets.size(0), rows_per_chunk):
        chunk = targets[start : start + rows_per_chunk].reshape(-1)
        if chunk.dtype != torch.long:
            chunk = chunk.long()
        histogram += torch.bincount(chunk, minlength=vocab_size)
        if start == 0 or start % max(rows_per_chunk * 50, 1) == 0:
            print(f"top_token_hist rows={min(start + rows_per_chunk, targets.size(0))}/{targets.size(0)}", flush=True)
    return torch.topk(histogram, k=min(count, vocab_size), largest=True, sorted=False).indices


def build_batch_schedule(total_examples: int, *, batch_size: int, steps: int, seed: int) -> list[torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    usable_total = total_examples - (total_examples % batch_size)
    if usable_total <= 0:
        raise RuntimeError("not enough examples for one batch")
    schedule: list[torch.Tensor] = []
    epoch_order = torch.randperm(total_examples, generator=generator)
    cursor = 0
    while len(schedule) < steps:
        if cursor + batch_size > usable_total:
            epoch_order = torch.randperm(total_examples, generator=generator)
            cursor = 0
        schedule.append(epoch_order[cursor : cursor + batch_size].clone())
        cursor += batch_size
    return schedule


def candidate_ids_with_targets(fixed_candidate_ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    target_ids = targets.detach().reshape(-1).to("cpu").unique(sorted=True)
    fixed_ids = fixed_candidate_ids.detach().to("cpu")
    return torch.cat((fixed_ids, target_ids)).unique(sorted=True).to(targets.device, non_blocking=True)


def linear_cross_entropy_sum_chunked(
    hidden: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    token_chunk_size: int,
) -> torch.Tensor:
    flat_hidden = hidden.reshape(-1, hidden.size(-1))
    flat_targets = targets.reshape(-1)
    if flat_targets.dtype != torch.long:
        flat_targets = flat_targets.long()
    if token_chunk_size <= 0 or flat_hidden.size(0) <= token_chunk_size:
        return F.cross_entropy(F.linear(flat_hidden, weight, bias), flat_targets, reduction="sum")
    loss_sum = None
    for start in range(0, flat_hidden.size(0), token_chunk_size):
        end = min(start + token_chunk_size, flat_hidden.size(0))
        logits = F.linear(flat_hidden[start:end], weight, bias)
        chunk_loss = F.cross_entropy(logits, flat_targets[start:end], reduction="sum")
        loss_sum = chunk_loss if loss_sum is None else loss_sum + chunk_loss
    if loss_sum is None:
        raise RuntimeError("empty chunked cross entropy")
    return loss_sum


def anchor_loss(
    model: CausalConvFactorizedLM,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    *,
    fixed_candidate_ids: torch.Tensor,
    config: TrainConfig,
) -> tuple[torch.Tensor, int, int]:
    candidate_ids = candidate_ids_with_targets(fixed_candidate_ids, targets)
    candidate_map = torch.full((config.vocab_size,), -1, dtype=torch.long, device=targets.device)
    candidate_map[candidate_ids] = torch.arange(candidate_ids.numel(), dtype=torch.long, device=targets.device)
    reduced_targets = candidate_map[targets.reshape(-1)]
    if bool((reduced_targets < 0).any()):
        raise RuntimeError("candidate set missed a target token")
    reduced_targets = reduced_targets.view_as(targets)
    hidden = model.factor_down(model.features(input_ids))
    candidate_weight = model.factor_up.weight.index_select(0, candidate_ids)
    candidate_bias = model.factor_up.bias.index_select(0, candidate_ids) if model.factor_up.bias is not None else None
    sampled_sum = linear_cross_entropy_sum_chunked(
        hidden,
        reduced_targets,
        candidate_weight,
        candidate_bias,
        token_chunk_size=config.token_chunk_size,
    )
    sampled_loss = sampled_sum / targets.numel()
    anchor_hidden = hidden[:, :: config.token_stride, :]
    anchor_targets = targets[:, :: config.token_stride]
    anchor_sum = linear_cross_entropy_sum_chunked(
        anchor_hidden,
        anchor_targets,
        model.factor_up.weight,
        model.factor_up.bias,
        token_chunk_size=config.token_chunk_size,
    )
    full_anchor_loss = anchor_sum / anchor_targets.numel()
    return 0.5 * (sampled_loss + full_anchor_loss), int(targets.numel()), int(candidate_ids.numel())


@torch.inference_mode()
def evaluate_full_loss(
    model: CausalConvFactorizedLM,
    val_inputs: torch.Tensor,
    val_targets: torch.Tensor,
    *,
    config: TrainConfig,
    device: torch.device,
    autocast_kwargs: dict[str, Any],
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for row in range(val_inputs.size(0)):
        batch_inputs = val_inputs[row : row + 1].to(device, non_blocking=True)
        batch_targets = val_targets[row : row + 1].to(device, non_blocking=True)
        if batch_inputs.dtype != torch.long:
            batch_inputs = batch_inputs.long()
        if batch_targets.dtype != torch.long:
            batch_targets = batch_targets.long()
        with torch.autocast(**autocast_kwargs):
            hidden = model.factor_down(model.features(batch_inputs))
            for start in range(0, hidden.size(1), config.full_eval_token_chunk_size):
                end = min(start + config.full_eval_token_chunk_size, hidden.size(1))
                logits = model.factor_up(hidden[:, start:end, :])
                chunk_targets = batch_targets[:, start:end]
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), chunk_targets.reshape(-1))
                token_count = int(chunk_targets.numel())
                total_loss += float(loss.item()) * token_count
                total_tokens += token_count
    return total_loss / max(total_tokens, 1)


def save_checkpoint(
    path: Path,
    *,
    config: TrainConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    tokens_seen: int,
    history: list[dict[str, float]],
    step_times: list[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "benchmark": "standalone_longseq_anchor_train",
            "config": {**asdict(config), "cache_path": str(config.cache_path), "output_dir": str(config.output_dir), "resume_checkpoint": str(config.resume_checkpoint) if config.resume_checkpoint else None},
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "step": step,
            "tokens_seen": tokens_seen,
            "history": history,
            "step_times": step_times,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        tmp,
    )
    tmp.replace(path)


def train(config: TrainConfig) -> None:
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if config.amp_dtype == "bf16" else torch.float16
    autocast_kwargs = {"device_type": "cuda", "dtype": dtype, "enabled": device.type == "cuda"}
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"
    result_path = output_dir / "result.json"
    checkpoint_path = output_dir / "checkpoint.pt"

    write_json_atomic(
        state_path,
        {
            "status": "loading_cache",
            "run_name": config.run_name,
            "cache_path": str(config.cache_path),
            "train_steps": config.train_steps,
        },
    )
    train_inputs, train_targets, val_inputs, val_targets, vocab_size = load_cache(config)
    if vocab_size != config.vocab_size:
        raise RuntimeError(f"cache vocab size {vocab_size} != configured vocab size {config.vocab_size}")
    fixed_candidate_ids = top_token_ids(train_targets, count=config.sampled_vocab_size, vocab_size=config.vocab_size)
    schedule = build_batch_schedule(
        len(train_inputs),
        batch_size=config.batch_size,
        steps=config.train_steps,
        seed=config.seed,
    )

    model = CausalConvFactorizedLM(config).to(device)
    parameter_count = count_parameters(model)
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, fused=device.type == "cuda")
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler(device="cuda", enabled=device.type == "cuda" and config.amp_dtype == "fp16")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    history: list[dict[str, float]] = []
    step_times: list[float] = []
    tokens_seen = 0
    start_step = 1

    if config.resume_checkpoint is not None and config.resume_checkpoint.exists():
        checkpoint = torch.load(config.resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint.get("scaler_state", {}))
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        cuda_rng_state_all = checkpoint.get("cuda_rng_state_all")
        if cuda_rng_state_all is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng_state_all])
        history = list(checkpoint.get("history", []))
        step_times = list(checkpoint.get("step_times", []))
        tokens_seen = int(checkpoint.get("tokens_seen", 0))
        start_step = int(checkpoint.get("step", 0)) + 1
        print(f"RESUMED step={start_step - 1} tokens={tokens_seen}", flush=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    write_json_atomic(
        state_path,
        {
            "status": "running",
            "run_name": config.run_name,
            "step": start_step - 1,
            "train_steps": config.train_steps,
            "tokens_seen": tokens_seen,
            "parameter_count": parameter_count,
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "learning_rate": config.learning_rate,
            "min_learning_rate": config.min_learning_rate,
            "warmup_steps": config.warmup_steps,
        },
    )
    print(
        f"START run={config.run_name} device={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'} "
        f"params={parameter_count:,} seq={config.sequence_length} steps={config.train_steps}",
        flush=True,
    )

    latest_val_loss = float("nan")
    latest_train_loss = float("nan")
    candidate_sizes: list[int] = []
    for step in range(start_step, config.train_steps + 1):
        batch_indices = schedule[step - 1]
        batch_inputs = train_inputs.index_select(0, batch_indices).to(device, non_blocking=True)
        batch_targets = train_targets.index_select(0, batch_indices).to(device, non_blocking=True)
        if batch_inputs.dtype != torch.long:
            batch_inputs = batch_inputs.long()
        if batch_targets.dtype != torch.long:
            batch_targets = batch_targets.long()
        current_lr = scheduled_learning_rate(config, step)
        set_optimizer_lr(optimizer, current_lr)
        step_start = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(**autocast_kwargs):
            loss, token_count, candidate_size = anchor_loss(
                model,
                batch_inputs,
                batch_targets,
                fixed_candidate_ids=fixed_candidate_ids,
                config=config,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_duration = time.perf_counter() - step_start
        step_times.append(step_duration)
        tokens_seen += token_count
        candidate_sizes.append(candidate_size)
        latest_train_loss = float(loss.detach().item())
        pure_time = sum(step_times)
        pure_tps = tokens_seen / max(pure_time, 1e-9)

        if step == 1 or step % 100 == 0:
            write_json_atomic(
                state_path,
                {
                    "status": "running",
                    "run_name": config.run_name,
                    "step": step,
                    "train_steps": config.train_steps,
                    "tokens_seen": tokens_seen,
                    "latest_train_loss": latest_train_loss,
                    "latest_val_loss": latest_val_loss,
                    "latest_learning_rate": current_lr,
                    "pure_train_tok_per_sec": pure_tps,
                    "step_time_ms": step_duration * 1000.0,
                    "peak_vram_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else None,
                    "checkpoint_path": str(checkpoint_path),
                },
            )
            if step == 1 or step % 500 == 0:
                print(
                    f"TRAIN step={step}/{config.train_steps} tokens={tokens_seen} "
                    f"loss={latest_train_loss:.4f} lr={current_lr:.6g} pure_tok_s={pure_tps:.0f}",
                    flush=True,
                )

        should_eval = step % config.eval_interval == 0 or step == config.train_steps
        if should_eval:
            eval_start = time.perf_counter()
            latest_val_loss = evaluate_full_loss(
                model,
                val_inputs,
                val_targets,
                config=config,
                device=device,
                autocast_kwargs=autocast_kwargs,
            )
            eval_seconds = time.perf_counter() - eval_start
            history.append(
                {
                    "step": float(step),
                    "tokens_seen": float(tokens_seen),
                    "train_loss": latest_train_loss,
                    "val_loss": latest_val_loss,
                    "learning_rate": current_lr,
                }
            )
            print(
                f"EVAL step={step}/{config.train_steps} tokens={tokens_seen} "
                f"train={latest_train_loss:.4f} val={latest_val_loss:.4f} eval_s={eval_seconds:.1f}",
                flush=True,
            )
            write_json_atomic(
                state_path,
                {
                    "status": "running",
                    "run_name": config.run_name,
                    "step": step,
                    "train_steps": config.train_steps,
                    "tokens_seen": tokens_seen,
                    "latest_train_loss": latest_train_loss,
                    "latest_val_loss": latest_val_loss,
                    "latest_learning_rate": current_lr,
                    "pure_train_tok_per_sec": pure_tps,
                    "peak_vram_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else None,
                    "history": history,
                    "checkpoint_path": str(checkpoint_path),
                },
            )

        should_checkpoint = (
            (config.checkpoint_interval > 0 and step % config.checkpoint_interval == 0)
            or should_eval
            or step == config.train_steps
        )
        if should_checkpoint:
            save_checkpoint(
                checkpoint_path,
                config=config,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                step=step,
                tokens_seen=tokens_seen,
                history=history,
                step_times=step_times,
            )
            if config.milestone_checkpoint_interval > 0 and step % config.milestone_checkpoint_interval == 0:
                milestone = output_dir / f"checkpoint.step{step}_tokens{tokens_seen}.pt"
                save_checkpoint(
                    milestone,
                    config=config,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=step,
                    tokens_seen=tokens_seen,
                    history=history,
                    step_times=step_times,
                )
                print(f"MILESTONE {milestone}", flush=True)

    pure_time = sum(step_times)
    result = {
        "benchmark": "standalone_longseq_anchor_train",
        "config": {**asdict(config), "cache_path": str(config.cache_path), "output_dir": str(config.output_dir), "resume_checkpoint": str(config.resume_checkpoint) if config.resume_checkpoint else None},
        "report": {
            "parameter_count": parameter_count,
            "train_tokens_seen": tokens_seen,
            "final_train_loss": latest_train_loss,
            "final_val_loss": latest_val_loss,
            "pure_train_tok_per_sec": tokens_seen / max(pure_time, 1e-9),
            "step_time_mean_ms": statistics.fmean(step_times) * 1000.0,
            "step_time_median_ms": statistics.median(step_times) * 1000.0,
            "peak_vram_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else None,
            "candidate_size_mean": statistics.fmean(candidate_sizes) if candidate_sizes else None,
            "history": history,
            "checkpoint_path": str(checkpoint_path),
        },
    }
    write_json_atomic(result_path, result)
    write_json_atomic(
        state_path,
        {
            "status": "completed",
            "run_name": config.run_name,
            "step": config.train_steps,
            "train_steps": config.train_steps,
            "tokens_seen": tokens_seen,
            "final_train_loss": latest_train_loss,
            "final_val_loss": latest_val_loss,
            "pure_train_tok_per_sec": result["report"]["pure_train_tok_per_sec"],
            "peak_vram_mb": result["report"]["peak_vram_mb"],
            "checkpoint_path": str(checkpoint_path),
            "result_path": str(result_path),
        },
    )
    print("RESULT " + json.dumps(result["report"], sort_keys=True), flush=True)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Standalone long-sequence anchor16 trainer.")
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--train-steps", type=int, default=196_851)
    parser.add_argument("--eval-interval", type=int, default=9_843)
    parser.add_argument("--checkpoint-interval", type=int, default=9_843)
    parser.add_argument("--milestone-checkpoint-interval", type=int, default=59_058)
    parser.add_argument("--val-blocks", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=10_160)
    parser.add_argument("--embedding-dim", type=int, default=1_831)
    parser.add_argument(
        "--block-type",
        choices=(
            "relu_square",
            "gated",
            "memory",
            "landmark_attention",
            "multi_scale",
            "adaptive_multi_scale",
            "token_gated_multi_scale",
            "multi_scale_memory",
            "multi_scale_lowrank_conv_memory",
            "multi_scale_lowrank_conv_memory_swiglu",
            "multi_scale_lowrank_conv_memory_silu_square",
            "memory_threshold_basis_lowrank",
            "token_gated_lowrank_conv_memory",
            "dilated_multi_scale",
        ),
        default="relu_square",
    )
    parser.add_argument("--conv-layers", type=int, default=1)
    parser.add_argument("--conv-kernel-size", type=int, default=7)
    parser.add_argument("--conv-rank", type=int, default=531)
    parser.add_argument("--memory-rank", type=int, default=64)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--landmark-stride", type=int, default=128)
    parser.add_argument("--sampled-vocab-size", type=int, default=16_384)
    parser.add_argument("--token-stride", type=int, default=16)
    parser.add_argument("--token-chunk-size", type=int, default=1_024)
    parser.add_argument("--full-eval-token-chunk-size", type=int, default=1_024)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--min-learning-rate", type=float, default=6e-5)
    parser.add_argument("--warmup-steps", type=int, default=2_000)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    return TrainConfig(
        cache_path=args.cache_path,
        output_dir=args.output_dir,
        run_name=args.run_name,
        train_steps=args.train_steps,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        milestone_checkpoint_interval=args.milestone_checkpoint_interval,
        val_blocks=args.val_blocks,
        sequence_length=args.sequence_length,
        embedding_dim=args.embedding_dim,
        block_type=args.block_type,
        conv_layers=args.conv_layers,
        conv_kernel_size=args.conv_kernel_size,
        conv_rank=args.conv_rank,
        memory_rank=args.memory_rank,
        attention_heads=args.attention_heads,
        landmark_stride=args.landmark_stride,
        sampled_vocab_size=args.sampled_vocab_size,
        token_stride=args.token_stride,
        token_chunk_size=args.token_chunk_size,
        full_eval_token_chunk_size=args.full_eval_token_chunk_size,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_steps=args.warmup_steps,
        resume_checkpoint=args.resume_checkpoint,
    )


if __name__ == "__main__":
    train(parse_args())
