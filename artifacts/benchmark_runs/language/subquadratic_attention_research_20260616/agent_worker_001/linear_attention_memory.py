from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


class CausalChunkedLinearAttentionMemory(nn.Module):
    """Causal kernelized linear attention branch for [batch, tokens, dim] states."""

    def __init__(
        self,
        *,
        dim: int,
        feature_rank: int = 32,
        value_rank: int = 64,
        chunk_size: int = 1024,
        dropout: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if feature_rank <= 0:
            raise ValueError("feature_rank must be positive")
        if value_rank <= 0:
            raise ValueError("value_rank must be positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.dim = dim
        self.feature_rank = feature_rank
        self.value_rank = value_rank
        self.chunk_size = chunk_size
        self.eps = eps

        self.norm = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, feature_rank, bias=False)
        self.key = nn.Linear(dim, feature_rank, bias=False)
        self.value = nn.Linear(dim, value_rank, bias=False)
        self.out = nn.Linear(value_rank, dim, bias=False)
        self.gate = nn.Linear(dim, 1)
        self.dropout = nn.Dropout(dropout)
        nn.init.constant_(self.gate.bias, -2.0)

    @staticmethod
    def feature_map(tensor: torch.Tensor) -> torch.Tensor:
        return F.elu(tensor) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected [batch, tokens, dim], got shape {tuple(x.shape)}")
        if x.size(-1) != self.dim:
            raise ValueError(f"last dim {x.size(-1)} does not match configured dim {self.dim}")

        source = self.norm(x)
        q = self.feature_map(self.query(source)) / math.sqrt(self.feature_rank)
        k = self.feature_map(self.key(source))
        v = self.value(source)

        state_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
        q_state = q.to(state_dtype)
        k_state = k.to(state_dtype)
        v_state = v.to(state_dtype)

        outputs = []
        kv_state: torch.Tensor | None = None
        z_state: torch.Tensor | None = None
        token_count = x.size(1)
        for start in range(0, token_count, self.chunk_size):
            end = min(start + self.chunk_size, token_count)
            q_chunk = q_state[:, start:end, :]
            k_chunk = k_state[:, start:end, :]
            v_chunk = v_state[:, start:end, :]

            kv_update = k_chunk.unsqueeze(-1) * v_chunk.unsqueeze(-2)
            kv_prefix = torch.cumsum(kv_update, dim=1)
            z_prefix = torch.cumsum(k_chunk, dim=1)
            if kv_state is not None and z_state is not None:
                kv_prefix = kv_prefix + kv_state.unsqueeze(1)
                z_prefix = z_prefix + z_state.unsqueeze(1)

            numerator = torch.einsum("btr,btrm->btm", q_chunk, kv_prefix)
            denominator = torch.einsum("btr,btr->bt", q_chunk, z_prefix).unsqueeze(-1).clamp_min(self.eps)
            outputs.append(numerator / denominator)
            kv_state = kv_prefix[:, -1, :, :]
            z_state = z_prefix[:, -1, :]

        attended = torch.cat(outputs, dim=1).to(dtype=source.dtype)
        gated = self.out(attended) * torch.sigmoid(self.gate(source))
        return x + self.dropout(gated)


class CausalMultiScaleLowRankConvMemoryLinearAttentionBlock(nn.Module):
    """Best current conv-memory block plus a subquadratic causal linear-attention branch."""

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
        attention_feature_rank: int = 32,
        attention_value_rank: int = 64,
        attention_chunk_size: int = 1024,
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
        self.linear_attention = CausalChunkedLinearAttentionMemory(
            dim=dim,
            feature_rank=attention_feature_rank,
            value_rank=attention_value_rank,
            chunk_size=attention_chunk_size,
            dropout=dropout,
        )
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

        x = self.linear_attention(x)

        hidden = F.relu(self.ffn_in(self.ffn_norm(x))).square()
        return x + self.dropout(self.ffn_out(hidden))


def linear_attention_cost(
    *,
    dim: int,
    tokens: int,
    feature_rank: int,
    value_rank: int,
    chunk_size: int,
) -> dict[str, int]:
    chunk = min(tokens, chunk_size)
    return {
        "projection_multiply_adds_approx": tokens * dim * (2 * feature_rank + 2 * value_rank),
        "state_update_multiply_adds_approx": tokens * feature_rank * value_rank,
        "max_state_elements_training_chunked": chunk * feature_rank * value_rank,
        "state_elements_streaming_inference": feature_rank * value_rank + feature_rank,
    }
