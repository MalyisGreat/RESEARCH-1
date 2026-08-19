from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class DeepWaveDeltaConfig:
    vocab_size: int = 50_257
    dim: int = 768
    depth: int = 12
    ffn_expansion: int = 2
    output_rank: int = 256
    local_memory_rank: int = 96
    local_memory_kernel: int = 128
    conv_kernels: tuple[int, ...] = (3, 7, 15)
    recurrent_layers: int = 4
    recurrent_heads: int = 8
    key_dim: int = 64
    value_dim: int = 64
    min_half_life: float = 32.0
    max_half_life: float = 8192.0
    dropout: float = 0.0
    use_fused_delta: bool = True

    def validate(self) -> None:
        if min(self.vocab_size, self.dim, self.depth, self.output_rank) <= 0:
            raise ValueError("vocab_size, dim, depth, and output_rank must be positive")
        if not 0 <= self.recurrent_layers <= self.depth:
            raise ValueError("recurrent_layers must be between zero and depth")
        if min(self.recurrent_heads, self.key_dim, self.value_dim) <= 0:
            raise ValueError("recurrent state dimensions must be positive")
        if not self.conv_kernels or any(kernel <= 0 for kernel in self.conv_kernels):
            raise ValueError("conv_kernels must contain positive integers")
        if self.local_memory_kernel <= 0 or self.local_memory_rank <= 0:
            raise ValueError("local memory dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 < self.min_half_life <= self.max_half_life:
            raise ValueError("invalid recurrent half-life range")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PRESETS: dict[str, dict[str, Any]] = {
    # The 10M preset deliberately spends little width on the 50K embedding so it
    # can be used for cheap algorithm screens without changing tokenization.
    "10m": {
        "dim": 128,
        "depth": 8,
        "ffn_expansion": 2,
        "output_rank": 48,
        "local_memory_rank": 32,
        "recurrent_layers": 2,
        "recurrent_heads": 4,
        "key_dim": 32,
        "value_dim": 32,
    },
    "100m": {
        "dim": 704,
        "depth": 12,
        "ffn_expansion": 2,
        "output_rank": 256,
        "local_memory_rank": 96,
        "recurrent_layers": 4,
        "recurrent_heads": 8,
        "key_dim": 64,
        "value_dim": 64,
    },
    "350m": {
        "dim": 1_408,
        "depth": 16,
        "ffn_expansion": 2,
        "output_rank": 512,
        "local_memory_rank": 192,
        "recurrent_layers": 4,
        "recurrent_heads": 8,
        "key_dim": 64,
        "value_dim": 64,
    },
}


def config_from_preset(name: str, **overrides: Any) -> DeepWaveDeltaConfig:
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {tuple(PRESETS)}")
    values = {**PRESETS[name], **overrides}
    config = DeepWaveDeltaConfig(**values)
    config.validate()
    return config


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


class HeadRMSNorm(nn.Module):
    def __init__(self, heads: int, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(heads, dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


def _logit(value: torch.Tensor) -> torch.Tensor:
    return torch.log(value) - torch.log1p(-value)


def recurrent_layer_indices(depth: int, count: int) -> tuple[int, ...]:
    if count == 0:
        return ()
    # End each approximately equal-depth stage with a persistent-memory update.
    return tuple(math.ceil((index + 1) * depth / count) - 1 for index in range(count))


class CausalDepthwiseConv(nn.Module):
    def __init__(self, dim: int, kernel_size: int, dilation: int = 1, bias: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(dim, dim, kernel_size, dilation=dilation, groups=dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x.transpose(1, 2), (self.left_padding, 0))).transpose(1, 2)

    def step(
        self, x: torch.Tensor, state: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.size(1) != 1:
            raise ValueError("step expects [batch, 1, dim]")
        current = x.transpose(1, 2)
        if state is None:
            state = current.new_zeros(current.size(0), self.dim, self.left_padding)
        window = torch.cat((state, current), dim=-1)
        output = self.conv(window).transpose(1, 2)
        next_state = window[:, :, -self.left_padding :] if self.left_padding else window[:, :, :0]
        return output, next_state


class AdaptiveLocalMixer(nn.Module):
    """Causal local mixing with nonlinear, token-conditioned scale competition."""

    def __init__(self, dim: int, kernels: tuple[int, ...], dilation: int) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        self.branches = nn.ModuleList(CausalDepthwiseConv(dim, kernel, dilation) for kernel in kernels)
        self.scale_gate = nn.Linear(dim, len(kernels), bias=True)
        self.output = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.scale_gate.weight)
        nn.init.zeros_(self.scale_gate.bias)

    def _mix(self, source: torch.Tensor, outputs: list[torch.Tensor]) -> torch.Tensor:
        weights = torch.softmax(self.scale_gate(source).float(), dim=-1).to(source.dtype)
        activated = torch.stack([F.relu(output).square() for output in outputs], dim=-2)
        return self.output((activated * weights.unsqueeze(-1)).sum(dim=-2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        source = self.norm(x)
        return self._mix(source, [branch(source) for branch in self.branches])

    def step(
        self, x: torch.Tensor, states: list[torch.Tensor | None] | None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        source = self.norm(x)
        if states is None:
            states = [None] * len(self.branches)
        if len(states) != len(self.branches):
            raise ValueError("local convolution cache does not match branch count")
        outputs: list[torch.Tensor] = []
        next_states: list[torch.Tensor] = []
        for branch, state in zip(self.branches, states):
            output, next_state = branch.step(source, state)
            outputs.append(output)
            next_states.append(next_state)
        return self._mix(source, outputs), next_states


class LowRankLocalMemory(nn.Module):
    """The original finite-window low-rank path, retained alongside global state."""

    def __init__(self, dim: int, rank: int, kernel_size: int) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        self.down = nn.Linear(dim, rank, bias=False)
        self.fir = CausalDepthwiseConv(rank, kernel_size, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(F.silu(self.fir(self.down(self.norm(x)))))

    def step(
        self, x: torch.Tensor, state: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, next_state = self.fir.step(self.down(self.norm(x)), state)
        return self.up(F.silu(hidden)), next_state


class DeltaRuleMemory(nn.Module):
    """Large, normalized delta-rule state with parallel training and cached inference."""

    def __init__(
        self,
        dim: int,
        *,
        heads: int,
        key_dim: int,
        value_dim: int,
        min_half_life: float,
        max_half_life: float,
        use_fused: bool,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.use_fused = use_fused
        key_width = heads * key_dim
        value_width = heads * value_dim
        self.query = nn.Linear(dim, key_width, bias=False)
        self.key = nn.Linear(dim, key_width, bias=False)
        self.value = nn.Linear(dim, value_width, bias=False)
        self.retention = nn.Linear(dim, heads, bias=True)
        self.write = nn.Linear(dim, heads, bias=True)
        self.output_gate = nn.Linear(dim, value_width, bias=True)
        self.head_norm = HeadRMSNorm(heads, value_dim)
        self.output = nn.Linear(value_width, dim, bias=False)

        half_lives = torch.logspace(math.log10(min_half_life), math.log10(max_half_life), heads)
        retention = torch.exp(math.log(0.5) / half_lives).clamp(1e-5, 1.0 - 1e-5)
        nn.init.zeros_(self.retention.weight)
        with torch.no_grad():
            self.retention.bias.copy_(_logit(retention))
        nn.init.zeros_(self.write.weight)
        nn.init.zeros_(self.write.bias)
        nn.init.zeros_(self.output_gate.weight)
        nn.init.constant_(self.output_gate.bias, 1.0)

    @property
    def state_size(self) -> int:
        return self.heads * self.key_dim * self.value_dim

    def _project(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, _ = x.shape
        v = self.value(x).view(batch, length, self.heads, self.value_dim)
        q = self.query(x).view(batch, length, self.heads, self.key_dim)
        k = self.key(x).view(batch, length, self.heads, self.key_dim)
        q = F.normalize(q.float(), dim=-1).to(v.dtype)
        k = F.normalize(k.float(), dim=-1).to(v.dtype)
        log_retention = F.logsigmoid(self.retention(x)).to(v.dtype)
        # Keep writes away from the dead extremes while allowing token-specific plasticity.
        write = (0.05 + 0.90 * torch.sigmoid(self.write(x))).to(v.dtype)
        return q, k, v, log_retention, write

    @staticmethod
    def reference_delta(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        log_retention: torch.Tensor,
        write: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, heads, key_dim = q.shape
        value_dim = v.size(-1)
        state = initial_state
        if state is None:
            state = q.new_zeros(batch, heads, key_dim, value_dim)
        outputs: list[torch.Tensor] = []
        for position in range(length):
            q_t = q[:, position]
            k_t = k[:, position]
            v_t = v[:, position]
            retention_t = log_retention[:, position].exp().unsqueeze(-1).unsqueeze(-1)
            state = retention_t * state
            prediction = torch.einsum("bhk,bhkv->bhv", k_t, state)
            error = v_t - prediction
            update = torch.einsum("bhk,bhv->bhkv", k_t, error)
            state = state + write[:, position].unsqueeze(-1).unsqueeze(-1) * update
            outputs.append(torch.einsum("bhk,bhkv->bhv", q_t, state))
        return torch.stack(outputs, dim=1), state

    def _finish(self, x: torch.Tensor, retrieved: torch.Tensor) -> torch.Tensor:
        batch, length, _, _ = retrieved.shape
        retrieved = self.head_norm(retrieved)
        gate = F.silu(self.output_gate(x)).view(batch, length, self.heads, self.value_dim)
        return self.output((retrieved * gate).reshape(batch, length, -1))

    def forward(
        self,
        x: torch.Tensor,
        *,
        initial_state: torch.Tensor | None = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        q, k, v, log_retention, write = self._project(x)
        can_fuse = self.use_fused and x.is_cuda
        if can_fuse:
            try:
                from fla.ops.gated_delta_rule import chunk_gated_delta_rule
            except ImportError as error:
                raise RuntimeError("fused delta training requires fla-core") from error
            retrieved, final_state = chunk_gated_delta_rule(
                q,
                k,
                v,
                log_retention,
                write,
                scale=1.0,
                initial_state=initial_state,
                output_final_state=return_state,
                use_qk_l2norm_in_kernel=False,
            )
        else:
            retrieved, final_state = self.reference_delta(q, k, v, log_retention, write, initial_state)
        output = self._finish(x, retrieved)
        return (output, final_state) if return_state else output

    def step(
        self, x: torch.Tensor, state: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q, k, v, log_retention, write = self._project(x)
        retrieved, next_state = self.reference_delta(q, k, v, log_retention, write, state)
        return self._finish(x, retrieved), next_state


class StableSquaredFFN(nn.Module):
    def __init__(self, dim: int, expansion: int) -> None:
        super().__init__()
        hidden = dim * expansion
        self.norm = RMSNorm(dim)
        self.value = nn.Linear(dim, hidden, bias=True)
        self.gate = nn.Linear(dim, hidden, bias=True)
        self.output = nn.Linear(hidden, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        source = self.norm(x)
        value = F.relu(self.value(source)).square()
        return self.output(value * torch.sigmoid(self.gate(source)))


class DeepWaveDeltaBlock(nn.Module):
    def __init__(self, config: DeepWaveDeltaConfig, layer_index: int, recurrent: bool) -> None:
        super().__init__()
        dilation = 2 ** (layer_index % 4)
        self.local = AdaptiveLocalMixer(config.dim, config.conv_kernels, dilation)
        self.local_memory = LowRankLocalMemory(
            config.dim, config.local_memory_rank, config.local_memory_kernel
        )
        self.recurrent_norm = RMSNorm(config.dim) if recurrent else None
        self.recurrent = (
            DeltaRuleMemory(
                config.dim,
                heads=config.recurrent_heads,
                key_dim=config.key_dim,
                value_dim=config.value_dim,
                min_half_life=config.min_half_life,
                max_half_life=config.max_half_life,
                use_fused=config.use_fused_delta,
            )
            if recurrent
            else None
        )
        self.competition = nn.Linear(config.dim, config.dim, bias=True) if recurrent else None
        self.ffn = StableSquaredFFN(config.dim, config.ffn_expansion)
        self.dropout = nn.Dropout(config.dropout)
        if self.competition is not None:
            nn.init.zeros_(self.competition.weight)
            nn.init.constant_(self.competition.bias, math.log(0.25 / 0.75))

    def _global_update(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None,
        return_state: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.recurrent is None or self.recurrent_norm is None or self.competition is None:
            return x, None
        source = self.recurrent_norm(x)
        result = self.recurrent(source, initial_state=state, return_state=return_state)
        if return_state:
            memory, next_state = result
        else:
            memory, next_state = result, None
        # Bounded channel-wise competition prevents recurrent reads from swamping local evidence.
        mixture = 2.0 * torch.sigmoid(self.competition(source))
        return x + self.dropout(mixture * memory), next_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.local(x))
        x = x + self.dropout(self.local_memory(x))
        x, _ = self._global_update(x, None, False)
        return x + self.dropout(self.ffn(x))

    def step(
        self, x: torch.Tensor, cache: dict[str, Any] | None
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        cache = cache or {}
        local, conv_state = self.local.step(x, cache.get("conv"))
        x = x + self.dropout(local)
        local_memory, fir_state = self.local_memory.step(x, cache.get("fir"))
        x = x + self.dropout(local_memory)
        recurrent_state = cache.get("recurrent")
        if self.recurrent is not None and self.recurrent_norm is not None and self.competition is not None:
            source = self.recurrent_norm(x)
            memory, recurrent_state = self.recurrent.step(source, recurrent_state)
            x = x + self.dropout(2.0 * torch.sigmoid(self.competition(source)) * memory)
        x = x + self.dropout(self.ffn(x))
        return x, {"conv": conv_state, "fir": fir_state, "recurrent": recurrent_state}


class DeepWaveDeltaLM(nn.Module):
    """Deep dual-memory LM compatible with the project's factorized-vocabulary loss."""

    def __init__(self, config: DeepWaveDeltaConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        recurrent_indices = set(recurrent_layer_indices(config.depth, config.recurrent_layers))
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList(
            DeepWaveDeltaBlock(config, index, index in recurrent_indices) for index in range(config.depth)
        )
        self.final_norm = RMSNorm(config.dim)
        self.factor_down = nn.Linear(config.dim, config.output_rank, bias=False)
        self.factor_up = nn.Linear(config.output_rank, config.vocab_size, bias=True)

    def features(self, input_ids: torch.Tensor) -> torch.Tensor:
        states = self.embedding(input_ids)
        for block in self.blocks:
            states = block(states)
        return self.final_norm(states)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.factor_up(self.factor_down(self.features(input_ids)))

    def step(
        self, token_ids: torch.Tensor, cache: list[dict[str, Any]] | None = None
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        if token_ids.ndim == 1:
            token_ids = token_ids[:, None]
        if token_ids.ndim != 2 or token_ids.size(1) != 1:
            raise ValueError("step expects token IDs shaped [batch] or [batch, 1]")
        if cache is None:
            cache = [{} for _ in self.blocks]
        if len(cache) != len(self.blocks):
            raise ValueError("model cache does not match block count")
        states = self.embedding(token_ids)
        next_cache: list[dict[str, Any]] = []
        for block, block_cache in zip(self.blocks, cache):
            states, next_block_cache = block.step(states, block_cache)
            next_cache.append(next_block_cache)
        features = self.final_norm(states)
        return self.factor_up(self.factor_down(features)), next_cache

    @property
    def recurrent_state_size(self) -> int:
        return sum(
            block.recurrent.state_size for block in self.blocks if block.recurrent is not None
        )

    def parameter_report(self) -> dict[str, int]:
        groups = {
            "embedding": sum(parameter.numel() for parameter in self.embedding.parameters()),
            "blocks": sum(parameter.numel() for parameter in self.blocks.parameters()),
            "output": sum(parameter.numel() for parameter in self.factor_down.parameters())
            + sum(parameter.numel() for parameter in self.factor_up.parameters()),
            "final_norm": sum(parameter.numel() for parameter in self.final_norm.parameters()),
        }
        groups["total"] = sum(parameter.numel() for parameter in self.parameters())
        groups["recurrent_state_values"] = self.recurrent_state_size
        return groups
