from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import torch
import torch.nn.functional as F
from torch import nn


ROOT = Path(r"E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language")
TRAINER_PATH = ROOT / "standalone_longseq_anchor_train.py"
ARTIFACT_ROOT = ROOT / "neuron_search_20260605" / "manual_self"
DEFAULT_CACHE_PATH = ROOT / "neuron_search_20260605" / "screen_cache_synth_seq255_train768_val64_gpt2.pt"
CACHE_PATH = Path(os.environ.get("MANUAL_SEARCH_CACHE", str(DEFAULT_CACHE_PATH)))


def load_trainer():
    spec = importlib.util.spec_from_file_location("longseq_trainer_manual_search", TRAINER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import trainer from {TRAINER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


trainer = load_trainer()
ORIGINAL_MAKE_MIXER_BLOCK = trainer.make_mixer_block


def causal_depthwise(layer: nn.Conv1d, x_bt_c: torch.Tensor, left_padding: int) -> torch.Tensor:
    return layer(F.pad(x_bt_c, (left_padding, 0))).transpose(1, 2)


class LowRankConvMemoryVariantBase(nn.Module):
    """Local experiment base matching CausalMultiScaleLowRankConvMemoryBlock structure."""

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
        self.dim = dim
        self.expansion = expansion
        self.memory_rank = memory_rank
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

    def conv_and_memory(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        conv_input = self.conv_norm(x).transpose(1, 2)
        outputs = [
            causal_depthwise(layer, conv_input, left_padding)
            for layer, left_padding in zip(self.depthwise_layers, self.left_paddings)
        ]
        conv_stack = torch.stack(outputs, dim=0)
        conv_output = conv_stack.mean(dim=0)
        x = x + self.dropout(self.mix(F.relu(conv_output).square()))

        memory_input = self.memory_down(self.memory_norm(x)).transpose(1, 2)
        memory_output = self.memory_depthwise(F.pad(memory_input, (self.memory_left_padding, 0))).transpose(1, 2)
        x = x + self.dropout(self.memory_up(F.silu(memory_output)))
        return x, conv_output, memory_output

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        return F.relu(self.ffn_in(self.ffn_norm(x))).square()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, conv_output, memory_output = self.conv_and_memory(x)
        hidden = self.neuron(x, conv_output, memory_output)
        return x + self.dropout(self.ffn_out(hidden))


def channel_dropout(hidden: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    if not training or p <= 0.0:
        return hidden
    keep = 1.0 - p
    mask = hidden.new_empty(hidden.size(0), 1, hidden.size(2)).bernoulli_(keep).div_(keep)
    return hidden * mask


def tensor_dropout(hidden: torch.Tensor, p: torch.Tensor, training: bool) -> torch.Tensor:
    if not training:
        return hidden
    keep = (1.0 - p).clamp_min(0.05)
    mask = torch.rand_like(hidden).lt(keep).to(hidden.dtype).div(keep)
    return hidden * mask


class HiddenDropSquareNeuronBlock(LowRankConvMemoryVariantBase):
    """Baseline square neuron with train-time hidden activation dropout."""

    hidden_dropout_p = 0.05

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, conv_output, memory_output = self.conv_and_memory(x)
        hidden = self.neuron(x, conv_output, memory_output)
        hidden = F.dropout(hidden, p=self.hidden_dropout_p, training=self.training)
        return x + self.dropout(self.ffn_out(hidden))


class HiddenDropLowSquareNeuronBlock(HiddenDropSquareNeuronBlock):
    """Baseline square neuron with lighter train-time hidden activation dropout."""

    hidden_dropout_p = 0.02


class HiddenDropMidSquareNeuronBlock(HiddenDropSquareNeuronBlock):
    """Baseline square neuron with medium train-time hidden activation dropout."""

    hidden_dropout_p = 0.075


class HiddenDropHighSquareNeuronBlock(HiddenDropSquareNeuronBlock):
    """Baseline square neuron with stronger train-time hidden activation dropout."""

    hidden_dropout_p = 0.10


class HiddenDropUltraSquareNeuronBlock(HiddenDropSquareNeuronBlock):
    """Baseline square neuron with very strong train-time hidden activation dropout."""

    hidden_dropout_p = 0.15


class HiddenDropExtremeSquareNeuronBlock(HiddenDropSquareNeuronBlock):
    """Baseline square neuron with extreme train-time hidden activation dropout."""

    hidden_dropout_p = 0.20


class HiddenDropP25SquareNeuronBlock(HiddenDropSquareNeuronBlock):
    """Baseline square neuron with p=0.25 train-time hidden activation dropout."""

    hidden_dropout_p = 0.25


class HiddenDropP30SquareNeuronBlock(HiddenDropSquareNeuronBlock):
    """Baseline square neuron with p=0.30 train-time hidden activation dropout."""

    hidden_dropout_p = 0.30


class HiddenDropP35SquareNeuronBlock(HiddenDropSquareNeuronBlock):
    """Baseline square neuron with p=0.35 train-time hidden activation dropout."""

    hidden_dropout_p = 0.35


class HiddenDropP40SquareNeuronBlock(HiddenDropSquareNeuronBlock):
    """Baseline square neuron with p=0.40 train-time hidden activation dropout."""

    hidden_dropout_p = 0.40


class ChannelDropSquareNeuronBlock(LowRankConvMemoryVariantBase):
    """Baseline square neuron with sequence-shared train-time channel dropout."""

    hidden_dropout_p = 0.05

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, conv_output, memory_output = self.conv_and_memory(x)
        hidden = self.neuron(x, conv_output, memory_output)
        hidden = channel_dropout(hidden, self.hidden_dropout_p, self.training)
        return x + self.dropout(self.ffn_out(hidden))


class MemoryEnergyDropSquareNeuronBlock(LowRankConvMemoryVariantBase):
    """Memory-energy conditioned hidden dropout for the squared neuron."""

    min_dropout_p = 0.02
    dropout_span = 0.08

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, conv_output, memory_output = self.conv_and_memory(x)
        hidden = self.neuron(x, conv_output, memory_output)
        if self.training:
            energy = memory_output.float().square().mean(dim=-1, keepdim=True).sqrt()
            centered = energy - energy.mean(dim=1, keepdim=True)
            scale = energy.std(dim=1, keepdim=True, unbiased=False).clamp_min(1.0e-4)
            p = self.min_dropout_p + self.dropout_span * torch.sigmoid(centered / scale)
            hidden = tensor_dropout(hidden, p.to(dtype=hidden.dtype), self.training)
        return x + self.dropout(self.ffn_out(hidden))


class MemoryThresholdNeuronBlock(LowRankConvMemoryVariantBase):
    """Low-rank memory predicts per-token FFN threshold and gain."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        hidden_dim = self.expansion * self.dim
        self.mem_to_threshold = nn.Linear(self.memory_rank, hidden_dim)
        self.mem_to_gain = nn.Linear(self.memory_rank, hidden_dim)
        nn.init.zeros_(self.mem_to_threshold.weight)
        nn.init.zeros_(self.mem_to_threshold.bias)
        nn.init.zeros_(self.mem_to_gain.weight)
        nn.init.zeros_(self.mem_to_gain.bias)

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        threshold = 0.25 * torch.tanh(self.mem_to_threshold(memory_output))
        gain = 1.0 + 0.25 * torch.tanh(self.mem_to_gain(memory_output))
        return gain * F.relu(z - threshold).square()


class ConvDisagreementNeuronBlock(LowRankConvMemoryVariantBase):
    """Conv branch disagreement controls squared-neuron curvature."""

    def conv_and_memory(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        conv_input = self.conv_norm(x).transpose(1, 2)
        outputs = [
            causal_depthwise(layer, conv_input, left_padding)
            for layer, left_padding in zip(self.depthwise_layers, self.left_paddings)
        ]
        conv_stack = torch.stack(outputs, dim=0)
        self._branch_disagreement = conv_stack.var(dim=0, unbiased=False).mean(dim=-1, keepdim=True)
        conv_output = conv_stack.mean(dim=0)
        x = x + self.dropout(self.mix(F.relu(conv_output).square()))
        memory_input = self.memory_down(self.memory_norm(x)).transpose(1, 2)
        memory_output = self.memory_depthwise(F.pad(memory_input, (self.memory_left_padding, 0))).transpose(1, 2)
        x = x + self.dropout(self.memory_up(F.silu(memory_output)))
        return x, conv_output, memory_output

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        disagreement = torch.log1p(self._branch_disagreement)
        curvature = 1.0 + 0.5 * torch.tanh(disagreement)
        return F.relu(z).square() / curvature


class AdaptiveBasisNeuronBlock(LowRankConvMemoryVariantBase):
    """Token-local learned mixture of three nonlinear basis responses."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.basis_router = nn.Linear(self.dim, 3)
        nn.init.zeros_(self.basis_router.weight)
        nn.init.zeros_(self.basis_router.bias)

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        weights = torch.softmax(self.basis_router(x), dim=-1).unsqueeze(-1)
        b0 = F.relu(z).square()
        b1 = z * torch.tanh(z)
        b2 = F.relu(z) * torch.sigmoid(1.5 * z)
        bases = torch.stack((b0, b1, b2), dim=-2)
        return (weights * bases).sum(dim=-2)


class RankCompetitionNeuronBlock(LowRankConvMemoryVariantBase):
    """Expanded channels inhibit each other in groups before squaring."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        hidden_dim = self.expansion * self.dim
        self.groups = 8 if hidden_dim % 8 == 0 else 4
        self.inhibit = nn.Parameter(torch.tensor(0.25))

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        b, t, h = z.shape
        grouped = F.relu(z).view(b, t, self.groups, h // self.groups)
        pressure = grouped.mean(dim=-1, keepdim=True)
        competed = F.relu(grouped - torch.sigmoid(self.inhibit) * pressure)
        return competed.reshape(b, t, h).square()


class RankCompetitionMildNeuronBlock(RankCompetitionNeuronBlock):
    """Rank competition with lower initial inhibition pressure."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        nn.init.constant_(self.inhibit, -0.5)


class RankCompetitionSoftNeuronBlock(RankCompetitionNeuronBlock):
    """Rank competition with very soft initial inhibition pressure."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        nn.init.constant_(self.inhibit, -1.5)


class RankCompetitionUltraSoftNeuronBlock(RankCompetitionNeuronBlock):
    """Rank competition with near-baseline initial inhibition pressure."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        nn.init.constant_(self.inhibit, -2.5)


class RankCompetitionFeatherNeuronBlock(RankCompetitionNeuronBlock):
    """Rank competition with minimal initial inhibition pressure."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        nn.init.constant_(self.inhibit, -3.5)


class RankCompetitionFixedFeatherNeuronBlock(RankCompetitionNeuronBlock):
    """Rank competition with fixed minimal inhibition pressure."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        del self._parameters["inhibit"]
        self.register_buffer("inhibit", torch.tensor(-3.5))


class RankCompetitionFixedTraceNeuronBlock(RankCompetitionNeuronBlock):
    """Rank competition with fixed trace-level inhibition pressure."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        del self._parameters["inhibit"]
        self.register_buffer("inhibit", torch.tensor(-4.5))


class RankCompetitionFixedDustNeuronBlock(RankCompetitionNeuronBlock):
    """Rank competition with fixed near-zero inhibition pressure."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        del self._parameters["inhibit"]
        self.register_buffer("inhibit", torch.tensor(-5.5))


class RankCompetitionFixedFeatherHiddenDropNeuronBlock(RankCompetitionFixedFeatherNeuronBlock):
    """Fixed feather-pressure competition with train-time hidden activation dropout."""

    hidden_dropout_p = 0.05

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, conv_output, memory_output = self.conv_and_memory(x)
        hidden = self.neuron(x, conv_output, memory_output)
        hidden = F.dropout(hidden, p=self.hidden_dropout_p, training=self.training)
        return x + self.dropout(self.ffn_out(hidden))


class RankCompetitionFixedFeatherChannelDropNeuronBlock(RankCompetitionFixedFeatherNeuronBlock):
    """Fixed feather-pressure competition with sequence-shared train-time channel dropout."""

    hidden_dropout_p = 0.05

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, conv_output, memory_output = self.conv_and_memory(x)
        hidden = self.neuron(x, conv_output, memory_output)
        hidden = channel_dropout(hidden, self.hidden_dropout_p, self.training)
        return x + self.dropout(self.ffn_out(hidden))


class StatefulThresholdNeuronBlock(LowRankConvMemoryVariantBase):
    """Causal running mean of preactivations provides a dynamic threshold."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.state_scale = nn.Parameter(torch.tensor(0.2))

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        positions = torch.arange(1, z.size(1) + 1, device=z.device, dtype=z.dtype).view(1, -1, 1)
        causal_mean = z.cumsum(dim=1) / positions
        threshold = torch.tanh(self.state_scale) * causal_mean
        return F.relu(z - threshold).square()


class PhaseAmplitudeNeuronBlock(LowRankConvMemoryVariantBase):
    """Separate amplitude and signed carrier, then remap to FFN width."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        hidden_dim = self.expansion * self.dim
        self.ffn_phase = nn.Linear(self.dim, hidden_dim)
        self.ffn_amp = nn.Linear(self.dim, hidden_dim)

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        u = self.ffn_norm(x)
        amp = F.softplus(self.ffn_amp(u))
        phase = torch.tanh(self.ffn_phase(u))
        carrier = torch.sign(phase) * phase.square()
        return amp * carrier


class PhaseAmplitudeNeutralNeuronBlock(LowRankConvMemoryVariantBase):
    """Parameter-neutral signed-amplitude neuron using the existing FFN projection."""

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        phase = torch.tanh(z)
        signed_carrier = phase * phase.abs()
        return F.softplus(z) * signed_carrier


class PhaseAmplitudeOneExtraNeuronBlock(LowRankConvMemoryVariantBase):
    """Use the baseline FFN projection for amplitude and one extra projection for signed phase."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        hidden_dim = self.expansion * self.dim
        self.ffn_phase = nn.Linear(self.dim, hidden_dim)

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        u = self.ffn_norm(x)
        amp = F.softplus(self.ffn_in(u))
        phase = torch.tanh(self.ffn_phase(u))
        return amp * phase * phase.abs()


class PhaseAmplitudeReplaceNeuronBlock(LowRankConvMemoryVariantBase):
    """Replace the baseline FFN projection with separate amplitude and phase projections."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        del self.ffn_in
        hidden_dim = self.expansion * self.dim
        self.ffn_phase = nn.Linear(self.dim, hidden_dim)
        self.ffn_amp = nn.Linear(self.dim, hidden_dim)

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        u = self.ffn_norm(x)
        amp = F.softplus(self.ffn_amp(u))
        phase = torch.tanh(self.ffn_phase(u))
        return amp * phase * phase.abs()


class PhaseResidualBlendNeuronBlock(LowRankConvMemoryVariantBase):
    """Baseline square neuron with a small learned signed-phase residual."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        hidden_dim = self.expansion * self.dim
        self.phase_mix = nn.Parameter(torch.full((hidden_dim,), -2.0))

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base = F.relu(z).square()
        phase = torch.tanh(z)
        signed = F.softplus(z) * phase * phase.abs()
        mix = torch.sigmoid(self.phase_mix).view(1, 1, -1)
        return base + mix * signed


class PhaseResidualBlendTinyNeuronBlock(PhaseResidualBlendNeuronBlock):
    """Same residual blend with a much smaller initial phase correction."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        nn.init.constant_(self.phase_mix, -4.0)


class PhaseResidualBlendLargeNeuronBlock(PhaseResidualBlendNeuronBlock):
    """Same residual blend with a larger initial phase correction."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        nn.init.constant_(self.phase_mix, -1.0)


class PhaseResidualBlendHalfNeuronBlock(PhaseResidualBlendNeuronBlock):
    """Same residual blend initialized at a half-strength phase correction."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        nn.init.constant_(self.phase_mix, 0.0)


class PhaseResidualBlendQuarterNeuronBlock(PhaseResidualBlendNeuronBlock):
    """Same residual blend initialized to exactly one-quarter signed-phase mix."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        nn.init.constant_(self.phase_mix, math.log(0.25 / 0.75))


class PhaseResidualBlendNormedNeuronBlock(PhaseResidualBlendNeuronBlock):
    """Signed-phase residual with token-local RMS limiting."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        nn.init.constant_(self.phase_mix, 0.0)

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base = F.relu(z).square()
        phase = torch.tanh(z)
        signed = F.softplus(z) * phase * phase.abs()
        signed_rms = signed.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        limited = signed / (1.0 + signed_rms)
        mix = torch.sigmoid(self.phase_mix).view(1, 1, -1)
        return base + mix * limited


class PhaseResidualBlendCenteredNeuronBlock(PhaseResidualBlendNeuronBlock):
    """Signed-phase residual with per-token channel mean removed."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        nn.init.constant_(self.phase_mix, 0.0)

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base = F.relu(z).square()
        phase = torch.tanh(z)
        signed = F.softplus(z) * phase * phase.abs()
        centered = signed - signed.mean(dim=-1, keepdim=True)
        mix = torch.sigmoid(self.phase_mix).view(1, 1, -1)
        return base + mix * centered


class PhaseResidualBoundaryBlendNeuronBlock(PhaseResidualBlendNeuronBlock):
    """Signed-phase residual concentrated near the ReLU-square decision boundary."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        nn.init.constant_(self.phase_mix, 0.0)

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base = F.relu(z).square()
        phase = torch.tanh(z)
        signed = F.softplus(z) * phase * phase.abs()
        boundary = z.abs().add(1.0).reciprocal()
        mix = torch.sigmoid(self.phase_mix).view(1, 1, -1)
        return base + mix * boundary * signed


class PhaseResidualBoundaryCenteredNeuronBlock(PhaseResidualBoundaryBlendNeuronBlock):
    """Boundary-local signed residual with per-token channel mean removed."""

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base = F.relu(z).square()
        phase = torch.tanh(z)
        signed = F.softplus(z) * phase * phase.abs()
        boundary = z.abs().add(1.0).reciprocal()
        local_signed = boundary * signed
        centered = local_signed - local_signed.mean(dim=-1, keepdim=True)
        mix = torch.sigmoid(self.phase_mix).view(1, 1, -1)
        return base + mix * centered


class PhaseResidualMemoryGateNeuronBlock(PhaseResidualBlendNeuronBlock):
    """Causal low-rank memory gates the signed-phase residual per token and channel."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        hidden_dim = self.expansion * self.dim
        nn.init.constant_(self.phase_mix, 0.0)
        self.memory_phase_gate = nn.Linear(self.memory_rank, hidden_dim)
        nn.init.zeros_(self.memory_phase_gate.weight)
        nn.init.zeros_(self.memory_phase_gate.bias)

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base = F.relu(z).square()
        phase = torch.tanh(z)
        signed = F.softplus(z) * phase * phase.abs()
        memory_gate = 0.5 * torch.tanh(self.memory_phase_gate(memory_output))
        gate = torch.sigmoid(self.phase_mix.view(1, 1, -1) + memory_gate)
        return base + gate * signed


class PhaseResidualMemoryScalarGateNeuronBlock(PhaseResidualBlendNeuronBlock):
    """Causal memory produces one scalar signed-phase residual gate per token."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        nn.init.constant_(self.phase_mix, 0.0)
        self.memory_phase_gate = nn.Linear(self.memory_rank, 1)
        nn.init.zeros_(self.memory_phase_gate.weight)
        nn.init.zeros_(self.memory_phase_gate.bias)

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base = F.relu(z).square()
        phase = torch.tanh(z)
        signed = F.softplus(z) * phase * phase.abs()
        memory_gate = 0.5 * torch.tanh(self.memory_phase_gate(memory_output))
        gate = torch.sigmoid(self.phase_mix.view(1, 1, -1) + memory_gate)
        return base + gate * signed


class PhaseResidualMemoryGroupGateNeuronBlock(PhaseResidualBlendNeuronBlock):
    """Causal memory gates signed-phase residuals in channel groups."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        hidden_dim = self.expansion * self.dim
        self.gate_groups = 16 if hidden_dim % 16 == 0 else 8
        nn.init.constant_(self.phase_mix, 0.0)
        self.memory_phase_gate = nn.Linear(self.memory_rank, self.gate_groups)
        nn.init.zeros_(self.memory_phase_gate.weight)
        nn.init.zeros_(self.memory_phase_gate.bias)

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base = F.relu(z).square()
        phase = torch.tanh(z)
        signed = F.softplus(z) * phase * phase.abs()
        group_gate = 0.5 * torch.tanh(self.memory_phase_gate(memory_output))
        expanded_gate = group_gate.repeat_interleave(signed.size(-1) // self.gate_groups, dim=-1)
        gate = torch.sigmoid(self.phase_mix.view(1, 1, -1) + expanded_gate)
        return base + gate * signed


class PhaseResidualMemoryCenteredGateNeuronBlock(PhaseResidualMemoryGateNeuronBlock):
    """Per-channel memory gate applied to a centered signed-phase residual."""

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base = F.relu(z).square()
        phase = torch.tanh(z)
        signed = F.softplus(z) * phase * phase.abs()
        centered = signed - signed.mean(dim=-1, keepdim=True)
        memory_gate = 0.5 * torch.tanh(self.memory_phase_gate(memory_output))
        gate = torch.sigmoid(self.phase_mix.view(1, 1, -1) + memory_gate)
        return base + gate * centered


class PhaseResidualMemoryScalarCenteredGateNeuronBlock(PhaseResidualMemoryScalarGateNeuronBlock):
    """Scalar memory gate applied to a centered signed-phase residual."""

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base = F.relu(z).square()
        phase = torch.tanh(z)
        signed = F.softplus(z) * phase * phase.abs()
        centered = signed - signed.mean(dim=-1, keepdim=True)
        memory_gate = 0.5 * torch.tanh(self.memory_phase_gate(memory_output))
        gate = torch.sigmoid(self.phase_mix.view(1, 1, -1) + memory_gate)
        return base + gate * centered


class PhaseResidualMemoryBoundaryGateNeuronBlock(PhaseResidualMemoryGateNeuronBlock):
    """Causal memory gates only boundary-local signed residual features."""

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base = F.relu(z).square()
        phase = torch.tanh(z)
        signed = F.softplus(z) * phase * phase.abs()
        boundary = z.abs().add(1.0).reciprocal()
        memory_gate = 0.5 * torch.tanh(self.memory_phase_gate(memory_output))
        gate = torch.sigmoid(self.phase_mix.view(1, 1, -1) + memory_gate)
        return base + gate * boundary * signed


class PhaseResidualMemoryBoundaryCenteredGateNeuronBlock(PhaseResidualMemoryGateNeuronBlock):
    """Memory-gated boundary-local residual with channel-centering."""

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base = F.relu(z).square()
        phase = torch.tanh(z)
        signed = F.softplus(z) * phase * phase.abs()
        boundary = z.abs().add(1.0).reciprocal()
        local_signed = boundary * signed
        centered = local_signed - local_signed.mean(dim=-1, keepdim=True)
        memory_gate = 0.5 * torch.tanh(self.memory_phase_gate(memory_output))
        gate = torch.sigmoid(self.phase_mix.view(1, 1, -1) + memory_gate)
        return base + gate * centered


class PhaseResidualMemoryBoundaryScalarGateNeuronBlock(PhaseResidualMemoryScalarGateNeuronBlock):
    """Token-wise memory gate over boundary-local signed residual features."""

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base = F.relu(z).square()
        phase = torch.tanh(z)
        signed = F.softplus(z) * phase * phase.abs()
        boundary = z.abs().add(1.0).reciprocal()
        memory_gate = 0.5 * torch.tanh(self.memory_phase_gate(memory_output))
        gate = torch.sigmoid(self.phase_mix.view(1, 1, -1) + memory_gate)
        return base + gate * boundary * signed


class RankCompetitionMemoryGateNeuronBlock(PhaseResidualMemoryGateNeuronBlock):
    """Rank-competitive squared base plus memory-gated signed residual."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        hidden_dim = self.expansion * self.dim
        self.groups = 8 if hidden_dim % 8 == 0 else 4
        self.inhibit = nn.Parameter(torch.tensor(0.25))

    def rank_competition_base(self, z: torch.Tensor) -> torch.Tensor:
        b, t, h = z.shape
        grouped = F.relu(z).view(b, t, self.groups, h // self.groups)
        pressure = grouped.mean(dim=-1, keepdim=True)
        competed = F.relu(grouped - torch.sigmoid(self.inhibit) * pressure)
        return competed.reshape(b, t, h).square()

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base = self.rank_competition_base(z)
        phase = torch.tanh(z)
        signed = F.softplus(z) * phase * phase.abs()
        memory_gate = 0.5 * torch.tanh(self.memory_phase_gate(memory_output))
        gate = torch.sigmoid(self.phase_mix.view(1, 1, -1) + memory_gate)
        return base + gate * signed


class RankCompetitionMemoryCenteredGateNeuronBlock(RankCompetitionMemoryGateNeuronBlock):
    """Rank-competitive squared base plus centered memory-gated signed residual."""

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base = self.rank_competition_base(z)
        phase = torch.tanh(z)
        signed = F.softplus(z) * phase * phase.abs()
        centered = signed - signed.mean(dim=-1, keepdim=True)
        memory_gate = 0.5 * torch.tanh(self.memory_phase_gate(memory_output))
        gate = torch.sigmoid(self.phase_mix.view(1, 1, -1) + memory_gate)
        return base + gate * centered


class RankCompetitionCenteredResidualNeuronBlock(RankCompetitionNeuronBlock):
    """Rank-competitive squared base plus a learned centered signed residual."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        hidden_dim = self.expansion * self.dim
        self.phase_mix = nn.Parameter(torch.zeros(hidden_dim))

    def rank_competition_base(self, z: torch.Tensor) -> torch.Tensor:
        b, t, h = z.shape
        grouped = F.relu(z).view(b, t, self.groups, h // self.groups)
        pressure = grouped.mean(dim=-1, keepdim=True)
        competed = F.relu(grouped - torch.sigmoid(self.inhibit) * pressure)
        return competed.reshape(b, t, h).square()

    def centered_residual(self, z: torch.Tensor) -> torch.Tensor:
        phase = torch.tanh(z)
        signed = F.softplus(z) * phase * phase.abs()
        return signed - signed.mean(dim=-1, keepdim=True)

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base = self.rank_competition_base(z)
        centered = self.centered_residual(z)
        gate = torch.sigmoid(self.phase_mix.view(1, 1, -1))
        return base + gate * centered


class RankCompetitionMildCenteredResidualNeuronBlock(RankCompetitionCenteredResidualNeuronBlock):
    """Centered residual on a mildly inhibited rank-competition base."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        nn.init.constant_(self.inhibit, -0.5)


class RankCompetitionSoftCenteredResidualNeuronBlock(RankCompetitionCenteredResidualNeuronBlock):
    """Centered residual on a softly inhibited rank-competition base."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        nn.init.constant_(self.inhibit, -1.5)


class RankCompetitionMemoryScalarCenteredGateNeuronBlock(RankCompetitionCenteredResidualNeuronBlock):
    """Rank-competitive centered residual gated by one memory scalar per token."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.memory_phase_gate = nn.Linear(self.memory_rank, 1)
        nn.init.zeros_(self.memory_phase_gate.weight)
        nn.init.zeros_(self.memory_phase_gate.bias)

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base = self.rank_competition_base(z)
        centered = self.centered_residual(z)
        memory_gate = 0.5 * torch.tanh(self.memory_phase_gate(memory_output))
        gate = torch.sigmoid(self.phase_mix.view(1, 1, -1) + memory_gate)
        return base + gate * centered


class RankCompetitionMemoryGroupCenteredGateNeuronBlock(RankCompetitionCenteredResidualNeuronBlock):
    """Rank-competitive centered residual gated by coarse memory channel groups."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        hidden_dim = self.expansion * self.dim
        self.gate_groups = 16 if hidden_dim % 16 == 0 else 8
        self.memory_phase_gate = nn.Linear(self.memory_rank, self.gate_groups)
        nn.init.zeros_(self.memory_phase_gate.weight)
        nn.init.zeros_(self.memory_phase_gate.bias)

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base = self.rank_competition_base(z)
        centered = self.centered_residual(z)
        group_gate = 0.5 * torch.tanh(self.memory_phase_gate(memory_output))
        expanded_gate = group_gate.repeat_interleave(centered.size(-1) // self.gate_groups, dim=-1)
        gate = torch.sigmoid(self.phase_mix.view(1, 1, -1) + expanded_gate)
        return base + gate * centered


class RankCompetitionMemoryFactorCenteredGateNeuronBlock(RankCompetitionCenteredResidualNeuronBlock):
    """Rank-competitive centered residual gated by a factorized memory-to-channel route."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        hidden_dim = self.expansion * self.dim
        gate_rank = min(16, max(4, self.memory_rank // 4))
        self.memory_gate_down = nn.Linear(self.memory_rank, gate_rank)
        self.memory_gate_up = nn.Linear(gate_rank, hidden_dim, bias=False)
        nn.init.zeros_(self.memory_gate_up.weight)

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base = self.rank_competition_base(z)
        centered = self.centered_residual(z)
        route = self.memory_gate_up(F.silu(self.memory_gate_down(memory_output)))
        memory_gate = 0.5 * torch.tanh(route)
        gate = torch.sigmoid(self.phase_mix.view(1, 1, -1) + memory_gate)
        return base + gate * centered


class RankCompetitionMemorySmallCenteredGateNeuronBlock(RankCompetitionMemoryCenteredGateNeuronBlock):
    """Full per-channel memory gate with smaller initial residual and route amplitude."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        nn.init.constant_(self.phase_mix, -1.0)

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base = self.rank_competition_base(z)
        phase = torch.tanh(z)
        signed = F.softplus(z) * phase * phase.abs()
        centered = signed - signed.mean(dim=-1, keepdim=True)
        memory_gate = 0.25 * torch.tanh(self.memory_phase_gate(memory_output))
        gate = torch.sigmoid(self.phase_mix.view(1, 1, -1) + memory_gate)
        return base + gate * centered


class RankCompetitionMemoryNormedCenteredGateNeuronBlock(RankCompetitionMemoryCenteredGateNeuronBlock):
    """Full per-channel memory gate with token-local route and residual normalization."""

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base = self.rank_competition_base(z)
        phase = torch.tanh(z)
        signed = F.softplus(z) * phase * phase.abs()
        centered = signed - signed.mean(dim=-1, keepdim=True)
        centered_rms = centered.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        limited = centered / (1.0 + centered_rms)
        route = torch.tanh(self.memory_phase_gate(memory_output))
        route = route - route.mean(dim=-1, keepdim=True)
        route_rms = route.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        memory_gate = 0.5 * route / (1.0 + route_rms)
        gate = torch.sigmoid(self.phase_mix.view(1, 1, -1) + memory_gate)
        return base + gate * limited


class RankCompetitionMemoryUncertaintyCenteredGateNeuronBlock(RankCompetitionMemoryCenteredGateNeuronBlock):
    """Memory residual is routed only near the rank-competition decision boundary."""

    def competition_base_and_uncertainty(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, h = z.shape
        grouped = F.relu(z).view(b, t, self.groups, h // self.groups)
        pressure = grouped.mean(dim=-1, keepdim=True)
        threshold = torch.sigmoid(self.inhibit) * pressure
        margin = grouped - threshold
        base = F.relu(margin).reshape(b, t, h).square()
        scale = pressure.abs().add(1e-3)
        uncertainty = torch.exp(-2.0 * margin.abs() / scale).reshape(b, t, h)
        return base, uncertainty

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base, uncertainty = self.competition_base_and_uncertainty(z)
        phase = torch.tanh(z)
        signed = F.softplus(z) * phase * phase.abs()
        centered = signed - signed.mean(dim=-1, keepdim=True)
        memory_gate = 0.5 * torch.tanh(self.memory_phase_gate(memory_output))
        gate = torch.sigmoid(self.phase_mix.view(1, 1, -1) + memory_gate)
        return base + gate * uncertainty * centered


class RankCompetitionMemorySuppressedCenteredGateNeuronBlock(RankCompetitionMemoryCenteredGateNeuronBlock):
    """Memory residual is routed preferentially through channels suppressed by competition."""

    def competition_base_and_suppressed(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, h = z.shape
        grouped = F.relu(z).view(b, t, self.groups, h // self.groups)
        pressure = grouped.mean(dim=-1, keepdim=True)
        threshold = torch.sigmoid(self.inhibit) * pressure
        margin = grouped - threshold
        base = F.relu(margin).reshape(b, t, h).square()
        scale = pressure.abs().add(1e-3)
        suppressed = torch.sigmoid(-4.0 * margin / scale).reshape(b, t, h)
        return base, suppressed

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base, suppressed = self.competition_base_and_suppressed(z)
        phase = torch.tanh(z)
        signed = F.softplus(z) * phase * phase.abs()
        centered = signed - signed.mean(dim=-1, keepdim=True)
        memory_gate = 0.5 * torch.tanh(self.memory_phase_gate(memory_output))
        gate = torch.sigmoid(self.phase_mix.view(1, 1, -1) + memory_gate)
        return base + gate * suppressed * centered


class RankCompetitionMemorySuppressedSmallCenteredGateNeuronBlock(RankCompetitionMemorySuppressedCenteredGateNeuronBlock):
    """Suppressed-channel memory route with lower initial route strength."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        nn.init.constant_(self.phase_mix, -1.0)

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base, suppressed = self.competition_base_and_suppressed(z)
        phase = torch.tanh(z)
        signed = F.softplus(z) * phase * phase.abs()
        centered = signed - signed.mean(dim=-1, keepdim=True)
        memory_gate = 0.25 * torch.tanh(self.memory_phase_gate(memory_output))
        gate = torch.sigmoid(self.phase_mix.view(1, 1, -1) + memory_gate)
        return base + gate * suppressed * centered


class RankCompetitionMemorySuppressedBoundedCenteredGateNeuronBlock(RankCompetitionMemorySuppressedCenteredGateNeuronBlock):
    """Suppressed-channel memory route with a hard-bounded centered residual."""

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base, suppressed = self.competition_base_and_suppressed(z)
        phase = torch.tanh(z)
        signed = F.softplus(z) * phase * phase.abs()
        centered = signed - signed.mean(dim=-1, keepdim=True)
        bounded = torch.tanh(centered)
        memory_gate = 0.5 * torch.tanh(self.memory_phase_gate(memory_output))
        gate = torch.sigmoid(self.phase_mix.view(1, 1, -1) + memory_gate)
        return base + gate * suppressed * bounded


class RankCompetitionMemorySuppressedBoundedSmallCenteredGateNeuronBlock(
    RankCompetitionMemorySuppressedBoundedCenteredGateNeuronBlock
):
    """Suppressed-channel memory route with bounded residual and smaller gate."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        nn.init.constant_(self.phase_mix, -1.0)

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base, suppressed = self.competition_base_and_suppressed(z)
        phase = torch.tanh(z)
        signed = F.softplus(z) * phase * phase.abs()
        centered = signed - signed.mean(dim=-1, keepdim=True)
        bounded = torch.tanh(centered)
        memory_gate = 0.25 * torch.tanh(self.memory_phase_gate(memory_output))
        gate = torch.sigmoid(self.phase_mix.view(1, 1, -1) + memory_gate)
        return base + gate * suppressed * bounded


class RankCompetitionMemorySuppressedNormedCenteredGateNeuronBlock(RankCompetitionMemorySuppressedCenteredGateNeuronBlock):
    """Suppressed-channel memory route with RMS-limited centered residual."""

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base, suppressed = self.competition_base_and_suppressed(z)
        phase = torch.tanh(z)
        signed = F.softplus(z) * phase * phase.abs()
        centered = signed - signed.mean(dim=-1, keepdim=True)
        centered_rms = centered.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        limited = centered / (1.0 + centered_rms)
        memory_gate = 0.5 * torch.tanh(self.memory_phase_gate(memory_output))
        gate = torch.sigmoid(self.phase_mix.view(1, 1, -1) + memory_gate)
        return base + gate * suppressed * limited


class RankCompetitionMemorySuppressedEnergyMatchedCenteredGateNeuronBlock(
    RankCompetitionMemorySuppressedCenteredGateNeuronBlock
):
    """Suppressed-channel memory route with residual energy matched to the base neuron."""

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base, suppressed = self.competition_base_and_suppressed(z)
        phase = torch.tanh(z)
        signed = F.softplus(z) * phase * phase.abs()
        centered = signed - signed.mean(dim=-1, keepdim=True)
        residual = suppressed * torch.tanh(centered)
        base_rms = base.float().pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt().detach().to(base.dtype)
        residual_rms = residual.float().pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt().to(base.dtype)
        matched = residual * (base_rms / (base_rms + residual_rms))
        memory_gate = 0.25 * torch.tanh(self.memory_phase_gate(memory_output))
        gate = torch.sigmoid(self.phase_mix.view(1, 1, -1) + memory_gate)
        return base + gate * matched


class RankCompetitionMemorySuppressedStopgradMaskCenteredGateNeuronBlock(
    RankCompetitionMemorySuppressedCenteredGateNeuronBlock
):
    """Suppressed-channel memory route with detached competition mask."""

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base, suppressed = self.competition_base_and_suppressed(z)
        phase = torch.tanh(z)
        signed = F.softplus(z) * phase * phase.abs()
        centered = signed - signed.mean(dim=-1, keepdim=True)
        bounded = torch.tanh(centered)
        memory_gate = 0.5 * torch.tanh(self.memory_phase_gate(memory_output))
        gate = torch.sigmoid(self.phase_mix.view(1, 1, -1) + memory_gate)
        return base + gate * suppressed.detach() * bounded


class RankCompetitionMemorySuppressedAuxHeadCenteredGateNeuronBlock(
    RankCompetitionMemorySuppressedCenteredGateNeuronBlock
):
    """Suppressed-channel memory route projected through a separate zero-init output head."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        hidden_dim = self.expansion * self.dim
        self.residual_out = nn.Linear(hidden_dim, self.dim, bias=False)
        self.residual_head_scale = nn.Parameter(torch.tensor(-1.0))
        nn.init.zeros_(self.residual_out.weight)

    def neuron_parts(
        self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.ffn_in(self.ffn_norm(x))
        base, suppressed = self.competition_base_and_suppressed(z)
        phase = torch.tanh(z)
        signed = F.softplus(z) * phase * phase.abs()
        centered = signed - signed.mean(dim=-1, keepdim=True)
        bounded = torch.tanh(centered)
        memory_gate = 0.25 * torch.tanh(self.memory_phase_gate(memory_output))
        gate = torch.sigmoid(self.phase_mix.view(1, 1, -1) + memory_gate)
        return base, gate * suppressed * bounded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, conv_output, memory_output = self.conv_and_memory(x)
        base, residual = self.neuron_parts(x, conv_output, memory_output)
        residual_gain = 0.25 * torch.sigmoid(self.residual_head_scale)
        return x + self.dropout(self.ffn_out(base) + residual_gain * self.residual_out(residual))


class RankCompetitionMemoryWithinGroupCenteredGateNeuronBlock(RankCompetitionMemoryCenteredGateNeuronBlock):
    """Full memory gate over a signed residual centered inside competition groups."""

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base = self.rank_competition_base(z)
        b, t, h = z.shape
        phase = torch.tanh(z)
        signed = F.softplus(z) * phase * phase.abs()
        grouped = signed.view(b, t, self.groups, h // self.groups)
        centered = (grouped - grouped.mean(dim=-1, keepdim=True)).reshape(b, t, h)
        memory_gate = 0.5 * torch.tanh(self.memory_phase_gate(memory_output))
        gate = torch.sigmoid(self.phase_mix.view(1, 1, -1) + memory_gate)
        return base + gate * centered


class PhaseResidualMemoryNormedGateNeuronBlock(PhaseResidualMemoryGateNeuronBlock):
    """Per-channel memory gate applied to an RMS-limited signed-phase residual."""

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base = F.relu(z).square()
        phase = torch.tanh(z)
        signed = F.softplus(z) * phase * phase.abs()
        signed_rms = signed.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        limited = signed / (1.0 + signed_rms)
        memory_gate = 0.5 * torch.tanh(self.memory_phase_gate(memory_output))
        gate = torch.sigmoid(self.phase_mix.view(1, 1, -1) + memory_gate)
        return base + gate * limited


class PhaseResidualMemorySmallGateNeuronBlock(PhaseResidualMemoryGateNeuronBlock):
    """Per-channel memory gate with lower initial mix and lower memory amplitude."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        nn.init.constant_(self.phase_mix, -1.0)

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base = F.relu(z).square()
        phase = torch.tanh(z)
        signed = F.softplus(z) * phase * phase.abs()
        memory_gate = 0.25 * torch.tanh(self.memory_phase_gate(memory_output))
        gate = torch.sigmoid(self.phase_mix.view(1, 1, -1) + memory_gate)
        return base + gate * signed


class PhaseResidualMemoryTinyGateNeuronBlock(PhaseResidualMemoryGateNeuronBlock):
    """Per-channel memory gate as a small perturbation around the baseline neuron."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        nn.init.constant_(self.phase_mix, -2.0)

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base = F.relu(z).square()
        phase = torch.tanh(z)
        signed = F.softplus(z) * phase * phase.abs()
        memory_gate = 0.25 * torch.tanh(self.memory_phase_gate(memory_output))
        gate = torch.sigmoid(self.phase_mix.view(1, 1, -1) + memory_gate)
        return base + gate * signed


class PhaseResidualMemoryZeroMeanGateNeuronBlock(PhaseResidualMemoryGateNeuronBlock):
    """Per-channel memory gate with token-wise gate mean removed before sigmoid."""

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base = F.relu(z).square()
        phase = torch.tanh(z)
        signed = F.softplus(z) * phase * phase.abs()
        memory_gate = 0.5 * torch.tanh(self.memory_phase_gate(memory_output))
        memory_gate = memory_gate - memory_gate.mean(dim=-1, keepdim=True)
        gate = torch.sigmoid(self.phase_mix.view(1, 1, -1) + memory_gate)
        return base + gate * signed


class PhaseResidualMemoryBoundedGateNeuronBlock(PhaseResidualMemoryGateNeuronBlock):
    """Per-channel memory gate constrained to a residual-safe range."""

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base = F.relu(z).square()
        phase = torch.tanh(z)
        signed = F.softplus(z) * phase * phase.abs()
        memory_gate = 0.5 * torch.tanh(self.memory_phase_gate(memory_output))
        gate = 0.25 + 0.5 * torch.sigmoid(self.phase_mix.view(1, 1, -1) + memory_gate)
        return base + gate * signed


class PhaseResidualMemoryBoundedCenteredGateNeuronBlock(PhaseResidualMemoryGateNeuronBlock):
    """Bounded memory gate applied to channel-centered signed residuals."""

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base = F.relu(z).square()
        phase = torch.tanh(z)
        signed = F.softplus(z) * phase * phase.abs()
        centered = signed - signed.mean(dim=-1, keepdim=True)
        memory_gate = 0.5 * torch.tanh(self.memory_phase_gate(memory_output))
        gate = 0.25 + 0.5 * torch.sigmoid(self.phase_mix.view(1, 1, -1) + memory_gate)
        return base + gate * centered


class PhaseResidualMemoryRmsGateNeuronBlock(PhaseResidualMemoryGateNeuronBlock):
    """Per-channel memory gate normalized by token-local routing RMS."""

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base = F.relu(z).square()
        phase = torch.tanh(z)
        signed = F.softplus(z) * phase * phase.abs()
        memory_gate = torch.tanh(self.memory_phase_gate(memory_output))
        gate_rms = memory_gate.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        memory_gate = 0.5 * memory_gate / (1.0 + gate_rms)
        gate = torch.sigmoid(self.phase_mix.view(1, 1, -1) + memory_gate)
        return base + gate * signed


class PhaseResidualMemorySparseGateNeuronBlock(PhaseResidualMemoryGateNeuronBlock):
    """Only high-confidence memory routes alter the signed residual gate."""

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base = F.relu(z).square()
        phase = torch.tanh(z)
        signed = F.softplus(z) * phase * phase.abs()
        route = torch.tanh(self.memory_phase_gate(memory_output))
        route = route - route.mean(dim=-1, keepdim=True)
        threshold = route.abs().mean(dim=-1, keepdim=True)
        sparse_route = route.sign() * F.relu(route.abs() - threshold)
        gate = torch.sigmoid(self.phase_mix.view(1, 1, -1) + 0.5 * sparse_route)
        return base + gate * signed


class PhaseResidualMemorySparseCenteredGateNeuronBlock(PhaseResidualMemorySparseGateNeuronBlock):
    """Sparse memory route over a channel-centered signed residual."""

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base = F.relu(z).square()
        phase = torch.tanh(z)
        signed = F.softplus(z) * phase * phase.abs()
        centered = signed - signed.mean(dim=-1, keepdim=True)
        route = torch.tanh(self.memory_phase_gate(memory_output))
        route = route - route.mean(dim=-1, keepdim=True)
        threshold = route.abs().mean(dim=-1, keepdim=True)
        sparse_route = route.sign() * F.relu(route.abs() - threshold)
        gate = torch.sigmoid(self.phase_mix.view(1, 1, -1) + 0.5 * sparse_route)
        return base + gate * centered


class PhaseResidualMemoryDeltaGateNeuronBlock(PhaseResidualMemoryGateNeuronBlock):
    """Use causal changes in memory state, not static memory state, to route residuals."""

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base = F.relu(z).square()
        phase = torch.tanh(z)
        signed = F.softplus(z) * phase * phase.abs()
        previous = F.pad(memory_output[:, :-1, :], (0, 0, 1, 0))
        delta = memory_output - previous
        memory_gate = 0.5 * torch.tanh(self.memory_phase_gate(delta))
        gate = torch.sigmoid(self.phase_mix.view(1, 1, -1) + memory_gate)
        return base + gate * signed


class MemorySquareGainNeuronBlock(LowRankConvMemoryVariantBase):
    """Causal low-rank memory multiplicatively gates the baseline squared FFN features."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        hidden_dim = self.expansion * self.dim
        self.memory_gain = nn.Linear(self.memory_rank, hidden_dim)
        nn.init.zeros_(self.memory_gain.weight)
        nn.init.zeros_(self.memory_gain.bias)

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base = F.relu(z).square()
        gain = 1.0 + 0.25 * torch.tanh(self.memory_gain(memory_output))
        return base * gain


class MemorySquareCenteredGainNeuronBlock(MemorySquareGainNeuronBlock):
    """Memory square gain with token-wise gain mean removed before modulation."""

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base = F.relu(z).square()
        raw_gain = torch.tanh(self.memory_gain(memory_output))
        raw_gain = raw_gain - raw_gain.mean(dim=-1, keepdim=True)
        gain = 1.0 + 0.25 * raw_gain
        return base * gain


class MemorySquareSmallGainNeuronBlock(MemorySquareGainNeuronBlock):
    """Memory square gain with smaller modulation amplitude."""

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base = F.relu(z).square()
        gain = 1.0 + 0.10 * torch.tanh(self.memory_gain(memory_output))
        return base * gain


class MemorySquareScalarGainNeuronBlock(LowRankConvMemoryVariantBase):
    """Causal low-rank memory produces one token-wise gain for all squared FFN features."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.memory_gain = nn.Linear(self.memory_rank, 1)
        nn.init.zeros_(self.memory_gain.weight)
        nn.init.zeros_(self.memory_gain.bias)

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base = F.relu(z).square()
        gain = 1.0 + 0.25 * torch.tanh(self.memory_gain(memory_output))
        return base * gain


def causal_center(x: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(1, x.size(1) + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
    return x - x.cumsum(dim=1) / positions


class PhaseResidualMemoryNoveltyGateNeuronBlock(PhaseResidualMemoryGateNeuronBlock):
    """Gate signed phase residuals from memory novelty rather than raw memory state."""

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base = F.relu(z).square()
        phase = torch.tanh(z)
        signed = F.softplus(z) * phase * phase.abs()
        novelty = causal_center(memory_output)
        memory_gate = 0.5 * torch.tanh(self.memory_phase_gate(novelty))
        gate = torch.sigmoid(self.phase_mix.view(1, 1, -1) + memory_gate)
        return base + gate * signed


class PhaseResidualMemoryNoveltyCenteredGateNeuronBlock(PhaseResidualMemoryGateNeuronBlock):
    """Memory-novelty gate applied to channel-centered signed phase residuals."""

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base = F.relu(z).square()
        phase = torch.tanh(z)
        signed = F.softplus(z) * phase * phase.abs()
        centered = signed - signed.mean(dim=-1, keepdim=True)
        novelty = causal_center(memory_output)
        memory_gate = 0.5 * torch.tanh(self.memory_phase_gate(novelty))
        gate = torch.sigmoid(self.phase_mix.view(1, 1, -1) + memory_gate)
        return base + gate * centered


class PhaseResidualMemoryNoveltyTinyGateNeuronBlock(PhaseResidualMemoryNoveltyGateNeuronBlock):
    """Memory-novelty gate with lower initial signed phase residual strength."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        nn.init.constant_(self.phase_mix, -1.0)


class MemorySquareNoveltyGainNeuronBlock(MemorySquareGainNeuronBlock):
    """Memory novelty multiplicatively gates baseline squared FFN features."""

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base = F.relu(z).square()
        novelty = causal_center(memory_output)
        gain = 1.0 + 0.25 * torch.tanh(self.memory_gain(novelty))
        return base * gain


class PhaseResidualMemoryConvAgreementNeuronBlock(PhaseResidualBlendNeuronBlock):
    """Signed phase residual gated by agreement between conv context and causal memory."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        nn.init.constant_(self.phase_mix, 0.0)
        self.agreement_scale = nn.Parameter(torch.tensor(1.0))

    def memory_context(self, memory_output: torch.Tensor) -> torch.Tensor:
        return self.memory_up(F.silu(memory_output))

    def agreement(self, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        memory_vec = self.memory_context(memory_output)
        conv_unit = F.normalize(conv_output.float(), dim=-1, eps=1e-4)
        memory_unit = F.normalize(memory_vec.float(), dim=-1, eps=1e-4)
        return (conv_unit * memory_unit).mean(dim=-1, keepdim=True).to(conv_output.dtype)

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base = F.relu(z).square()
        phase = torch.tanh(z)
        signed = F.softplus(z) * phase * phase.abs()
        agreement = self.agreement(conv_output, memory_output)
        gate = torch.sigmoid(self.phase_mix.view(1, 1, -1) + self.agreement_scale * agreement)
        return base + gate * signed


class PhaseResidualMemoryConvDisagreementDampNeuronBlock(PhaseResidualMemoryConvAgreementNeuronBlock):
    """Suppress signed phase residuals when conv and memory disagree."""

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base = F.relu(z).square()
        phase = torch.tanh(z)
        signed = F.softplus(z) * phase * phase.abs()
        agreement = self.agreement(conv_output, memory_output)
        disagreement = F.relu(-agreement)
        gate = torch.sigmoid(self.phase_mix.view(1, 1, -1) - F.softplus(self.agreement_scale) * disagreement)
        return base + gate * signed


class PhaseResidualMemoryConvCenteredAgreementNeuronBlock(PhaseResidualMemoryConvAgreementNeuronBlock):
    """Agreement-gated phase residual with channel-centered signed carrier."""

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base = F.relu(z).square()
        phase = torch.tanh(z)
        signed = F.softplus(z) * phase * phase.abs()
        centered = signed - signed.mean(dim=-1, keepdim=True)
        agreement = self.agreement(conv_output, memory_output)
        gate = torch.sigmoid(self.phase_mix.view(1, 1, -1) + self.agreement_scale * agreement)
        return base + gate * centered


class MemorySquareConvAgreementGainNeuronBlock(MemorySquareScalarGainNeuronBlock):
    """Token-wise square gain controlled by conv-memory agreement."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.agreement_scale = nn.Parameter(torch.tensor(0.25))

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base = F.relu(z).square()
        memory_vec = self.memory_up(F.silu(memory_output))
        conv_unit = F.normalize(conv_output.float(), dim=-1, eps=1e-4)
        memory_unit = F.normalize(memory_vec.float(), dim=-1, eps=1e-4)
        agreement = (conv_unit * memory_unit).mean(dim=-1, keepdim=True).to(base.dtype)
        gain = 1.0 + torch.tanh(self.agreement_scale) * agreement
        return base * gain


class PhaseResidualConvEnergyGateNeuronBlock(PhaseResidualBlendNeuronBlock):
    """Local conv energy suppresses the signed-phase residual in high-curvature contexts."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        nn.init.constant_(self.phase_mix, 0.0)
        self.energy_scale = nn.Parameter(torch.tensor(0.25))

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        base = F.relu(z).square()
        phase = torch.tanh(z)
        signed = F.softplus(z) * phase * phase.abs()
        conv_energy = torch.log1p(conv_output.pow(2).mean(dim=-1, keepdim=True))
        gate = torch.sigmoid(self.phase_mix.view(1, 1, -1) - F.softplus(self.energy_scale) * conv_energy)
        return base + gate * signed


class StableSquareNeuronBlock(LowRankConvMemoryVariantBase):
    """Squared activation normalized by local RMS with a small linear leak."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.leak = nn.Parameter(torch.tensor(0.05))

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        positive = F.relu(z)
        squared = positive.square()
        rms = squared.pow(2).mean(dim=-1, keepdim=True).add(1e-4).sqrt()
        return squared / (1.0 + rms) + torch.sigmoid(self.leak) * 0.05 * z


class StableCompetitionNeuronBlock(LowRankConvMemoryVariantBase):
    """Rank competition plus RMS-stabilized square for a low-cost second iteration."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        hidden_dim = self.expansion * self.dim
        self.groups = 8 if hidden_dim % 8 == 0 else 4
        self.inhibit = nn.Parameter(torch.tensor(0.20))
        self.leak = nn.Parameter(torch.tensor(0.03))

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        z = self.ffn_in(self.ffn_norm(x))
        b, t, h = z.shape
        positive = F.relu(z).view(b, t, self.groups, h // self.groups)
        pressure = positive.mean(dim=-1, keepdim=True)
        competed = F.relu(positive - torch.sigmoid(self.inhibit) * pressure).reshape(b, t, h)
        squared = competed.square()
        rms = squared.pow(2).mean(dim=-1, keepdim=True).add(1e-4).sqrt()
        return squared / (1.0 + rms) + torch.sigmoid(self.leak) * 0.03 * z


class BottleneckAwareNeuronBlock(LowRankConvMemoryVariantBase):
    """Inject a low-rank residual basis into FFN features before the factorized output bottleneck."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        hidden_dim = self.expansion * self.dim
        bottleneck = max(4, self.memory_rank)
        self.bottleneck_down = nn.Linear(self.dim, bottleneck, bias=False)
        self.bottleneck_up = nn.Linear(bottleneck, hidden_dim, bias=False)
        self.bottleneck_gate = nn.Linear(self.dim, hidden_dim)
        nn.init.zeros_(self.bottleneck_up.weight)
        nn.init.zeros_(self.bottleneck_gate.weight)
        nn.init.constant_(self.bottleneck_gate.bias, -2.0)

    def neuron(self, x: torch.Tensor, conv_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        u = self.ffn_norm(x)
        z = self.ffn_in(u)
        low_rank_echo = self.bottleneck_up(F.silu(self.bottleneck_down(u)))
        gate = torch.sigmoid(self.bottleneck_gate(u))
        return F.relu(z + gate * low_rank_echo).square()


VARIANTS: dict[str, type[nn.Module]] = {
    "hidden_drop_square_neuron": HiddenDropSquareNeuronBlock,
    "hidden_drop_low_square_neuron": HiddenDropLowSquareNeuronBlock,
    "hidden_drop_mid_square_neuron": HiddenDropMidSquareNeuronBlock,
    "hidden_drop_high_square_neuron": HiddenDropHighSquareNeuronBlock,
    "hidden_drop_ultra_square_neuron": HiddenDropUltraSquareNeuronBlock,
    "hidden_drop_extreme_square_neuron": HiddenDropExtremeSquareNeuronBlock,
    "hidden_drop_p25_square_neuron": HiddenDropP25SquareNeuronBlock,
    "hidden_drop_p30_square_neuron": HiddenDropP30SquareNeuronBlock,
    "hidden_drop_p35_square_neuron": HiddenDropP35SquareNeuronBlock,
    "hidden_drop_p40_square_neuron": HiddenDropP40SquareNeuronBlock,
    "channel_drop_square_neuron": ChannelDropSquareNeuronBlock,
    "memory_energy_drop_square_neuron": MemoryEnergyDropSquareNeuronBlock,
    "mem_threshold_neuron": MemoryThresholdNeuronBlock,
    "conv_disagreement_neuron": ConvDisagreementNeuronBlock,
    "adaptive_basis_neuron": AdaptiveBasisNeuronBlock,
    "rank_competition_neuron": RankCompetitionNeuronBlock,
    "rank_competition_mild_neuron": RankCompetitionMildNeuronBlock,
    "rank_competition_soft_neuron": RankCompetitionSoftNeuronBlock,
    "rank_competition_ultrasoft_neuron": RankCompetitionUltraSoftNeuronBlock,
    "rank_competition_feather_neuron": RankCompetitionFeatherNeuronBlock,
    "rank_competition_fixed_feather_neuron": RankCompetitionFixedFeatherNeuronBlock,
    "rank_competition_fixed_trace_neuron": RankCompetitionFixedTraceNeuronBlock,
    "rank_competition_fixed_dust_neuron": RankCompetitionFixedDustNeuronBlock,
    "rank_competition_fixed_feather_hidden_drop_neuron": RankCompetitionFixedFeatherHiddenDropNeuronBlock,
    "rank_competition_fixed_feather_channel_drop_neuron": RankCompetitionFixedFeatherChannelDropNeuronBlock,
    "stateful_threshold_neuron": StatefulThresholdNeuronBlock,
    "phase_amplitude_neuron": PhaseAmplitudeNeuronBlock,
    "phase_amplitude_neutral_neuron": PhaseAmplitudeNeutralNeuronBlock,
    "phase_amplitude_one_extra_neuron": PhaseAmplitudeOneExtraNeuronBlock,
    "phase_amplitude_replace_neuron": PhaseAmplitudeReplaceNeuronBlock,
    "phase_residual_blend_neuron": PhaseResidualBlendNeuronBlock,
    "phase_residual_blend_tiny_neuron": PhaseResidualBlendTinyNeuronBlock,
    "phase_residual_blend_large_neuron": PhaseResidualBlendLargeNeuronBlock,
    "phase_residual_blend_half_neuron": PhaseResidualBlendHalfNeuronBlock,
    "phase_residual_blend_quarter_neuron": PhaseResidualBlendQuarterNeuronBlock,
    "phase_residual_blend_normed_neuron": PhaseResidualBlendNormedNeuronBlock,
    "phase_residual_blend_centered_neuron": PhaseResidualBlendCenteredNeuronBlock,
    "phase_residual_boundary_blend_neuron": PhaseResidualBoundaryBlendNeuronBlock,
    "phase_residual_boundary_centered_neuron": PhaseResidualBoundaryCenteredNeuronBlock,
    "phase_residual_memory_gate_neuron": PhaseResidualMemoryGateNeuronBlock,
    "phase_residual_memory_scalar_gate_neuron": PhaseResidualMemoryScalarGateNeuronBlock,
    "phase_residual_memory_group_gate_neuron": PhaseResidualMemoryGroupGateNeuronBlock,
    "phase_residual_memory_centered_gate_neuron": PhaseResidualMemoryCenteredGateNeuronBlock,
    "phase_residual_memory_scalar_centered_gate_neuron": PhaseResidualMemoryScalarCenteredGateNeuronBlock,
    "phase_residual_memory_boundary_gate_neuron": PhaseResidualMemoryBoundaryGateNeuronBlock,
    "phase_residual_memory_boundary_centered_gate_neuron": PhaseResidualMemoryBoundaryCenteredGateNeuronBlock,
    "phase_residual_memory_boundary_scalar_gate_neuron": PhaseResidualMemoryBoundaryScalarGateNeuronBlock,
    "rank_competition_memory_gate_neuron": RankCompetitionMemoryGateNeuronBlock,
    "rank_competition_memory_centered_gate_neuron": RankCompetitionMemoryCenteredGateNeuronBlock,
    "rank_competition_centered_residual_neuron": RankCompetitionCenteredResidualNeuronBlock,
    "rank_competition_mild_centered_residual_neuron": RankCompetitionMildCenteredResidualNeuronBlock,
    "rank_competition_soft_centered_residual_neuron": RankCompetitionSoftCenteredResidualNeuronBlock,
    "rank_competition_memory_scalar_centered_gate_neuron": RankCompetitionMemoryScalarCenteredGateNeuronBlock,
    "rank_competition_memory_group_centered_gate_neuron": RankCompetitionMemoryGroupCenteredGateNeuronBlock,
    "rank_competition_memory_factor_centered_gate_neuron": RankCompetitionMemoryFactorCenteredGateNeuronBlock,
    "rank_competition_memory_small_centered_gate_neuron": RankCompetitionMemorySmallCenteredGateNeuronBlock,
    "rank_competition_memory_normed_centered_gate_neuron": RankCompetitionMemoryNormedCenteredGateNeuronBlock,
    "rank_competition_memory_uncertainty_centered_gate_neuron": RankCompetitionMemoryUncertaintyCenteredGateNeuronBlock,
    "rank_competition_memory_suppressed_centered_gate_neuron": RankCompetitionMemorySuppressedCenteredGateNeuronBlock,
    "rank_competition_memory_suppressed_small_centered_gate_neuron": RankCompetitionMemorySuppressedSmallCenteredGateNeuronBlock,
    "rank_competition_memory_suppressed_bounded_centered_gate_neuron": RankCompetitionMemorySuppressedBoundedCenteredGateNeuronBlock,
    "rank_competition_memory_suppressed_bounded_small_centered_gate_neuron": RankCompetitionMemorySuppressedBoundedSmallCenteredGateNeuronBlock,
    "rank_competition_memory_suppressed_normed_centered_gate_neuron": RankCompetitionMemorySuppressedNormedCenteredGateNeuronBlock,
    "rank_competition_memory_suppressed_energy_matched_centered_gate_neuron": RankCompetitionMemorySuppressedEnergyMatchedCenteredGateNeuronBlock,
    "rank_competition_memory_suppressed_stopgrad_mask_centered_gate_neuron": RankCompetitionMemorySuppressedStopgradMaskCenteredGateNeuronBlock,
    "rank_competition_memory_suppressed_aux_head_centered_gate_neuron": RankCompetitionMemorySuppressedAuxHeadCenteredGateNeuronBlock,
    "rank_competition_memory_within_group_centered_gate_neuron": RankCompetitionMemoryWithinGroupCenteredGateNeuronBlock,
    "phase_residual_memory_normed_gate_neuron": PhaseResidualMemoryNormedGateNeuronBlock,
    "phase_residual_memory_small_gate_neuron": PhaseResidualMemorySmallGateNeuronBlock,
    "phase_residual_memory_tiny_gate_neuron": PhaseResidualMemoryTinyGateNeuronBlock,
    "phase_residual_memory_zero_mean_gate_neuron": PhaseResidualMemoryZeroMeanGateNeuronBlock,
    "phase_residual_memory_bounded_gate_neuron": PhaseResidualMemoryBoundedGateNeuronBlock,
    "phase_residual_memory_bounded_centered_gate_neuron": PhaseResidualMemoryBoundedCenteredGateNeuronBlock,
    "phase_residual_memory_rms_gate_neuron": PhaseResidualMemoryRmsGateNeuronBlock,
    "phase_residual_memory_sparse_gate_neuron": PhaseResidualMemorySparseGateNeuronBlock,
    "phase_residual_memory_sparse_centered_gate_neuron": PhaseResidualMemorySparseCenteredGateNeuronBlock,
    "phase_residual_memory_delta_gate_neuron": PhaseResidualMemoryDeltaGateNeuronBlock,
    "memory_square_gain_neuron": MemorySquareGainNeuronBlock,
    "memory_square_centered_gain_neuron": MemorySquareCenteredGainNeuronBlock,
    "memory_square_small_gain_neuron": MemorySquareSmallGainNeuronBlock,
    "memory_square_scalar_gain_neuron": MemorySquareScalarGainNeuronBlock,
    "phase_residual_memory_novelty_gate_neuron": PhaseResidualMemoryNoveltyGateNeuronBlock,
    "phase_residual_memory_novelty_centered_gate_neuron": PhaseResidualMemoryNoveltyCenteredGateNeuronBlock,
    "phase_residual_memory_novelty_tiny_gate_neuron": PhaseResidualMemoryNoveltyTinyGateNeuronBlock,
    "memory_square_novelty_gain_neuron": MemorySquareNoveltyGainNeuronBlock,
    "phase_residual_memory_conv_agreement_neuron": PhaseResidualMemoryConvAgreementNeuronBlock,
    "phase_residual_memory_conv_disagreement_damp_neuron": PhaseResidualMemoryConvDisagreementDampNeuronBlock,
    "phase_residual_memory_conv_centered_agreement_neuron": PhaseResidualMemoryConvCenteredAgreementNeuronBlock,
    "memory_square_conv_agreement_gain_neuron": MemorySquareConvAgreementGainNeuronBlock,
    "phase_residual_conv_energy_gate_neuron": PhaseResidualConvEnergyGateNeuronBlock,
    "stable_square_neuron": StableSquareNeuronBlock,
    "stable_competition_neuron": StableCompetitionNeuronBlock,
    "bottleneck_aware_neuron": BottleneckAwareNeuronBlock,
}


def patched_make_mixer_block(config: Any, layer_index: int) -> nn.Module:
    if config.block_type not in VARIANTS:
        return ORIGINAL_MAKE_MIXER_BLOCK(config, layer_index)
    kwargs = {
        "dim": config.embedding_dim,
        "expansion": 2,
        "kernel_size": config.conv_kernel_size,
        "dilation": 2 ** (layer_index % 6),
        "dropout": 0.1,
        "memory_rank": config.memory_rank,
        "memory_kernel_size": config.landmark_stride,
    }
    return VARIANTS[config.block_type](**kwargs)


trainer.make_mixer_block = patched_make_mixer_block


def config_for(block_type: str, output_dir: Path, steps: int, eval_interval: int | None = None) -> Any:
    seed = int(os.environ.get("MANUAL_SEARCH_SEED", "13"))
    val_blocks = int(os.environ.get("MANUAL_SEARCH_VAL_BLOCKS", "2"))
    sequence_length = int(os.environ.get("MANUAL_SEARCH_SEQUENCE_LENGTH", "255"))
    embedding_dim = int(os.environ.get("MANUAL_SEARCH_EMBEDDING_DIM", "64"))
    conv_layers = int(os.environ.get("MANUAL_SEARCH_CONV_LAYERS", "1"))
    conv_kernel_size = int(os.environ.get("MANUAL_SEARCH_CONV_KERNEL_SIZE", "7"))
    conv_rank = int(os.environ.get("MANUAL_SEARCH_CONV_RANK", "32"))
    memory_rank = int(os.environ.get("MANUAL_SEARCH_MEMORY_RANK", "12"))
    landmark_stride = int(os.environ.get("MANUAL_SEARCH_LANDMARK_STRIDE", "32"))
    sampled_vocab_size = int(os.environ.get("MANUAL_SEARCH_SAMPLED_VOCAB_SIZE", "512"))
    token_stride = int(os.environ.get("MANUAL_SEARCH_TOKEN_STRIDE", "32"))
    token_chunk_size = int(os.environ.get("MANUAL_SEARCH_TOKEN_CHUNK_SIZE", "128"))
    full_eval_token_chunk_size = int(os.environ.get("MANUAL_SEARCH_FULL_EVAL_TOKEN_CHUNK_SIZE", str(token_chunk_size)))
    return trainer.TrainConfig(
        cache_path=CACHE_PATH,
        output_dir=output_dir,
        run_name=output_dir.name,
        sequence_length=sequence_length,
        seed=seed,
        train_steps=steps,
        eval_interval=eval_interval or steps,
        checkpoint_interval=0,
        milestone_checkpoint_interval=0,
        val_blocks=val_blocks,
        embedding_dim=embedding_dim,
        block_type=block_type,
        conv_layers=conv_layers,
        conv_kernel_size=conv_kernel_size,
        conv_rank=conv_rank,
        memory_rank=memory_rank,
        landmark_stride=landmark_stride,
        sampled_vocab_size=sampled_vocab_size,
        token_stride=token_stride,
        token_chunk_size=token_chunk_size,
        full_eval_token_chunk_size=full_eval_token_chunk_size,
        learning_rate=6e-4,
        min_learning_rate=1e-5,
        warmup_steps=max(2, steps // 4),
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def finite_grad_test(block_type: str) -> dict[str, Any]:
    torch.manual_seed(1234)
    cfg = config_for(block_type, ARTIFACT_ROOT / "grad_tmp", steps=1)
    block = patched_make_mixer_block(cfg, 0)
    block.train()
    x = torch.randn(2, 17, cfg.embedding_dim, requires_grad=True)
    y = block(x)
    loss = y.float().pow(2).mean()
    loss.backward()
    finite_output = bool(torch.isfinite(y).all().item())
    finite_grads = True
    max_grad = 0.0
    for param in block.parameters():
        if param.grad is not None:
            finite_grads = finite_grads and bool(torch.isfinite(param.grad).all().item())
            max_grad = max(max_grad, float(param.grad.detach().abs().max().item()))
    return {
        "block_type": block_type,
        "finite_output": finite_output,
        "finite_grads": finite_grads,
        "loss": float(loss.detach().item()),
        "max_grad": max_grad,
    }


def parameter_count(block_type: str) -> int:
    cfg = config_for(block_type, ARTIFACT_ROOT / "param_tmp", steps=1)
    model = trainer.CausalConvFactorizedLM(cfg)
    return int(trainer.count_parameters(model))


def finite_report(report: dict[str, Any]) -> bool:
    return math.isfinite(float(report["final_train_loss"])) and math.isfinite(float(report["final_val_loss"]))


def run_training_screen(block_type: str, name: str, steps: int) -> dict[str, Any]:
    out = ARTIFACT_ROOT / name
    result_path = out / "result.json"
    if os.environ.get("MANUAL_SEARCH_REUSE_EXISTING", "0") == "1" and result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        report = result["report"]
        return {
            "ok": finite_report(report),
            "reused_existing": True,
            "block_type": block_type,
            "output_dir": str(out),
            "elapsed_s": 0.0,
            "command": f"reuse existing {result_path}",
            "final_train_loss": float(report["final_train_loss"]),
            "final_val_loss": float(report["final_val_loss"]),
            "pure_train_tok_per_sec": float(report["pure_train_tok_per_sec"]),
            "peak_vram_mb": report.get("peak_vram_mb"),
            "parameter_count": int(report["parameter_count"]),
            "result_path": str(result_path),
        }
    cfg = config_for(block_type, out, steps=steps)
    cmd_text = f"trainer.train({asdict(cfg)})"
    start = time.perf_counter()
    try:
        trainer.train(cfg)
        elapsed = time.perf_counter() - start
        result = json.loads(result_path.read_text(encoding="utf-8"))
        report = result["report"]
        return {
            "ok": finite_report(report),
            "block_type": block_type,
            "output_dir": str(out),
            "elapsed_s": elapsed,
            "command": cmd_text,
            "final_train_loss": float(report["final_train_loss"]),
            "final_val_loss": float(report["final_val_loss"]),
            "pure_train_tok_per_sec": float(report["pure_train_tok_per_sec"]),
            "peak_vram_mb": report.get("peak_vram_mb"),
            "parameter_count": int(report["parameter_count"]),
            "result_path": str(result_path),
        }
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return {
            "ok": False,
            "block_type": block_type,
            "output_dir": str(out),
            "elapsed_s": elapsed,
            "command": cmd_text,
            "error": repr(exc),
        }


DESIGN_NOTES: dict[str, dict[str, str]] = {
    "hidden_drop_square_neuron": {
        "design": "Hidden-dropout squared neuron",
        "equations": "h=dropout(relu(Wf norm(x))^2, p=0.05 during train); y=ffn_out(h)",
        "why": "The 2048 failures repeatedly show lower train loss but worse validation, so this regularizes the FFN hidden path without changing tokenizer, objective, or eval metric.",
        "novelty": "Train-time activation thinning inside the block-local squared-neuron expansion rather than adding memory-route capacity.",
    },
    "hidden_drop_low_square_neuron": {
        "design": "Low hidden-dropout squared neuron",
        "equations": "h=dropout(relu(Wf norm(x))^2, p=0.02 during train); y=ffn_out(h)",
        "why": "Tests whether a lighter block-local hidden regularizer can reduce overfit while preserving the near-baseline 2048 validation level.",
        "novelty": "Very light train-time activation thinning on the squared-neuron FFN hidden state.",
    },
    "hidden_drop_mid_square_neuron": {
        "design": "Medium hidden-dropout squared neuron",
        "equations": "h=dropout(relu(Wf norm(x))^2, p=0.075 during train); y=ffn_out(h)",
        "why": "The p=0.05 version won multiple 2048-step seeds while p=0.02 was too weak at 1024, so this checks whether the regularization optimum is higher.",
        "novelty": "A stronger zero-parameter train-time thinning point for the block-local squared-neuron expansion.",
    },
    "hidden_drop_high_square_neuron": {
        "design": "High hidden-dropout squared neuron",
        "equations": "h=dropout(relu(Wf norm(x))^2, p=0.10 during train); y=ffn_out(h)",
        "why": "Tests whether the hidden-drop win is robust to stronger anti-coadaptation or whether p=0.05 is near the safe edge.",
        "novelty": "Aggressive train-time activation thinning inside the long-context conv-memory FFN without changing inference math.",
    },
    "hidden_drop_ultra_square_neuron": {
        "design": "Ultra hidden-dropout squared neuron",
        "equations": "h=dropout(relu(Wf norm(x))^2, p=0.15 during train); y=ffn_out(h)",
        "why": "The p=0.05, p=0.075, and p=0.10 curve improved on seed 31, so this checks whether stronger regularization is still beneficial or starts to underfit.",
        "novelty": "Very strong train-time thinning on the squared-neuron expansion with baseline-equivalent inference math.",
    },
    "hidden_drop_extreme_square_neuron": {
        "design": "Extreme hidden-dropout squared neuron",
        "equations": "h=dropout(relu(Wf norm(x))^2, p=0.20 during train); y=ffn_out(h)",
        "why": "Bounds the useful dropout-strength range and tests whether the apparent monotonic improvement breaks under aggressive hidden regularization.",
        "novelty": "Stress-test endpoint for the zero-parameter hidden regularization curve inside the conv-memory block.",
    },
    "hidden_drop_p25_square_neuron": {
        "design": "p=0.25 hidden-dropout squared neuron",
        "equations": "h=dropout(relu(Wf norm(x))^2, p=0.25 during train); y=ffn_out(h)",
        "why": "The 2048 curve continued improving through p=0.20, so this probes the next stronger point before committing scale-up evidence.",
        "novelty": "High-strength zero-parameter hidden activation thinning for the squared-neuron conv-memory FFN.",
    },
    "hidden_drop_p30_square_neuron": {
        "design": "p=0.30 hidden-dropout squared neuron",
        "equations": "h=dropout(relu(Wf norm(x))^2, p=0.30 during train); y=ffn_out(h)",
        "why": "Tests whether very high train-time thinning still improves 2048-step generalization or crosses into underfitting.",
        "novelty": "Upper curve endpoint for squared-neuron hidden regularization with baseline-equivalent inference math.",
    },
    "hidden_drop_p35_square_neuron": {
        "design": "p=0.35 hidden-dropout squared neuron",
        "equations": "h=dropout(relu(Wf norm(x))^2, p=0.35 during train); y=ffn_out(h)",
        "why": "The 2048 curve improved through p=0.30 even as 1024 weakened, so this tests whether later generalization still benefits from heavier hidden thinning.",
        "novelty": "Very high train-time activation thinning endpoint for the squared-neuron conv-memory FFN.",
    },
    "hidden_drop_p40_square_neuron": {
        "design": "p=0.40 hidden-dropout squared neuron",
        "equations": "h=dropout(relu(Wf norm(x))^2, p=0.40 during train); y=ffn_out(h)",
        "why": "Stress-tests the high-dropout regime to locate the underfitting boundary before scale-up recommendations.",
        "novelty": "Aggressive zero-parameter hidden regularization curve endpoint with baseline-equivalent inference math.",
    },
    "channel_drop_square_neuron": {
        "design": "Sequence-shared channel-drop squared neuron",
        "equations": "h=m_channel*relu(Wf norm(x))^2 where m_channel is shared over the sequence during train",
        "why": "Regularizes whole expanded channels consistently across the long sequence, targeting channel co-adaptation rather than token noise.",
        "novelty": "Block-local sequence-shared hidden channel dropout for a causal conv-memory FFN.",
    },
    "memory_energy_drop_square_neuron": {
        "design": "Memory-energy conditioned hidden-dropout squared neuron",
        "equations": "e=rms(memory); p=0.02+0.08*sigmoid((e-mean_seq(e))/std_seq(e)); h=dropout_tensor(relu(Wf norm(x))^2,p) during train",
        "why": "The long-context memory path identifies tokens with stronger causal state. This regularizes the FFN hidden expansion most where memory energy is high, targeting memory-neuron co-adaptation while keeping inference identical to baseline.",
        "novelty": "Zero-parameter stochastic coupling between low-rank causal memory energy and squared-neuron hidden activation thinning.",
    },
    "mem_threshold_neuron": {
        "design": "Memory-threshold neuron",
        "equations": "m=memory_conv(memory_down(norm(x))); z=Wf norm(x); h=(1+0.25 tanh(Wg m)) * relu(z-0.25 tanh(Wt m))^2",
        "why": "The existing memory path is additive. This lets causal memory change the FFN decision surface at the same token, allowing long-range evidence to alter neuron thresholds without attention.",
        "novelty": "Low-rank causal memory modulates the activation threshold/curvature inside the FFN rather than only adding a residual memory vector.",
    },
    "conv_disagreement_neuron": {
        "design": "Conv-disagreement curvature neuron",
        "equations": "d=mean_channel(var_branch(conv_k(x))); z=Wf norm(x); h=relu(z)^2/(1+0.5 tanh(log(1+d)))",
        "why": "Multi-scale branch disagreement marks uncertain local context. The neuron dampens curvature there to stabilize updates over ambiguous spans.",
        "novelty": "Uses disagreement among existing causal conv branches as a local context signal controlling neuron curvature.",
    },
    "adaptive_basis_neuron": {
        "design": "Adaptive nonlinear basis neuron",
        "equations": "w=softmax(Wr x); h=sum_i w_i basis_i(z), bases=[relu(z)^2, z*tanh(z), relu(z)*sigmoid(1.5z)]",
        "why": "A block can choose sharper, signed, or smoother responses per token without expensive routing or seq attention.",
        "novelty": "Token-local mixture of nonlinear bases, not a single known activation replacement.",
    },
    "rank_competition_neuron": {
        "design": "Grouped rank-competition neuron",
        "equations": "g=reshape(relu(z)); p=mean_group(g); h=relu(g-sigmoid(a)p)^2",
        "why": "The low-rank head and memory force compression; group competition may allocate FFN expansion capacity to fewer stronger features.",
        "novelty": "Cheap local channel inhibition before the squared nonlinearity, no top-k or sorting.",
    },
    "rank_competition_mild_neuron": {
        "design": "Mild grouped rank-competition neuron",
        "equations": "g=reshape(relu(z)); p=mean_group(g); h=relu(g-sigmoid(a)p)^2, a initialized to -0.5",
        "why": "The original rank competition wins shorter gates but loses at 2048, so this lowers initial pressure without changing the comparison objective.",
        "novelty": "Softer learnable group inhibition as a low-cost channel competition prior.",
    },
    "rank_competition_soft_neuron": {
        "design": "Soft grouped rank-competition neuron",
        "equations": "g=reshape(relu(z)); p=mean_group(g); h=relu(g-sigmoid(a)p)^2, a initialized to -1.5",
        "why": "Tests a near-baseline competition regime after aggressive pressure showed longer-screen overfit.",
        "novelty": "Very soft local channel competition before the squared neuron.",
    },
    "rank_competition_ultrasoft_neuron": {
        "design": "Ultra-soft grouped rank-competition neuron",
        "equations": "g=reshape(relu(z)); p=mean_group(g); h=relu(g-sigmoid(a)p)^2, a initialized to -2.5",
        "why": "The 2048 pressure curve improved as inhibition softened, so this probes the near-baseline region for a small but stable win.",
        "novelty": "Minimal local channel competition as a learnable perturbation of the original squared neuron.",
    },
    "rank_competition_feather_neuron": {
        "design": "Feather-pressure grouped rank-competition neuron",
        "equations": "g=reshape(relu(z)); p=mean_group(g); h=relu(g-sigmoid(a)p)^2, a initialized to -3.5",
        "why": "Tests whether only a very weak competition prior is enough to help without the longer-screen overfit seen in stronger pressure settings.",
        "novelty": "Near-zero group pressure that can still learn if useful.",
    },
    "rank_competition_fixed_feather_neuron": {
        "design": "Fixed feather-pressure grouped rank-competition neuron",
        "equations": "g=reshape(relu(z)); p=mean_group(g); h=relu(g-sigmoid(-3.5)p)^2",
        "why": "The learned feather-pressure run nearly matched the 2048 baseline but its inhibition drifted upward; this tests the same weak prior without train-fit drift.",
        "novelty": "Fixed near-zero local channel competition as a non-capacity regularizing perturbation.",
    },
    "rank_competition_fixed_trace_neuron": {
        "design": "Fixed trace-pressure grouped rank-competition neuron",
        "equations": "g=reshape(relu(z)); p=mean_group(g); h=relu(g-sigmoid(-4.5)p)^2",
        "why": "Probes a weaker fixed pressure after the 2048 curve approached the baseline as inhibition decreased.",
        "novelty": "Trace-level fixed group pressure inside the squared FFN neuron.",
    },
    "rank_competition_fixed_dust_neuron": {
        "design": "Fixed dust-pressure grouped rank-competition neuron",
        "equations": "g=reshape(relu(z)); p=mean_group(g); h=relu(g-sigmoid(-5.5)p)^2",
        "why": "Tests whether an almost-zero competition prior can produce a small stable validation improvement without lowering train loss too much.",
        "novelty": "Near-baseline fixed channel pressure as a block-local inductive-bias probe.",
    },
    "rank_competition_fixed_feather_hidden_drop_neuron": {
        "design": "Fixed feather-pressure hidden-drop rank-competition neuron",
        "equations": "h=dropout(fixed_feather_rank_competition(z), p=0.05 during train)",
        "why": "Fixed feather was the closest 2048 miss; this adds train-time hidden regularization to reduce the remaining lower-train/worse-val gap.",
        "novelty": "Combines fixed near-zero channel competition with train-time hidden activation thinning.",
    },
    "rank_competition_fixed_feather_channel_drop_neuron": {
        "design": "Fixed feather-pressure channel-drop rank-competition neuron",
        "equations": "h=m_channel*fixed_feather_rank_competition(z) with sequence-shared channel mask during train",
        "why": "Tests whether sequence-shared channel regularization is more appropriate for long-context conv-memory blocks than token-independent dropout.",
        "novelty": "Fixed near-zero channel competition plus sequence-shared FFN expansion channel dropout.",
    },
    "stateful_threshold_neuron": {
        "design": "Causal stateful threshold neuron",
        "equations": "c_t=mean_{i<=t} z_i; h=relu(z_t-tanh(a)c_t)^2",
        "why": "Long-context linear models need causal state. This gives every FFN neuron a parallel causal running threshold.",
        "novelty": "Uses cumsum-derived causal preactivation state inside the neuron with no recurrent Python loop.",
    },
    "phase_amplitude_neuron": {
        "design": "Signed amplitude-phase neuron",
        "equations": "amp=softplus(Wa u); phase=tanh(Wp u); h=amp * sign(phase)*phase^2",
        "why": "Separating magnitude from signed carrier may preserve directional information lost by nonnegative squared ReLU features.",
        "novelty": "Signed-amplitude carrier neuron inside this conv-memory block rather than a standard gate.",
    },
    "stable_square_neuron": {
        "design": "RMS-stabilized square neuron",
        "equations": "s=relu(z)^2; h=s/(1+rms_channel(s)) + 0.05 sigmoid(a) z",
        "why": "Keeps square-like sample efficiency while bounding large channel curvature and adding a small gradient path for inactive units.",
        "novelty": "Local curvature normalization and leak around the squared neuron, not a simple activation swap.",
    },
    "stable_competition_neuron": {
        "design": "Stable competition neuron",
        "equations": "g=relu(z); c=relu(g-sigmoid(a)mean_group(g)); h=c^2/(1+rms_channel(c^2))+0.03 sigmoid(b)z",
        "why": "Combines the two cheap first-pass signals: channel competition for allocation and RMS stabilization for curvature control.",
        "novelty": "A grouped inhibitory square neuron with local curvature normalization, still linear in sequence length.",
    },
    "phase_amplitude_neutral_neuron": {
        "design": "Parameter-neutral signed amplitude neuron",
        "equations": "z=Wf norm(x); phase=tanh(z); h=softplus(z)*phase*abs(phase)",
        "why": "Tests whether the phase-amplitude signal was real or just extra parameters by reusing the baseline FFN projection.",
        "novelty": "Signed amplitude carrier with no extra projection parameters.",
    },
    "phase_amplitude_one_extra_neuron": {
        "design": "One-extra-projection phase-amplitude neuron",
        "equations": "amp=softplus(Wf norm(x)); phase=tanh(Wp norm(x)); h=amp*phase*abs(phase)",
        "why": "Tests whether separate signed carrier is the key mechanism with only one added projection instead of two.",
        "novelty": "Uses the baseline FFN projection as amplitude and adds a distinct signed phase carrier.",
    },
    "phase_amplitude_replace_neuron": {
        "design": "Replacement phase-amplitude neuron",
        "equations": "amp=softplus(Wa norm(x)); phase=tanh(Wp norm(x)); h=amp*phase*abs(phase)",
        "why": "Removes the unused baseline FFN input projection from the first phase-amplitude variant while preserving separate amplitude/phase channels.",
        "novelty": "Separate amplitude and signed carrier projections replace, rather than stack on top of, the squared FFN input.",
    },
    "phase_residual_blend_neuron": {
        "design": "Phase residual blend neuron",
        "equations": "base=relu(z)^2; signed=softplus(z)*tanh(z)*abs(tanh(z)); h=base+sigmoid(a_c)*signed",
        "why": "Preserves the known-working squared neuron while allowing each channel to learn a small signed phase correction.",
        "novelty": "A conservative per-channel signed-phase residual around the baseline neuron, designed after full replacement proved seed-unstable on real text.",
    },
    "phase_residual_blend_tiny_neuron": {
        "design": "Tiny phase residual blend neuron",
        "equations": "base=relu(z)^2; signed=softplus(z)*tanh(z)*abs(tanh(z)); h=base+sigmoid(a_c)*signed, a_c=-4 init",
        "why": "Tests whether the positive residual-blend signal is best as an almost-baseline perturbation.",
        "novelty": "Strength ablation of the signed-phase residual correction.",
    },
    "phase_residual_blend_large_neuron": {
        "design": "Large phase residual blend neuron",
        "equations": "base=relu(z)^2; signed=softplus(z)*tanh(z)*abs(tanh(z)); h=base+sigmoid(a_c)*signed, a_c=-1 init",
        "why": "Tests whether the signed-phase residual should influence the neuron more strongly.",
        "novelty": "Strength ablation of the signed-phase residual correction.",
    },
    "phase_residual_blend_half_neuron": {
        "design": "Half phase residual blend neuron",
        "equations": "base=relu(z)^2; signed=softplus(z)*tanh(z)*abs(tanh(z)); h=base+sigmoid(a_c)*signed, a_c=0 init",
        "why": "Tests whether high signed-phase mixing causes the seed instability seen in full phase replacement.",
        "novelty": "Strength ablation of the signed-phase residual correction.",
    },
    "phase_residual_blend_quarter_neuron": {
        "design": "Quarter phase residual blend neuron",
        "equations": "base=relu(z)^2; signed=softplus(z)*tanh(z)*abs(tanh(z)); h=base+0.25-learnable signed",
        "why": "Targets the 40M/1024 overfit signal by testing a fixed midpoint between large and half residual strength.",
        "novelty": "A precise residual-strength ablation after half strength showed mixed long-sequence validation behavior.",
    },
    "phase_residual_blend_normed_neuron": {
        "design": "RMS-limited phase residual blend neuron",
        "equations": "base=relu(z)^2; signed=softplus(z)*tanh(z)*abs(tanh(z)); h=base+sigmoid(a_c)*signed/(1+rms_channel(signed))",
        "why": "Keeps the signed phase path but prevents it from lowering train loss by growing unchecked residual scale.",
        "novelty": "Token-local residual normalization around the baseline squared neuron, not an activation swap.",
    },
    "phase_residual_blend_centered_neuron": {
        "design": "Centered phase residual blend neuron",
        "equations": "base=relu(z)^2; signed=softplus(z)*tanh(z)*abs(tanh(z)); h=base+sigmoid(a_c)*(signed-mean_channel(signed))",
        "why": "Removes token-wise signed residual bias so the phase path reallocates channel evidence instead of shifting all FFN features.",
        "novelty": "Channel-centered signed phase correction inside the block-local neuron.",
    },
    "phase_residual_boundary_blend_neuron": {
        "design": "Boundary-local phase residual blend neuron",
        "equations": "base=relu(z)^2; signed=softplus(z)*tanh(z)*abs(tanh(z)); boundary=1/(1+abs(z)); h=base+sigmoid(a_c)*boundary*signed",
        "why": "The signed residual helped short screens but overfit at longer gates; this confines it to uncertain near-threshold FFN channels.",
        "novelty": "A decision-boundary-local signed residual around the squared-ReLU neuron, not an activation swap.",
    },
    "phase_residual_boundary_centered_neuron": {
        "design": "Centered boundary-local phase residual neuron",
        "equations": "local=boundary*softplus(z)*tanh(z)*abs(tanh(z)); h=relu(z)^2+sigmoid(a_c)*(local-mean_channel(local))",
        "why": "Combines the strongest short-screen centered residual with a boundary mask that avoids modifying high-confidence FFN activations.",
        "novelty": "Channel-centered signed residual restricted to the FFN activation boundary.",
    },
    "phase_residual_memory_gate_neuron": {
        "design": "Memory-gated phase residual neuron",
        "equations": "gate=sigmoid(a_c+0.5*tanh(Wm memory)); h=relu(z)^2+gate*softplus(z)*tanh(z)*abs(tanh(z))",
        "why": "Lets the causal low-rank memory decide when signed phase information helps long-context prediction.",
        "novelty": "Couples the existing causal memory state directly into the per-channel neuron residual gate.",
    },
    "phase_residual_memory_scalar_gate_neuron": {
        "design": "Scalar memory-gated phase residual neuron",
        "equations": "g_t=sigmoid(a_c+0.5*tanh(wm memory_t)); h=relu(z)^2+g_t*signed",
        "why": "Tests whether the strong memory-gate signal needs expensive per-channel routing or only a token-level causal gain.",
        "novelty": "Low-rank causal memory controls a token-wise residual gain with only one added gate channel.",
    },
    "phase_residual_memory_group_gate_neuron": {
        "design": "Group memory-gated phase residual neuron",
        "equations": "g_{t,group}=sigmoid(a_c+0.5*tanh(Wm memory_t)); h=relu(z)^2+repeat(g)*signed",
        "why": "Splits the difference between cheap scalar memory gates and expensive per-channel gates.",
        "novelty": "Memory-conditioned channel-group residual routing inside the local neuron.",
    },
    "phase_residual_memory_centered_gate_neuron": {
        "design": "Centered memory-gated phase residual neuron",
        "equations": "centered=signed-mean_channel(signed); h=relu(z)^2+gate(memory)*centered",
        "why": "Targets the seed 13 overfit by removing the phase residual's token-wise channel bias while preserving memory routing.",
        "novelty": "Combines causal-memory residual routing with channel-centered signed phase features.",
    },
    "phase_residual_memory_scalar_centered_gate_neuron": {
        "design": "Scalar centered memory-gated phase residual neuron",
        "equations": "centered=signed-mean_channel(signed); h=relu(z)^2+g_t*centered",
        "why": "Tests a low-parameter memory-conditioned centered residual that may keep the cheap centered benefit without per-channel overfit.",
        "novelty": "Token-wise causal-memory gate over a centered signed phase correction.",
    },
    "phase_residual_memory_boundary_gate_neuron": {
        "design": "Memory-gated boundary-local phase residual neuron",
        "equations": "boundary=1/(1+abs(z)); gate=sigmoid(a_c+0.5*tanh(Wm memory)); h=relu(z)^2+gate*boundary*signed",
        "why": "Keeps causal memory routing but prevents it from rewriting high-confidence expanded FFN features that drove train/val separation.",
        "novelty": "Causal memory gates a residual that is localized by the FFN preactivation boundary.",
    },
    "phase_residual_memory_boundary_centered_gate_neuron": {
        "design": "Centered memory-gated boundary-local phase residual neuron",
        "equations": "local=boundary*signed; centered=local-mean_channel(local); h=relu(z)^2+gate(memory)*centered",
        "why": "Tests whether memory routing needs both boundary localization and channel-centering to avoid seed-specific overfit.",
        "novelty": "Boundary-local, channel-centered signed residual routed by causal low-rank memory.",
    },
    "phase_residual_memory_boundary_scalar_gate_neuron": {
        "design": "Scalar memory-gated boundary-local phase residual neuron",
        "equations": "g_t=sigmoid(a_c+0.5*tanh(wm memory_t)); h=relu(z)^2+g_t*boundary*signed",
        "why": "Cheap test of whether token-level memory confidence is enough when the residual is restricted to threshold-local FFN features.",
        "novelty": "Single-channel causal memory route over decision-boundary-local signed residuals.",
    },
    "rank_competition_memory_gate_neuron": {
        "design": "Rank-competition memory-gated phase residual neuron",
        "equations": "base=relu(group(z)-sigmoid(a)*mean_group(relu(z)))^2; h=base+sigmoid(p+0.5*tanh(Wm memory))*signed",
        "why": "Rank competition was the first cheap 1024-step 3/3 baseline winner; this tests whether it can stabilize the stronger memory-gated signed residual.",
        "novelty": "Combines group-wise FFN channel competition with causal-memory residual routing inside one block-local primitive.",
    },
    "rank_competition_memory_centered_gate_neuron": {
        "design": "Rank-competition centered memory-gated phase residual neuron",
        "equations": "base=rank_competition(relu(z))^2; centered=signed-mean_channel(signed); h=base+gate(memory)*centered",
        "why": "Tests whether rank competition plus zero-mean signed residual routing can keep the robust rank win while improving memory-gate seed stability.",
        "novelty": "Rank-competitive squared base with centered causal-memory signed residual routing.",
    },
    "rank_competition_centered_residual_neuron": {
        "design": "Rank-competition centered residual neuron",
        "equations": "base=rank_competition(relu(z))^2; centered=signed-mean_channel(signed); h=base+sigmoid(p)*centered",
        "why": "Isolates whether the hybrid gain comes from rank competition plus centered signed residual before adding memory routing cost.",
        "novelty": "Cheap zero-mean signed residual on top of a rank-competitive squared base.",
    },
    "rank_competition_mild_centered_residual_neuron": {
        "design": "Mild rank-competition centered residual neuron",
        "equations": "base=mild_rank_competition(z); centered=signed-mean_channel(signed); h=base+sigmoid(p)*centered",
        "why": "The centered residual was the closest stable 2048 miss; this tests whether softer pressure fixes the final validation regression.",
        "novelty": "Zero-mean signed residual on a softer group-competition base.",
    },
    "rank_competition_soft_centered_residual_neuron": {
        "design": "Soft rank-competition centered residual neuron",
        "equations": "base=soft_rank_competition(z); centered=signed-mean_channel(signed); h=base+sigmoid(p)*centered",
        "why": "Tests a near-baseline pressure setting with the centered residual that had the closest 2048 miss.",
        "novelty": "Centered signed residual combined with very soft local channel competition.",
    },
    "rank_competition_memory_scalar_centered_gate_neuron": {
        "design": "Scalar-memory rank-competition centered gate neuron",
        "equations": "g_t=sigmoid(p+0.5*tanh(wm memory_t)); h=rank_competition(z)+g_t*centered",
        "why": "Tests whether token-level memory confidence is enough to recover the full hybrid's gain at a fraction of the parameters.",
        "novelty": "A single causal-memory route gates a rank-competitive zero-mean signed residual.",
    },
    "rank_competition_memory_group_centered_gate_neuron": {
        "design": "Group-memory rank-competition centered gate neuron",
        "equations": "g_{t,k}=sigmoid(p_k+0.5*tanh(Wg memory_t)_k); h=rank_competition(z)+repeat(g)*centered",
        "why": "Keeps memory-conditioned channel selectivity while avoiding the full memory-rank to FFN-width gate.",
        "novelty": "Coarse causal-memory channel routing coupled to rank competition and centered signed residuals.",
    },
    "rank_competition_memory_factor_centered_gate_neuron": {
        "design": "Factorized-memory rank-competition centered gate neuron",
        "equations": "r=U silu(D memory_t); g=sigmoid(p+0.5*tanh(r)); h=rank_competition(z)+g*centered",
        "why": "Approximates the full per-channel memory gate with a low-rank route to test whether the expensive map is necessary.",
        "novelty": "Low-rank memory-to-neuron gating over a rank-competitive zero-mean signed residual.",
    },
    "rank_competition_memory_small_centered_gate_neuron": {
        "design": "Small full-memory rank-competition centered gate neuron",
        "equations": "gate=sigmoid(-1+0.25*tanh(Wm memory)); h=rank_competition(z)+gate*centered",
        "why": "Targets the seed 31 train/val split by keeping the full per-channel memory route but reducing its initial strength.",
        "novelty": "Conservative per-channel causal-memory routing over a rank-competitive centered signed residual.",
    },
    "rank_competition_memory_normed_centered_gate_neuron": {
        "design": "Normed full-memory rank-competition centered gate neuron",
        "equations": "limited=centered/(1+rms(centered)); route=zero_mean(tanh(Wm memory)); h=rank_competition(z)+sigmoid(p+0.5*route/(1+rms(route)))*limited",
        "why": "Targets seed-specific overfit by limiting both the signed residual scale and the per-token memory-route scale.",
        "novelty": "Coupled token-local normalization of memory route and centered signed residual inside the rank-competitive block.",
    },
    "rank_competition_memory_uncertainty_centered_gate_neuron": {
        "design": "Competition-uncertainty memory-routed centered residual neuron",
        "equations": "u=exp(-2*abs(relu(z)-threshold)/pressure); h=rank_competition(z)+gate(memory)*u*centered",
        "why": "The seed 31 failure looks like the memory route rewriting confident features; this only routes signed residuals near the competition boundary.",
        "novelty": "Rank-competition margin acts as a differentiable uncertainty mask for causal-memory residual routing.",
    },
    "rank_competition_memory_suppressed_centered_gate_neuron": {
        "design": "Competition-suppressed memory-routed centered residual neuron",
        "equations": "s=sigmoid(-4*(relu(z)-threshold)/pressure); h=rank_competition(z)+gate(memory)*s*centered",
        "why": "Tests whether memory should only recover information from channels suppressed by local rank competition rather than modifying winners.",
        "novelty": "Causal memory is restricted to the loser side of the rank-competition neuron.",
    },
    "rank_competition_memory_suppressed_small_centered_gate_neuron": {
        "design": "Small-gain competition-suppressed memory-routed centered residual neuron",
        "equations": "s=sigmoid(-4*margin/pressure); gate=sigmoid(-1+0.25*tanh(Wm memory)); h=rank_competition(z)+gate*s*centered",
        "why": "The 1024-step suppressed route was robust but blew up at 2048, so this keeps the same loser-channel route while reducing route amplitude.",
        "novelty": "A causal-memory residual explicitly limited to suppressed competition channels with conservative learnable route strength.",
    },
    "rank_competition_memory_suppressed_bounded_centered_gate_neuron": {
        "design": "Bounded competition-suppressed memory-routed centered residual neuron",
        "equations": "s=sigmoid(-4*margin/pressure); h=rank_competition(z)+gate(memory)*s*tanh(centered)",
        "why": "Targets the 2048 NaN by hard-bounding the signed residual while preserving the full memory gate and suppressed-channel routing.",
        "novelty": "Combines rank-competition loser-channel routing with a bounded signed residual to stabilize long linear-sequence training.",
    },
    "rank_competition_memory_suppressed_bounded_small_centered_gate_neuron": {
        "design": "Bounded small-gain competition-suppressed memory-routed centered residual neuron",
        "equations": "s=sigmoid(-4*margin/pressure); gate=sigmoid(-1+0.25*tanh(Wm memory)); h=rank_competition(z)+gate*s*tanh(centered)",
        "why": "Tests whether both failure controls are needed: bounded residual magnitude and smaller memory-route strength.",
        "novelty": "A double-constrained loser-channel memory recovery path for rank-competitive low-rank conv-memory blocks.",
    },
    "rank_competition_memory_suppressed_normed_centered_gate_neuron": {
        "design": "RMS-limited competition-suppressed memory-routed centered residual neuron",
        "equations": "limited=centered/(1+rms(centered)); h=rank_competition(z)+gate(memory)*s*limited",
        "why": "Keeps more amplitude information than tanh while preventing token-local centered residuals from dominating longer runs.",
        "novelty": "Rank-competition loser-channel routing with token-local residual RMS limiting.",
    },
    "rank_competition_memory_suppressed_energy_matched_centered_gate_neuron": {
        "design": "Energy-matched competition-suppressed memory-routed centered residual neuron",
        "equations": "r=s*tanh(centered); r'=r*rms(base).detach()/(rms(base).detach()+rms(r)); h=rank_competition(z)+gate(memory)*r'",
        "why": "The bounded route still diverged at 2048, so this constrains residual energy relative to the current base neuron rather than bounding values independently.",
        "novelty": "A loser-channel causal-memory residual whose token-local RMS is tied to the rank-competitive base activation.",
    },
    "rank_competition_memory_suppressed_stopgrad_mask_centered_gate_neuron": {
        "design": "Detached-mask competition-suppressed memory-routed centered residual neuron",
        "equations": "s=stopgrad(sigmoid(-4*margin/pressure)); h=rank_competition(z)+gate(memory)*s*tanh(centered)",
        "why": "Tests whether 2048 divergence comes from gradients gaming the competition mask rather than the signed residual value alone.",
        "novelty": "Allows residual learning while making the suppression route a local selector instead of a trainable escape path.",
    },
    "rank_competition_memory_suppressed_aux_head_centered_gate_neuron": {
        "design": "Auxiliary-head competition-suppressed memory-routed centered residual neuron",
        "equations": "y=ffn_out(base)+alpha*residual_out(gate(memory)*s*tanh(centered)); residual_out starts at zero",
        "why": "The failing variants mix signed memory residuals through the main FFN output projection; this isolates that route behind a controlled zero-init head.",
        "novelty": "Separate low-amplitude output projection for rank-competition loser-channel memory recovery inside the conv-memory block.",
    },
    "rank_competition_memory_within_group_centered_gate_neuron": {
        "design": "Within-group centered memory-routed rank residual neuron",
        "equations": "centered_g=signed_g-mean_group(signed_g); h=rank_competition(z)+gate(memory)*centered_g",
        "why": "The base competition is group-local, so centering the residual within the same groups may reduce token-wide shifts that hurt seed 31.",
        "novelty": "Aligns signed residual centering with the rank-competition groups instead of whole-token channel centering.",
    },
    "phase_residual_memory_normed_gate_neuron": {
        "design": "Normed memory-gated phase residual neuron",
        "equations": "limited=signed/(1+rms_channel(signed)); h=relu(z)^2+gate(memory)*limited",
        "why": "Keeps memory routing but limits residual scale to address the low-train/high-val overfit pattern.",
        "novelty": "Causal-memory gating plus token-local residual RMS limiting.",
    },
    "phase_residual_memory_small_gate_neuron": {
        "design": "Small memory-gated phase residual neuron",
        "equations": "gate=sigmoid(-1+0.25*tanh(Wm memory)); h=relu(z)^2+gate*signed",
        "why": "Targets seed 13/31 overfit by reducing both initial signed residual strength and memory-gate amplitude.",
        "novelty": "Constrained causal-memory residual routing after the unconstrained memory gate proved seed-mixed.",
    },
    "phase_residual_memory_tiny_gate_neuron": {
        "design": "Tiny memory-gated phase residual neuron",
        "equations": "gate=sigmoid(-2+0.25*tanh(Wm memory)); h=relu(z)^2+gate*signed",
        "why": "Tests whether memory routing is useful only as a small perturbation around the baseline square neuron.",
        "novelty": "Low-amplitude causal-memory gate over the signed phase residual.",
    },
    "phase_residual_memory_zero_mean_gate_neuron": {
        "design": "Zero-mean memory-gated phase residual neuron",
        "equations": "r=0.5*tanh(Wm memory); gate=sigmoid(a_c+r-mean_channel(r)); h=relu(z)^2+gate*signed",
        "why": "Keeps per-channel memory routing while removing token-wise global gate shifts that may drive overfit.",
        "novelty": "Channel-centered memory routing gate inside the FFN neuron.",
    },
    "phase_residual_memory_bounded_gate_neuron": {
        "design": "Bounded memory-gated phase residual neuron",
        "equations": "r=0.5*tanh(Wm memory); gate=0.25+0.5*sigmoid(a_c+r); h=relu(z)^2+gate*signed",
        "why": "Keeps the current memory-gate lead's per-channel routing but prevents the learned gate from collapsing toward 0 or 1 on unlucky seeds.",
        "novelty": "Causal memory controls a residual gate with an explicit functional floor and ceiling instead of an unconstrained sigmoid.",
    },
    "phase_residual_memory_bounded_centered_gate_neuron": {
        "design": "Bounded centered memory-gated phase residual neuron",
        "equations": "centered=signed-mean_channel(signed); gate=0.25+0.5*sigmoid(a_c+0.5*tanh(Wm memory)); h=relu(z)^2+gate*centered",
        "why": "Combines bias-free centered residuals with a bounded memory gate to reduce seed-specific over-routing while preserving channel competition.",
        "novelty": "A range-limited causal-memory route over zero-mean signed residual features.",
    },
    "phase_residual_memory_rms_gate_neuron": {
        "design": "RMS-normalized memory-gated phase residual neuron",
        "equations": "r=tanh(Wm memory); r'=0.5*r/(1+rms_channel(r)); h=relu(z)^2+sigmoid(a_c+r')*signed",
        "why": "Targets seed 13/31 failures by making the memory gate depend on direction more than raw learned routing magnitude.",
        "novelty": "Token-local normalization of the causal-memory routing logits before they enter the neuron gate.",
    },
    "phase_residual_memory_sparse_gate_neuron": {
        "design": "Sparse-confidence memory-gated phase residual neuron",
        "equations": "r=tanh(Wm memory)-mean_channel; s=sign(r)*relu(abs(r)-mean_channel(abs(r))); h=relu(z)^2+sigmoid(a_c+0.5*s)*signed",
        "why": "Forces memory routing to act only on above-average channel evidence instead of globally shifting the residual path.",
        "novelty": "Soft thresholded per-token channel competition inside the memory-to-neuron gate.",
    },
    "phase_residual_memory_sparse_centered_gate_neuron": {
        "design": "Sparse centered memory-gated phase residual neuron",
        "equations": "centered=signed-mean_channel(signed); s=sparse_centered(tanh(Wm memory)); h=relu(z)^2+sigmoid(a_c+0.5*s)*centered",
        "why": "Tests whether both the routed residual and the route itself need zero-mean competition to keep validation gains at longer horizons.",
        "novelty": "Coupled sparse channel competition in the memory route and centered signed residual feature.",
    },
    "phase_residual_memory_delta_gate_neuron": {
        "design": "Causal memory-delta gated phase residual neuron",
        "equations": "delta_t=memory_t-memory_{t-1}; gate=sigmoid(a_c+0.5*tanh(Wm delta_t)); h=relu(z)^2+gate*signed",
        "why": "Static memory gates may learn seed-specific biases; routing from memory changes emphasizes transitions and long-context updates.",
        "novelty": "A causal finite-difference memory state controls the FFN signed residual gate.",
    },
    "memory_square_gain_neuron": {
        "design": "Memory-gated square neuron",
        "equations": "h=relu(z)^2 * (1+0.25*tanh(Wm memory))",
        "why": "Tests memory-neuron coupling without adding the signed phase residual path that produced seed-mixed overfit.",
        "novelty": "Causal low-rank memory multiplicatively modulates the existing squared FFN feature map.",
    },
    "memory_square_centered_gain_neuron": {
        "design": "Centered memory-gated square neuron",
        "equations": "r=tanh(Wm memory)-mean_channel(tanh(Wm memory)); h=relu(z)^2*(1+0.25*r)",
        "why": "Preserves per-channel memory routing while preventing global token-wise FFN gain shifts.",
        "novelty": "Zero-mean memory gain over the baseline square neuron.",
    },
    "memory_square_small_gain_neuron": {
        "design": "Small memory-gated square neuron",
        "equations": "h=relu(z)^2 * (1+0.10*tanh(Wm memory))",
        "why": "Tests whether a conservative memory gain improves validation without the stronger gate's overfit.",
        "novelty": "Low-amplitude causal-memory gain over baseline squared features.",
    },
    "memory_square_scalar_gain_neuron": {
        "design": "Scalar memory-gated square neuron",
        "equations": "g_t=1+0.25*tanh(wm memory_t); h=relu(z)^2*g_t",
        "why": "Tests whether token-wise memory confidence is enough without expensive per-channel memory routing.",
        "novelty": "A single causal-memory gain applied to all expanded FFN channels at each token.",
    },
    "phase_residual_memory_novelty_gate_neuron": {
        "design": "Memory-novelty gated phase residual neuron",
        "equations": "novelty_t=m_t-mean_{i<=t}(m_i); gate=sigmoid(a_c+0.5*tanh(W novelty_t)); h=relu(z)^2+gate*signed",
        "why": "Targets seed-specific overfit by routing only from changes in causal memory state, not static memory bias.",
        "novelty": "Uses causal memory novelty as the neuron residual routing signal.",
    },
    "phase_residual_memory_novelty_centered_gate_neuron": {
        "design": "Centered memory-novelty gated phase residual neuron",
        "equations": "centered=signed-mean_channel(signed); novelty_t=m_t-mean_{i<=t}(m_i); h=relu(z)^2+gate(novelty_t)*centered",
        "why": "Combines the 128-step centered-residual signal with novelty-only memory routing to reduce long-run overfit.",
        "novelty": "Memory-novelty routing over channel-centered signed phase features.",
    },
    "phase_residual_memory_novelty_tiny_gate_neuron": {
        "design": "Tiny memory-novelty gated phase residual neuron",
        "equations": "novelty_t=m_t-mean_{i<=t}(m_i); gate=sigmoid(-1+0.5*tanh(W novelty_t)); h=relu(z)^2+gate*signed",
        "why": "Tests whether novelty routing should enter as a smaller perturbation after raw/tiny gates were seed mixed.",
        "novelty": "Low-initial-strength causal memory novelty gate.",
    },
    "memory_square_novelty_gain_neuron": {
        "design": "Memory-novelty square gain neuron",
        "equations": "novelty_t=m_t-mean_{i<=t}(m_i); h=relu(z)^2*(1+0.25*tanh(W novelty_t))",
        "why": "Tests memory-neuron coupling through novelty gain without the signed phase residual path.",
        "novelty": "Causal memory novelty multiplicatively modulates baseline squared features.",
    },
    "phase_residual_memory_conv_agreement_neuron": {
        "design": "Memory-conv agreement phase residual neuron",
        "equations": "a=mean(norm(conv)*norm(memory_up(memory))); h=relu(z)^2+sigmoid(p+a)*signed",
        "why": "Lets signed phase corrections through only when local causal conv context agrees with low-rank memory context.",
        "novelty": "Agreement between the block's two non-FFN context paths controls the neuron residual.",
    },
    "phase_residual_memory_conv_disagreement_damp_neuron": {
        "design": "Memory-conv disagreement damped phase residual neuron",
        "equations": "d=relu(-mean(norm(conv)*norm(memory_up(memory)))); h=relu(z)^2+sigmoid(p-softplus(s)d)*signed",
        "why": "Targets seed overfit by damping signed residuals specifically when memory and local convolution conflict.",
        "novelty": "Conflict between causal conv and causal memory acts as a residual suppressor.",
    },
    "phase_residual_memory_conv_centered_agreement_neuron": {
        "design": "Centered memory-conv agreement phase residual neuron",
        "equations": "centered=signed-mean_channel(signed); a=agreement(conv,memory); h=relu(z)^2+sigmoid(p+a)*centered",
        "why": "Combines the useful centered signed residual with context agreement gating to reduce unconstrained shifts.",
        "novelty": "Conv-memory agreement gates a channel-centered signed phase residual.",
    },
    "memory_square_conv_agreement_gain_neuron": {
        "design": "Memory-conv agreement square gain neuron",
        "equations": "a=mean(norm(conv)*norm(memory_up(memory))); h=relu(z)^2*(1+tanh(s)a)",
        "why": "Tests the agreement signal on the baseline squared neuron without signed phase residuals.",
        "novelty": "Block-local conv-memory agreement directly modulates squared FFN features.",
    },
    "phase_residual_conv_energy_gate_neuron": {
        "design": "Conv-energy-gated phase residual neuron",
        "equations": "e=log(1+mean_channel(conv^2)); gate=sigmoid(a_c-softplus(s)e); h=relu(z)^2+gate*signed",
        "why": "High conv energy marks sharp local context where the signed residual may overfit; the gate suppresses it there.",
        "novelty": "Uses block-local causal conv energy as a continuous phase-residual gain control.",
    },
    "bottleneck_aware_neuron": {
        "design": "Bottleneck-aware low-rank echo neuron",
        "equations": "e=U silu(D norm(x)); h=relu(z+sigmoid(G norm(x)) e)^2",
        "why": "The factorized output bottleneck rewards features already compressible into low-rank subspaces; this adds a block-local low-rank echo before FFN squaring.",
        "novelty": "Block-local low-rank basis injection aimed at the output compression bottleneck without touching the head or loss.",
    },
}


def main() -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    commands: list[str] = []
    write_json(ARTIFACT_ROOT / "design_notes.json", DESIGN_NOTES)
    compile_cmd = [sys.executable, "-B", "-m", "py_compile", str(Path(__file__))]
    subprocess.run(compile_cmd, check=True)
    commands.append(" ".join(compile_cmd))

    selected_env = os.environ.get("MANUAL_SEARCH_BLOCKS", "").strip()
    if selected_env:
        selected_variants = [name.strip() for name in selected_env.split(",") if name.strip()]
        unknown = [name for name in selected_variants if name not in VARIANTS]
        if unknown:
            raise ValueError(f"unknown selected variants: {unknown}")
    else:
        selected_variants = list(VARIANTS.keys())

    block_types = ["multi_scale_lowrank_conv_memory", *selected_variants]
    grad = {block_type: finite_grad_test(block_type) for block_type in block_types}
    params = {block_type: parameter_count(block_type) for block_type in block_types}
    write_json(ARTIFACT_ROOT / "grad_and_params.json", {"grad": grad, "params": params})

    steps = int(os.environ.get("MANUAL_SEARCH_STEPS", "12"))
    seed = int(os.environ.get("MANUAL_SEARCH_SEED", "13"))
    tag = os.environ.get("MANUAL_SEARCH_TAG", "").strip()
    tag_part = f"_{tag}" if tag else ""
    run_prefix = f"screen_{steps}_seed{seed}{tag_part}"
    screens: dict[str, Any] = {}
    screens["multi_scale_lowrank_conv_memory"] = run_training_screen(
        "multi_scale_lowrank_conv_memory", f"{run_prefix}_baseline", steps
    )
    for block_type in selected_variants:
        screens[block_type] = run_training_screen(block_type, f"{run_prefix}_{block_type}", steps)
    screen_results_path = ARTIFACT_ROOT / f"{run_prefix}_screen_results.json"
    if os.environ.get("MANUAL_SEARCH_REUSE_EXISTING", "0") == "1" and screen_results_path.exists():
        previous_screens = json.loads(screen_results_path.read_text(encoding="utf-8"))
        previous_screens.update(screens)
        screens = previous_screens
    write_json(screen_results_path, screens)

    baseline = screens["multi_scale_lowrank_conv_memory"]
    rows: list[dict[str, Any]] = []
    if baseline.get("ok"):
        base_loss = baseline["final_val_loss"]
        base_tps = baseline["pure_train_tok_per_sec"]
        for block_type, result in screens.items():
            if block_type == "multi_scale_lowrank_conv_memory":
                continue
            row = {
                "block_type": block_type,
                "ok": result.get("ok", False),
                "parameter_count": params.get(block_type, result.get("parameter_count")),
                "param_delta": params.get(block_type, result.get("parameter_count", 0))
                - params.get("multi_scale_lowrank_conv_memory", baseline.get("parameter_count", 0)),
                "grad_ok": grad.get(block_type, {}).get("finite_output", True)
                and grad.get(block_type, {}).get("finite_grads", True),
            }
            if result.get("ok"):
                row.update(
                    {
                        "final_train_loss": result["final_train_loss"],
                        "final_val_loss": result["final_val_loss"],
                        "val_delta_vs_baseline": result["final_val_loss"] - base_loss,
                        "tok_per_sec": result["pure_train_tok_per_sec"],
                        "speed_ratio_vs_baseline": result["pure_train_tok_per_sec"] / max(base_tps, 1e-9),
                    }
                )
            else:
                row["error"] = result.get("error", "unknown")
            rows.append(row)
        rows.sort(key=lambda r: (not r.get("ok", False), r.get("val_delta_vs_baseline", math.inf)))
    write_json(ARTIFACT_ROOT / f"{run_prefix}_ranked_results.json", rows)

    lines = [
        "# Manual Neuron Search",
        "",
        f"Trainer: `{TRAINER_PATH}`",
        f"Cache: `{CACHE_PATH}`",
        f"Steps per screen: `{steps}`",
        f"Seed: `{seed}`",
        "",
        "## Baseline",
        "",
        "```json",
        json.dumps(baseline, indent=2, sort_keys=True),
        "```",
        "",
        "## Ranked Variants",
        "",
    ]
    for row in rows:
        note = DESIGN_NOTES.get(row["block_type"], {})
        lines.append(f"### {row['block_type']}")
        lines.append("")
        lines.append(f"- Design: {note.get('design', '')}")
        lines.append(f"- Equations: `{note.get('equations', '')}`")
        lines.append(f"- Why: {note.get('why', '')}")
        lines.append(f"- Novelty: {note.get('novelty', '')}")
        lines.append(f"- Metrics: `{json.dumps(row, sort_keys=True)}`")
        if row.get("ok") and row.get("val_delta_vs_baseline", 1.0) < 0:
            lines.append("- Decision: keep for longer local screen.")
        elif row.get("ok") and row.get("speed_ratio_vs_baseline", 0.0) > 1.05:
            lines.append("- Decision: keep only as a speed/VRAM ablation if loss gap is small.")
        else:
            lines.append("- Decision: kill for now unless a second version changes the mechanism.")
        lines.append("")
    (ARTIFACT_ROOT / "writeup.md").write_text("\n".join(lines), encoding="utf-8")

    commands.append(f"set CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '-1')} && {sys.executable} {Path(__file__)}")
    with (ARTIFACT_ROOT / "commands.txt").open("a", encoding="utf-8") as handle:
        handle.write("\n".join(commands) + "\n")
    print(json.dumps({"artifact_root": str(ARTIFACT_ROOT), "steps": steps, "ranked": rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
