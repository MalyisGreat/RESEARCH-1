from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F
from torch import nn

os.environ.setdefault("WAVE10_HEAD_MULTIPLIER", "1")
os.environ.setdefault("WAVE10_IDENTITY_HEAD", "1")
os.environ.setdefault("WAVE10_RESIDUAL_HEAD", "0")

import compressed_head_rotating_collapsed_train as experiment


BaseModel = experiment.CompressedHeadModel


class SortedInductionModel(BaseModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        initial_scale = 2.0
        self.induction_log_scale = nn.Parameter(torch.tensor(math.log(math.expm1(initial_scale))))
        self.induction_gate = nn.Linear(config.conv_rank, 1)
        nn.init.zeros_(self.induction_gate.weight)
        nn.init.zeros_(self.induction_gate.bias)
        self._induction_tokens: torch.Tensor | None = None
        self._induction_valid: torch.Tensor | None = None

    def build_induction_index(self, input_ids: torch.Tensor) -> None:
        batch, length = input_ids.shape
        flat_tokens = input_ids.reshape(-1)
        batch_ids = torch.arange(batch, device=input_ids.device).repeat_interleave(length)
        keys = flat_tokens + batch_ids * self.vocab_size
        order = torch.argsort(keys, stable=True)
        sorted_keys = keys.index_select(0, order)
        previous_positions = torch.roll(order, 1)
        valid_sorted = torch.ones_like(sorted_keys, dtype=torch.bool)
        valid_sorted[0] = False
        valid_sorted[1:] = sorted_keys[1:] == sorted_keys[:-1]
        previous_local = torch.remainder(previous_positions, length)
        valid_sorted &= previous_local + 1 < length
        source_next = (previous_positions + 1).clamp(max=flat_tokens.numel() - 1)
        retrieved_sorted = flat_tokens.index_select(0, source_next)
        retrieved = torch.zeros_like(flat_tokens)
        valid = torch.zeros_like(flat_tokens, dtype=torch.bool)
        retrieved.scatter_(0, order, retrieved_sorted)
        valid.scatter_(0, order, valid_sorted)
        self._induction_tokens = retrieved.view(batch, length)
        self._induction_valid = valid.view(batch, length)

    def features(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.build_induction_index(input_ids)
        return super().features(input_ids)

    def candidate_logits(
        self,
        *,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        logits = super().candidate_logits(
            hidden=hidden,
            input_ids=input_ids,
            candidate_ids=candidate_ids,
            weight=weight,
            bias=bias,
            positions=positions,
        )
        if self._induction_tokens is None or self._induction_valid is None:
            raise RuntimeError("induction index missing; call features first")
        retrieved = self._induction_tokens.index_select(1, positions)
        valid = self._induction_valid.index_select(1, positions)
        if candidate_ids.numel() == self.vocab_size:
            candidate_indices = retrieved
        else:
            candidate_map = torch.full(
                (self.vocab_size,), -1, device=candidate_ids.device, dtype=torch.long
            )
            candidate_map[candidate_ids] = torch.arange(candidate_ids.numel(), device=candidate_ids.device)
            candidate_indices = candidate_map[retrieved]
            valid = valid & (candidate_indices >= 0)
            candidate_indices = candidate_indices.clamp(min=0)
        scale = F.softplus(self.induction_log_scale).to(logits.dtype)
        gate = torch.sigmoid(self.induction_gate(hidden)).to(logits.dtype)
        values = scale * gate * valid.unsqueeze(-1).to(logits.dtype)
        return logits.scatter_add(2, candidate_indices.unsqueeze(-1), values)


experiment.experiment.trainer.CausalConvFactorizedLM = SortedInductionModel


if __name__ == "__main__":
    experiment.experiment.trainer.train(experiment.experiment.trainer.parse_args())
