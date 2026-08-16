from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F
from torch import nn

import compressed_head_rotating_collapsed_train as experiment


BaseModel = experiment.CompressedHeadModel


class SparseExactRecallModel(BaseModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        initial_scale = 2.0
        self.sparse_recall_log_scale = nn.Parameter(torch.tensor(math.log(math.expm1(initial_scale))))
        self.sparse_recall_gate = nn.Linear(config.conv_rank, 1)
        nn.init.zeros_(self.sparse_recall_gate.weight)
        nn.init.zeros_(self.sparse_recall_gate.bias)
        self.sparse_recall_window = int(os.environ.get("SPARSE_RECALL_WINDOW", "512"))
        self.sparse_recall_tau = float(os.environ.get("SPARSE_RECALL_TAU", "0"))

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
        if candidate_ids.numel() != self.vocab_size:
            return logits
        offsets = torch.arange(self.sparse_recall_window, device=input_ids.device)
        source_positions = positions[:, None] - offsets[None, :]
        valid = source_positions >= 0
        safe_positions = source_positions.clamp(min=0)
        source_tokens = input_ids.index_select(1, safe_positions.reshape(-1)).view(
            input_ids.size(0), positions.numel(), self.sparse_recall_window
        )
        sorted_tokens, sort_order = source_tokens.sort(dim=-1)
        indices = torch.arange(self.sparse_recall_window, device=input_ids.device).view(1, 1, -1)
        starts_run = torch.ones_like(sorted_tokens, dtype=torch.bool)
        starts_run[..., 1:] = sorted_tokens[..., 1:] != sorted_tokens[..., :-1]
        ends_run = torch.ones_like(sorted_tokens, dtype=torch.bool)
        ends_run[..., :-1] = sorted_tokens[..., :-1] != sorted_tokens[..., 1:]
        run_starts = torch.where(starts_run, indices, 0).cummax(dim=-1).values
        end_markers = torch.where(ends_run, indices, self.sparse_recall_window)
        run_ends = torch.flip(torch.cummin(torch.flip(end_markers, dims=(-1,)), dim=-1).values, dims=(-1,))
        sorted_weights = (run_ends - run_starts + 1).to(logits.dtype).reciprocal()
        occurrence_weights = torch.empty_like(sorted_weights).scatter_(2, sort_order, sorted_weights)
        if self.sparse_recall_tau > 0:
            recency = torch.exp(-offsets.to(torch.float32) / self.sparse_recall_tau).to(logits.dtype)
            occurrence_weights = occurrence_weights * recency.view(1, 1, -1)
        scale = F.softplus(self.sparse_recall_log_scale).to(logits.dtype)
        gate = torch.sigmoid(self.sparse_recall_gate(hidden)).to(logits.dtype)
        source_values = (scale * gate).expand(-1, -1, self.sparse_recall_window)
        source_values = source_values * occurrence_weights
        source_values = source_values * valid.view(1, positions.numel(), self.sparse_recall_window).to(logits.dtype)
        return logits.scatter_add(2, source_tokens, source_values)


experiment.experiment.trainer.CausalConvFactorizedLM = SparseExactRecallModel


if __name__ == "__main__":
    experiment.experiment.trainer.train(experiment.experiment.trainer.parse_args())
