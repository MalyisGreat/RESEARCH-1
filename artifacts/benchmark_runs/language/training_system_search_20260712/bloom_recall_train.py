from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

import compressed_head_rotating_collapsed_train as experiment
from sketch_recall_probe import decode_direct


BaseModel = experiment.CompressedHeadModel


class BloomRecallModel(BaseModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        initial_scale = 2.0
        self.bloom_log_scale = nn.Parameter(torch.tensor(math.log(math.expm1(initial_scale))))
        self.bloom_gate = nn.Linear(config.conv_rank, 1)
        nn.init.zeros_(self.bloom_gate.weight)
        nn.init.zeros_(self.bloom_gate.bias)
        self.bloom_tables = 3
        self.bloom_buckets = 4_096
        self.bloom_window = 512

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
        evidence = decode_direct(
            input_ids,
            candidate_ids,
            positions,
            self.bloom_window,
            self.bloom_tables,
            self.bloom_buckets,
            logits.dtype,
        )
        scale = F.softplus(self.bloom_log_scale).to(logits.dtype)
        gate = torch.sigmoid(self.bloom_gate(hidden)).to(logits.dtype)
        return logits + evidence * scale * gate


experiment.experiment.trainer.CausalConvFactorizedLM = BloomRecallModel


if __name__ == "__main__":
    experiment.experiment.trainer.train(experiment.experiment.trainer.parse_args())
