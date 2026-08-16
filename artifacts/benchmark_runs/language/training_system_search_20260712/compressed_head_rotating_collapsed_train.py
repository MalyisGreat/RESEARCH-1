from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import nn

import rotating_anchor_collapsed_train as experiment


BaseModel = experiment.trainer.CausalConvFactorizedLM


class CompressedHeadModel(BaseModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.identity_head = os.environ.get("WAVE10_IDENTITY_HEAD", "0") == "1"
        multiplier = int(os.environ.get("WAVE10_HEAD_MULTIPLIER", "2"))
        if multiplier not in {1, 2, 3}:
            raise ValueError("WAVE10_HEAD_MULTIPLIER must be 1, 2, or 3")
        hidden_dim = multiplier * config.embedding_dim
        self.head_fc = nn.Linear(config.embedding_dim, hidden_dim)
        self.head_proj = nn.Linear(hidden_dim, config.embedding_dim)
        if self.identity_head:
            self.head_fc = nn.Identity()
            self.head_proj = nn.Identity()
        self.residual_head = os.environ.get("WAVE10_RESIDUAL_HEAD", "0") == "1"
        initial_scale = float(os.environ.get("WAVE10_HEAD_INITIAL_SCALE", "1.0"))
        self.head_scale = nn.Parameter(torch.tensor(initial_scale)) if self.residual_head else None

    def features(self, input_ids):
        states = self.embedding(input_ids)
        for block in self.blocks:
            states = block(states)
        normalized = self.final_norm(states)
        if self.identity_head:
            return normalized
        transformed = self.head_proj(F.relu(self.head_fc(normalized)).square())
        return normalized + self.head_scale * transformed if self.residual_head else transformed


experiment.trainer.CausalConvFactorizedLM = CompressedHeadModel


if __name__ == "__main__":
    experiment.trainer.train(experiment.trainer.parse_args())
