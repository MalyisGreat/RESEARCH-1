from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F
from torch import nn

import sorted_induction_train as experiment


BaseModel = experiment.SortedInductionModel


class SemanticSuccessorModel(BaseModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        rng_state = torch.random.get_rng_state()
        self.semantic_rank = 32
        self.semantic_tables = int(os.environ.get("SEMANTIC_TABLES", "2"))
        self.semantic_candidates_per_table = int(os.environ.get("SEMANTIC_CANDIDATES", "3"))
        self.semantic_key = nn.Linear(config.embedding_dim, self.semantic_rank, bias=False)
        rotations = torch.randn(self.semantic_tables, self.semantic_rank, 16)
        rotations = F.normalize(rotations, dim=1)
        self.register_buffer("semantic_rotations", rotations, persistent=True)
        self.semantic_log_scale = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))
        self.semantic_gate = nn.Linear(config.conv_rank, 1)
        self.semantic_log_temperature = nn.Parameter(torch.tensor(math.log(0.25)))
        self.semantic_stop_gradient = os.environ.get("SEMANTIC_STOP_GRADIENT", "0") == "1"
        nn.init.zeros_(self.semantic_gate.weight)
        nn.init.constant_(self.semantic_gate.bias, -1.0)
        # Preserve the parent model's dropout RNG trajectory for strict controls.
        torch.random.set_rng_state(rng_state)
        self._semantic_features: torch.Tensor | None = None
        self._semantic_predecessors: torch.Tensor | None = None
        self._semantic_valid: torch.Tensor | None = None

    def build_semantic_index(self, features: torch.Tensor) -> None:
        semantic = F.normalize(self.semantic_key(features), dim=-1)
        if self.semantic_stop_gradient:
            semantic = semantic.detach()
        batch, length, _ = semantic.shape
        predecessor_tables = []
        valid_tables = []
        for batch_index in range(batch):
            batch_predecessors = []
            batch_valid = []
            detached = semantic[batch_index].detach()
            for table in range(self.semantic_tables):
                projection = detached @ self.semantic_rotations[table]
                buckets = torch.cat((projection, -projection), dim=-1).argmax(dim=-1)
                order = torch.argsort(buckets, stable=True)
                sorted_buckets = buckets.index_select(0, order)
                for shift in range(1, self.semantic_candidates_per_table + 1):
                    previous = torch.roll(order, shift)
                    valid_sorted = torch.arange(length, device=features.device) >= shift
                    valid_sorted &= sorted_buckets == torch.roll(sorted_buckets, shift)
                    predecessor = torch.zeros(length, device=features.device, dtype=torch.long)
                    valid = torch.zeros(length, device=features.device, dtype=torch.bool)
                    predecessor.scatter_(0, order, previous)
                    valid.scatter_(0, order, valid_sorted)
                    batch_predecessors.append(predecessor)
                    batch_valid.append(valid)
            predecessor_tables.append(torch.stack(batch_predecessors, dim=-1))
            valid_tables.append(torch.stack(batch_valid, dim=-1))
        self._semantic_features = semantic
        self._semantic_predecessors = torch.stack(predecessor_tables, dim=0)
        self._semantic_valid = torch.stack(valid_tables, dim=0)

    def features(self, input_ids: torch.Tensor) -> torch.Tensor:
        features = super().features(input_ids)
        self.build_semantic_index(features)
        return features

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
        if self._semantic_features is None or self._semantic_predecessors is None or self._semantic_valid is None:
            raise RuntimeError("semantic index missing; call features first")
        query = self._semantic_features.index_select(1, positions)
        predecessors = self._semantic_predecessors.index_select(1, positions)
        valid = self._semantic_valid.index_select(1, positions)
        batch, position_count, candidate_count = predecessors.shape
        batch_offsets = torch.arange(batch, device=hidden.device).view(batch, 1, 1) * input_ids.size(1)
        flat_predecessors = (predecessors + batch_offsets).reshape(-1)
        predecessor_keys = self._semantic_features.reshape(-1, self.semantic_rank).index_select(0, flat_predecessors)
        predecessor_keys = predecessor_keys.view(batch, position_count, candidate_count, self.semantic_rank)
        similarity = (query.unsqueeze(2) * predecessor_keys).sum(dim=-1)
        temperature = self.semantic_log_temperature.exp().clamp_min(0.05)
        similarity = similarity / temperature
        similarity = similarity.masked_fill(~valid, -30.0)
        weights = torch.softmax(similarity, dim=-1) * valid.to(similarity.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        successor_positions = (predecessors + 1).clamp(max=input_ids.size(1) - 1)
        flat_successors = (successor_positions + batch_offsets).reshape(-1)
        successor_tokens = input_ids.reshape(-1).index_select(0, flat_successors).view_as(predecessors)
        if candidate_ids.numel() == self.vocab_size:
            candidate_indices = successor_tokens
        else:
            candidate_map = torch.full((self.vocab_size,), -1, device=hidden.device, dtype=torch.long)
            candidate_map[candidate_ids] = torch.arange(candidate_ids.numel(), device=hidden.device)
            candidate_indices = candidate_map[successor_tokens]
            valid = valid & (candidate_indices >= 0)
            candidate_indices = candidate_indices.clamp(min=0)
            weights = weights * valid.to(weights.dtype)
        scale = F.softplus(self.semantic_log_scale).to(logits.dtype)
        gate = torch.sigmoid(self.semantic_gate(hidden)).to(logits.dtype)
        confidence = self._semantic_confidence(logits, weights, valid).to(logits.dtype)
        values = weights.to(logits.dtype) * scale * gate * confidence
        return logits.scatter_add(2, candidate_indices, values)

    def _semantic_confidence(
        self,
        logits: torch.Tensor,
        weights: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Extension point for confidence experiments; baseline is always enabled."""
        return torch.ones_like(weights[..., :1])


experiment.experiment.experiment.trainer.CausalConvFactorizedLM = SemanticSuccessorModel


if __name__ == "__main__":
    trainer = experiment.experiment.experiment.trainer
    trainer.train(trainer.parse_args())
