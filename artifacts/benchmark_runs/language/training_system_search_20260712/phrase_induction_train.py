from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F
from torch import nn

import sorted_induction_train as experiment


BaseModel = experiment.SortedInductionModel


class PhraseInductionModel(BaseModel):
    """Exact causal induction over multiple token-context orders."""

    def __init__(self, config) -> None:
        super().__init__(config)
        orders_text = os.environ.get("PHRASE_ORDERS", "2,3,4")
        self.phrase_orders = tuple(int(value) for value in orders_text.split(",") if value)
        self.phrase_history = int(os.environ.get("PHRASE_HISTORY", "1"))
        if not self.phrase_orders or any(order < 2 for order in self.phrase_orders):
            raise ValueError("PHRASE_ORDERS must contain comma-separated integers >= 2")
        if any(order > 4 for order in self.phrase_orders):
            raise ValueError("collision-free GPT-2 phrase keys support orders up to 4 in int64")
        if self.phrase_history < 1:
            raise ValueError("PHRASE_HISTORY must be at least one")
        initial_scale = float(os.environ.get("PHRASE_INITIAL_SCALE", "1.0"))
        initial_gate_bias = float(os.environ.get("PHRASE_INITIAL_GATE_BIAS", "-1.0"))
        self.phrase_log_scales = nn.ParameterDict()
        self.phrase_gates = nn.ModuleDict()
        rng_state = torch.random.get_rng_state()
        for order in self.phrase_orders:
            self.phrase_log_scales[str(order)] = nn.Parameter(
                torch.tensor(math.log(math.expm1(initial_scale)))
            )
            gate = nn.Linear(config.conv_rank, 1)
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, initial_gate_bias)
            self.phrase_gates[str(order)] = gate
        # Training uses global CUDA RNG for dropout. Extra modules must not change
        # the matched baseline's stochastic trajectory.
        torch.random.set_rng_state(rng_state)
        self._phrase_tokens: dict[int, torch.Tensor] = {}
        self._phrase_valid: dict[int, torch.Tensor] = {}

    def _build_order_index(self, input_ids: torch.Tensor, order: int) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length = input_ids.shape
        keys = torch.zeros((batch, length), device=input_ids.device, dtype=torch.long)
        for offset in range(order):
            shifted = torch.zeros_like(input_ids)
            if offset == 0:
                shifted.copy_(input_ids)
            else:
                shifted[:, offset:] = input_ids[:, :-offset]
            keys = keys * self.vocab_size + shifted

        batch_retrieved = []
        batch_valid = []
        sorted_slots = torch.arange(length, device=input_ids.device)
        for batch_index in range(batch):
            local_keys = keys[batch_index]
            order_indices = torch.argsort(local_keys, stable=True)
            sorted_keys = local_keys.index_select(0, order_indices)
            retrieved_history = []
            valid_history = []
            for history_offset in range(1, self.phrase_history + 1):
                previous_positions = torch.roll(order_indices, history_offset)
                valid_sorted = sorted_slots >= history_offset
                valid_sorted &= sorted_keys == torch.roll(sorted_keys, history_offset)
                valid_sorted &= order_indices >= order - 1
                valid_sorted &= previous_positions >= order - 1
                valid_sorted &= previous_positions + 1 < length

                source_next = (previous_positions + 1).clamp(max=length - 1)
                retrieved_sorted = input_ids[batch_index].index_select(0, source_next)
                retrieved = torch.zeros_like(input_ids[batch_index])
                valid = torch.zeros_like(input_ids[batch_index], dtype=torch.bool)
                retrieved.scatter_(0, order_indices, retrieved_sorted)
                valid.scatter_(0, order_indices, valid_sorted)
                retrieved_history.append(retrieved)
                valid_history.append(valid)
            batch_retrieved.append(torch.stack(retrieved_history, dim=-1))
            batch_valid.append(torch.stack(valid_history, dim=-1))
        return torch.stack(batch_retrieved, dim=0), torch.stack(batch_valid, dim=0)

    def features(self, input_ids: torch.Tensor) -> torch.Tensor:
        features = super().features(input_ids)
        self._phrase_tokens.clear()
        self._phrase_valid.clear()
        for order in self.phrase_orders:
            tokens, valid = self._build_order_index(input_ids, order)
            self._phrase_tokens[order] = tokens
            self._phrase_valid[order] = valid
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
        base_logits = logits
        candidate_map = None
        if candidate_ids.numel() != self.vocab_size:
            candidate_map = torch.full(
                (self.vocab_size,), -1, device=candidate_ids.device, dtype=torch.long
            )
            candidate_map[candidate_ids] = torch.arange(candidate_ids.numel(), device=candidate_ids.device)

        for order in self.phrase_orders:
            retrieved = self._phrase_tokens[order].index_select(1, positions)
            valid = self._phrase_valid[order].index_select(1, positions)
            if candidate_map is None:
                candidate_indices = retrieved
            else:
                candidate_indices = candidate_map[retrieved]
                valid = valid & (candidate_indices >= 0)
                candidate_indices = candidate_indices.clamp(min=0)
            scale = F.softplus(self.phrase_log_scales[str(order)]).to(logits.dtype)
            gate = torch.sigmoid(self.phrase_gates[str(order)](hidden)).to(logits.dtype)
            weights = valid.to(logits.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
            values = scale * gate * weights
            logits = logits.scatter_add(2, candidate_indices, values)
        return self._bound_phrase_logits(base_logits, logits)

    def _bound_phrase_logits(self, base_logits: torch.Tensor, recall_logits: torch.Tensor) -> torch.Tensor:
        """Extension point for experiments; the baseline remains exactly unbounded."""
        return recall_logits


experiment.experiment.experiment.trainer.CausalConvFactorizedLM = PhraseInductionModel


if __name__ == "__main__":
    trainer = experiment.experiment.experiment.trainer
    trainer.train(trainer.parse_args())
