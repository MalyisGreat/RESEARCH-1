from __future__ import annotations

import os

import torch

import phrase_induction_train as phrase


class BoundedPhraseRecallModel(phrase.PhraseInductionModel):
    """Bound total exact-recall influence relative to the base-logit scale."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.recall_bound_ratio = float(os.environ.get("RECALL_BOUND_RATIO", "0.20"))

    def _bound_phrase_logits(self, base_logits: torch.Tensor, recall_logits: torch.Tensor) -> torch.Tensor:
        delta = recall_logits - base_logits
        scale = base_logits.float().std(dim=-1, keepdim=True).clamp_min(1e-3)
        cap = (self.recall_bound_ratio * scale).to(delta.dtype)
        return base_logits + cap * torch.tanh(delta / cap)


trainer = phrase.experiment.experiment.experiment.trainer
trainer.CausalConvFactorizedLM = BoundedPhraseRecallModel


if __name__ == "__main__":
    trainer.train(trainer.parse_args())
