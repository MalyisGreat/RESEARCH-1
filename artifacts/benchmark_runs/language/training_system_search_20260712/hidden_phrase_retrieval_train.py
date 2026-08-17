from __future__ import annotations

import math
import os

import torch
from torch import nn

import phrase_induction_train as phrase


class HiddenPhraseRetrievalModel(phrase.PhraseInductionModel):
    """Fuse exact retrieved successors into features instead of into vocabulary logits."""

    def __init__(self, config) -> None:
        super().__init__(config)
        rng_state = torch.random.get_rng_state()
        self.hidden_retrieval_proj = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False)
        self.hidden_retrieval_norm = nn.LayerNorm(config.embedding_dim)
        initial_gate = float(os.environ.get("HIDDEN_RETRIEVAL_INITIAL_GATE", "0.1"))
        self.hidden_retrieval_logit = nn.Parameter(torch.tensor(math.log(initial_gate / (1.0 - initial_gate))))
        nn.init.eye_(self.hidden_retrieval_proj.weight)
        torch.random.set_rng_state(rng_state)

    def features(self, input_ids: torch.Tensor) -> torch.Tensor:
        features = super().features(input_ids)
        retrieved_sum = torch.zeros_like(features)
        retrieved_count = torch.zeros((*features.shape[:2], 1), device=features.device, dtype=features.dtype)
        for order in self.phrase_orders:
            tokens = self._phrase_tokens[order]
            valid = self._phrase_valid[order]
            embeddings = self.embedding(tokens)
            mask = valid.unsqueeze(-1).to(embeddings.dtype)
            retrieved_sum = retrieved_sum + (embeddings * mask).sum(dim=2)
            retrieved_count = retrieved_count + mask.sum(dim=2)
        retrieved = retrieved_sum / retrieved_count.clamp_min(1.0)
        retrieved = self.hidden_retrieval_proj(self.hidden_retrieval_norm(retrieved))
        return features + torch.sigmoid(self.hidden_retrieval_logit) * retrieved

    def candidate_logits(self, **kwargs) -> torch.Tensor:
        # Skip PhraseInductionModel's direct logit boost to isolate hidden-state recall.
        return super(phrase.PhraseInductionModel, self).candidate_logits(**kwargs)


trainer = phrase.experiment.experiment.experiment.trainer
trainer.CausalConvFactorizedLM = HiddenPhraseRetrievalModel


if __name__ == "__main__":
    trainer.train(trainer.parse_args())
