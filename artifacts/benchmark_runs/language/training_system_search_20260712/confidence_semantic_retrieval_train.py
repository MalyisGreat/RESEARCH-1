from __future__ import annotations

import os

import torch

import phrase_induction_train as phrase
import phrase_semantic_induction_train as combined


class ConfidenceSemanticRetrievalModel(combined.PhraseSemanticInductionModel):
    """Enable semantic recall only for confident matches at uncertain positions."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.retrieval_confidence_threshold = float(os.environ.get("RETRIEVAL_CONFIDENCE_THRESHOLD", "0.55"))
        self.retrieval_confidence_temperature = float(os.environ.get("RETRIEVAL_CONFIDENCE_TEMPERATURE", "0.08"))
        self.retrieval_entropy_threshold = float(os.environ.get("RETRIEVAL_ENTROPY_THRESHOLD", "0.55"))
        self.retrieval_entropy_temperature = float(os.environ.get("RETRIEVAL_ENTROPY_TEMPERATURE", "0.08"))

    def _semantic_confidence(
        self,
        logits: torch.Tensor,
        weights: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        match_confidence = weights.max(dim=-1, keepdim=True).values
        match_gate = torch.sigmoid(
            (match_confidence - self.retrieval_confidence_threshold) / self.retrieval_confidence_temperature
        )
        probabilities = torch.softmax(logits.float(), dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-9).log()).sum(dim=-1, keepdim=True)
        normalized_entropy = entropy / torch.log(torch.tensor(logits.size(-1), device=logits.device))
        uncertainty_gate = torch.sigmoid(
            (normalized_entropy - self.retrieval_entropy_threshold) / self.retrieval_entropy_temperature
        )
        any_valid = valid.any(dim=-1, keepdim=True).to(match_gate.dtype)
        return match_gate * uncertainty_gate * any_valid


trainer = phrase.experiment.experiment.experiment.trainer
trainer.CausalConvFactorizedLM = ConfidenceSemanticRetrievalModel


if __name__ == "__main__":
    trainer.train(trainer.parse_args())
