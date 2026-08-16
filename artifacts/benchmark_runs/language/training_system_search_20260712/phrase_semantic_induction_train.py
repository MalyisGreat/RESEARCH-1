from __future__ import annotations

import phrase_induction_train as phrase
import semantic_successor_train as semantic


class PhraseSemanticInductionModel(
    semantic.SemanticSuccessorModel,
    phrase.PhraseInductionModel,
):
    """Wave10 with exact unigram/phrase induction and learned semantic retrieval."""


trainer = phrase.experiment.experiment.experiment.trainer
trainer.CausalConvFactorizedLM = PhraseSemanticInductionModel


if __name__ == "__main__":
    trainer.train(trainer.parse_args())
