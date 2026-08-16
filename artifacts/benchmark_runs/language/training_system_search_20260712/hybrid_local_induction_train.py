from __future__ import annotations

import local_attention_memory_train as local_attention
import sorted_induction_train as induction


trainer = induction.experiment.experiment.trainer
trainer.CausalMultiScaleLowRankConvMemoryBlock = local_attention.LocalAttentionDropIn
trainer.CausalConvFactorizedLM = induction.SortedInductionModel


if __name__ == "__main__":
    trainer.train(trainer.parse_args())
