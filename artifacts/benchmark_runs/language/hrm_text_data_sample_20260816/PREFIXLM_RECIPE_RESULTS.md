# Official-Data PrefixLM Recipe Screen

Date: 2026-08-16  
GPU: NVIDIA GeForce RTX 2080 SUPER  
Data: 24,360 examples from the official cleaned HRM-Text mixture  
Tokenizer: official HRM-Text 65,536-token BPE  
Training: seed 13, 500 steps, batch 4, maximum length 192, 110,868 supervised response tokens

## Contract

This screen implements the authors' sequence contract rather than ordinary next-token pretraining:

- beginning-of-question + condition + instruction + end-of-question
- response + end-of-answer
- bidirectional attention within the instruction prefix
- causal response attention over prefix and preceding response tokens
- response-only cross-entropy
- pre-RMSNorm, RoPE, gated multi-head attention, SwiGLU and tied embeddings/output head

All models have 8,914,568 parameters. The control has two distinct Transformer blocks. HRM has one high and one low block, reused for two high cycles and three low cycles per high cycle.

## Results

| Condition | 20.7K tokens | 42.5K | 65.0K | 87.2K | 110.9K | Response tok/s | Peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Transformer control | **7.9440** | **7.0125** | **6.8318** | **6.7676** | **6.7374** | **7,364** | 726.5 MB |
| HRM full backprop | 8.0672 | 7.1446 | 6.9773 | 6.9081 | 6.8785 | 2,985 | 739.8 MB |
| HRM 2→5 credit warmup | 8.2164 | 7.1811 | 7.0017 | 6.9313 | 6.9004 | 2,972 | 740.1 MB |

At the endpoint, full-backprop HRM is 0.1411 loss worse than the matched Transformer and 2.47 times slower. Credit warmup is another 0.0219 worse than full HRM. No HRM crossover appears anywhere in the measured curve.

## Interpretation

Using authentic task-completion data and PrefixLM materially changes the experiment, but does not make this tiny HRM configuration competitive. The negative result is internally strong for this 8.9M-parameter, 111K-supervised-token regime because parameter count, tokenizer, examples, schedule and objective are matched.

It does not falsify the official HRM-Text claim. The published model is roughly 1B parameters and trains on about 40B tokens. This screen is over 100 times smaller in parameters and more than 300,000 times smaller in supervised-token exposure. Credit warmup is designed for a long training horizon; here it consumes 20% of only 500 steps and never recovers its early deficit.

## Decision

Do not scale this small HRM configuration locally. For an immediate local model, use the Transformer control on this data. For the main Wave research program, continue the dense hidden-state retrieval fresh-data comparison. A serious HRM-Text replication should use the official repository on Hopper GPUs at substantially larger scale rather than further modifying this miniature implementation.
