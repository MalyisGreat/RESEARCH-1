# HRM-Text Stability Recipe Transfer Test

Date: 2026-08-16  
GPU: NVIDIA GeForce RTX 2080 SUPER  
Protocol: seed 13, existing FineWeb-Edu cache, 1,000 steps / 10.16M token exposures, full-vocabulary validation every 250 steps

This experiment tests whether the architecture-side stability mechanisms in the official HRM-Text implementation rescue the local subquadratic HRM-Wave model. It is not an exact HRM-Text reproduction.

## Implemented

- Two slow H cycles and three fast L cycles with shared weights
- H: subquadratic landmark attention; L: Wave10 conv-memory
- Pre-RMSNorm in place of LayerNorm
- Official-style truncated deep-credit schedule: backpropagation depth warms from 2 to 5 recurrent steps over the first 20% of training
- Separate initialized low-level state
- Truncated LeCun-normal initialization
- Same local optimizer, cache, sampled/full-anchor objective and full-vocabulary evaluation

## Results

| Condition | 2.54M | 5.08M | 7.62M | 10.16M | Tok/s | VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phrase 2+3 control, prior 3-seed mean | 6.8853 | 6.6475 | 6.5359 | **6.5022** | 647k* | 1,559 MB |
| Prior HRM-Wave add, 3-seed mean | 7.0066 | **6.8302** | 6.8298 | 6.9063 | 229k* | 3,135 MB |
| Official-style initialization and independent L state | 7.6010 | 7.3133 | 7.0965 | 7.0451 | 72k | 2,579 MB |
| Compatible initialization and embedded L state | 7.0852 | 6.9231 | **6.9080** | 6.9613 | 71k | 2,579 MB |

`*` Prior runs were measured on a different GPU/session, so their throughput is contextual rather than a clean paired timing result.

## Interpretation

The wholesale recipe is stable but learns much too slowly under the local learning rate and squared-ReLU output head. Restoring the initialization and low-state convention compatible with Wave improves loss substantially, but does not eliminate the late validation upturn and remains worse than the earlier HRM-Wave model.

The stability mechanisms are therefore not the source of HRM-Text's reported advantage in isolation. The remaining major differences are full learned attention at both recurrent levels, gated attention and SwiGLU Transformer blocks, PrefixLM masking, instruction-response/task-completion data, the official output head, and a much larger training scale.

PrefixLM/task-completion was deliberately not simulated by masking arbitrary halves of raw FineWeb blocks. Without real instruction/response boundaries, that would waste half the tokens and would not test HRM-Text's data/objective claim.

## Decision

Kill both local stability-transfer variants. Do not spend a 100M-token budget on them. If HRM-Text is pursued further, use a small official-style Transformer HRM and a matched Transformer control on an actual instruction-response dataset. For the current Wave research direction, dense hidden-state retrieval remains the stronger candidate for fresh-data scale testing.
