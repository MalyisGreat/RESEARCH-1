# Local Wave10 Optimization Screen

Date: 2026-08-16
GPU: NVIDIA GeForce RTX 2080 SUPER
Cache: `real_cache_finewebedu_sample_seq10160_train192_val8_gpt2.pt`
Model: 38.3M parameters, Wave10, sequence length 10,160, seed 13
Budget: 1,000 steps / 10.16M token exposures unless noted

This is a systems and regression screen on the small repeated local cache. It is not fresh-data scaling evidence.

## Results

| Variant | Tokens | Full-vocab loss | Tok/s | Peak VRAM | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Phrase orders 2+3 | 10.16M | 6.503814 | 220,740 | 1,558.6 MB | current local control |
| Phrase orders 2+3+4 | 10.16M | 6.498715 | 217,230 | 1,558.7 MB | tiny quality gain, retain as ablation |
| Packed phrase orders 2+3+4 | 10.16M | 6.498715 | 217,793 | 1,558.7 MB | kill: 0.26% speed gain is noise-level |
| Phrase orders 2+3, batch 2 | 10.16M | 6.617114 | 186,561 | 2,871.1 MB | kill: slower and worse |
| Phrase orders 2+3 + semantic | 10.16M | 6.497948 | 214,698 | 1,571.8 MB | kill as always-on path: repetition regression |

The packed implementation exactly matches the reference retrieval masks and every retrieved token at valid positions on CPU and CUDA.

Evaluation cadence was tested at end-only versus steps 250/500/750/1000. Training and final validation were identical under the current code, so evaluation does not perturb the current training trajectory.

## Repetition

Six fixed prompts were sampled for 64 tokens with temperature 0.8 and top-k 40.

| Variant | Repeat-1 | Repeat-2 | Repeat-3 | Repeat-4 |
| --- | ---: | ---: | ---: | ---: |
| Phrase orders 2+3 | 40.10% | 17.20% | 6.18% | 2.46% |
| Phrase orders 2+3 + semantic | 46.35% | 23.81% | 11.02% | 6.56% |

The semantic path gains only 0.00587 validation loss in this current-code screen while materially worsening every repetition measure. It should not be scaled in always-on logit-injection form.

## Reproducibility Warning

The current source does not reproduce the older published 1,000-step phrase losses (`6.4013` phrase-only and `6.3766` combined) despite matching the recorded high-level configuration. Current isolated runs are around `6.50`. The old result artifacts did not record a source commit or source-tree hash, so code drift cannot yet be localized. Future results must record the Git commit, dirty-tree state, and trainer-file hash.

## Next Tests

1. Confidence-trigger semantic retrieval only when phrase confidence is low or base entropy is high.
2. Bound recall contribution relative to base-logit standard deviation.
3. Inject retrieved features into hidden state and process them before the output head.
4. Run the best variants on fresh, deduplicated data; do not promote from this repeated cache.
