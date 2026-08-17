# Retrieval Quality Screen, Round 2

Date: 2026-08-16  
GPU: NVIDIA GeForce RTX 2080 SUPER  
Protocol: same cache, seed 13, 38M-class Wave10, sequence length 10,160, 1,000 steps / 10.16M token exposures, end-only full-vocabulary validation

This is a controlled local screen on a small repeated cache. It is evidence for ranking mechanisms, not evidence of fresh-data scaling.

## Results

| Rank | Variant | Full-vocab loss | Params | Tok/s | Peak VRAM | Decision |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Hidden phrase retrieval, dense mixer | **6.487678** | 38,588,760 | 190,403 | 1,601.9 MB | keep; longer fresh-data screen |
| 2 | Always-on semantic logits (prior round) | 6.497948 | 38,342,170 | 214,698 | 1,571.8 MB | kill; severe repetition |
| 3 | Soft confidence semantic logits | 6.499533 | 38,342,170 | 206,333 | 1,571.8 MB | retain as ablation |
| 4 | Strict confidence semantic logits | 6.501544 | 38,342,170 | 208,839 | 1,571.8 MB | kill; dominated by soft gate |
| 5 | Phrase 2+3 control | 6.503814 | 38,325,591 | 220,740 | 1,558.6 MB | control |
| 6 | Bounded phrase logits, ratio 0.20 | 6.509974 | 38,325,591 | 213,462 | 1,558.6 MB | kill; worse loss, no resource gain |
| 7 | Hidden phrase retrieval, diagonal mixer | 6.514760 | 38,327,128 | 193,121 | 1,618.4 MB | kill; cross-channel mixer is necessary |

## Repetition

Six fixed prompts, 64 generated tokens, temperature 0.8, top-k 40.

| Variant | Repeat-1 | Repeat-2 | Repeat-3 | Repeat-4 |
| --- | ---: | ---: | ---: | ---: |
| Phrase 2+3 control | 40.10% | 17.20% | 6.18% | 2.46% |
| Always-on semantic logits | 46.35% | 23.81% | 11.02% | 6.56% |
| Bounded phrase logits | 40.63% | 17.20% | 5.65% | 2.19% |
| Strict confidence semantic | 42.45% | 18.25% | 6.99% | 3.01% |
| Soft confidence semantic | 40.10% | 17.20% | 6.18% | 2.73% |
| Hidden phrase retrieval, dense | **38.28%** | 15.87% | 6.45% | 3.01% |
| Hidden phrase retrieval, diagonal | 39.06% | **15.34%** | **4.84%** | **1.91%** |

The dense hidden path is the only candidate that improves validation materially and improves unigram/bigram diversity. Its slight 3/4-gram regression is small compared with the always-on semantic failure, but requires a larger-prompt evaluation before scale-up. The diagonal version is diverse but loses too much validation quality.

## Interpretation

Direct logit recall is a copying shortcut. Bounding it weakens useful recall along with repetition, while semantic confidence gating can recover the control's repetition profile but yields only a small loss improvement. Hidden retrieval is more promising because the model can transform retrieved content through learned cross-channel mixing before prediction. The dense projection matters: replacing it with channel-wise gains reverses the loss gain.

The dense hidden model adds 263,169 parameters (0.69%), uses 43.3 MB more peak VRAM (2.78%), and is 13.7% slower than the phrase control. Its validation improvement is 0.01614 (0.25%). This is promising but not yet a breakthrough, especially given the repeated-cache limitation.

## Next Experiment

Run the dense hidden candidate and phrase control for 50M fresh, deduplicated tokens with at least three seeds or a paired token schedule. Add exact-match, repeated-span, rare-token, number/code-token, and retrieval-coverage metrics. Do not scale to billions until the loss advantage persists on fresh data and across seeds.
