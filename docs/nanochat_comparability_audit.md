# NanoChat Comparability Audit

Audit date: 2026-07-18

## Verdict

The historical `nanochat_watch` result is a useful matched local architecture probe, but it is not a faithful replication of Karpathy's NanoChat training system. It must not be used to claim that `partial_untied`, Wave10, or a later architecture beats official NanoChat in quality, training efficiency, or speed.

The local `NanochatMiniLM` closely reproduces many block-level operations from NanoChat: untied embeddings, rotary position encoding, QK normalization, `SSSL` sliding-window attention, value embeddings, squared ReLU MLPs, learned residual/input mixing, smear, backout, and logit soft-capping. The largest discrepancies are in scale geometry and the training system around the block.

## Audited References

- Local implementation: [`arc_tactic3/language_nanochat_actual_compare.py`](../arc_tactic3/language_nanochat_actual_compare.py)
- Local 50M artifact: [`artifacts/watch_runs/nanochat_watch_50m_20260328_retry2/final.json`](../artifacts/watch_runs/nanochat_watch_50m_20260328_retry2/final.json)
- Upstream NanoChat commit available when the local run was made: [`d8bbddb`](https://github.com/karpathy/nanochat/tree/d8bbddb07dcbe212b66554400b72ad9a53deb31b)
- Current upstream snapshot inspected: [`92d63d4`](https://github.com/karpathy/nanochat/tree/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd)

## Material Differences

| Dimension | Local `nanochat_watch` | Official NanoChat on 2026-03-27 | Consequence |
| --- | --- | --- | --- |
| Model geometry | 4 layers, width 40, 4 heads, head dimension 10 | Width is `depth * 64`, rounded to head dimension 128; depth 4 gives width 256 and 2 heads | The local model is far outside NanoChat's tuned depth-scaling family |
| Parameters | 8.13M total; about 99% are token, output, and value-embedding tables | Official depth 4 is about 36.7M total | Equal local parameter count does not represent an official NanoChat scale point |
| Tokenizer | GPT-2, 50,257 tokens | NanoChat-trained byte-level BPE, 32,768 tokens | Loss and parameter allocation change; raw token loss is not directly comparable |
| Data | 18M cached FineWeb-Edu training tokens reused for 50M token exposures | Streaming ClimbMix documents | Different distribution and document boundaries; the local run made about 2.78 passes over its training cache |
| Packing | Fixed contiguous 128-token blocks | BOS-aligned best-fit document packing at 2048 tokens | Local training loses the official context and packing behavior |
| Context | 127 prediction positions | 2048 prediction positions | Attention, induction, and long-range learning are tested in different regimes |
| Optimizer | Fused AdamW over all parameters, LR 0.002, weight decay 0.0001 | Muon for block matrices; separate AdamW groups and learning rates for embeddings, output, value embeddings, and scalars | This omits a central part of NanoChat's learning recipe |
| Schedule | Constant LR, no warmup or warmdown | 40-step warmup, hold, then 65% linear warmdown; Muon momentum and weight decay are also scheduled | Optimization trajectories are not comparable |
| Precision/kernel | Autocast and explicit dense attention mask | Explicit compute dtype and Flash Attention path | The local speed and VRAM numbers are not representative of upstream |
| Evaluation | Full-vocab GPT-2-token cross-entropy on 512 cached blocks | Tokenizer-invariant validation bits per byte over a larger stream plus DCLM CORE | The reported quality numbers answer different questions |

The later local `fair_cosine_sssl` sweep only changed AdamW decay grouping and added a generic cosine schedule. It still did not implement NanoChat's Muon/AdamW parameter groups, per-group learning rates, momentum schedule, weight-decay schedule, tokenizer, data loader, context, or scaling rule.

## What The Old Result Does Establish

At the same local seed, GPT-2-token cache, sequence length, token exposure count, and approximately 8M total parameters:

| Model | Final validation loss | Train tokens/s | Peak VRAM |
| --- | ---: | ---: | ---: |
| `partial_untied` | 5.3370 | 37.5k | 2201.9 MB |
| NanoChat-inspired mini-port | 5.3958 | 39.9k | 3931.2 MB |

This supports a narrow statement: `partial_untied` performed better in that local microbenchmark. It does not support a comparison against an upstream NanoChat model.

## Required Comparison

Use two complementary tracks and keep their labels separate.

### Track A: Upstream Reference

Run unmodified upstream NanoChat at depth 4:

- upstream commit pinned in the result manifest
- official 32,768-token tokenizer
- official ClimbMix train/validation split
- sequence length 2048
- official BOS-aligned best-fit loader
- official Muon/AdamW optimizer and all schedules
- official auto-computed batch size and 12 scaling tokens per scaling parameter
- report validation BPB, CORE, tokens/s, FLOPs, wall time, and peak VRAM

This is the reference-faithful NanoChat number. Do not replace its optimizer or data pipeline in the name of matching the candidate.

### Track B: Controlled Architecture Comparison

Train the upstream NanoChat block and the candidate architecture with:

- identical tokenizer and raw document split
- identical packed batches in identical order
- identical context length and full-vocabulary next-token objective
- matched non-embedding parameter count and a separately reported total parameter count
- matched token budget and evaluation checkpoints
- at least three seeds for the promoted scale
- validation BPB as the primary loss metric
- CORE, copying/recall probes, speed, peak VRAM, and estimated FLOPs as secondary metrics

Give each architecture its best justified optimizer recipe, but add an optimizer ablation if the winning conclusion depends on that choice.

## Promotion Rule

Do not say the candidate beats NanoChat unless it beats the pinned upstream reference or controlled upstream block on validation BPB at matched training FLOPs or matched wall time, and the result survives at least three seeds. Raw GPT-2-token loss from the historical harness is not eligible for that claim.
