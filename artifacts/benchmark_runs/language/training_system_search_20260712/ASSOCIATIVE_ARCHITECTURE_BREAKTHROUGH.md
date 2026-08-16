# Quality-First Associative Architecture Search

Date: 2026-07-12

## Decision

The promoted architecture is **Wave10 + exact multi-order phrase induction (orders 2, 3, 4) + learned semantic successor retrieval**.

This is the first result in this search that is both materially better than the sorted-unigram control and stable across two seeds. It is a promising architecture breakthrough, not yet a field-level SOTA claim. A larger independent-data screen is required before billion-token scaling.

## Architecture

Wave10 remains the dense representation path: collapsed multi-scale causal depthwise convolution, low-rank causal convolutional memory, and ReLU-squared FFNs. The new associative layer adds two complementary causal operators to the logits.

For phrase order `o` in `{2, 3, 4}`, define the collision-free key

`k_t^o = (x_{t-o+1}, ..., x_t)`.

Find the latest earlier position `p < t` with `k_p^o = k_t^o`, retrieve its observed successor `s_t^o = x_{p+1}`, and add

`softplus(alpha_o) * sigmoid(g_o(h_t)) * one_hot(s_t^o)`

to the logits. Sorting is stable and causal. No target or future token is used.

The semantic path computes

`z_t = normalize(W_sem h_t)`

and uses two signed random-projection LSH tables to find three prior contextual candidates per table. It scores candidates by learned cosine similarity, retrieves their observed successors, and adds a bounded softmax-weighted logit contribution. This handles contexts that are similar without having identical token phrases.

Training complexity is `O(n log n)` with `O(n)` temporary state and no sequence-by-sequence attention matrix. Exact phrase decoding can be incremental with constant-size hash lookups per order. The semantic prototype can also maintain per-bucket recent candidates incrementally, but the current generation code still rebuilds its index and must be replaced before long-context inference.

## Controlled Protocol

- Cache: `real_cache_finewebedu_sample_seq10160_train192_val8_gpt2.pt`
- Sequence length: 10,160
- Model: dim 512, rank 192, two Wave10 blocks, memory rank 64
- Same tokenizer, sampled-vocabulary objective, rotating full-vocabulary anchors, optimizer, schedule, validation blocks, and seeds
- Full-vocabulary validation loss is the promotion metric
- Short screen: 350 steps / 3.556M tokens
- Longer screen: 1,000 steps / 10.16M tokens
- RNG-preserving constructors ensure added zero-initialized modules do not alter the control's dropout trajectory

## Ranked 350-Step Results

| Rank | Tested variant | Seed | Val loss | Gain vs sorted control | Params | Tok/s | Decision |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | phrase 2+3+4 + semantic t2/k3 | 13 | **6.686475** | **0.104495** | 38,343,140 | 95,876 | promote |
| 2 | phrase 2+3+4 + semantic t2/k3 | 17 | **6.688571** | **0.106261** | 38,343,140 | 89,643 | confirm |
| 3 | phrase 2+3 + semantic t2/k3 | 13 | 6.699442 | 0.091527 | 38,342,946 | 95,062 | keep |
| 4 | phrase 2+3+4 | 13 | 6.705375 | 0.085594 | 38,326,561 | 116,870 | speed-quality option |
| 5 | phrase 2+3 + semantic t1/k2 | 13 | 6.709652 | 0.081318 | 38,342,946 | 77,970* | keep only as ablation |
| 6 | phrase 2+3, two-history vote | 13 | 6.716642 | 0.074328 | 38,326,367 | 97,922 | kill: poor marginal cost |
| 7 | phrase 2+3 | 13 | 6.719643 | 0.071326 | 38,326,367 | 119,203 | keep |
| 8 | bigram only | 13 | 6.743964 | 0.047006 | 38,326,173 | 123,226 | useful ablation |
| 9 | trigram only | 13 | 6.760852 | 0.030118 | 38,326,173 | 127,190 | useful ablation |
| 10 | semantic t2/k3 only | 13 | 6.765415 | 0.025555 | 38,342,558 | 113,649 | keep as component |
| 11 | self-selective memory | 13 | 6.790956 | 0.000014 | 38,326,491 | 119,567 | kill: noise-level |
| 12 | sorted unigram control | 13 | 6.790970 | - | 38,325,979 | 152,939 | control |
| 13 | rank-wise selective memory | 13 | 6.804175 | -0.013205 | 38,457,307 | 99,900 | kill |

`*` Two unrelated local training processes contaminated some throughput measurements. Loss comparisons remain deterministic; speed must be rerun on an isolated GPU.

Seed-17 sorted control: `6.794833`. Seed-17 final candidate: `6.688571`.

## Longer Result

At 1,000 steps / 10.16M tokens:

| Variant | Val loss | Tok/s | Peak VRAM |
|---|---:|---:|---:|
| sorted unigram control | 6.477781 | 161,560 | 1,621.8 MB |
| phrase 2+3 | 6.401321 | 114,029 | 1,622.2 MB |
| phrase 2+3 + semantic t2/k3 | **6.376553** | 91,146 | 1,761.8 MB |

The combined model improves loss by `0.101228` at equal tokens. Phrase-only is the stronger time-to-quality profile; the combined model is the stronger token-efficiency and final-quality profile. Order 4 was discovered after this longer run and has only been validated at 350 steps.

## Intervention Evidence

Interventions use the trained 1,000-step phrase-2+3 + semantic checkpoint and full-vocabulary validation over 81,280 tokens.

| Intervention | Full-vocab loss | Top-1 |
|---|---:|---:|
| full model | **6.376553** | **16.970%** |
| semantic disabled | 6.407180 | 16.574% |
| phrase disabled | 6.455828 | 15.952% |
| semantic and phrase disabled | 6.492637 | 15.492% |
| all induction disabled | 6.617764 | 14.375% |

The two new paths have independent value: semantic contributes `0.030627` and phrase contributes `0.079275` when disabled individually. Bigram and trigram recall-opportunity top-1 accuracy is about `60.8%` in the full model. Disabling both new paths reduces it to `37.6%` and `34.5%` respectively.

Loss improves in every frequency bucket relative to disabling semantic and phrase:

| Training frequency | Full model | New paths disabled | Gain |
|---|---:|---:|---:|
| 0-10 | 11.0897 | 11.2958 | 0.2061 |
| 11-100 | 9.1680 | 9.3075 | 0.1396 |
| 101-1,000 | 7.3969 | 7.5219 | 0.1250 |
| over 1,000 | 3.8041 | 3.8896 | 0.0855 |

This is not merely a common-token copy improvement.

## Rejected Directions

- Rank-wise and self-selective convolutional memory: no meaningful validation gain.
- Local block attention, including replacement and dual-path variants: worse loss with no sufficient systems benefit.
- Fenwick semantic memory and diagonal RLS memory: correct and finite, but impractical on this GPU implementation.
- Fused Gated DeltaNet through FLA/Triton on Turing: finite but over an order of magnitude too slow with available kernels.
- Two-history phrase voting: only `0.0030` loss gain and materially worse speed/VRAM.
- Plain activation swaps remain rejected from prior work.

## Cost and Limits

- Final 350-step candidate adds 17,161 parameters, only `0.0448%` over the control.
- Observed peak VRAM rises from about 1.622 GB to 1.762 GB.
- Observed throughput is roughly 37% lower in the short screen, although shared-machine contamination requires an isolated rerun.
- The current cache has only 192 training blocks and eight validation blocks. Repeated epochs make this a screening result, not scaling proof.
- Exact phrase recall is specialized. Learned semantic retrieval broadens the mechanism, but reasoning, state tracking, and assistant behavior are not established by these tiny runs.
- Current semantic generation rebuilds the prefix index; incremental inference is required.

## Required Next Gate

Do not launch a billion-token run from this evidence alone. Run a fresh 50-100M-token, 40M-parameter comparison on substantially more uncached training and validation blocks:

1. sorted unigram control;
2. phrase 2+3+4;
3. phrase 2+3+4 + semantic t2/k3;
4. seeds 13, 17, and 23;
5. full-vocabulary validation every 5-10M tokens;
6. isolated GPU timing and equal-wall-time comparison;
7. rare-token, number/name/code, repeated-phrase, semantic-retrieval, repetition, and copying metrics.

Promote to 80M+ and billion-token training only if the combined model keeps at least `0.05` full-vocabulary loss gain across seeds and the phrase-only model retains a time-to-quality advantage.

## Artifacts

- `phrase_induction_train.py`: batch-safe exact multi-order phrase operator
- `semantic_successor_train.py`: learned semantic LSH successor operator
- `phrase_semantic_induction_train.py`: combined architecture
- `evaluate_associative_breakthrough.py`: intervention and frequency-bucket evaluation
- `associative_breakthrough_interventions_1000_seed13.json`: intervention output
- `phrase234_semantic_t2k3_stride24_350_seed13/`: final seed-13 checkpoint/result
- `phrase234_semantic_t2k3_stride24_350_seed17/`: final seed-17 checkpoint/result
- `phrase23_semantic_t2k3_stride24_1000_seed13/`: longer combined checkpoint/result
- `phrase_induction_orders23_stride24_1000_seed13/`: longer phrase-only checkpoint/result
