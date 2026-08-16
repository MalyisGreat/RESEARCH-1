# H100 Wave10 350M Full-Vocab Proof Run

This package sets up a 350M-parameter Wave10 run for H100/H200 hardware.

The training objective is still the fast sampled+anchor objective from the existing trainer. The validation metric is full-vocabulary cross entropy over every validation token, so the tracked validation loss is not a sampled-vocab proxy.

## Chosen Architecture

- Block: `CausalMultiScaleLowRankConvMemoryBlock`
- Parameters: about `350,418,513`
- Sequence length: `10,160`
- Width: `2,304`
- Layers: `4`
- Multi-scale causal depthwise kernels: `3`, `7`, `15`
- Low-rank causal memory: rank `256`, kernel `128`
- Factorized vocab head rank: `1,536`
- Sampled training candidate set: `32,768`
- Full-vocab validation: enabled at every eval

## Learning Rate

Default schedule is cosine with:

- peak LR `2e-4`
- floor LR `2e-5`
- warmup `2,000` steps

This is deliberately lower than the 76M run because the model is roughly 4.6x larger. If the first 100M-token probe is stable but slow to improve, the next LR probe should test `2.5e-4` peak with the same floor.

## H100 Launch

```bash
export CACHE_PATH=/workspace/research1/cache/finewebedu_train5000m_val64_seq10160_gpt2.pt
export RUN_ROOT=/workspace/research1/runs
bash launch_h100_smoke.sh
```

If the smoke run is finite and VRAM is below the target, launch the full run:

```bash
export CACHE_PATH=/workspace/research1/cache/finewebedu_train5000m_val64_seq10160_gpt2.pt
export RUN_ROOT=/workspace/research1/runs
bash launch_h100_wave10_350m_fullvocab.sh
```

On H200, try `BATCH_SIZE=8` after the batch-4 smoke is clean.

## Outputs

Each run writes:

- `run_meta.json`: config, hardware, PyTorch/CUDA metadata
- `state.json`: live status
- `metrics.csv`: train and eval rows
- `metrics.jsonl`: append-only metric stream
- `checkpoint.pt`: latest checkpoint
- `checkpoint.step*_tokens*.pt`: milestones
- `result.json`: final report

The important column is `val_loss_full_vocab`. That is the number to compare against GPT-2-style baselines.

## Do Not Claim Success From This Alone

This run proves only whether the architecture scales on full-vocab validation loss. A real proof still needs matched GPT-2-small or GPT-2-medium baselines, sample audits, fresh validation, and long-context probes.
