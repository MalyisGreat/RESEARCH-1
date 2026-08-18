# GiveMeANode H100 Qualification

This is a bounded systems qualification, not the 5B-token proof run.

## Remote resources

- Workspace: `default`
- Data bucket: `research1-data`
- Cache object: `finewebedu/fresh_after2b_train1000106586_val325152_seq10160_gpt2.pt`
- Qualification node: `research1-h100-qual`
- Shape: `h100-1`, clock locked, 100 GiB ephemeral scratch
- Image: `pytorch-2.13-cuda12.9`

The uploaded cache contains 1,000,106,586 fresh GPT-2 tokens after the earlier
2B-token range. It is an int32 `torch.save` archive and is reused across arms.

## Qualification arms

`launch_h100_qualification.sh` runs two matched 350M-parameter arms:

1. Existing three-branch Wave10 convolution.
2. Exact collapsed Wave10 convolution.

Both use BF16, the optimized device-local candidate construction, full-vocab
validation, and no checkpoint writes. Defaults are deliberately bounded to 20
steps, batch size 1, two validation blocks, an 8,192-token training-loss chunk,
and a 4,096-token validation chunk.

The local proof before upload requires:

- identical legacy and optimized candidate loss;
- identical gradients;
- identical candidate count;
- finite forward/backward;
- collapsed-block feature error below `1e-6`.

## Remote command

```bash
export CACHE_PATH="$HOME/data/fresh_after2b_train1000106586_val325152_seq10160_gpt2.pt"
export RUN_ROOT="$HOME/research1-runs/h100-qualification"
bash artifacts/benchmark_runs/language/h100_wave10_350m_fullvocab_20260616/launch_h100_qualification.sh
```

Do not start the 5B run until the qualification records throughput, peak VRAM,
finite gradients, full-vocab loss, and a successful compiled follow-up for the
winning convolution path.
