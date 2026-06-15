# Artifact Notes

## Included

This repo includes:

- source code for the language-model architecture experiments
- test files for those experiments
- JSON / JSONL / text / PNG benchmark artifacts used to support the README claims
- full 50M-token watch histories for the final `partial_untied` and `nanochat_small` runs
- lightweight long-sequence research artifacts such as result JSON, CSV summaries, plots, logs, and writeups

## Not Committed Directly

Some very large derived dataset caches and training checkpoints were not committed directly to GitHub:

- `fineweb_edu_first20m_gpt2tokens_cache.pt` (about 80 MB)
- `fineweb_edu_first100m_gpt2tokens_cache.pt` (about 400 MB)
- long-sequence FineWeb-Edu token caches, some several GB each
- PyTorch checkpoint files from the long-sequence and neuron-search runs

Reason:

- they are derived caches rather than hand-authored research artifacts
- many files exceed normal GitHub file-size limits
- keeping them out of the repo makes the public snapshot much more usable

The current local inventory of GitHub-excluded large files is recorded in:

- [`large_artifacts_manifest.md`](./large_artifacts_manifest.md)

The code and artifact trail needed to understand the experiments is preserved here even without the raw cache files.
