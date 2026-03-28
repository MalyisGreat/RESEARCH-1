# Artifact Notes

## Included

This repo includes:

- source code for the language-model architecture experiments
- test files for those experiments
- JSON / JSONL / text / PNG benchmark artifacts used to support the README claims
- full 50M-token watch histories for the final `partial_untied` and `nanochat_small` runs

## Not Committed Directly

Some very large derived dataset caches were not committed directly to GitHub:

- `fineweb_edu_first20m_gpt2tokens_cache.pt` (about 80 MB)
- `fineweb_edu_first100m_gpt2tokens_cache.pt` (about 400 MB)

Reason:

- they are derived caches rather than hand-authored research artifacts
- the 100M cache exceeds normal GitHub file-size limits
- keeping them out of the repo makes the public snapshot much more usable

The code and artifact trail needed to understand the experiments is preserved here even without the raw cache files.
