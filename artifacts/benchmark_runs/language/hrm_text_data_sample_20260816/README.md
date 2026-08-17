# Official HRM-Text Data Sample

This directory contains a small deterministic sample from the official cleaned HRM-Text pretraining dataset:

`sapientinc/HRM-Text-data-io-cleaned-20260515`

The full dataset contains roughly 104 million instruction-response rows and is the cleaned output published by the authors' `sapientinc/data_io` pipeline. Its schema is:

- `instruction`: PrefixLM-visible task or prompt
- `response`: causally generated completion and loss-bearing region
- `condition`: task formatting condition, commonly `direct`

The sample uses evenly spaced 100-row pages across each split instead of taking the first rows. The repository is ordered by source, and a prefix-only sample would overrepresent the first mathematical dataset.

Files and exact row offsets, split sizes and SHA-256 hashes are recorded in `manifest.json`. Run `download_sample.py` to reproduce or refresh the sample.

This is sufficient for inspecting formatting, building a small PrefixLM cache and running pipeline smoke tests. It is not large enough for a meaningful language-model comparison.

## Local Source Subset

`download_source_files.py` also downloads 24,360 exact examples from four small files in the official mixture:

| Component | Rows | Purpose |
| --- | ---: | --- |
| No Robots | 10,000 | General instruction following and longer responses |
| GSM8K | 7,473 | Grade-school mathematical reasoning |
| OpenBookQA | 5,457 | Science question answering |
| FLAN ARC-Challenge | 1,430 | FLAN-formatted multiple-choice and task templates |

The four files occupy 17,292,089 bytes. Their upstream URLs and SHA-256 hashes are stored in `official_source_subset/manifest.json`.

The authors' exact BPE tokenizer is stored locally at `official_tokenizer/tokenizer.json`; its source commit and hash are recorded beside it. Raw upstream source files and the tokenizer are ignored by Git to avoid duplicating third-party datasets in this research repository. Download scripts, manifests and the 2,400-row inspection sample remain reproducible.
