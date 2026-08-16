# 3080 LZ-sketch short screen summary

- Hub job: `20260614-231543-4d3784`
- Worker: `mwstroud-mwstr-6aea1cf3`, RTX 3080 only
- Cache: `finewebedu_fresh_after2b_train3000289275_val325152_seq10160_gpt2.pt`
- Scratch steps: 8; warm-start steps: 4; val blocks: 1; sampled vocab: 32768; seq length: 10160

| name | val loss | delta vs matched baseline | tok/s | peak VRAM MB | params | orders | slots | tables |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| scratch_lz_4096 | 12.8678 | -0.2492 | 431 | 12,932 | 76,219,801 | 0,1,2,4 | 4096 | 2 |
| scratch_lz_3hash | 12.8993 | -0.2177 | 307 | 12,625 | 76,219,801 | 0,1,2,4 | 2048 | 3 |
| scratch_lz_o0_1_2_4_8 | 12.9026 | -0.2143 | 345 | 13,339 | 76,220,123 | 0,1,2,4,8 | 2048 | 2 |
| scratch_lz_o0_1_2_4 | 12.9034 | -0.2135 | 484 | 12,299 | 76,219,801 | 0,1,2,4 | 2048 | 2 |
| scratch_lz_o0_1 | 12.9083 | -0.2087 | 1,220 | 10,225 | 76,219,157 | 0,1 | 2048 | 2 |
| scratch_lz_o0 | 12.9773 | -0.1397 | 6,514 | 9,189 | 76,218,835 | 0 | 2048 | 2 |
| scratch_baseline | 13.1170 | 0.0000 | 28,861 | 8,069 | 76,218,513 |  | 0 | 0 |
| warm_lz_o0_1_2_4 | 4.4193 | -0.0091 | 463 | 12,295 | 76,219,801 | 0,1,2,4 | 2048 | 2 |
| warm_baseline | 4.4284 | 0.0000 | 31,081 | 8,062 | 76,218,513 |  | 0 | 0 |

Interpretation: the 4096-slot 2-hash sketch was the best scratch short-screen row, but all multi-order sketch rows are much slower than baseline. Warm-start LZ helped only slightly in this very short screen.
