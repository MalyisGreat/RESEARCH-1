# Research Decision Board - 2026-06-04

## Current Quality Read

The 160M long-sequence anchor model continued from 2B to 5B total tokens on fresh post-2B data. Validation loss improved from `4.642203100236851` at 2B tokens to `4.58411523998957` at 5B tokens. This is real progress but not enough to produce a capable language model: the 40-sample batch still shows prompt drift, weak direct QA, weak code response, and frequent medical/encyclopedia/religious boilerplate.

Artifacts:

- `longseq_anchor16_160m_5b_fresh_after2b_20260604_samples_loss/loss_curve_160m_5b.png`
- `longseq_anchor16_160m_5b_fresh_after2b_20260604_samples_loss/loss_history_160m_5b.json`
- `longseq_anchor16_160m_5b_fresh_after2b_20260604_samples_loss/samples_160m_5b_more_40_20260604.txt`
- `longseq_anchor16_160m_5b_fresh_after2b_20260604_samples_loss/samples_160m_5b_more_120_20260605_after_patch.txt`
- 400-sample expansion from the 5B checkpoint completed on the 3080 as hub job `20260604-200442-2d507b`; retrieval job `20260604-200628-a65f35` copied it back successfully.
- `longseq_anchor16_160m_5b_fresh_after2b_20260604_samples_loss/samples_160m_5b_more_400_20260605.txt` contains 400 completions, 271,688 bytes, with 54 EOS endings. Simple checks found 166/400 with medical/health drift and 188/400 with citation/research boilerplate; all ten `Question: What is machine learning? Answer:` completions failed the basic direct-answer check.
- 5B 120-sample expansion job `20260604-183836-ba8772` failed before the sampler was patched to load the checkpoint on CPU first.
- Patched 5B 120-sample expansion job `20260604-191936-68c0db` generated the full sample file, then failed only while printing Unicode text to cp1252 stdout. Retrieval job `20260604-193555-669a22` copied the file back successfully.
- `sample_longseq_checkpoint.py` was patched again to stop printing the entire sample body after writing the UTF-8 output file.

## Completed Architecture Evidence

Dense stride4 is the current keeper direction:

- Host 2080 dense stride4 60M: `val=5.4645802578901055`, `tok/s=75602`.
- Host 2080 gated stride4 60M: `val=5.524196946495787`, `tok/s=75087`.
- Host 2080 gated stride8 45M: `val=5.742143817492358`, `tok/s=105909`.
- Host 2080 memory stride4 60M: `val=5.466395527484223`, `tok/s=68291`.
- 2060 dense stride4 30M: `val=5.876664426383071`, `tok/s=72393`.
- 2060 gated stride8 30M: `val=6.0484`.
- 2060 landmark attention stride4 30M: `val=5.920425334220796`, `tok/s=64849`.
- 3080 gated stride8 100M fresh-after-2B: `val=5.339258316722442`, `tok/s=111127`.
- 3080 dense stride4 100M fresh-after-2B: `val=5.23925779752844`, `tok/s=108736`.
- 3080 dense stride4 fresh continuation 300M: `val=4.9111`, `tok/s=108751`.
- 3080 dense stride4 fresh continuation 400M: `val=4.8028`, `tok/s=108756`.
- 3080 dense stride4 fresh continuation 500M: `val=4.7682`, `tok/s=108760`.
- 4060 dense stride4 80M: `val=5.332662227210098`, `tok/s=67782`.
- 4060 dense stride4 continuation 160M: `val=5.1698`, `tok/s=67770`.
- 4060 dense stride4 continuation 240M: `val=4.9761`, `tok/s=67815`.
- 4060 dense stride4 continuation 300M: `val=4.9309`, `tok/s=67825`.
- 4060 gated stride8 80M: `val=5.470948456239513`, `tok/s=85287`.
- 4060 dense stride8 continuation 300M: `val=5.010951595149171`, `tok/s=81811`.
- Host 2080 landmark attention stride4 60M: `val=5.50221453407618`, `tok/s=64557`.
- Host 2080 multi-scale stride4 60M: `val=5.382352503829115`, `tok/s=69356.6957`, `peak_vram=4312 MB`.
- Host 2080 multi-scale stride4 continuation 100M: `val=5.2701041966285604`, `tok/s=68559.4367`, `peak_vram=4507 MB`.
- Host 2080 multi-scale stride4 continuation 200M: `val=5.0059`, `tok/s~68688`.
- Host 2080 multi-scale stride4 continuation 300M: `val=4.879228965097212`, `tok/s=68884`, `peak_vram=4507 MB`.
- 2060 multi-scale stride4 30M: `val=5.822688090284978`, `tok/s=69390.8578`, `peak_vram=3524 MB`.
- 2060 multi-scale stride4 continuation 100M: `val=5.3690`, `tok/s=69467`.
- 2060 multi-scale stride4 continuation 300M: `val=4.996778944721372`, `tok/s=69486`.
- 3080 multi-scale stride4 100M fresh-after-2B: `val=5.1293`, `tok/s=96983`.
- 3080 multi-scale stride4 continuation 300M: `val=4.813975762053738`, `tok/s=97046`.
- 4060 multi-scale stride4 80M: `val=5.2106`, `tok/s~61404`.
- 2060 dilated multi-scale stride4 30M: `val=5.8486`, `tok/s~67419`.
- 4060 dilated multi-scale stride4 80M: `val=5.2467`, `tok/s~59281`.
- Host 2080 dilated multi-scale stride4 60M: `val=5.420864054130444`, `tok/s=66067`, `peak_vram=4336 MB`.
- Host 2080 adaptive multi-scale stride4 60M: `val=5.380507930088544`, `tok/s=66740.9664`, `peak_vram=4363 MB`.
- 2060 adaptive multi-scale stride4 30M: `val=5.8203269276562635`, `tok/s=67530`, `peak_vram=3561 MB`.
- 2060 multi-scale stride8 30M: `val=5.8959`, `tok/s~76673`.

Pruning decisions:

- Gated block is rejected for now. It is faster at stride8, but it loses loss badly across host, 2060, 4060, and 3080 screens.
- Memory block is not promoted from the host screen. It tied dense stride4 on validation loss and was slower.
- Landmark attention is rejected for now. It directly targeted prompt/topic drift, but the host screen was both worse and slower than dense stride4.
- 2060 landmark attention confirms the rejection: it lost to 2060 dense stride4 and was slower.
- Multi-scale conv is promoted for longer screens. It beat dense stride4 on both host 2080 and 2060 validation loss, with lower VRAM, but it is slower than dense stride4.
- Dense stride4 should be promoted to longer continuation and fresh-cache comparisons.
- Dilated multi-scale is rejected for now. It beat 2060 dense stride4 30M, but lost to plain multi-scale on both 2060 30M and 4060 80M, while also running slower.
- Multi-scale stride8 is rejected as a quality direction on the 2060: it is faster, but `val=5.8959` is worse than dense stride4 and plain multi-scale stride4.
- Adaptive multi-scale is not promoted yet. It barely beat host plain multi-scale at 60M (`5.3805` vs `5.3824`) and barely beat 2060 plain multi-scale at 30M (`5.8203` vs `5.8227`), but both were slower. Treat as weak evidence, not a keeper direction yet.

## Active Runs

- 3080: `wave6_3080_multiscale_stride4_continue500m_total49215`, hub job `20260604-202043-ffd781`, resumed from 300M at step `29529` and is actively training toward 500M.
- 4060: `wave6_4060_multiscale_stride4_continue300m`, hub job `20260604-191304-3fe8cd`, running from the 80M plain multi-scale checkpoint; latest eval `240M` tokens, `val=4.8355`, `pure_tok_s~61413`.
- 2060: `wave9_2060_multiscale_memory_stride4_30m`, hub job `20260604-201956-a0f255`, running.
- 2080 host: `wave9_host2080_multiscale_memory_stride4_60m`, local pid `34680`, running after the adaptive screen completed; latest state around `53.8M/60.0M` tokens.

Queued follow-ons:

- 4060: dense stride4 continuation requeue, hub job `20260604-191304-307977`.
- 3080: an earlier `wave6_3080_multiscale_stride4_continue500m` job `20260604-201852-c613d3` no-oped because `train_steps` was supplied as additional rather than total post-resume steps; corrected total-step run is active as `20260604-202043-ffd781`.
- 2060: adaptive multi-scale 30M cross-check, hub job `20260604-200613-b5ffe3`, completed `val=5.8203269276562635`.
- 2060: `wave8_2060_multiscale_stride8_30m`, hub job `20260604-191352-f03dcf`, completed `val=5.8959`; rejected as a quality direction.
- 2060: deep/narrow multi-scale job `20260604-191017-d3a4eb` failed immediately with return code `3221226505`; treat that configuration as unsafe/rejected for now.

## Rejected Hypothesis: Landmark Attention

The main observed sample limitation is not only loss; it is poor prompt adherence and long-range/topic conditioning. The new `landmark_attention` block keeps the cheap causal conv-mixer path but adds attention over causal prefix landmarks every `landmark_stride` tokens. It is opt-in and checkpoint-compatible for old runs.

Host screen command shape:

- `block_type=landmark_attention`
- `embedding_dim=640`
- `conv_layers=2`
- `conv_rank=224`
- `attention_heads=4`
- `landmark_stride=128`
- `sampled_vocab_size=32768`
- `token_stride=4`
- `train_tokens=60004960`

Result:

- Rejected for now: host validation `5.50221453407618`, below dense stride4 `5.4645802578901055`, and throughput `64557` tok/s, below dense stride4 `75602` tok/s.

## Promoted Hypothesis: Multi-Scale Conv

The `multi_scale` block stays inside the convolutional advantage but mixes parallel causal depthwise kernels: short, medium, and longer windows. This is intended to improve local/mid-range context mixing without the landmark-attention slowdown or the gated-block loss hit.

Host screen command shape:

- `block_type=multi_scale`
- `embedding_dim=640`
- `conv_layers=2`
- `conv_rank=224`
- `sampled_vocab_size=32768`
- `token_stride=4`
- `train_tokens=60004960`

Result:

- Promoted for longer screens. Host 60M reached `val=5.382352503829115`, beating host dense stride4 `5.4645802578901055`.
- Host continuation to 100M reached `val=5.2701041966285604`, showing continued improvement.
- 2060 30M reached `val=5.822688090284978`, beating 2060 dense stride4 `5.876664426383071`.
- Tradeoff: speed is lower than dense stride4, but VRAM is better. Keep it as a quality/VRAM direction, not a throughput direction.

Sampling read:

- `wave6_multiscale_20260604/samples_host2080_multiscale_continue100m_80_20260604.txt` contains 80 samples from the 100M local checkpoint.
- Samples remain far below useful instruction quality: they are fluent-looking but incoherent, with weak direct QA/code behavior, generic topic drift, and quote-encoding artifacts.

## New Hypothesis Under Test: Dilated Multi-Scale Conv

The `dilated_multi_scale` block keeps the same depthwise-conv/factorized-output style, but mixes short, normal, 2x-dilated, and 4x-dilated causal branches. The goal is to widen effective context without landmark attention's slowdown or the old memory block's weak averaging.

Implementation status:

- Added as opt-in `block_type=dilated_multi_scale`; defaults and old checkpoint compatibility are preserved.
- `python -m py_compile standalone_longseq_anchor_train.py sample_longseq_checkpoint.py` passed.
- Tiny CPU forward smoke passed with finite logits.

Queued or active screens:

- 2060 30M screen: `wave7_2060_dilated_multiscale_stride4_30m`, hub job `20260604-184957-41c58c`, completed `val=5.8486`.
- 4060 80M screen: `wave7_4060_dilated_multiscale_stride4_80m`, hub job `20260604-185056-ce4bdc`, completed `val=5.2467`.
- Host 2080 60M screen: `wave7_host2080_dilated_multiscale_stride4_60m`, completed `val=5.420864054130444`.

## Staged Hypothesis: Adaptive Multi-Scale Conv

The `adaptive_multi_scale` block keeps the same convolution-only branch set as plain `multi_scale`, but replaces fixed branch averaging with learned per-channel softmax branch weights. This tests whether the model benefits from choosing short, medium, or long local windows by channel while preserving the cheap depthwise-conv/factorized-output shape.

Implementation status:

- Added as opt-in `block_type=adaptive_multi_scale`; defaults and old checkpoint compatibility are preserved.
- `python -m py_compile standalone_longseq_anchor_train.py sample_longseq_checkpoint.py` passed.
- Tiny CPU forward smoke passed with finite activations: `shape=(1, 64, 96)`, `params=210081`.
- Host 2080 60M screen completed at `val=5.380507930088544`, `tok/s=66740.9664`, `peak_vram=4363 MB`.
- 2060 30M cross-check completed at `val=5.8203269276562635`, only `0.0023611626287145` better than plain 2060 multi-scale 30M (`5.822688090284978`) and slower.
- Read: adaptive branch weighting is a weak positive on loss but a negative on throughput. Do not promote unless a longer run shows a real margin.

## Staged Hypothesis: Multi-Scale Plus Causal Memory

The `multi_scale_memory` block combines the promoted plain multi-scale depthwise branches with the low-rank causal prefix summary from the earlier memory block. The standalone memory block tied dense and was slower, but this tests whether cheap global summary helps after the stronger local mixer is already present.

Implementation status:

- Added as opt-in `block_type=multi_scale_memory`; defaults and old checkpoint compatibility are preserved.
- `python -m py_compile standalone_longseq_anchor_train.py sample_longseq_checkpoint.py` passed.
- Tiny CPU forward smoke passed with finite activations: `shape=(1, 64, 96)`, `params=216033`.
- Host 2080 60M screen staged as `wave9_host2080_multiscale_memory_stride4_60m`; watcher pid `14804` is waiting for adaptive pid `38792`.

## June 5 Update: Low-Rank Causal Conv Memory

The stronger follow-up to `multi_scale_memory` is `multi_scale_lowrank_conv_memory`: plain multi-scale causal depthwise branches plus a low-rank causal convolution memory path. This keeps the memory path local/causal instead of a prefix mean.

Results:

- 2060 30M: `val=5.635960840146373`, `tok/s=66266`, `peak_vram=3609 MB`. This is a large win over 2060 plain multi-scale 30M (`5.822688090284978`), adaptive multi-scale 30M (`5.8203269276562635`), and prefix-memory 30M (`5.815523709790913`).
- 2060 100M: `val=5.148364779517406`, `tok/s=66458.5253`, `peak_vram=3753 MB`. This confirms the 30M win at a longer token count.
- 2060 200M intermediate continuation eval: `val=4.9206`.
- Host 2080 60M: `val=5.162740405025132`, `tok/s=67253.6216`, `peak_vram=4411 MB`. This beats host plain multi-scale 60M (`5.382352503829115`), adaptive multi-scale 60M (`5.380507930088544`), and prefix-memory 60M (`5.366613806545578`).
- 2060 continuation to 300M is queued: hub job `20260604-205817-df624b`, resumed from the 100M checkpoint.
- Host 2080 continuation to 100M completed at `val=5.045041977952471`, `tok/s=67075.5945`, `peak_vram=4604 MB`. This is much better than host plain multi-scale 100M (`5.2701041966285604`).
- Host 2080 continuation to 300M is active: `wave10_host2080_lowrank_conv_memory_stride4_continue300m`, local pid `36816`, resumed from the 100M checkpoint. Intermediate 200M eval: `val=4.8113`.

Decision:

- Promote `multi_scale_lowrank_conv_memory` as the best new architecture signal so far. It has a much larger validation-loss margin than adaptive or prefix-memory variants while preserving roughly the same throughput class as plain multi-scale.
- Keep dense and plain multi-scale comparisons running until equal-token results finish; do not promote based only on short-screen wins.

## June 5 Update: 5B Continuation Sampling

The 160M LongSeq 2B to 5B continuation did improve validation loss, but sample quality remains poor.

Metrics:

- 2B checkpoint validation loss: `4.642203100236851`.
- 5B checkpoint validation loss: `4.58411523998957`.
- Approximate perplexity improved from about `103.8` to `97.9`.
- Local loss graph: `longseq_anchor16_160m_5b_fresh_after2b_20260604_samples_loss/loss_curve_160m_5b.png`.
- Existing 400-sample artifact: `longseq_anchor16_160m_5b_fresh_after2b_20260604_samples_loss/samples_160m_5b_more_400_20260605.txt`.
- 400-sample quick parse: `400` sections, `54` EOS markers, average completion block about `585` characters.
- Additional 1000-sample artifact: `longseq_anchor16_160m_5b_fresh_after2b_20260604_samples_loss/samples_160m_5b_more_1000_20260605.txt`.
- 1000-sample quick parse: `1000` sections, `168` EOS markers, average completion block about `657` characters, `68` copyright/URL-like boilerplate flags, `47` URL flags, and `10` encoding-artifact flags.

Quality read:

- More tokens made the model smoother and lowered loss, but it is still not instruction-useful. Samples drift across medical/science/news boilerplate, frequently ignore direct QA/code prompts, and include encoding artifacts.
- The larger 1000-sample batch completed on the 3080 worker as hub job `20260604-204438-be4ced` and was copied locally.

Current active/queued work:

- 3080: `wave6_3080_multiscale_stride4_continue500m_total49215_resume_after_cuda` completed the 500M eval at `val=4.743149240138963`, then started the queued low-rank-conv memory fresh 100M screen, hub job `20260604-210436-c98623`.
- 4060 Ti: user requested immediate stop. Dense stride4 500M comparison job `20260604-191304-307977` and queued token-gated screen `20260604-210310-789b63` are both cancelled. Do not queue new 4060 Ti work unless the user explicitly re-enables it.
- 2060: low-rank-conv memory continuation to 300M is running, hub job `20260604-205817-df624b`.
- Host 2080: low-rank-conv memory continuation to 300M, local pid `36816`.
- Host 2080 queued watcher: local pid `32884` waits for pid `36816`, then starts `wave12_host2080_token_gated_lowrank_conv_memory_stride4_60m`.
- 3080 completed: 1000-sample 5B batch, hub job `20260604-204438-be4ced`.

## June 5 Update: Token-Gated Multi-Scale Screen

The next broad architecture probe is `token_gated_multi_scale`. It keeps the same multi-scale causal depthwise branches, but uses a small per-token learned softmax gate to choose the branch mix dynamically. This differs from `adaptive_multi_scale`, which had fixed per-channel branch weights, and avoids adding attention or changing the factorized output path.

Implementation status:

- Added as opt-in `block_type=token_gated_multi_scale`; default behavior and existing checkpoints are unchanged.
- `python -m py_compile standalone_longseq_anchor_train.py sample_longseq_checkpoint.py` passed.
- CPU forward smoke passed: output `shape=(1, 64, 96)`, `params=327334`, finite activations.

Queued screens:

- 4060 Ti screen cancelled by user stop request: `wave11_4060_token_gated_multiscale_stride4_80m`, hub job `20260604-210310-789b63`.
- 3080: `wave10_3080_lowrank_conv_memory_stride4_100m_fresh`, hub job `20260604-210436-c98623`, running after the completed multi-scale 500M comparison to test the promoted low-rank-conv memory block at 3080 scale on the fresh-after-2B cache.

## June 5 Update: Token-Gated Low-Rank Memory Hybrid

The next hybrid probe is `token_gated_lowrank_conv_memory`. It combines per-token multi-scale branch gating with the best current low-rank causal convolution memory path. This keeps the core advantages: causal convolutional mixing, low-rank memory, sampled/factorized output, and opt-in checkpoint compatibility.

Implementation status:

- Added as opt-in `block_type=token_gated_lowrank_conv_memory`; default behavior and existing checkpoints are unchanged.
- `python -m py_compile standalone_longseq_anchor_train.py sample_longseq_checkpoint.py` passed.
- CPU forward smoke passed: output `shape=(1, 64, 96)`, `params=334374`, finite activations.

Queued screens:

- 3080: `wave12_3080_token_gated_lowrank_conv_memory_stride4_100m_fresh`, hub job `20260604-211828-a2c452`, queued behind the current 3080 low-rank-conv memory 100M screen.
- Host 2080: `wave12_host2080_token_gated_lowrank_conv_memory_stride4_60m`, watcher pid `32884`, queued behind the current host 300M continuation.

## June 5 Update: Host 2080 Shifted Off Local GPU

User requested stopping the local 2080 immediately and shifting the current 2080 continuation onto the 3080 after the 3080's current job.

Actions taken:

- Stopped local host 2080 trainer pid `36816`.
- Stopped local host 2080 watcher pid `32884`; no local host follow-on should start.
- Verified no local Python trainer or PowerShell watcher remains for the longseq host runs.
- Found recoverable host checkpoint at `wave10_lowrank_conv_memory_20260604/wave10_host2080_lowrank_conv_memory_stride4_continue300m/checkpoint.pt`.
- Checkpoint metadata: `step=19686`, `tokens_seen=200009760`, last eval `val=4.81127757251732`.
- Verified the checkpoint and matching base cache are reachable from the local file server for the 3080 worker.

3080 queue after reorder:

- Current running 3080 job remains `wave12_3080_token_gated_lowrank_conv_memory_stride4_100m_fresh`, hub job `20260604-211828-a2c452`.
- Canceled the earlier queued 3080 continuation `20260604-213323-7e8c48` before start so the shifted host run can take the next 3080 slot.
- Queued shifted host 2080 continuation on the 3080 as `wave10_host2080_lowrank_conv_memory_stride4_continue300m_shifted_to_3080`, hub job `20260604-213727-768154`. It resumes from the host checkpoint at 200.0M tokens and targets the original `29529` train-step / 300M-token endpoint.
- Requeued the 3080 low-rank-conv memory continuation behind the shifted host run as hub job `20260604-213809-65a479`.
- 4060 Ti remains excluded by user request; do not queue new work there unless explicitly re-enabled.

## June 5 Update: Hub Recovery And Current Scaling Policy

The house compute hub and local static file server both dropped offline after the 2080-to-3080 queue shift. The first shifted 3080 assignment became stale before printing its own start marker, so it was canceled and replaced with a retry using explicit `curl.exe` download logs.

Infrastructure recovery:

- Restarted the local file server on port `8790`; verified it serves `standalone_longseq_anchor_train.py`.
- Patched `house_compute_hub/hub.py` so repeated Windows `PermissionError` during `hub-state.json` replacement no longer breaks worker HTTP requests.
- Added `house_compute_hub/recover-services.ps1` to restart the hub/file server in detached mode and skip already-listening ports.
- Restarted the hub outside the sandbox; this was required because sandboxed temp-file replacement under `E:\CODEXRESEARCH` denied even simple replace operations.
- Verified the 3080 worker `mwstroud-mwstr-6aea1cf3` is visible again at `192.168.68.88`.

Current policy:

- Promote and scale `multi_scale_lowrank_conv_memory`; it remains the best architecture signal.
- Keep using the 2060 for short broad screens so the search does not over-focus on one idea.
- Do not use the 4060 Ti unless the user explicitly re-enables it.
- Do not use the local host 2080; its interrupted run is now represented by the shifted 3080 continuation.

Results and decisions:

- 2060 low-rank-conv memory rescue completed as hub job `20260605-010705-476939`, run `wave10_2060_lowrank_conv_memory_stride4_continue300m_resume_after_hubdrop`: final `val=4.796805142980861`, `train=4.8634796142578125`, `tok/s=66461.8683`, `tokens=300014640`. This strengthens the promotion case for `multi_scale_lowrank_conv_memory`.
- 2060 token-gated low-rank memory screen completed as hub job `20260605-010823-a88076`, run `wave12_2060_token_gated_lowrank_conv_memory_stride4_30m`: final `val=5.638194144288386`, `tok/s=64275.5311`, `peak_vram=3667 MB`. This is essentially flat/slightly worse than the earlier 2060 low-rank 30M baseline `val=5.635960840146373`, and slower, so do not promote token gating yet.
- Stale 3080 token-gated job `20260604-211828-a2c452` was canceled after reaching only step `6500/9843`; no final validation was produced.
- Stale shifted 3080 assignment `20260604-213727-768154` was canceled after it showed no execution output and `0%` GPU.

Current active/queued work:

- 3080 active: `wave10_host2080_lowrank_conv_memory_stride4_continue300m_shifted_to_3080_retry1`, hub job `20260605-011144-50d666`, resumed from host checkpoint `step=19686`, `tokens=200009760`. Latest observed training: step `23500/29529`, `tokens=238760000`, `loss=4.6962`, `tok/s=72961`, GPU util around `96-98%`.
- 3080 queued next: `wave10_3080_lowrank_conv_memory_stride4_continue300m_fresh_retry1`, hub job `20260605-011211-b755cd`.
- 2060 active: `wave13_2060_lowrank_conv_memory_kernel15_stride4_30m`, hub job `20260605-011722-0e3c28`, testing longer local kernel `15`.
- 2060 queued next: `wave13_2060_lowrank_conv_memory_memoryrank128_stride4_30m`, hub job `20260605-011722-017e5b`, testing `memory_rank=128`.

## June 5 Update: 3080 Long Run Corrected To Best Architecture

User corrected the 3080 long-run target: do not use the old dense 80M run; use the most promising architecture found so far. That remains `multi_scale_lowrank_conv_memory`.

Actions taken:

- Canceled the old 3080 300M-only continuation `wave10_3080_lowrank_conv_memory_stride4_continue300m_fresh_retry1`, hub job `20260605-011211-b755cd`, after it had reached step `11500`, `tokens=116840000`, latest loss `5.1288`.
- Queued the corrected 3080 long run as `wave10_3080_lowrank_conv_memory_5b_fresh_after5b_20260605`, hub job `20260605-013134-c2223c`.
- The long run resumes from the 76.2M checkpoint `wave10_3080_lowrank_conv_memory_stride4_100m_fresh/checkpoint.pt`, whose prior 100M result was `val=4.928917782959037`.
- Target is total step `492126`, or `5,000,000,160` training tokens. Because this architecture checkpoint is at about `100,004,880` tokens, the new run will add about `4.900B` tokens to reach the 5B total.
- Fresh train cache is intentionally not the existing fresh-after-2B cache. It uses offset `5,000,817,438` and cache path `finewebedu_fresh_after5b_lowrank76m_train4900477563_val325152_seq10160_gpt2.pt`, so it skips beyond the previous cached token window before collecting the new training tokens.
- Setup logs confirm the 3080 downloaded the updated trainer and fresh-continuation script, passed syntax checks, found Python dependencies, and started `FRESH_CACHE_BUILD_START`.

Additional 2060 research results:

- Kernel-size 15 screen `wave13_2060_lowrank_conv_memory_kernel15_stride4_30m`, hub job `20260605-011722-0e3c28`, finished with `val=5.632237777418978`, `train=5.691011905670166`, `tok/s=65108.9538`, `params=40,461,585`. This is a small improvement over the earlier 30M baseline but not large enough by itself to promote.
- Memory-rank 128 screen `wave13_2060_lowrank_conv_memory_memoryrank128_stride4_30m`, hub job `20260605-011722-017e5b`, finished with `val=5.625197367405328`, `train=5.689725399017334`, `tok/s=64534.1514`, `params=40,584,465`. This is the best short 2060 variant so far and should get follow-up.
- Queued combined follow-up `wave13_2060_lowrank_conv_memory_memoryrank128_landmark64_stride4_30m`, hub job `20260605-013355-bca176`, to test whether `memory_rank=128` combines with denser landmarks at `landmark_stride=64`.
- Denser-landmark combined follow-up `wave13_2060_lowrank_conv_memory_memoryrank128_landmark64_stride4_30m`, hub job `20260605-013355-bca176`, finished with `val=5.63443731745397`, `train=5.690289497375488`, `tok/s=65907.8350`, `params=40,568,081`. This regressed versus `memory_rank=128` alone, so do not promote the `landmark_stride=64` combination.
- Queued next short screen `wave13_2060_lowrank_conv_memory_memoryrank128_kernel15_stride4_30m`, hub job `20260605-014244-e3b907`, to test whether the two individually helpful changes, `memory_rank=128` and `conv_kernel_size=15`, combine.
- Kernel-15 plus memory-rank-128 screen `wave13_2060_lowrank_conv_memory_memoryrank128_kernel15_stride4_30m`, hub job `20260605-014244-e3b907`, finished with `val=5.625654260051532`, `train=5.675204277038574`, `tok/s=63150.3639`, `params=40,609,041`. This is close to but slightly worse than `memory_rank=128` with kernel 7, and is slower, so do not promote the kernel-15 combination.
- Promoted `memory_rank=128`, `conv_kernel_size=7`, `landmark_stride=128` to a 100M confirmation on the 2060 as `wave13_2060_lowrank_conv_memory_memoryrank128_stride4_100m`, hub job `20260605-015212-267bfb`.

Current policy after correction:

- 3080: dedicate to the corrected 5B run for `multi_scale_lowrank_conv_memory` 76.2M.
- 2060: continue short architecture research screens around low-rank memory variants.
- 4060 Ti: still excluded by user request.
- Local host 2080: still not used.
