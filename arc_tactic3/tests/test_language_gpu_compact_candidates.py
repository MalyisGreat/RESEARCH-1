from __future__ import annotations

from pathlib import Path

import torch

from arc_tactic3.language_gpu_compact_candidates import (
    DenseValueLocalGlobalPartialLM,
    DensePartialWindowedPartialLM,
    DenseValueWindowedPartialLM,
    FusedCompactPartialUntiedLM,
    FusedLearnedChunkWriterPartialLM,
    FusedSharedLocalGlobalPartialLM,
    FusedWindowedCompactPartialUntiedLM,
    GPUCompactCandidatesConfig,
    _scatter_attention_to_partial,
    run_gpu_compact_candidates,
)
from arc_tactic3.language_recurrent_nano_tricks import PartialUntiedAssociativeLM


def test_scatter_attention_to_partial_matches_reference() -> None:
    attention = torch.tensor(
        [[
            [0.10, 0.20, 0.30, 0.40],
            [0.40, 0.30, 0.20, 0.10],
        ]],
        dtype=torch.float32,
    )
    token_ids = torch.tensor([[5, 2, 5, 9]], dtype=torch.long)
    token_to_partial = torch.full((10,), -1, dtype=torch.long)
    token_to_partial[2] = 0
    token_to_partial[5] = 1
    token_to_partial[9] = 2
    partial = _scatter_attention_to_partial(
        attention=attention,
        token_ids=token_ids,
        token_to_partial=token_to_partial,
        partial_size=3,
    )
    reference = torch.tensor(
        [[
            [0.20, 0.40, 0.40],
            [0.30, 0.60, 0.10],
        ]],
        dtype=torch.float32,
    )
    assert torch.allclose(partial, reference)


def test_fused_compact_partial_matches_baseline_when_subset_is_full_vocab() -> None:
    torch.manual_seed(0)
    vocab_size = 17
    untied = torch.arange(vocab_size)
    compact = FusedCompactPartialUntiedLM(
        vocab_size=vocab_size,
        embedding_dim=8,
        hidden_dim=12,
        memory_dim=8,
        dropout=0.0,
        max_length=7,
        untied_token_ids=untied,
    )
    baseline = PartialUntiedAssociativeLM(
        vocab_size=vocab_size,
        embedding_dim=8,
        hidden_dim=12,
        memory_dim=8,
        dropout=0.0,
        max_length=7,
        untied_token_ids=untied,
    )
    baseline.load_state_dict(compact.state_dict(), strict=False)
    input_ids = torch.randint(0, vocab_size, (2, 7))
    compact.eval()
    baseline.eval()
    with torch.no_grad():
        compact_logits = compact(input_ids)
        baseline_logits = baseline(input_ids)
    assert torch.allclose(compact_logits, baseline_logits, atol=1e-5, rtol=1e-5)


def test_compact_candidate_forwards_have_expected_shape() -> None:
    torch.manual_seed(1)
    vocab_size = 23
    untied = torch.arange(8)
    input_ids = torch.randint(0, vocab_size, (2, 9))
    models = [
        FusedCompactPartialUntiedLM(
            vocab_size=vocab_size,
            embedding_dim=8,
            hidden_dim=12,
            memory_dim=8,
            dropout=0.0,
            max_length=9,
            untied_token_ids=untied,
        ),
        FusedWindowedCompactPartialUntiedLM(
            vocab_size=vocab_size,
            embedding_dim=8,
            hidden_dim=12,
            memory_dim=8,
            dropout=0.0,
            max_length=9,
            untied_token_ids=untied,
            window_size=4,
        ),
        DenseValueWindowedPartialLM(
            vocab_size=vocab_size,
            embedding_dim=8,
            hidden_dim=12,
            memory_dim=8,
            dropout=0.0,
            max_length=9,
            untied_token_ids=untied,
            window_size=4,
        ),
        DensePartialWindowedPartialLM(
            vocab_size=vocab_size,
            embedding_dim=8,
            hidden_dim=12,
            memory_dim=8,
            dropout=0.0,
            max_length=9,
            untied_token_ids=untied,
            window_size=4,
        ),
        FusedLearnedChunkWriterPartialLM(
            vocab_size=vocab_size,
            embedding_dim=8,
            hidden_dim=12,
            memory_dim=8,
            dropout=0.0,
            max_length=9,
            chunk_size=3,
            untied_token_ids=untied,
        ),
        FusedSharedLocalGlobalPartialLM(
            vocab_size=vocab_size,
            embedding_dim=8,
            hidden_dim=12,
            memory_dim=8,
            dropout=0.0,
            max_length=9,
            untied_token_ids=untied,
            local_window=4,
            older_chunk_size=3,
        ),
        DenseValueLocalGlobalPartialLM(
            vocab_size=vocab_size,
            embedding_dim=8,
            hidden_dim=12,
            memory_dim=8,
            dropout=0.0,
            max_length=9,
            untied_token_ids=untied,
            local_window=4,
            older_chunk_size=3,
        ),
    ]
    for model in models:
        logits = model(input_ids)
        assert logits.shape == (2, 9, vocab_size)
        assert torch.isfinite(logits).all()


def test_run_gpu_compact_candidates_cpu_smoke(tmp_path: Path) -> None:
    sequence_length = 7
    block_size = sequence_length + 1
    vocab_size = 32
    train_tokens = torch.randint(0, vocab_size, (block_size * 24,))
    val_tokens = torch.randint(0, vocab_size, (block_size * 8,))
    cache_path = tmp_path / "tiny_cache.pt"
    torch.save(
        {
            "train_tokens": train_tokens,
            "val_tokens": val_tokens,
            "vocab_size": vocab_size,
        },
        cache_path,
    )
    config = GPUCompactCandidatesConfig(
        cache_path=cache_path,
        train_blocks=12,
        val_blocks=4,
        train_steps=1,
        eval_interval=1,
        seed=7,
        device="cpu",
        batch_size=4,
        eval_batch_size=4,
        sequence_length=sequence_length,
        embedding_dim=16,
        hidden_dim=24,
        memory_dim=16,
        partial_token_count=8,
        use_amp=False,
        pin_memory=False,
        use_fused_adamw=False,
        cache_dataset_on_device=False,
    )
    payload = run_gpu_compact_candidates(config, seeds=[7])
    assert payload["benchmark"] == "language_gpu_compact_candidates"
    assert set(payload["mean_results"]) == {
        "partial_untied",
        "compact_partial",
        "compact_window32",
        "dense_value_window32",
        "dense_partial_window32",
        "compact_chunk_writer",
        "compact_local_global",
        "dense_value_local_global",
    }
    for result in payload["mean_results"].values():
        assert result["final_val_loss"] > 0.0
