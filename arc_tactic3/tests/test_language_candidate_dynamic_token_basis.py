from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import torch

from arc_tactic3 import language_candidate_dynamic_token_basis as candidate


def _make_cache(path: Path, *, vocab_size: int = 128, sequence_length: int = 7) -> None:
    block_size = sequence_length + 1
    train_tokens = torch.arange(block_size * 32, dtype=torch.long) % vocab_size
    val_tokens = torch.arange(block_size * 32, block_size * 40, dtype=torch.long) % vocab_size
    torch.save({"train_tokens": train_tokens, "val_tokens": val_tokens, "vocab_size": vocab_size}, path)


def test_dynamic_token_basis_model_produces_expected_shape() -> None:
    model = candidate.DynamicTokenBasisAssociativeLM(
        vocab_size=64,
        embedding_dim=16,
        hidden_dim=24,
        memory_dim=16,
        dropout=0.0,
        max_length=7,
        basis_rank=8,
        routing_experts=3,
        routing_top_k=2,
    )
    input_ids = torch.randint(0, 64, (2, 7), dtype=torch.long)
    logits = model(input_ids)
    assert logits.shape == (2, 7, 64)
    assert torch.isfinite(logits).all()


def test_dynamic_token_basis_compare_smoke_cpu() -> None:
    temp_dir = Path(tempfile.mkdtemp(dir=Path.cwd()))
    cache_path = temp_dir / "cache.pt"
    _make_cache(cache_path)
    try:
        payload = candidate.run_dynamic_token_basis_compare(
            candidate.DynamicTokenBasisConfig(
                cache_path=cache_path,
                train_blocks=16,
                val_blocks=4,
                sequence_length=7,
                train_steps=2,
                eval_interval=1,
                seed=13,
                device="cpu",
                use_amp=False,
                pin_memory=False,
                use_fused_adamw=False,
                compute_val_bpb=False,
                recurrent_embedding_dim=16,
                recurrent_hidden_dim=24,
                recurrent_memory_dim=16,
                partial_untied_tokens=16,
                token_basis_rank=8,
                routing_experts=3,
                routing_top_k=2,
                dropout=0.0,
            )
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    assert payload["benchmark"] == "language_candidate_dynamic_token_basis"
    assert payload["compare_target"] == "partial_untied"
    assert set(payload["results"]) == {"partial_untied", "dynamic_token_basis"}
    for report in payload["results"].values():
        assert report["parameter_count"] > 0
        assert report["final_val_loss"] >= 0.0
        assert report["train_tok_per_sec"] > 0.0
        assert report["pure_train_tok_per_sec"] > 0.0


def test_topk_router_limits_active_experts() -> None:
    logits = torch.tensor(
        [
            [[3.0, 1.0, -1.0], [0.5, 0.4, 0.3]],
            [[-0.2, 1.5, 0.1], [2.0, 2.0, -3.0]],
        ]
    )
    weights = candidate._topk_router(logits, top_k=2)
    assert weights.shape == logits.shape
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights[..., 0]))
    assert torch.all((weights > 0).sum(dim=-1) <= 2)
