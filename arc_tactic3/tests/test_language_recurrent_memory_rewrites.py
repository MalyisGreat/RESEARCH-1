from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import torch

from arc_tactic3 import language_recurrent_memory_rewrites as rewrites


def _make_cache(path: Path, *, vocab_size: int = 128, sequence_length: int = 7) -> None:
    block_size = sequence_length + 1
    train_tokens = torch.arange(block_size * 32, dtype=torch.long) % vocab_size
    val_tokens = torch.arange(block_size * 32, block_size * 40, dtype=torch.long) % vocab_size
    torch.save({"train_tokens": train_tokens, "val_tokens": val_tokens, "vocab_size": vocab_size}, path)


def test_feature_retrieval_models_produce_expected_shape() -> None:
    config = rewrites.RecurrentMemoryRewriteConfig(
        cache_path=Path("unused.pt"),
        device="cpu",
        sequence_length=7,
        train_steps=2,
        eval_interval=1,
        recurrent_embedding_dim=16,
        recurrent_hidden_dim=24,
        recurrent_memory_dim=16,
        partial_untied_tokens=32,
        stride=2,
        chunk_size=4,
    )
    common = {
        "vocab_size": 64,
        "embedding_dim": config.recurrent_embedding_dim,
        "hidden_dim": config.recurrent_hidden_dim,
        "memory_dim": config.recurrent_memory_dim,
        "dropout": 0.0,
        "max_length": config.sequence_length,
    }
    input_ids = torch.randint(0, 64, (2, 7), dtype=torch.long)
    for mode in ("full", "stride", "chunk"):
        model = rewrites.FeatureRetrievalAssociativeLM(
            **common,
            source_mode=mode,
            stride=config.stride,
            chunk_size=config.chunk_size,
        )
        logits = model(input_ids)
        assert logits.shape == (2, 7, 64)
        assert torch.isfinite(logits).all()


def test_run_memory_rewrite_benchmark_smoke_cpu() -> None:
    temp_dir = Path(tempfile.mkdtemp(dir=Path.cwd()))
    cache_path = temp_dir / "cache.pt"
    _make_cache(cache_path)
    try:
        payload = rewrites.run_memory_rewrite_benchmark(
            rewrites.RecurrentMemoryRewriteConfig(
                cache_path=cache_path,
                train_blocks=16,
                val_blocks=4,
                sequence_length=7,
                train_steps=2,
                eval_interval=1,
                seed=13,
                device="cpu",
                partial_untied_tokens=32,
                recurrent_embedding_dim=16,
                recurrent_hidden_dim=24,
                recurrent_memory_dim=16,
                stride=2,
                chunk_size=4,
            )
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    assert payload["benchmark"] == "language_recurrent_memory_rewrites"
    assert set(payload["results"]) == {
        "recurrent_champion",
        "recurrent_relu2_legacy",
        "partial_untied",
        "feature_retrieval_full",
        "feature_retrieval_stride4",
        "feature_retrieval_chunk8",
        "chunk8_partial_hybrid",
        "chunk8_token_partial_memory",
    }
    for report in payload["results"].values():
        assert report["parameter_count"] > 0
        assert report["final_val_loss"] >= 0.0
        assert report["train_tok_per_sec"] > 0.0
        assert report["pure_train_tok_per_sec"] > 0.0
