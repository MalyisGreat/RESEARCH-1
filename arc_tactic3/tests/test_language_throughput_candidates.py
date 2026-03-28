from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import torch

from arc_tactic3 import language_throughput_candidates as candidates


def _make_cache(path: Path, *, vocab_size: int = 128, sequence_length: int = 7) -> None:
    block_size = sequence_length + 1
    train_tokens = torch.arange(block_size * 32, dtype=torch.long) % vocab_size
    val_tokens = torch.arange(block_size * 32, block_size * 40, dtype=torch.long) % vocab_size
    torch.save({"train_tokens": train_tokens, "val_tokens": val_tokens, "vocab_size": vocab_size}, path)


def test_all_candidate_models_produce_expected_shape() -> None:
    config = candidates.ThroughputCandidateConfig(
        cache_path=Path("unused.pt"),
        device="cpu",
        use_amp=False,
        pin_memory=False,
        use_fused_adamw=False,
        compute_val_bpb=False,
        sequence_length=7,
        embedding_dim=16,
        hidden_dim=24,
        memory_dim=16,
        dropout=0.0,
        window_size=3,
    )
    models = candidates._build_candidates(config, vocab_size=64)
    input_ids = torch.randint(0, 64, (2, 7), dtype=torch.long)
    for model in models.values():
        logits = model(input_ids)
        assert logits.shape == (2, 7, 64)


def test_run_throughput_candidates_smoke_cpu() -> None:
    temp_dir = Path(tempfile.mkdtemp(dir=Path.cwd()))
    cache_path = temp_dir / "cache.pt"
    _make_cache(cache_path)
    try:
        config = candidates.ThroughputCandidateConfig(
            cache_path=cache_path,
            device="cpu",
            use_amp=False,
            pin_memory=False,
            use_fused_adamw=False,
            compute_val_bpb=False,
            sequence_length=7,
            train_blocks=16,
            val_blocks=4,
            batch_size=4,
            eval_batch_size=4,
            train_steps=2,
            eval_interval=1,
            embedding_dim=16,
            hidden_dim=24,
            memory_dim=16,
            dropout=0.0,
            window_size=3,
        )
        payload = candidates.run_throughput_candidates(config)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    assert payload["benchmark"] == "language_throughput_candidates"
    assert set(payload["results"]) == {
        "baseline",
        "gru_only",
        "shared_projection",
        "windowed_32",
        "decayed_vote",
    }
    for report in payload["results"].values():
        assert report["parameter_count"] > 0
        assert report["final_val_loss"] >= 0.0
        assert report["train_tok_per_sec"] > 0.0
