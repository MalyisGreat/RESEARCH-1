from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import torch

from arc_tactic3 import language_compute_efficiency_matrix as matrix


def _make_cache(path: Path, *, vocab_size: int = 128, sequence_length: int = 7) -> None:
    block_size = sequence_length + 1
    train_tokens = torch.arange(block_size * 64, dtype=torch.long) % vocab_size
    val_tokens = torch.arange(block_size * 64, block_size * 72, dtype=torch.long) % vocab_size
    torch.save({"train_tokens": train_tokens, "val_tokens": val_tokens, "vocab_size": vocab_size}, path)


def test_run_efficiency_matrix_smoke_cpu() -> None:
    temp_dir = Path(tempfile.mkdtemp(dir=Path.cwd()))
    cache_path = temp_dir / "cache.pt"
    _make_cache(cache_path)
    try:
        config = matrix.EfficiencyMatrixConfig(
            cache_path=cache_path,
            train_blocks=32,
            val_blocks=8,
            sequence_length=7,
            train_steps=2,
            eval_interval=1,
            seed=13,
            device="cpu",
            partial_untied_tokens=32,
            recurrent_embedding_dim=16,
            recurrent_hidden_dim=24,
            recurrent_memory_dim=16,
        )
        payload = matrix.run_efficiency_matrix(config)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    assert payload["benchmark"] == "language_compute_efficiency_matrix"
    assert "default" in payload["results"]
    assert set(payload["results"]["default"]) == {"recurrent_champion", "partial_untied"}
    for variant_payload in payload["results"].values():
        for report in variant_payload.values():
            assert report["parameter_count"] > 0
            assert report["final_val_loss"] >= 0.0
            assert report["train_tok_per_sec"] > 0.0
