from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import torch

from arc_tactic3 import language_candidate_learned_compressor as candidate


def _make_cache(path: Path, *, vocab_size: int = 128, sequence_length: int = 7) -> None:
    block_size = sequence_length + 1
    train_tokens = torch.arange(block_size * 32, dtype=torch.long) % vocab_size
    val_tokens = torch.arange(block_size * 32, block_size * 40, dtype=torch.long) % vocab_size
    torch.save({"train_tokens": train_tokens, "val_tokens": val_tokens, "vocab_size": vocab_size}, path)


def test_learned_compressor_produces_valid_write_weights() -> None:
    model = candidate.LearnedChunkWriterPartialLM(
        vocab_size=64,
        embedding_dim=16,
        hidden_dim=24,
        memory_dim=16,
        dropout=0.0,
        max_length=7,
        chunk_size=4,
        untied_token_ids=torch.arange(0, 16, dtype=torch.long),
    )
    input_ids = torch.randint(0, 64, (2, 7), dtype=torch.long)
    logits = model(input_ids)
    assert logits.shape == (2, 7, 64)
    assert torch.isfinite(logits).all()

    debug = model.inspect_chunk_writer(input_ids)
    write_weights = debug["write_weights"]
    retain_gates = debug["retain_gates"]
    valid_mask = debug["valid_mask"]

    assert write_weights.shape == (2, 2, 4)
    assert retain_gates.shape == (2, 2, 1)
    assert torch.all((retain_gates >= 0.0) & (retain_gates <= 1.0))
    invalid_weights = write_weights.masked_select(~valid_mask.expand_as(write_weights))
    assert torch.allclose(invalid_weights, torch.zeros_like(invalid_weights), atol=1e-6)
    weight_sums = (write_weights * valid_mask.to(write_weights.dtype)).sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)


def test_run_learned_compressor_benchmark_smoke_cpu() -> None:
    temp_dir = Path(tempfile.mkdtemp(dir=Path.cwd()))
    cache_path = temp_dir / "cache.pt"
    _make_cache(cache_path)
    try:
        payload = candidate.run_learned_compressor_benchmark(
            candidate.LearnedCompressorCandidateConfig(
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
                partial_untied_tokens=32,
                recurrent_embedding_dim=16,
                recurrent_hidden_dim=24,
                recurrent_memory_dim=16,
                chunk_size=4,
            )
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    assert payload["benchmark"] == "language_candidate_learned_compressor"
    assert set(payload["results"]) == {
        "partial_untied",
        "chunk_mean_token_memory",
        "learned_chunk_writer",
    }
    for report in payload["results"].values():
        assert report["parameter_count"] > 0
        assert report["final_val_loss"] >= 0.0
        assert report["train_tok_per_sec"] > 0.0
        assert report["pure_train_tok_per_sec"] > 0.0
