from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import torch

from arc_tactic3 import language_candidate_slot_memory as slots


def _make_cache(path: Path, *, vocab_size: int = 128, sequence_length: int = 7) -> None:
    block_size = sequence_length + 1
    train_tokens = torch.arange(block_size * 32, dtype=torch.long) % vocab_size
    val_tokens = torch.arange(block_size * 32, block_size * 40, dtype=torch.long) % vocab_size
    torch.save({"train_tokens": train_tokens, "val_tokens": val_tokens, "vocab_size": vocab_size}, path)


def test_slot_memory_candidate_produces_expected_shape() -> None:
    model = slots.SlotMemoryRecurrentLM(
        vocab_size=64,
        embedding_dim=16,
        hidden_dim=24,
        dropout=0.0,
        slot_count=4,
        slot_dim=12,
    )
    input_ids = torch.randint(0, 64, (2, 7), dtype=torch.long)
    logits = model(input_ids)
    assert logits.shape == (2, 7, 64)
    assert torch.isfinite(logits).all()


def test_run_slot_memory_compare_smoke_cpu() -> None:
    temp_dir = Path(tempfile.mkdtemp(dir=Path.cwd()))
    cache_path = temp_dir / "cache.pt"
    _make_cache(cache_path)
    try:
        payload = slots.run_slot_memory_compare(
            slots.SlotMemoryCompareConfig(
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
                recurrent_memory_dim=12,
                slot_count=4,
                slot_dim=12,
                use_amp=False,
                pin_memory=False,
                use_fused_adamw=False,
            )
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    assert payload["benchmark"] == "language_candidate_slot_memory"
    assert set(payload["results"]) == {"partial_untied", "slot_memory_candidate"}
    for report in payload["results"].values():
        assert report["parameter_count"] > 0
        assert report["final_val_loss"] >= 0.0
        assert report["train_tok_per_sec"] > 0.0
        assert report["pure_train_tok_per_sec"] > 0.0
