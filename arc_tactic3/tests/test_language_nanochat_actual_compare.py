from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import torch

from arc_tactic3 import language_nanochat_actual_compare as compare


def _make_cache(path: Path, *, vocab_size: int = 128, sequence_length: int = 7) -> None:
    block_size = sequence_length + 1
    train_tokens = torch.arange(block_size * 32, dtype=torch.long) % vocab_size
    val_tokens = torch.arange(block_size * 32, block_size * 40, dtype=torch.long) % vocab_size
    torch.save({"train_tokens": train_tokens, "val_tokens": val_tokens, "vocab_size": vocab_size}, path)


def test_all_models_produce_expected_shape() -> None:
    config = compare.NanochatActualCompareConfig(
        cache_path=Path("unused.pt"),
        device="cpu",
        use_amp=False,
        pin_memory=False,
        use_fused_adamw=False,
        compute_val_bpb=False,
        sequence_length=7,
        recurrent_embedding_dim=16,
        recurrent_hidden_dim=24,
        recurrent_memory_dim=16,
        window_size=3,
        nano_n_layer=2,
        nano_n_head=4,
        nano_n_kv_head=4,
        nano_n_embd=16,
        nano_window_pattern="SL",
    )
    models = compare._build_models(config, vocab_size=64)
    input_ids = torch.randint(0, 64, (2, 7), dtype=torch.long)
    assert set(models) == {"recurrent_baseline", "recurrent_champion", "gru_only", "windowed_32", "nanochat_small"}
    for model in models.values():
        logits = model(input_ids)
        assert logits.shape == (2, 7, 64)
        assert torch.isfinite(logits).all()


def test_run_nanochat_actual_compare_smoke_cpu() -> None:
    temp_dir = Path(tempfile.mkdtemp(dir=Path.cwd()))
    cache_path = temp_dir / "cache.pt"
    _make_cache(cache_path)
    try:
        config = compare.NanochatActualCompareConfig(
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
            recurrent_embedding_dim=16,
            recurrent_hidden_dim=24,
            recurrent_memory_dim=16,
            window_size=3,
            nano_n_layer=2,
            nano_n_head=4,
            nano_n_kv_head=4,
            nano_n_embd=16,
            nano_window_pattern="L",
        )
        payload = compare.run_nanochat_actual_compare(config)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    assert payload["benchmark"] == "language_nanochat_actual_compare"
    assert payload["fairness"]["paired_batch_schedule"] is True
    assert set(payload["results"]) == {"recurrent_baseline", "recurrent_champion", "gru_only", "windowed_32", "nanochat_small"}
    for report in payload["results"].values():
        assert report["parameter_count"] > 0
        assert report["final_val_loss"] >= 0.0
        assert report["train_tokens_seen"] > 0
        assert report["train_tok_per_sec"] > 0.0
        assert report["paired_train_batches"] is True
        assert report["reseed_per_model"] is True
