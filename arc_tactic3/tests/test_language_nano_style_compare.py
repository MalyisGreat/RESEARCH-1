from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import torch

from arc_tactic3 import language_nano_style_compare as compare
from arc_tactic3.language_realtext_microbench import build_models, count_parameters


def test_default_target_models_are_near_five_million_parameters() -> None:
    config = compare.NanoStyleCompareConfig(
        cache_path=Path("unused.pt"),
        device="cpu",
        use_amp=False,
        pin_memory=False,
        use_fused_adamw=False,
        compute_val_bpb=False,
    )
    shared = compare._build_shared_config(config)
    models = build_models(shared, vocab_size=50_257)
    recurrent_params = count_parameters(models["associative_recurrent"])
    gpt_params = count_parameters(models["gpt2_like"])
    assert 4_800_000 <= recurrent_params <= 5_300_000
    assert 4_800_000 <= gpt_params <= 5_300_000


def test_run_nano_style_compare_on_cached_tokens_cpu() -> None:
    temp_dir = Path(tempfile.mkdtemp(dir=Path.cwd()))
    cache_path = temp_dir / "fineweb_cache.pt"
    block_size = 8
    train_tokens = torch.arange(0, block_size * 32, dtype=torch.long) % 256
    val_tokens = torch.arange(block_size * 32, block_size * 40, dtype=torch.long) % 256
    torch.save({"train_tokens": train_tokens, "val_tokens": val_tokens, "vocab_size": 256}, cache_path)
    try:
        config = compare.NanoStyleCompareConfig(
            cache_path=cache_path,
            device="cpu",
            use_amp=False,
            pin_memory=False,
            use_fused_adamw=False,
            compute_val_bpb=False,
            sequence_length=7,
            train_blocks=16,
            val_blocks=4,
            train_steps=2,
            eval_interval=1,
            batch_size=4,
            eval_batch_size=4,
            recurrent_embedding_dim=16,
            recurrent_hidden_dim=24,
            recurrent_memory_dim=16,
            gpt_d_model=16,
            gpt_heads=4,
            gpt_layers=1,
            gpt_ff_dim=80,
        )
        payload = compare.run_nano_style_compare(config)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    assert payload["benchmark"] == "language_nano_style_compare"
    assert payload["dataset"]["train_blocks"] == 16
    assert payload["dataset"]["val_blocks"] == 4
    assert payload["device"]["precision_mode"] == "fp32"
    for report in payload["models"].values():
        assert report["parameter_count"] > 0
        assert report["final_val_loss"] >= 0.0
        assert report["val_bits_per_token"] >= 0.0
        assert report["val_bpb"] is None
        assert report["train_tokens_seen"] > 0
        assert report["train_tok_per_sec"] > 0.0
        assert report["train_mfu_estimate"] is None
