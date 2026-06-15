from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from arc_tactic3 import language_partial_untied_muon_probe as probe


def test_split_muon_params_keeps_embeddings_and_heads_on_adamw() -> None:
    cache_dir = Path(tempfile.mkdtemp(dir=Path.cwd()))
    cache_path = cache_dir / "cache.pt"
    try:
        probe.make_synthetic_cache(
            cache_path,
            seed=7,
            vocab_size=64,
            sequence_length=7,
            train_blocks=16,
            val_blocks=4,
        )
        config = probe.PartialUntiedMuonProbeConfig(
            cache_path=cache_path,
            train_blocks=16,
            val_blocks=4,
            sequence_length=7,
            partial_untied_tokens=16,
            recurrent_embedding_dim=16,
            recurrent_hidden_dim=24,
            recurrent_memory_dim=12,
            device="cpu",
            use_fused_adamw=False,
            cache_dataset_on_device=False,
            pin_memory=False,
        )
        train_dataset, _, vocab_size = probe._load_cached_datasets(config)
        token_ids = probe._top_token_ids(train_dataset, count=16, vocab_size=vocab_size)
        model = probe._build_model(config, vocab_size=vocab_size, partial_token_ids=token_ids)
        _, _, split_report = probe._split_muon_params(model)
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)

    assert split_report["muon_parameter_count"] > 0
    assert split_report["adamw_parameter_count"] > 0
    assert "embedding.weight" not in split_report["muon_parameter_names"]
    assert "partial_head.weight" not in split_report["muon_parameter_names"]
    assert "encoder.weight_ih_l0" in split_report["muon_parameter_names"]


def test_partial_untied_muon_probe_smoke_cpu() -> None:
    cache_dir = Path(tempfile.mkdtemp(dir=Path.cwd()))
    cache_path = cache_dir / "cache.pt"
    try:
        probe.make_synthetic_cache(
            cache_path,
            seed=13,
            vocab_size=96,
            sequence_length=7,
            train_blocks=32,
            val_blocks=8,
        )
        payload = probe.run_partial_untied_muon_probe(
            probe.PartialUntiedMuonProbeConfig(
                cache_path=cache_path,
                train_blocks=32,
                val_blocks=8,
                sequence_length=7,
                train_steps=2,
                eval_interval=1,
                partial_untied_tokens=16,
                recurrent_embedding_dim=16,
                recurrent_hidden_dim=24,
                recurrent_memory_dim=12,
                device="cpu",
                use_amp=False,
                use_fused_adamw=False,
                cache_dataset_on_device=False,
                pin_memory=False,
                dropout=0.0,
            )
        )
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)

    assert payload["benchmark"] == "language_partial_untied_muon_probe"
    assert set(payload["results"]) == {"adamw", "muon_adamw"}
    assert payload["results"]["adamw"]["final_val_loss"] >= 0.0
    assert payload["results"]["muon_adamw"]["final_val_loss"] >= 0.0
    assert payload["results"]["muon_adamw"]["optimizer_report"]["muon_parameter_count"] > 0
