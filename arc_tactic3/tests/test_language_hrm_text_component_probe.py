from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest
import torch

from arc_tactic3 import language_hrm_text_component_probe as probe


def _make_cache(path: Path, *, vocab_size: int = 128, sequence_length: int = 7) -> None:
    block_size = sequence_length + 1
    train_tokens = torch.arange(block_size * 32, dtype=torch.long) % vocab_size
    val_tokens = torch.arange(block_size * 32, block_size * 40, dtype=torch.long) % vocab_size
    torch.save({"train_tokens": train_tokens, "val_tokens": val_tokens, "vocab_size": vocab_size}, path)


def test_hrm_text_component_model_produces_expected_shape() -> None:
    token_ids = torch.arange(16, dtype=torch.long)
    magic_norm_model = probe.MagicNormPartialUntiedAssociativeLM(
        vocab_size=64,
        embedding_dim=16,
        hidden_dim=24,
        memory_dim=12,
        dropout=0.0,
        max_length=7,
        untied_token_ids=token_ids,
    )
    hrm_model = probe.HRMTextComponentAssociativeLM(
        vocab_size=64,
        embedding_dim=16,
        hidden_dim=24,
        memory_dim=12,
        dropout=0.0,
        max_length=7,
        untied_token_ids=token_ids,
        latent_refine_steps=2,
        high_update_interval=2,
    )
    input_ids = torch.randint(0, 64, (2, 7), dtype=torch.long)
    for model in (magic_norm_model, hrm_model):
        logits = model(input_ids)
        assert logits.shape == (2, 7, 64)
        assert torch.isfinite(logits).all()


def test_hrm_text_component_rejects_invalid_refinement_settings() -> None:
    token_ids = torch.arange(8, dtype=torch.long)
    with pytest.raises(ValueError, match="latent_refine_steps"):
        probe.HRMTextComponentAssociativeLM(
            vocab_size=32,
            embedding_dim=8,
            hidden_dim=12,
            memory_dim=8,
            dropout=0.0,
            max_length=7,
            untied_token_ids=token_ids,
            latent_refine_steps=0,
            high_update_interval=1,
        )
    with pytest.raises(ValueError, match="high_update_interval"):
        probe.HRMTextComponentAssociativeLM(
            vocab_size=32,
            embedding_dim=8,
            hidden_dim=12,
            memory_dim=8,
            dropout=0.0,
            max_length=7,
            untied_token_ids=token_ids,
            latent_refine_steps=1,
            high_update_interval=0,
        )


def test_hrm_text_component_probe_smoke_cpu() -> None:
    temp_dir = Path(tempfile.mkdtemp(dir=Path.cwd()))
    cache_path = temp_dir / "cache.pt"
    _make_cache(cache_path)
    try:
        payload = probe.run_hrm_text_component_probe(
            probe.HRMTextComponentProbeConfig(
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
                recurrent_memory_dim=12,
                partial_untied_tokens=16,
                latent_refine_steps=2,
                high_update_interval=2,
                dropout=0.0,
            )
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    assert payload["benchmark"] == "language_hrm_text_component_probe"
    assert payload["compare_target"] == "partial_untied"
    assert payload["architecture_summary"]["status"] == "probe_only"
    assert set(payload["results"]) == {
        "partial_untied",
        "magic_norm_partial_untied",
        "hrm_text_component_probe",
    }
    for report in payload["results"].values():
        assert report["parameter_count"] > 0
        assert report["final_val_loss"] >= 0.0
        assert report["train_tok_per_sec"] > 0.0
        assert report["pure_train_tok_per_sec"] > 0.0
