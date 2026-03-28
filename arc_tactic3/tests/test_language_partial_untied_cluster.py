from __future__ import annotations

from pathlib import Path

import torch

from arc_tactic3.language_fastlearn_benchmark import count_parameters
from arc_tactic3.language_partial_untied_cluster import (
    PartialUntiedClusterConfig,
    _DEFAULT_PROMPTS,
    _default_cache_path,
    _hardware_profile_from_name,
    resolve_cluster_config,
)
from arc_tactic3.language_recurrent_nano_tricks import PartialUntiedAssociativeLM


def test_hardware_profile_defaults_for_h100() -> None:
    profile = _hardware_profile_from_name("NVIDIA H100 PCIe", 80.0)
    assert profile.name == "h100"
    assert profile.amp_dtype == "bf16"
    assert profile.batch_size >= 128
    assert profile.cache_dataset_on_device is True


def test_resolve_cluster_config_disables_cuda_features_on_cpu() -> None:
    config = PartialUntiedClusterConfig(output_dir=Path("tmp"), device="cpu", amp_dtype="auto")
    resolved, profile = resolve_cluster_config(config)
    assert profile.name == "cpu"
    assert resolved.use_amp is False
    assert resolved.pin_memory is False
    assert resolved.use_fused_adamw is False
    assert resolved.cache_dataset_on_device is False
    assert resolved.amp_dtype == "fp32"


def test_default_cache_path_contains_budget_and_tokenizer() -> None:
    config = PartialUntiedClusterConfig(
        output_dir=Path("runs/test"),
        train_tokens=98_000_000,
        val_tokens=2_000_000,
        sequence_length=127,
        tokenizer_name="gpt2",
    )
    cache_path = _default_cache_path(config)
    assert "finewebedu_train98000000_val2000000_seq127_gpt2.pt" in str(cache_path)


def test_default_partial_untied_cluster_size_is_about_20m_params() -> None:
    untied_token_ids = torch.arange(1024)
    model = PartialUntiedAssociativeLM(
        vocab_size=50_257,
        embedding_dim=320,
        hidden_dim=640,
        memory_dim=320,
        dropout=0.1,
        max_length=127,
        untied_token_ids=untied_token_ids,
    )
    assert 19_000_000 <= count_parameters(model) <= 21_000_000
    assert _DEFAULT_PROMPTS[0] == "The capital of France is"
