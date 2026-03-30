from __future__ import annotations

from pathlib import Path

import torch

from arc_tactic3.language_fastlearn_benchmark import count_parameters
from arc_tactic3.language_partial_untied_cluster import (
    PartialUntiedClusterConfig,
    _DEFAULT_PROMPTS,
    _apply_named_preset,
    _compute_resume_aware_rates,
    _default_cache_path,
    _estimate_partial_untied_parameter_count,
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


def test_hardware_profile_override_resolves_h100_without_local_gpu() -> None:
    config = PartialUntiedClusterConfig(
        output_dir=Path("tmp"),
        device="cuda",
        hardware_profile_override="h100",
        amp_dtype="auto",
    )
    resolved, profile = resolve_cluster_config(config)
    assert profile.name == "h100"
    assert resolved.batch_size == 128
    assert resolved.eval_batch_size == 192
    assert resolved.amp_dtype == "bf16"
    assert resolved.use_amp is True


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


def test_h100_100m_1b_preset_targets_about_100m_params() -> None:
    base = PartialUntiedClusterConfig(output_dir=Path("runs/test"), device="cuda")
    config = _apply_named_preset(base, "h100_100m_1b")
    assert config.total_tokens == 1_000_000_000
    assert config.train_tokens == 980_000_000
    assert config.val_tokens == 20_000_000
    assert config.batch_size == 192
    assert config.eval_batch_size == 256
    assert config.cache_dataset_on_device is True
    assert 99_000_000 <= _estimate_partial_untied_parameter_count(config) <= 101_000_000


def test_large_cache_budget_is_not_cached_on_device() -> None:
    config = PartialUntiedClusterConfig(
        output_dir=Path("runs/test"),
        device="cuda",
        hardware_profile_override="h100",
        total_tokens=5_000_000_000,
        train_tokens=4_900_000_000,
        val_tokens=100_000_000,
        cache_dataset_on_device=True,
    )
    resolved, _ = resolve_cluster_config(config)
    assert resolved.cache_dataset_on_device is False


def test_compute_resume_aware_rates_uses_resume_delta_without_timing_offsets() -> None:
    train_tok_per_sec, pure_train_tok_per_sec, wall_time_seconds, pure_train_time_seconds = _compute_resume_aware_rates(
        tokens_seen=650,
        resume_tokens_seen=500,
        elapsed_since_resume=3.0,
        step_times=[1.0, 1.0],
    )
    assert train_tok_per_sec == 50.0
    assert pure_train_tok_per_sec == 75.0
    assert wall_time_seconds == 3.0
    assert pure_train_time_seconds == 2.0


def test_compute_resume_aware_rates_preserves_total_timing_when_offsets_exist() -> None:
    train_tok_per_sec, pure_train_tok_per_sec, wall_time_seconds, pure_train_time_seconds = _compute_resume_aware_rates(
        tokens_seen=650,
        resume_tokens_seen=500,
        elapsed_since_resume=3.0,
        step_times=[1.0, 1.0],
        resume_wall_time_seconds=5.0,
        resume_pure_train_time_seconds=4.0,
    )
    assert train_tok_per_sec == 650 / 8.0
    assert pure_train_tok_per_sec == 650 / 6.0
    assert wall_time_seconds == 8.0
    assert pure_train_time_seconds == 6.0
