from __future__ import annotations

from pathlib import Path

from arc_tactic3.language_nanochat_cluster import (
    NanochatClusterConfig,
    _DEFAULT_PROMPTS,
    _apply_named_preset,
    _estimate_nanochat_parameter_count,
    resolve_cluster_config,
)


def test_resolve_nanochat_cluster_config_disables_cuda_features_on_cpu() -> None:
    config = NanochatClusterConfig(output_dir=Path("tmp"), device="cpu", amp_dtype="auto")
    resolved, profile = resolve_cluster_config(config)
    assert profile.name == "cpu"
    assert resolved.use_amp is False
    assert resolved.pin_memory is False
    assert resolved.use_fused_adamw is False
    assert resolved.cache_dataset_on_device is False
    assert resolved.amp_dtype == "fp32"


def test_nanochat_h100_override_resolves_without_local_gpu() -> None:
    config = NanochatClusterConfig(
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


def test_h100_100m_1b_nanochat_preset_targets_about_100m_params() -> None:
    base = NanochatClusterConfig(output_dir=Path("runs/test"), device="cuda")
    config = _apply_named_preset(base, "h100_100m_1b")
    assert config.total_tokens == 1_000_000_000
    assert config.train_tokens == 980_000_000
    assert config.val_tokens == 20_000_000
    assert config.batch_size == 192
    assert config.eval_batch_size == 256
    assert config.learning_rate == 6e-4
    assert config.warmup_steps == 1024
    assert config.lr_schedule == "cosine"
    assert config.min_lr_scale == 0.1
    assert config.cache_dataset_on_device is True
    assert _DEFAULT_PROMPTS[0] == "The capital of France is"
    assert 95_000_000 <= _estimate_nanochat_parameter_count(config) <= 105_000_000
