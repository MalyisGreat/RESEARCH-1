from __future__ import annotations

from pathlib import Path

from arc_tactic3.language_cluster_checkpoint_benchmark import (
    _coerce_config,
    _resolve_checkpoint_path,
)
from arc_tactic3.language_nanochat_cluster import NanochatClusterConfig
from arc_tactic3.language_partial_untied_cluster import PartialUntiedClusterConfig


def test_coerce_partial_cluster_config_restores_paths() -> None:
    config = _coerce_config(
        PartialUntiedClusterConfig,
        {
            "output_dir": "runs/partial",
            "cache_path": "cache/tokens.pt",
            "device": "cpu",
            "learning_rate": 1e-3,
            "unused_key": "ignored",
        },
    )
    assert isinstance(config.output_dir, Path)
    assert isinstance(config.cache_path, Path)
    assert config.output_dir == Path("runs/partial")
    assert config.cache_path == Path("cache/tokens.pt")
    assert config.learning_rate == 1e-3


def test_coerce_nanochat_cluster_config_uses_defaults_for_missing_fields() -> None:
    config = _coerce_config(
        NanochatClusterConfig,
        {
            "output_dir": "runs/nano",
            "device": "cpu",
        },
    )
    assert config.output_dir == Path("runs/nano")
    assert config.cache_path is None
    assert config.learning_rate == 2e-3


def test_resolve_checkpoint_path_prefers_explicit_path() -> None:
    explicit = Path("checkpoints/manual.pt")
    resolved = _resolve_checkpoint_path(
        run_dir=Path("runs/partial"),
        checkpoint_path=explicit,
        checkpoint_name="best.pt",
    )
    assert resolved == explicit


def test_resolve_checkpoint_path_from_run_dir() -> None:
    resolved = _resolve_checkpoint_path(
        run_dir=Path("runs/partial"),
        checkpoint_path=None,
        checkpoint_name="best.pt",
    )
    assert resolved == Path("runs/partial/checkpoints/best.pt")
