from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from arc_tactic3 import language_longseq_replay_probe as probe
from arc_tactic3.language_fastlearn_benchmark import count_parameters


def test_windowed_replay_partial_untied_outputs_finite_logits() -> None:
    input_ids = torch.randint(0, 64, (2, 31))
    models = [
        probe.WindowedReplayPartialUntiedLM(
            vocab_size=64,
            embedding_dim=16,
            hidden_dim=24,
            memory_dim=12,
            dropout=0.0,
            max_length=31,
            untied_token_ids=torch.arange(16),
            window_size=8,
        ),
        probe.DetachedReplayPartialUntiedLM(
            vocab_size=64,
            embedding_dim=16,
            hidden_dim=24,
            memory_dim=12,
            dropout=0.0,
            max_length=31,
            untied_token_ids=torch.arange(16),
        ),
    ]
    for model in models:
        logits = model(input_ids)
        assert logits.shape == (2, 31, 64)
        assert torch.isfinite(logits).all()


def test_longseq_cache_config_uses_block_counts() -> None:
    config = probe.LongSeqReplayProbeConfig(
        output_dir=Path("artifacts/test-longseq"),
        train_blocks=17,
        val_blocks=5,
        sequence_length=31,
        batch_size=4,
        eval_batch_size=4,
    )
    cache_config = probe._cache_config(config)
    assert cache_config.train_tokens == 17 * 32
    assert cache_config.val_tokens == 5 * 32
    assert cache_config.total_tokens == 22 * 32
    assert cache_config.sequence_length == 31


def test_comparison_reports_median_step_speed_ratio() -> None:
    baseline = {
        "final_val_loss": 8.0,
        "pure_train_tok_per_sec": 10.0,
        "step_time_median_ms": 200.0,
    }
    candidate = {
        "final_val_loss": 7.5,
        "pure_train_tok_per_sec": 25.0,
        "step_time_median_ms": 40.0,
    }
    comparison = probe._comparison_vs_baseline(candidate, baseline)
    assert comparison["loss_delta_vs_baseline"] == -0.5
    assert comparison["pure_speed_ratio_vs_baseline"] == 2.5
    assert comparison["median_step_speed_ratio_vs_baseline"] == 5.0
    assert comparison["baseline_step_time_median_ms"] == 200.0
    assert comparison["candidate_step_time_median_ms"] == 40.0


def test_cosine_learning_rate_schedule_warms_up_and_decays() -> None:
    config = probe.LongSeqReplayProbeConfig(
        train_steps=10,
        learning_rate=1.0,
        lr_schedule="cosine",
        warmup_steps=2,
        min_learning_rate=0.1,
    )
    assert probe._scheduled_learning_rate(config, 0) == 0.0
    assert probe._scheduled_learning_rate(config, 1) == 0.5
    assert probe._scheduled_learning_rate(config, 2) == 1.0
    assert probe._scheduled_learning_rate(config, 10) == pytest.approx(0.1)
    assert 0.1 < probe._scheduled_learning_rate(config, 6) < 1.0


def test_default_longseq_model_size_is_about_20m_params() -> None:
    config = probe.LongSeqReplayProbeConfig(sequence_length=63)
    model = probe._build_model(
        "partial_untied",
        config,
        vocab_size=50_257,
        partial_token_ids=torch.arange(config.partial_token_count),
    )
    assert 19_000_000 <= count_parameters(model) <= 21_000_000


def test_train_variant_can_skip_initial_eval_and_still_final_eval() -> None:
    config = probe.LongSeqReplayProbeConfig(
        sequence_length=5,
        batch_size=2,
        eval_batch_size=2,
        train_steps=1,
        eval_interval=999,
        initial_eval=False,
        device="cpu",
        use_amp=False,
        pin_memory=False,
        use_fused_adamw=False,
        cache_dataset_on_device=False,
        embedding_dim=8,
        hidden_dim=12,
        memory_dim=8,
        partial_token_count=4,
        dropout=0.0,
    )
    train_blocks = torch.randint(0, 32, (2, 6))
    val_blocks = torch.randint(0, 32, (2, 6))
    train_dataset = probe.TokenBlockDataset(train_blocks[:, :-1], train_blocks[:, 1:])
    val_dataset = probe.TokenBlockDataset(val_blocks[:, :-1], val_blocks[:, 1:])
    report = probe._train_variant(
        "partial_untied",
        train_dataset,
        val_dataset,
        vocab_size=32,
        partial_token_ids=torch.arange(config.partial_token_count),
        sampled_candidate_ids=torch.arange(config.sampled_vocab_size),
        batch_schedule=[torch.tensor([0, 1])],
        config=config,
    )
    assert math.isnan(report["initial_val_loss"])
    assert len(report["history"]) == 1
    assert report["history"][0]["step"] == 1.0
    assert torch.isfinite(torch.tensor(report["final_val_loss"]))


def test_train_variant_supports_sampled_vocab_validation() -> None:
    config = probe.LongSeqReplayProbeConfig(
        sequence_length=5,
        batch_size=2,
        eval_batch_size=2,
        train_steps=1,
        eval_interval=999,
        initial_eval=False,
        eval_loss_mode="sampled_vocab",
        device="cpu",
        use_amp=False,
        pin_memory=False,
        use_fused_adamw=False,
        cache_dataset_on_device=False,
        conv_embedding_dim=8,
        conv_layers=1,
        conv_rank=4,
        conv_kernel_size=3,
        sampled_vocab_size=8,
        dropout=0.0,
    )
    train_blocks = torch.randint(0, 32, (2, 6))
    val_blocks = torch.randint(0, 32, (2, 6))
    train_dataset = probe.TokenBlockDataset(train_blocks[:, :-1], train_blocks[:, 1:])
    val_dataset = probe.TokenBlockDataset(val_blocks[:, :-1], val_blocks[:, 1:])
    report = probe._train_variant(
        "causal_conv_mixer_sampled_vocab",
        train_dataset,
        val_dataset,
        vocab_size=32,
        partial_token_ids=torch.arange(config.partial_token_count),
        sampled_candidate_ids=torch.arange(config.sampled_vocab_size),
        batch_schedule=[torch.tensor([0, 1])],
        config=config,
    )
    assert report["validation_loss_mode"] == "sampled_vocab"
    assert torch.isfinite(torch.tensor(report["final_val_loss"]))
    assert report["sampled_eval_candidate_size_mean"] is not None


def test_train_variant_supports_sampled_vocab_anchor() -> None:
    config = probe.LongSeqReplayProbeConfig(
        sequence_length=5,
        batch_size=2,
        eval_batch_size=2,
        train_steps=1,
        eval_interval=999,
        initial_eval=False,
        device="cpu",
        use_amp=False,
        pin_memory=False,
        use_fused_adamw=False,
        cache_dataset_on_device=False,
        conv_embedding_dim=8,
        conv_layers=1,
        conv_rank=4,
        conv_kernel_size=3,
        sampled_vocab_size=8,
        full_loss_token_stride=2,
        dropout=0.0,
    )
    train_blocks = torch.randint(0, 32, (2, 6))
    val_blocks = torch.randint(0, 32, (2, 6))
    train_dataset = probe.TokenBlockDataset(train_blocks[:, :-1], train_blocks[:, 1:])
    val_dataset = probe.TokenBlockDataset(val_blocks[:, :-1], val_blocks[:, 1:])
    report = probe._train_variant(
        "causal_conv_mixer_sampled_vocab_anchor",
        train_dataset,
        val_dataset,
        vocab_size=32,
        partial_token_ids=torch.arange(config.partial_token_count),
        sampled_candidate_ids=torch.arange(config.sampled_vocab_size),
        batch_schedule=[torch.tensor([0, 1])],
        config=config,
    )
    assert report["validation_loss_mode"] == "full"
    assert torch.isfinite(torch.tensor(report["final_val_loss"]))
    assert report["sampled_candidate_size_mean"] is not None


def test_train_variant_writes_state_and_result_artifacts(tmp_path: Path) -> None:
    config = probe.LongSeqReplayProbeConfig(
        output_dir=tmp_path,
        sequence_length=5,
        batch_size=2,
        eval_batch_size=2,
        train_steps=1,
        eval_interval=999,
        initial_eval=False,
        device="cpu",
        use_amp=False,
        pin_memory=False,
        use_fused_adamw=False,
        cache_dataset_on_device=False,
        embedding_dim=8,
        hidden_dim=12,
        memory_dim=8,
        partial_token_count=4,
        dropout=0.0,
    )
    train_blocks = torch.randint(0, 32, (2, 6))
    val_blocks = torch.randint(0, 32, (2, 6))
    train_dataset = probe.TokenBlockDataset(train_blocks[:, :-1], train_blocks[:, 1:])
    val_dataset = probe.TokenBlockDataset(val_blocks[:, :-1], val_blocks[:, 1:])
    artifact_dir = tmp_path / "variant_results"
    report = probe._train_variant(
        "partial_untied",
        train_dataset,
        val_dataset,
        vocab_size=32,
        partial_token_ids=torch.arange(config.partial_token_count),
        sampled_candidate_ids=torch.arange(config.sampled_vocab_size),
        batch_schedule=[torch.tensor([0, 1])],
        config=config,
        variant_artifact_dir=artifact_dir,
    )
    state = json.loads((artifact_dir / "partial_untied.state.json").read_text(encoding="utf-8"))
    result = json.loads((artifact_dir / "partial_untied.json").read_text(encoding="utf-8"))
    assert state["status"] == "completed"
    assert state["final_val_loss"] == report["final_val_loss"]
    assert result["report"]["final_val_loss"] == report["final_val_loss"]


def test_train_variant_marks_slow_step_abort(tmp_path: Path) -> None:
    config = probe.LongSeqReplayProbeConfig(
        output_dir=tmp_path,
        sequence_length=5,
        batch_size=2,
        eval_batch_size=2,
        train_steps=1,
        eval_interval=999,
        initial_eval=False,
        max_step_seconds=0.0,
        device="cpu",
        use_amp=False,
        pin_memory=False,
        use_fused_adamw=False,
        cache_dataset_on_device=False,
        embedding_dim=8,
        hidden_dim=12,
        memory_dim=8,
        partial_token_count=4,
        dropout=0.0,
    )
    train_blocks = torch.randint(0, 32, (2, 6))
    val_blocks = torch.randint(0, 32, (2, 6))
    train_dataset = probe.TokenBlockDataset(train_blocks[:, :-1], train_blocks[:, 1:])
    val_dataset = probe.TokenBlockDataset(val_blocks[:, :-1], val_blocks[:, 1:])
    artifact_dir = tmp_path / "variant_results"
    with pytest.raises(RuntimeError, match="exceeding max_step_seconds"):
        probe._train_variant(
            "partial_untied",
            train_dataset,
            val_dataset,
            vocab_size=32,
            partial_token_ids=torch.arange(config.partial_token_count),
            sampled_candidate_ids=torch.arange(config.sampled_vocab_size),
            batch_schedule=[torch.tensor([0, 1])],
            config=config,
            variant_artifact_dir=artifact_dir,
        )
    state = json.loads((artifact_dir / "partial_untied.state.json").read_text(encoding="utf-8"))
    assert state["status"] == "aborted_slow_step"


def test_train_variant_can_resume_from_checkpoint_after_slow_abort(tmp_path: Path) -> None:
    base_kwargs = dict(
        output_dir=tmp_path,
        sequence_length=5,
        batch_size=2,
        eval_batch_size=2,
        train_steps=2,
        eval_interval=999,
        initial_eval=False,
        variant_checkpoint_interval=1,
        device="cpu",
        use_amp=False,
        pin_memory=False,
        use_fused_adamw=False,
        cache_dataset_on_device=False,
        embedding_dim=8,
        hidden_dim=12,
        memory_dim=8,
        partial_token_count=4,
        dropout=0.0,
    )
    abort_config = probe.LongSeqReplayProbeConfig(max_step_seconds=0.0, **base_kwargs)
    train_blocks = torch.randint(0, 32, (2, 6))
    val_blocks = torch.randint(0, 32, (2, 6))
    train_dataset = probe.TokenBlockDataset(train_blocks[:, :-1], train_blocks[:, 1:])
    val_dataset = probe.TokenBlockDataset(val_blocks[:, :-1], val_blocks[:, 1:])
    artifact_dir = tmp_path / "variant_results"
    schedule = [torch.tensor([0, 1]), torch.tensor([0, 1])]
    with pytest.raises(RuntimeError, match="exceeding max_step_seconds"):
        probe._train_variant(
            "partial_untied",
            train_dataset,
            val_dataset,
            vocab_size=32,
            partial_token_ids=torch.arange(abort_config.partial_token_count),
            sampled_candidate_ids=torch.arange(abort_config.sampled_vocab_size),
            batch_schedule=schedule,
            config=abort_config,
            variant_artifact_dir=artifact_dir,
        )
    assert (artifact_dir / "partial_untied.checkpoint.pt").exists()

    resume_config = probe.LongSeqReplayProbeConfig(
        resume_variant_checkpoints=True,
        **base_kwargs,
    )
    report = probe._train_variant(
        "partial_untied",
        train_dataset,
        val_dataset,
        vocab_size=32,
        partial_token_ids=torch.arange(resume_config.partial_token_count),
        sampled_candidate_ids=torch.arange(resume_config.sampled_vocab_size),
        batch_schedule=schedule,
        config=resume_config,
        variant_artifact_dir=artifact_dir,
    )
    assert report["train_tokens_seen"] == 20
    assert report["history"][-1]["step"] == 2.0
    assert torch.isfinite(torch.tensor(report["final_val_loss"]))


def test_train_variant_marks_slow_eval_abort(tmp_path: Path) -> None:
    config = probe.LongSeqReplayProbeConfig(
        output_dir=tmp_path,
        sequence_length=5,
        batch_size=2,
        eval_batch_size=2,
        train_steps=1,
        eval_interval=999,
        initial_eval=True,
        max_eval_seconds=0.0,
        device="cpu",
        use_amp=False,
        pin_memory=False,
        use_fused_adamw=False,
        cache_dataset_on_device=False,
        embedding_dim=8,
        hidden_dim=12,
        memory_dim=8,
        partial_token_count=4,
        dropout=0.0,
    )
    train_blocks = torch.randint(0, 32, (2, 6))
    val_blocks = torch.randint(0, 32, (2, 6))
    train_dataset = probe.TokenBlockDataset(train_blocks[:, :-1], train_blocks[:, 1:])
    val_dataset = probe.TokenBlockDataset(val_blocks[:, :-1], val_blocks[:, 1:])
    artifact_dir = tmp_path / "variant_results"
    with pytest.raises(RuntimeError, match="exceeding max_eval_seconds"):
        probe._train_variant(
            "partial_untied",
            train_dataset,
            val_dataset,
            vocab_size=32,
            partial_token_ids=torch.arange(config.partial_token_count),
            sampled_candidate_ids=torch.arange(config.sampled_vocab_size),
            batch_schedule=[torch.tensor([0, 1])],
            config=config,
            variant_artifact_dir=artifact_dir,
        )
    state = json.loads((artifact_dir / "partial_untied.state.json").read_text(encoding="utf-8"))
    assert state["status"] == "aborted_slow_eval"


def test_reusable_variant_report_loads_matching_config(tmp_path: Path) -> None:
    config = probe.LongSeqReplayProbeConfig(
        output_dir=tmp_path,
        sequence_length=5,
        train_steps=1,
        device="cpu",
        use_amp=False,
    )
    artifact_dir = tmp_path / "variant_results" / probe._run_slug(config)
    report = {"variant": "partial_untied", "final_val_loss": 1.25, "pure_train_tok_per_sec": 10.0}
    probe._write_json_atomic(
        artifact_dir / "partial_untied.json",
        {
            "benchmark": "language_longseq_replay_probe_variant",
            "variant": "partial_untied",
            "config": {
                **probe.asdict(config),
                "output_dir": str(config.output_dir),
                "cache_path": None,
            },
            "report": report,
        },
    )
    loaded = probe._load_reusable_variant_report("partial_untied", config, artifact_dir)
    assert loaded == report


def test_reusable_variant_report_treats_missing_eval_loss_mode_as_full(tmp_path: Path) -> None:
    config = probe.LongSeqReplayProbeConfig(
        output_dir=tmp_path,
        sequence_length=5,
        train_steps=1,
        device="cpu",
        use_amp=False,
    )
    saved_config = {
        **probe.asdict(config),
        "output_dir": str(config.output_dir),
        "cache_path": None,
    }
    saved_config.pop("eval_loss_mode")
    artifact_dir = tmp_path / "variant_results" / probe._run_slug(config)
    report = {"variant": "partial_untied", "final_val_loss": 1.25, "pure_train_tok_per_sec": 10.0}
    probe._write_json_atomic(
        artifact_dir / "partial_untied.json",
        {
            "benchmark": "language_longseq_replay_probe_variant",
            "variant": "partial_untied",
            "config": saved_config,
            "report": report,
        },
    )

    loaded = probe._load_reusable_variant_report("partial_untied", config, artifact_dir)
    assert loaded == report

    sampled_config = probe.LongSeqReplayProbeConfig(
        output_dir=tmp_path,
        sequence_length=5,
        train_steps=1,
        eval_loss_mode="sampled_vocab",
        device="cpu",
        use_amp=False,
    )
    with pytest.raises(ValueError, match="does not match current config"):
        probe._load_reusable_variant_report("partial_untied", sampled_config, artifact_dir)


def test_reusable_partial_baseline_ignores_candidate_only_knobs(tmp_path: Path) -> None:
    saved_config = probe.LongSeqReplayProbeConfig(
        output_dir=tmp_path,
        sequence_length=5,
        train_steps=1,
        sampled_vocab_size=4096,
        full_loss_interval=4,
        conv_kernel_size=7,
        device="cpu",
        use_amp=False,
    )
    current_config = probe.LongSeqReplayProbeConfig(
        output_dir=tmp_path,
        sequence_length=5,
        train_steps=1,
        sampled_vocab_size=16384,
        full_loss_interval=8,
        conv_kernel_size=31,
        device="cpu",
        use_amp=False,
    )
    artifact_dir = tmp_path / "variant_results"
    report = {"variant": "partial_untied", "final_val_loss": 1.25, "pure_train_tok_per_sec": 10.0}
    probe._write_json_atomic(
        artifact_dir / "partial_untied.json",
        {
            "benchmark": "language_longseq_replay_probe_variant",
            "variant": "partial_untied",
            "config": {
                **probe.asdict(saved_config),
                "output_dir": str(saved_config.output_dir),
                "cache_path": None,
            },
            "report": report,
        },
    )
    probe._write_json_atomic(
        artifact_dir / "causal_conv_mixer_sampled_vocab_full4.json",
        {
            "benchmark": "language_longseq_replay_probe_variant",
            "variant": "causal_conv_mixer_sampled_vocab_full4",
            "config": {
                **probe.asdict(saved_config),
                "output_dir": str(saved_config.output_dir),
                "cache_path": None,
            },
            "report": {"variant": "causal_conv_mixer_sampled_vocab_full4", "final_val_loss": 1.5},
        },
    )

    assert probe._load_reusable_variant_report("partial_untied", current_config, artifact_dir) == report
    with pytest.raises(ValueError, match="does not match current config"):
        probe._load_reusable_variant_report("causal_conv_mixer_sampled_vocab_full4", current_config, artifact_dir)


def test_reusable_variant_report_rejects_config_mismatch(tmp_path: Path) -> None:
    saved_config = probe.LongSeqReplayProbeConfig(
        output_dir=tmp_path,
        sequence_length=5,
        train_steps=1,
        device="cpu",
        use_amp=False,
    )
    current_config = probe.LongSeqReplayProbeConfig(
        output_dir=tmp_path,
        sequence_length=7,
        train_steps=1,
        device="cpu",
        use_amp=False,
    )
    artifact_dir = tmp_path / "variant_results"
    probe._write_json_atomic(
        artifact_dir / "partial_untied.json",
        {
            "benchmark": "language_longseq_replay_probe_variant",
            "variant": "partial_untied",
            "config": {
                **probe.asdict(saved_config),
                "output_dir": str(saved_config.output_dir),
                "cache_path": None,
            },
            "report": {"variant": "partial_untied", "final_val_loss": 1.25},
        },
    )
    with pytest.raises(ValueError, match="does not match current config"):
        probe._load_reusable_variant_report("partial_untied", current_config, artifact_dir)


def test_gpu_preflight_rejects_high_existing_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    config = probe.LongSeqReplayProbeConfig(device="cuda", max_gpu_used_mb=1200.0)
    monkeypatch.setattr(probe, "_current_gpu_used_mb", lambda device_name: 1500.0)
    with pytest.raises(RuntimeError, match="GPU preflight rejected run"):
        probe._enforce_gpu_preflight(config)


def test_gpu_preflight_allows_low_existing_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    config = probe.LongSeqReplayProbeConfig(device="cuda", max_gpu_used_mb=1200.0)
    monkeypatch.setattr(probe, "_current_gpu_used_mb", lambda device_name: 500.0)
    probe._enforce_gpu_preflight(config)


def test_factorized_head_variants_output_finite_logits() -> None:
    config = probe.LongSeqReplayProbeConfig(
        sequence_length=11,
        embedding_dim=16,
        hidden_dim=24,
        memory_dim=12,
        factorized_embedding_dim=16,
        factorized_hidden_dim=24,
        factorized_memory_dim=12,
        no_replay_embedding_dim=16,
        no_replay_hidden_dim=24,
        no_replay_rank=6,
        conv_embedding_dim=16,
        conv_layers=2,
        conv_rank=6,
        conv_kernel_size=3,
        partial_token_count=8,
        untied_rank=6,
        dropout=0.0,
    )
    input_ids = torch.randint(0, 64, (2, 11))
    for variant in (
        "factorized_untied",
        "low_rank_untied",
        "factorized_untied_20m",
        "low_rank_untied_20m",
        "factorized_no_replay_20m",
        "causal_conv_mixer_20m",
        "causal_conv_mixer_sampled_vocab",
        "causal_conv_mixer_sampled_vocab_full4",
        "causal_conv_mixer_sampled_vocab_anchor",
        "causal_conv_mixer_sampled_vocab_anchor16",
    ):
        model = probe._build_model(
            variant,
            config,
            vocab_size=64,
            partial_token_ids=torch.arange(config.partial_token_count),
        )
        logits = model(input_ids)
        assert logits.shape == (2, 11, 64)
        assert torch.isfinite(logits).all()


def test_factorized_20m_variant_size_matches_partial_untied() -> None:
    config = probe.LongSeqReplayProbeConfig(sequence_length=63)
    partial_model = probe._build_model(
        "partial_untied",
        config,
        vocab_size=50_257,
        partial_token_ids=torch.arange(config.partial_token_count),
    )
    factorized_model = probe._build_model(
        "factorized_untied_20m",
        config,
        vocab_size=50_257,
        partial_token_ids=torch.arange(config.partial_token_count),
    )
    assert abs(count_parameters(factorized_model) - count_parameters(partial_model)) < 100_000


def test_factorized_no_replay_20m_variant_size_matches_partial_untied() -> None:
    config = probe.LongSeqReplayProbeConfig(sequence_length=63)
    partial_model = probe._build_model(
        "partial_untied",
        config,
        vocab_size=50_257,
        partial_token_ids=torch.arange(config.partial_token_count),
    )
    no_replay_model = probe._build_model(
        "factorized_no_replay_20m",
        config,
        vocab_size=50_257,
        partial_token_ids=torch.arange(config.partial_token_count),
    )
    assert abs(count_parameters(no_replay_model) - count_parameters(partial_model)) < 100_000


def test_causal_conv_mixer_20m_variant_size_matches_partial_untied() -> None:
    config = probe.LongSeqReplayProbeConfig(sequence_length=63)
    partial_model = probe._build_model(
        "partial_untied",
        config,
        vocab_size=50_257,
        partial_token_ids=torch.arange(config.partial_token_count),
    )
    conv_model = probe._build_model(
        "causal_conv_mixer_20m",
        config,
        vocab_size=50_257,
        partial_token_ids=torch.arange(config.partial_token_count),
    )
    assert abs(count_parameters(conv_model) - count_parameters(partial_model)) < 100_000


def test_sampled_vocab_loss_includes_batch_targets() -> None:
    config = probe.LongSeqReplayProbeConfig(
        sequence_length=7,
        embedding_dim=16,
        hidden_dim=24,
        memory_dim=12,
        partial_token_count=8,
        sampled_vocab_size=8,
        dropout=0.0,
    )
    model = probe._build_model(
        "sampled_vocab4096",
        config,
        vocab_size=64,
        partial_token_ids=torch.arange(config.partial_token_count),
    )
    input_ids = torch.randint(0, 64, (2, 7))
    targets = torch.randint(0, 64, (2, 7))
    loss, token_count, candidate_count = probe._sampled_vocab_loss(
        model,
        input_ids,
        targets,
        fixed_candidate_ids=torch.arange(config.sampled_vocab_size),
        vocab_size=64,
    )
    assert torch.isfinite(loss)
    assert token_count == targets.numel()
    assert candidate_count >= torch.unique(targets).numel()


def test_factorized_sampled_vocab_loss_includes_batch_targets() -> None:
    config = probe.LongSeqReplayProbeConfig(
        sequence_length=7,
        conv_embedding_dim=16,
        conv_layers=1,
        conv_rank=6,
        conv_kernel_size=3,
        sampled_vocab_size=8,
        dropout=0.0,
    )
    model = probe._build_model(
        "causal_conv_mixer_sampled_vocab",
        config,
        vocab_size=64,
        partial_token_ids=torch.arange(config.partial_token_count),
    )
    input_ids = torch.randint(0, 64, (2, 7))
    targets = torch.randint(0, 64, (2, 7))
    assert isinstance(model, probe.CausalConvFactorizedLM)
    loss, token_count, candidate_count = probe._factorized_sampled_vocab_loss(
        model,
        input_ids,
        targets,
        fixed_candidate_ids=torch.arange(config.sampled_vocab_size),
        vocab_size=64,
    )
    assert torch.isfinite(loss)
    assert token_count == targets.numel()
    assert candidate_count >= torch.unique(targets).numel()


def test_factorized_sampled_vocab_anchor_loss_is_finite() -> None:
    config = probe.LongSeqReplayProbeConfig(
        sequence_length=7,
        conv_embedding_dim=16,
        conv_layers=1,
        conv_rank=6,
        conv_kernel_size=3,
        sampled_vocab_size=8,
        full_loss_token_stride=3,
        dropout=0.0,
    )
    model = probe._build_model(
        "causal_conv_mixer_sampled_vocab_anchor",
        config,
        vocab_size=64,
        partial_token_ids=torch.arange(config.partial_token_count),
    )
    input_ids = torch.randint(0, 64, (2, 7))
    targets = torch.randint(0, 64, (2, 7))
    assert isinstance(model, probe.CausalConvFactorizedLM)
    loss, token_count, candidate_count = probe._factorized_sampled_vocab_anchor_loss(
        model,
        input_ids,
        targets,
        fixed_candidate_ids=torch.arange(config.sampled_vocab_size),
        vocab_size=64,
        token_stride=config.full_loss_token_stride,
    )
    assert torch.isfinite(loss)
    assert token_count == targets.numel()
    assert candidate_count >= torch.unique(targets).numel()


def test_factorized_full_loss_chunked_matches_full_logits() -> None:
    config = probe.LongSeqReplayProbeConfig(
        sequence_length=7,
        conv_embedding_dim=16,
        conv_layers=1,
        conv_rank=6,
        conv_kernel_size=3,
        dropout=0.0,
    )
    model = probe._build_model(
        "causal_conv_mixer_sampled_vocab_anchor",
        config,
        vocab_size=64,
        partial_token_ids=torch.arange(config.partial_token_count),
    )
    input_ids = torch.randint(0, 64, (2, 7))
    targets = torch.randint(0, 64, (2, 7))
    logits = model(input_ids)
    full_loss, full_tokens = probe._loss_and_tokens(logits, targets)
    chunked_loss, chunked_tokens = probe._factorized_full_loss_chunked(
        model,
        input_ids,
        targets,
        token_chunk_size=3,
    )
    assert chunked_tokens == full_tokens
    assert torch.allclose(chunked_loss, full_loss, atol=1e-6)


def test_anchor_variant_suffix_overrides_full_loss_token_stride() -> None:
    config = probe.LongSeqReplayProbeConfig(full_loss_token_stride=8)
    assert probe._full_loss_token_stride_for_variant("causal_conv_mixer_sampled_vocab_anchor", config) == 8
    assert probe._full_loss_token_stride_for_variant("causal_conv_mixer_sampled_vocab_anchor16", config) == 16
    with pytest.raises(ValueError, match="Invalid anchor stride"):
        probe._full_loss_token_stride_for_variant("causal_conv_mixer_sampled_vocab_anchorx", config)


def test_hierarchical_softmax_loss_is_finite() -> None:
    model = probe.HierarchicalSoftmaxAssociativeLM(
        vocab_size=64,
        embedding_dim=16,
        hidden_dim=24,
        dropout=0.0,
        class_count=8,
    )
    input_ids = torch.randint(0, 64, (2, 7))
    targets = torch.randint(0, 64, (2, 7))
    loss, token_count = model.loss(input_ids, targets)
    assert torch.isfinite(loss)
    assert token_count == targets.numel()
