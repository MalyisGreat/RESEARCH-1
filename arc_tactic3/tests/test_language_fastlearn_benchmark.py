from __future__ import annotations

import math

from arc_tactic3.language_fastlearn_benchmark import (
    BenchmarkConfig,
    LanguageTaskDataset,
    build_language_tasks,
    build_models,
    fairness_summary,
    run_benchmark,
    run_scaling_sweep,
    set_global_seed,
)


def test_build_language_tasks_is_deterministic() -> None:
    tasks_a = build_language_tasks(seed=123, task_count=6, max_support=4, query_examples=4)
    tasks_b = build_language_tasks(seed=123, task_count=6, max_support=4, query_examples=4)
    assert tasks_a == tasks_b
    families = {task.family for task in tasks_a}
    assert families == {"agreement_marker", "order_map"}


def test_default_model_sizes_are_fairly_matched() -> None:
    config = BenchmarkConfig(device="cpu")
    tasks = build_language_tasks(seed=10, task_count=8, max_support=config.max_support, query_examples=config.query_examples)
    dataset = LanguageTaskDataset(tasks, max_support=config.max_support, query_examples=config.query_examples)
    models = build_models(config, max_length=dataset.max_sequence_length - 1)
    reports = {
        name: {"parameter_count_mean": sum(parameter.numel() for parameter in model.parameters())}
        for name, model in models.items()
    }
    fairness = fairness_summary(reports)
    assert fairness["parameter_gap_ok"] is True
    assert fairness["relative_parameter_gap"] <= 0.15


def test_run_benchmark_smoke_cpu() -> None:
    config = BenchmarkConfig(
        seed=7,
        train_tasks=16,
        val_tasks=4,
        test_tasks=4,
        max_support=4,
        query_examples=4,
        epochs=1,
        batch_size=8,
        eval_batch_size=8,
        device="cpu",
        support_shots=(0, 1, 2, 4),
        gru_embedding_dim=24,
        gru_hidden_dim=28,
        gru_memory_dim=24,
        gpt2_d_model=24,
        gpt2_heads=4,
        gpt2_layers=1,
        gpt2_ff_dim=96,
        sequence_focus_weight=0.1,
        rollout_loss_weight=0.1,
        rollout_teacher_forcing_prob=0.25,
    )
    payload = run_benchmark(seeds=[7], config=config)
    assert payload["benchmark"] == "language_fastlearn_compare"
    assert payload["fairness"]["parameter_gap_ok"] is True
    assert set(payload["models"].keys()) == {"fast_gru", "gpt2_like"}
    assert payload["winner_by_adaptation_auc"] in {"fast_gru", "gpt2_like"}
    assert payload["winner_by_autoregressive_adaptation_auc"] in {"fast_gru", "gpt2_like"}
    for model_payload in payload["models"].values():
        assert model_payload["adaptation_auc_mean"] >= 0.0
        assert model_payload["autoregressive_adaptation_auc_mean"] >= 0.0
        assert math.isfinite(model_payload["best_val_loss_mean"])
        assert set(model_payload["eval_by_shot"].keys()) == {"0", "1", "2", "4"}
        assert "train_sequence_focus_loss" in model_payload["history"][0]
        assert "val_sequence_focus_loss" in model_payload["history"][0]
        assert "train_rollout_loss" in model_payload["history"][0]
        assert "val_rollout_sequence_accuracy" in model_payload["history"][0]
        assert "rollout_weight" in model_payload["history"][0]
        assert "rollout_teacher_forcing_prob" in model_payload["history"][0]
        for metrics in model_payload["eval_by_shot"].values():
            assert 0.0 <= metrics["token_accuracy"] <= 1.0
            assert 0.0 <= metrics["sequence_accuracy"] <= 1.0
            assert 0.0 <= metrics["order_map_sequence_accuracy"] <= 1.0
            assert 0.0 <= metrics["agreement_marker_sequence_accuracy"] <= 1.0
            assert 0.0 <= metrics["autoregressive_token_accuracy"] <= 1.0
            assert 0.0 <= metrics["autoregressive_sequence_accuracy"] <= 1.0
            assert 0.0 <= metrics["order_map_autoregressive_sequence_accuracy"] <= 1.0
            assert 0.0 <= metrics["agreement_marker_autoregressive_sequence_accuracy"] <= 1.0


def test_fast_gru_uses_support_memory() -> None:
    set_global_seed(5)
    config = BenchmarkConfig(
        device="cpu",
        max_support=4,
        query_examples=4,
        gru_embedding_dim=24,
        gru_hidden_dim=28,
        gru_memory_dim=24,
        gpt2_d_model=24,
        gpt2_heads=4,
        gpt2_layers=1,
        gpt2_ff_dim=96,
    )
    tasks = build_language_tasks(seed=5, task_count=4, max_support=config.max_support, query_examples=config.query_examples)
    dataset = LanguageTaskDataset(tasks, max_support=config.max_support, query_examples=config.query_examples)
    batch = dataset[0]
    model = build_models(config, max_length=dataset.max_sequence_length - 1)["fast_gru"]
    query_ids = batch["query_ids"].unsqueeze(0)
    support_ids = batch["support_ids"].unsqueeze(0)
    support_targets = batch["support_targets"].unsqueeze(0)
    support_mask = batch["support_mask"].unsqueeze(0)
    with_support = model(
        support_ids=support_ids,
        support_targets=support_targets,
        support_mask=support_mask,
        support_present=batch["support_present"].unsqueeze(0),
        query_ids=query_ids,
    )
    without_support = model(
        support_ids=support_ids,
        support_targets=support_targets,
        support_mask=support_mask,
        support_present=batch["support_present"].unsqueeze(0).zero_(),
        query_ids=query_ids,
    )
    assert with_support.shape == without_support.shape
    assert not with_support.allclose(without_support)


def test_gpt2_like_uses_in_context_support() -> None:
    set_global_seed(11)
    config = BenchmarkConfig(
        device="cpu",
        max_support=4,
        query_examples=4,
        gru_embedding_dim=24,
        gru_hidden_dim=28,
        gru_memory_dim=24,
        gpt2_d_model=24,
        gpt2_heads=4,
        gpt2_layers=1,
        gpt2_ff_dim=96,
    )
    tasks = build_language_tasks(seed=11, task_count=4, max_support=config.max_support, query_examples=config.query_examples)
    dataset = LanguageTaskDataset(tasks, max_support=config.max_support, query_examples=config.query_examples)
    batch = dataset[0]
    model = build_models(config, max_length=dataset.max_sequence_length - 1)["gpt2_like"]
    with_support = model(
        support_ids=batch["support_ids"].unsqueeze(0),
        support_targets=batch["support_targets"].unsqueeze(0),
        support_mask=batch["support_mask"].unsqueeze(0),
        support_present=batch["support_present"].unsqueeze(0),
        query_ids=batch["query_ids"].unsqueeze(0),
    )
    without_support = model(
        support_ids=batch["support_ids"].unsqueeze(0),
        support_targets=batch["support_targets"].unsqueeze(0),
        support_mask=batch["support_mask"].unsqueeze(0),
        support_present=batch["support_present"].unsqueeze(0).zero_(),
        query_ids=batch["query_ids"].unsqueeze(0),
    )
    assert with_support.shape == without_support.shape
    assert not with_support.allclose(without_support)


def test_run_scaling_sweep_smoke_cpu() -> None:
    config = BenchmarkConfig(
        seed=7,
        train_tasks=8,
        val_tasks=4,
        test_tasks=4,
        max_support=4,
        query_examples=4,
        epochs=1,
        batch_size=4,
        eval_batch_size=4,
        device="cpu",
        support_shots=(0, 1, 2, 4),
    )
    payload = run_scaling_sweep(seeds=[7], config=config, scales=("small", "medium"))
    assert payload["benchmark"] == "language_fastlearn_scaling"
    assert payload["scales"] == ["small", "medium"]
    for scale in payload["results"].values():
        assert scale["winner_by_adaptation_auc"] in {"fast_gru", "gpt2_like"}
        assert scale["fairness"]["parameter_gap_ok"] is True
        for model_payload in scale["models"].values():
            assert model_payload["adaptation_auc_mean"] >= 0.0
            assert model_payload["autoregressive_adaptation_auc_mean"] >= 0.0
            assert 0.0 <= model_payload["shot8_token_accuracy"] <= 1.0
            assert 0.0 <= model_payload["shot8_sequence_accuracy"] <= 1.0
            assert 0.0 <= model_payload["shot8_autoregressive_token_accuracy"] <= 1.0
            assert 0.0 <= model_payload["shot8_autoregressive_sequence_accuracy"] <= 1.0
