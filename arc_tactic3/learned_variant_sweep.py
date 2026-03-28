from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean

from .benchmark import evaluate_split
from .dsl import rename_button_labels, reorder_available_buttons
from .learned_mechanic_agent import (
    LearnedMechanicAgent,
    TrainConfig,
    compute_prediction_metrics,
    generate_trace_samples,
    train_mechanic_model,
)
from .protocol import BenchmarkSplit, build_protocol, clone_case_with_seed


def limit_split(split: BenchmarkSplit, case_limit: int | None) -> BenchmarkSplit:
    if case_limit is None:
        return split
    return BenchmarkSplit(name=split.name, cases=split.cases[:case_limit])


def augmented_train_cases(base_cases: tuple, *, seed_offset: int) -> tuple:
    augmented = list(base_cases)
    for index, case in enumerate(base_cases):
        cloned = clone_case_with_seed(
            case,
            env_suffix=f"aug{seed_offset}",
            seed=7000 + seed_offset + index * 19,
            remap_each_level=True,
        )
        reordered = reorder_available_buttons(cloned, tuple(reversed(cloned.config.available_buttons)))
        renamed = rename_button_labels(
            reordered,
            tuple(f"aug_{seed_offset}_{slot}" for slot, _ in enumerate(reordered.config.available_buttons)),
        )
        augmented.append(renamed)
    return tuple(augmented)


def hard_mean(split_scores: dict[str, dict[str, float]]) -> float:
    keys = tuple(name for name in ("test_transfer", "test_remapped", "test_ood") if name in split_scores)
    return float(mean(split_scores[name]["score"] for name in keys))


def screen_mean(split_scores: dict[str, dict[str, float]]) -> float:
    keys = ("val", "test_transfer")
    return float(mean(split_scores[name]["score"] for name in keys))


def load_active_champion_value() -> float | None:
    objective_path = Path(__file__).resolve().parent / ".optimization-lab" / "live" / "current_objective.json"
    if not objective_path.exists():
        return None
    try:
        payload = json.loads(objective_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    value = payload.get("champion_value")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def run_variant(
    *,
    config: TrainConfig,
    train_cases: tuple,
    val_cases: tuple,
    eval_splits: dict[str, BenchmarkSplit],
    agent_kwargs: dict[str, object],
    seed_bump: int,
) -> dict[str, object]:
    train_samples = generate_trace_samples(
        train_cases,
        traces_per_level=config.train_traces_per_level,
        rollout_steps=config.rollout_steps,
        seed=config.seed + seed_bump,
        text_manual_probability=config.text_manual_probability,
        teacher_rollin_probability=config.teacher_rollin_probability,
    )
    val_samples = generate_trace_samples(
        val_cases,
        traces_per_level=config.eval_traces_per_level,
        rollout_steps=config.rollout_steps,
        seed=config.seed + seed_bump + 1,
        text_manual_probability=config.text_manual_probability,
        teacher_rollin_probability=config.teacher_rollin_probability,
    )
    model, tokenizer, metrics = train_mechanic_model(train_samples, val_samples, config=config)
    prediction_metrics = {
        split_name: compute_prediction_metrics(
            model,
            tokenizer,
            generate_trace_samples(
                split.cases,
                traces_per_level=config.eval_traces_per_level,
                rollout_steps=config.rollout_steps,
                seed=config.seed + seed_bump + 40 + index,
                text_manual_probability=config.text_manual_probability,
                teacher_rollin_probability=config.teacher_rollin_probability,
            ),
            batch_size=config.batch_size,
            device=config.device,
        )
        for index, (split_name, split) in enumerate(eval_splits.items())
    }
    learned_factory = lambda: LearnedMechanicAgent(
        model,
        tokenizer,
        device=config.device,
        explore_budget=config.explore_budget,
        confidence_threshold=config.confidence_threshold,
        **agent_kwargs,
    )
    split_scores = {split.name: asdict(evaluate_split(learned_factory, split)) for split in eval_splits.values()}
    primary_name = "test_transfer" if "test_transfer" in split_scores else next(iter(split_scores))
    return {
        "config": asdict(config),
        "agent_kwargs": agent_kwargs,
        "sample_counts": {"train": len(train_samples), "val": len(val_samples)},
        "training": metrics,
        "prediction_metrics": prediction_metrics,
        "split_scores": split_scores,
        "primary_metric_name": primary_name,
        "primary_metric": float(split_scores[primary_name]["score"]),
        "hard_mean": hard_mean(split_scores),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run five tracked learned-agent tweaks against a locked synthetic benchmark.")
    parser.add_argument("--replicas-per-case", type=int, default=1)
    parser.add_argument("--train-case-limit", type=int, default=4)
    parser.add_argument("--val-case-limit", type=int, default=4)
    parser.add_argument("--eval-case-limit", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = TrainConfig(
        epochs=2,
        train_traces_per_level=1,
        eval_traces_per_level=1,
        rollout_steps=4,
        batch_size=64,
        embedding_dim=24,
        hidden_dim=48,
        seed=1337,
        pooling="last",
    )
    protocol = build_protocol(replicas_per_case=args.replicas_per_case)
    train_cases = protocol.train.cases[: args.train_case_limit]
    val_cases = protocol.val.cases[: args.val_case_limit]
    screen_splits = {
        split.name: limit_split(split, args.eval_case_limit)
        for split in protocol.splits()
        if split.name in {"val", "test_transfer"}
    }
    guardrail_splits = {
        split.name: limit_split(split, max(1, args.eval_case_limit))
        for split in protocol.splits()
        if split.name in {"test_remapped", "test_ood"}
    }

    champion_config = replace(
        base,
        train_traces_per_level=3,
        rollout_steps=6,
        hidden_dim=72,
        num_layers=2,
        pooling="hybrid",
        action_loss_weight=1.0,
        effect_loss_weight=1.0,
        teacher_rollin_probability=0.05,
        learning_rate=5e-4,
        weight_decay=2e-4,
    )
    champion_agent_kwargs = {
        "use_symbolic_posterior": True,
        "symbolic_transfer": False,
        "symbolic_transfer_mode": "full",
        "symbolic_summary_transfer": False,
        "symbolic_summary_transfer_mode": "family_click",
        "symbolic_family_prior": True,
        "symbolic_click_prior": False,
        "symbolic_direction_prior": True,
        "symbolic_plan_confidence": 0.45,
        "symbolic_prior_floor": 0.01,
        "symbolic_prior_power": 1.5,
        "symbolic_family_weight": 1.0,
        "symbolic_click_weight": 0.0,
        "symbolic_direction_weight": 1.0,
        "symbolic_reprioritize": True,
        "symbolic_reprioritize_uncertainty": 0.6,
        "symbolic_plan_commit_steps": 2,
        "symbolic_plan_commit_confidence": 0.5,
        "symbolic_plan_commit_uncertainty_ceiling": 0.55,
    }
    baseline = run_variant(
        config=champion_config,
        train_cases=train_cases,
        val_cases=val_cases,
        eval_splits=screen_splits,
        agent_kwargs=champion_agent_kwargs,
        seed_bump=0,
    )
    baseline_primary = float(baseline["primary_metric"])
    baseline_screen_mean = float(screen_mean(baseline["split_scores"]))
    baseline_guardrail_hard_mean = float(baseline["hard_mean"])
    active_champion_value = load_active_champion_value()
    incumbent_primary = max(
        baseline_primary,
        active_champion_value if active_champion_value is not None else baseline_primary,
    )

    variants = [
        {
            "name": "conservative_affordance_click_only_solved",
            "rationale": "Transfer only solved-level click affordances and state-change summaries into the posterior, while keeping direction and family inference fresh per level.",
            "config": champion_config,
            "agent_kwargs": {
                **champion_agent_kwargs,
                "symbolic_affordance_transfer": True,
                "symbolic_affordance_prior_mode": "click_only",
                "symbolic_affordance_bonus_weight": 0.25,
                "symbolic_transfer_requires_solved": True,
            },
            "train_cases": train_cases,
        },
        {
            "name": "local_affordance_bonus_only",
            "rationale": "Use affordance memory only as an action bonus and never as a posterior prior, testing whether transition summaries help only at action selection time.",
            "config": champion_config,
            "agent_kwargs": {
                **champion_agent_kwargs,
                "symbolic_affordance_transfer": True,
                "symbolic_affordance_prior_mode": "none",
                "symbolic_affordance_bonus_weight": 0.45,
                "symbolic_transfer_confidence_floor": 0.75,
                "symbolic_transfer_uncertainty_ceiling": 0.45,
            },
            "train_cases": train_cases,
        },
        {
            "name": "structural_affordance_click_only_scaled",
            "rationale": "Scale traces and capacity, then transfer click-only affordances with a stronger bonus to test whether the safer abstraction benefits from more evidence.",
            "config": replace(
                champion_config,
                train_traces_per_level=4,
                rollout_steps=7,
                hidden_dim=88,
                num_layers=3,
                learning_rate=4e-4,
                weight_decay=2e-4,
            ),
            "agent_kwargs": {
                **champion_agent_kwargs,
                "symbolic_affordance_transfer": True,
                "symbolic_affordance_prior_mode": "click_only",
                "symbolic_affordance_bonus_weight": 0.45,
                "symbolic_transfer_confidence_floor": 0.72,
                "symbolic_transfer_uncertainty_ceiling": 0.5,
            },
            "train_cases": train_cases,
        },
        {
            "name": "contrarian_affordance_family_only_low_bonus",
            "rationale": "Transfer only family-level affordance summaries with a low action bonus, testing whether object-level click priors were the harmful part of v19.",
            "config": champion_config,
            "agent_kwargs": {
                **champion_agent_kwargs,
                "symbolic_affordance_transfer": True,
                "symbolic_affordance_prior_mode": "family_only",
                "symbolic_affordance_bonus_weight": 0.15,
                "symbolic_transfer_confidence_floor": 0.72,
                "symbolic_transfer_uncertainty_ceiling": 0.48,
            },
            "train_cases": train_cases,
        },
        {
            "name": "synthesis_affordance_click_only_augmented",
            "rationale": "Blend the stable augmented champion with click-only affordance priors, safe summary transfer, and train-only augmentation for the strongest safe hybrid.",
            "config": replace(
                champion_config,
                train_traces_per_level=4,
                rollout_steps=6,
                hidden_dim=80,
                num_layers=2,
                teacher_rollin_probability=0.03,
                learning_rate=4e-4,
                weight_decay=2e-4,
            ),
            "agent_kwargs": {
                **champion_agent_kwargs,
                "symbolic_summary_transfer": True,
                "symbolic_affordance_transfer": True,
                "symbolic_affordance_prior_mode": "click_only",
                "symbolic_affordance_bonus_weight": 0.35,
                "symbolic_transfer_requires_solved": True,
                "symbolic_transfer_confidence_floor": 0.72,
                "symbolic_transfer_uncertainty_ceiling": 0.48,
                "symbolic_summary_transfer_mode": "family_click",
            },
            "train_cases": augmented_train_cases(train_cases, seed_offset=418),
        },
    ]

    results: dict[str, object] = {}
    champion_primary = incumbent_primary
    champion_name = "baseline_current"

    for index, variant in enumerate(variants, start=1):
        result = run_variant(
            config=variant["config"],
            train_cases=variant["train_cases"],
            val_cases=val_cases,
            eval_splits=screen_splits,
            agent_kwargs=variant["agent_kwargs"],
            seed_bump=index * 97,
        )
        primary = float(result["primary_metric"])
        candidate_screen_mean = float(screen_mean(result["split_scores"]))
        improves_primary = primary >= champion_primary * 1.01 if champion_primary > 0 else primary > champion_primary
        preserves_screen_mean = candidate_screen_mean >= baseline_screen_mean
        verification_runs = []
        guardrail_metrics = None
        preserves_guardrails = False
        if improves_primary and preserves_screen_mean:
            verification = run_variant(
                config=replace(variant["config"], seed=variant["config"].seed + 1000),
                train_cases=variant["train_cases"],
                val_cases=val_cases,
                eval_splits=screen_splits,
                agent_kwargs=variant["agent_kwargs"],
                seed_bump=index * 131,
            )
            verification_runs.append(
                {
                    "primary_metric": verification["primary_metric"],
                    "screen_mean": screen_mean(verification["split_scores"]),
                    "split_scores": verification["split_scores"],
                }
            )
            improves_primary = improves_primary and float(verification["primary_metric"]) >= champion_primary * 1.01
            preserves_screen_mean = preserves_screen_mean and float(screen_mean(verification["split_scores"])) >= baseline_screen_mean
            if improves_primary and preserves_screen_mean:
                guardrail = run_variant(
                    config=variant["config"],
                    train_cases=variant["train_cases"],
                    val_cases=val_cases,
                    eval_splits=guardrail_splits,
                    agent_kwargs=variant["agent_kwargs"],
                    seed_bump=index * 149,
                )
                guardrail_metrics = {
                    "hard_mean": guardrail["hard_mean"],
                    "split_scores": guardrail["split_scores"],
                }
                preserves_guardrails = float(guardrail["hard_mean"]) >= baseline_guardrail_hard_mean
        accepted = improves_primary and preserves_screen_mean and guardrail_metrics is not None and preserves_guardrails
        result.update(
            {
                "rationale": variant["rationale"],
                "delta_vs_baseline": primary - baseline_primary,
                "delta_vs_champion": primary - champion_primary,
                "accepted": accepted,
                "verification_runs": verification_runs,
                "screen_mean": candidate_screen_mean,
                "guardrail_metrics": guardrail_metrics,
                "preserves_guardrails": preserves_guardrails,
            }
        )
        results[variant["name"]] = result
        if accepted:
            champion_primary = primary
            champion_name = variant["name"]

    summary = {
        "primary_metric": "test_transfer score",
        "screen_metric": "mean(val, test_transfer)",
        "guardrail_metric": "mean(test_remapped, test_ood)",
        "baseline_name": "baseline_current",
        "baseline": baseline,
        "active_champion_value": active_champion_value,
        "incumbent_primary": incumbent_primary,
        "champion_name": champion_name,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
