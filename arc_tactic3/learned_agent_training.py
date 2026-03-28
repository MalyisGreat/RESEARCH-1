from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .agents import RandomAgent, TACTICAgent
from .benchmark import SplitReport, evaluate_split
from .learned_mechanic_agent import (
    LearnedMechanicAgent,
    TrainConfig,
    compute_prediction_metrics,
    generate_trace_samples,
    save_checkpoint,
    train_mechanic_model,
)
from .protocol import BenchmarkSplit, build_protocol


def limit_split(split: BenchmarkSplit, case_limit: int | None) -> BenchmarkSplit:
    if case_limit is None:
        return split
    return BenchmarkSplit(name=split.name, cases=split.cases[:case_limit])


def protocol_for_eval(protocol, split_names: set[str], case_limit: int | None) -> dict[str, BenchmarkSplit]:
    return {split.name: limit_split(split, case_limit) for split in protocol.splits() if split.name in split_names}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the learned mixed-environment mechanic agent.")
    parser.add_argument("--replicas-per-case", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-traces-per-level", type=int, default=6)
    parser.add_argument("--eval-traces-per-level", type=int, default=3)
    parser.add_argument("--rollout-steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--pooling", choices=["last", "attention", "hybrid"], default="last")
    parser.add_argument("--explore-budget", type=int, default=5)
    parser.add_argument("--confidence-threshold", type=float, default=0.58)
    parser.add_argument("--text-manual-probability", type=float, default=0.0)
    parser.add_argument("--teacher-rollin-probability", type=float, default=0.0)
    parser.add_argument("--direction-loss-weight", type=float, default=1.0)
    parser.add_argument("--effect-loss-weight", type=float, default=0.75)
    parser.add_argument("--action-loss-weight", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--eval-split", action="append", default=["val", "test_transfer"])
    parser.add_argument("--eval-case-limit", type=int, default=None)
    parser.add_argument("--compare-baselines", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    train_config = TrainConfig(
        seed=args.seed,
        train_traces_per_level=args.train_traces_per_level,
        eval_traces_per_level=args.eval_traces_per_level,
        rollout_steps=args.rollout_steps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        pooling=args.pooling,
        explore_budget=args.explore_budget,
        confidence_threshold=args.confidence_threshold,
        text_manual_probability=args.text_manual_probability,
        teacher_rollin_probability=args.teacher_rollin_probability,
        direction_loss_weight=args.direction_loss_weight,
        effect_loss_weight=args.effect_loss_weight,
        action_loss_weight=args.action_loss_weight,
    )

    protocol = build_protocol(replicas_per_case=args.replicas_per_case)
    eval_splits = protocol_for_eval(protocol, set(args.eval_split), args.eval_case_limit)
    train_samples = generate_trace_samples(
        protocol.train.cases,
        traces_per_level=train_config.train_traces_per_level,
        rollout_steps=train_config.rollout_steps,
        seed=train_config.seed,
        text_manual_probability=train_config.text_manual_probability,
        teacher_rollin_probability=train_config.teacher_rollin_probability,
    )
    val_samples = generate_trace_samples(
        protocol.val.cases,
        traces_per_level=train_config.eval_traces_per_level,
        rollout_steps=train_config.rollout_steps,
        seed=train_config.seed + 1,
        text_manual_probability=train_config.text_manual_probability,
        teacher_rollin_probability=train_config.teacher_rollin_probability,
    )
    test_samples = {
        split.name: generate_trace_samples(
            split.cases,
            traces_per_level=train_config.eval_traces_per_level,
            rollout_steps=train_config.rollout_steps,
            seed=train_config.seed + 10 + index,
            text_manual_probability=train_config.text_manual_probability,
            teacher_rollin_probability=train_config.teacher_rollin_probability,
        )
        for index, split in enumerate(eval_splits.values())
    }

    model, tokenizer, metrics = train_mechanic_model(train_samples, val_samples, config=train_config)
    prediction_metrics = {
        name: compute_prediction_metrics(
            model,
            tokenizer,
            samples,
            batch_size=train_config.batch_size,
            device=train_config.device,
        )
        for name, samples in test_samples.items()
    }

    learned_factory = lambda: LearnedMechanicAgent(
        model,
        tokenizer,
        device=train_config.device,
        explore_budget=train_config.explore_budget,
        confidence_threshold=train_config.confidence_threshold,
    )
    learned_scores = {
        split.name: evaluate_split(learned_factory, split)
        for split in eval_splits.values()
    }
    baseline_scores: dict[str, dict[str, object]] = {}
    if args.compare_baselines:
        tactic_scores = {
            split.name: evaluate_split(TACTICAgent, split)
            for split in eval_splits.values()
        }
        random_scores = {
            split.name: evaluate_split(lambda: RandomAgent(seed=train_config.seed), split)
            for split in eval_splits.values()
        }
        baseline_scores = {
            "tactic": {name: asdict(report) for name, report in tactic_scores.items()},
            "random": {name: asdict(report) for name, report in random_scores.items()},
        }

    payload = {
        "train_config": asdict(train_config),
        "eval_splits": sorted(eval_splits.keys()),
        "sample_counts": {
            "train": len(train_samples),
            "val": len(val_samples),
            **{name: len(samples) for name, samples in test_samples.items()},
        },
        "training": metrics,
        "prediction_metrics": prediction_metrics,
        "protocol_scores": {
            "learned_mechanic": {name: asdict(report) for name, report in learned_scores.items()},
            **baseline_scores,
        },
    }

    if args.checkpoint is not None:
        save_checkpoint(args.checkpoint, model, tokenizer, train_config, payload)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
