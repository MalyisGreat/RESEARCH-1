from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from arc_tactic3.language_fastlearn_benchmark import set_global_seed
from arc_tactic3.language_nanochat_actual_compare import _load_cached_datasets, _shared_realtext_config, _train_candidate
from arc_tactic3.language_realtext_microbench import _build_train_batch_schedule
from arc_tactic3.language_recurrent_nano_tricks import (
    PartialUntiedAssociativeLM,
    ReLU2HeadAssociativeLM,
    RecurrentNanoTricksConfig,
    _top_token_ids,
)


@dataclass(frozen=True, slots=True)
class EfficiencyMatrixConfig:
    cache_path: Path
    train_blocks: int = 2048
    val_blocks: int = 128
    sequence_length: int = 127
    train_steps: int = 16
    eval_interval: int = 8
    seed: int = 13
    device: str = "cuda"
    compute_val_bpb: bool = False
    partial_untied_tokens: int = 1024
    recurrent_embedding_dim: int = 144
    recurrent_hidden_dim: int = 288
    recurrent_memory_dim: int = 144


def _variant_overrides() -> dict[str, dict[str, object]]:
    return {
        "default": {},
        "no_amp": {"use_amp": False},
        "no_fused_adamw": {"use_fused_adamw": False},
        "no_pin_memory": {"pin_memory": False},
        "cache_on_device": {"cache_dataset_on_device": True},
        "tensor_eval": {"tensor_batching": True},
    }


def _build_models(config: RecurrentNanoTricksConfig, *, vocab_size: int, partial_token_ids) -> dict[str, object]:
    common = {
        "vocab_size": vocab_size,
        "embedding_dim": config.recurrent_embedding_dim,
        "hidden_dim": config.recurrent_hidden_dim,
        "memory_dim": config.recurrent_memory_dim,
        "dropout": config.dropout,
        "max_length": config.sequence_length,
    }
    return {
        "recurrent_champion": ReLU2HeadAssociativeLM(**common),
        "partial_untied": PartialUntiedAssociativeLM(untied_token_ids=partial_token_ids, **common),
    }


def run_efficiency_matrix(config: EfficiencyMatrixConfig) -> dict[str, Any]:
    base_cfg = RecurrentNanoTricksConfig(
        cache_path=config.cache_path,
        train_blocks=config.train_blocks,
        val_blocks=config.val_blocks,
        sequence_length=config.sequence_length,
        train_steps=config.train_steps,
        eval_interval=config.eval_interval,
        seed=config.seed,
        device=config.device,
        compute_val_bpb=config.compute_val_bpb,
        partial_untied_tokens=config.partial_untied_tokens,
        recurrent_embedding_dim=config.recurrent_embedding_dim,
        recurrent_hidden_dim=config.recurrent_hidden_dim,
        recurrent_memory_dim=config.recurrent_memory_dim,
    )
    set_global_seed(config.seed)
    train_dataset, val_dataset, vocab_size = _load_cached_datasets(base_cfg)
    partial_token_ids = _top_token_ids(train_dataset, count=base_cfg.partial_untied_tokens, vocab_size=vocab_size)
    total_subruns = len(_variant_overrides()) * 2
    completed = 0
    results: dict[str, dict[str, dict[str, Any]]] = {}

    for variant_name, overrides in _variant_overrides().items():
        variant_cfg = RecurrentNanoTricksConfig(**(asdict(base_cfg) | overrides))
        shared_cfg = _shared_realtext_config(variant_cfg)
        batch_schedule = _build_train_batch_schedule(
            len(train_dataset),
            batch_size=shared_cfg.batch_size,
            steps=shared_cfg.train_steps,
            seed=shared_cfg.seed if shared_cfg.train_schedule_seed is None else shared_cfg.train_schedule_seed,
            drop_last=True,
        )
        models = _build_models(variant_cfg, vocab_size=vocab_size, partial_token_ids=partial_token_ids)
        results[variant_name] = {}
        for model_name, model in models.items():
            set_global_seed(config.seed)
            report = _train_candidate(
                model,
                train_dataset,
                val_dataset,
                model_name=model_name,
                tokenizer=None,
                config=shared_cfg,
                compute_val_bpb=False,
                batch_schedule=batch_schedule,
            )
            results[variant_name][model_name] = report
            completed += 1
            print(
                json.dumps(
                    {
                        "progress": f"{completed}/{total_subruns}",
                        "variant": variant_name,
                        "model": model_name,
                        "final_val_loss": report["final_val_loss"],
                        "train_tok_per_sec": report["train_tok_per_sec"],
                        "pure_train_tok_per_sec": report["pure_train_tok_per_sec"],
                        "peak_vram_mb": report["peak_vram_mb"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    summary: dict[str, dict[str, Any]] = {}
    for variant_name, reports in results.items():
        champion = reports["recurrent_champion"]
        partial = reports["partial_untied"]
        summary[variant_name] = {
            "recurrent_champion": {
                "final_val_loss": champion["final_val_loss"],
                "train_tok_per_sec": champion["train_tok_per_sec"],
                "pure_train_tok_per_sec": champion["pure_train_tok_per_sec"],
                "peak_vram_mb": champion["peak_vram_mb"],
                "parameter_count": champion["parameter_count"],
            },
            "partial_untied": {
                "final_val_loss": partial["final_val_loss"],
                "train_tok_per_sec": partial["train_tok_per_sec"],
                "pure_train_tok_per_sec": partial["pure_train_tok_per_sec"],
                "peak_vram_mb": partial["peak_vram_mb"],
                "parameter_count": partial["parameter_count"],
            },
            "delta": {
                "partial_loss_minus_champion": partial["final_val_loss"] - champion["final_val_loss"],
                "partial_tok_per_sec_minus_champion": partial["train_tok_per_sec"] - champion["train_tok_per_sec"],
                "partial_pure_train_tok_per_sec_minus_champion": partial["pure_train_tok_per_sec"] - champion["pure_train_tok_per_sec"],
            },
        }
    return {
        "benchmark": "language_compute_efficiency_matrix",
        "config": {
            "cache_path": str(config.cache_path),
            "train_blocks": config.train_blocks,
            "val_blocks": config.val_blocks,
            "train_steps": config.train_steps,
            "eval_interval": config.eval_interval,
            "sequence_length": config.sequence_length,
            "seed": config.seed,
            "device": config.device,
            "compute_val_bpb": config.compute_val_bpb,
            "partial_untied_tokens": config.partial_untied_tokens,
            "recurrent_embedding_dim": config.recurrent_embedding_dim,
            "recurrent_hidden_dim": config.recurrent_hidden_dim,
            "recurrent_memory_dim": config.recurrent_memory_dim,
        },
        "results": results,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small compute-efficiency matrix for recurrent candidates.")
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--train-blocks", type=int, default=2048)
    parser.add_argument("--val-blocks", type=int, default=128)
    parser.add_argument("--train-steps", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=127)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--recurrent-embedding-dim", type=int, default=144)
    parser.add_argument("--recurrent-hidden-dim", type=int, default=288)
    parser.add_argument("--recurrent-memory-dim", type=int, default=144)
    parser.add_argument("--partial-untied-tokens", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = EfficiencyMatrixConfig(
        cache_path=args.cache_path,
        train_blocks=args.train_blocks,
        val_blocks=args.val_blocks,
        train_steps=args.train_steps,
        eval_interval=args.eval_interval,
        sequence_length=args.sequence_length,
        seed=args.seed,
        device=args.device,
        recurrent_embedding_dim=args.recurrent_embedding_dim,
        recurrent_hidden_dim=args.recurrent_hidden_dim,
        recurrent_memory_dim=args.recurrent_memory_dim,
        partial_untied_tokens=args.partial_untied_tokens,
    )
    payload = run_efficiency_matrix(config)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
