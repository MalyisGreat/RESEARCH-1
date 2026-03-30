from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from arc_tactic3.language_docaware_head_to_head import DocAwareHeadToHeadConfig, run_docaware_head_to_head
from arc_tactic3.language_fastlearn_benchmark import set_global_seed


@dataclass(frozen=True, slots=True)
class ScreeningCandidate:
    label: str
    sequence_length: int = 127
    target_tokens: int = 30_000_000
    partial_learning_rate: float = 1e-3
    partial_warmup_steps: int = 1024
    partial_lr_schedule: str = "cosine"
    partial_min_lr_scale: float = 0.1
    partial_batch_size: int = 16
    partial_eval_batch_size: int = 16
    factorized_embedding_dim: int = 128
    factorized_hidden_dim: int = 896
    factorized_memory_dim: int = 128
    factorized_untied_rank: int = 192
    notes: str = ""


def default_candidates() -> list[ScreeningCandidate]:
    return [
        ScreeningCandidate(
            label="factorized20m_fastbatch_seq127",
            notes="Baseline 20M factorized model with the optimized doc_reset batcher path.",
        ),
        ScreeningCandidate(
            label="factorized20m_seq256",
            sequence_length=256,
            partial_warmup_steps=512,
            notes="Same 20M factorized model, but with longer 256-token context.",
        ),
        ScreeningCandidate(
            label="factorized30m_seq127",
            factorized_embedding_dim=224,
            factorized_hidden_dim=1408,
            factorized_memory_dim=224,
            factorized_untied_rank=192,
            notes="Conservative scale step to ~30M params.",
        ),
        ScreeningCandidate(
            label="factorized40m_lr13_w768",
            partial_learning_rate=1.3e-3,
            partial_warmup_steps=768,
            factorized_embedding_dim=256,
            factorized_hidden_dim=1536,
            factorized_memory_dim=256,
            factorized_untied_rank=320,
            notes="40M factorized model with a moderately more aggressive recipe.",
        ),
        ScreeningCandidate(
            label="factorized40m_lr15_w512",
            partial_learning_rate=1.5e-3,
            partial_warmup_steps=512,
            factorized_embedding_dim=256,
            factorized_hidden_dim=1536,
            factorized_memory_dim=256,
            factorized_untied_rank=320,
            notes="40M factorized model with a more aggressive LR and shorter warmup.",
        ),
    ]


def semifinal50m_candidates() -> list[ScreeningCandidate]:
    return [
        ScreeningCandidate(
            label="factorized20m_fastbatch_seq127_50m",
            sequence_length=127,
            target_tokens=50_000_000,
            notes="50M semifinal control from fastbatch seq127 screen.",
        ),
        ScreeningCandidate(
            label="factorized20m_seq256_50m",
            sequence_length=256,
            target_tokens=50_000_000,
            partial_warmup_steps=512,
            notes="50M semifinal challenger from seq256 screen.",
        ),
    ]


def _preset_candidates(name: str) -> list[ScreeningCandidate]:
    if name == "default":
        return default_candidates()
    if name == "semifinal50m":
        return semifinal50m_candidates()
    raise ValueError(f"Unsupported preset: {name}")


def run_screening_portfolio(
    *,
    cache_path: Path,
    output_dir: Path,
    seed: int,
    device: str,
    sample_prompt: str | None,
    sample_every_tokens: int,
    sample_generation_tokens: int,
    candidates: list[ScreeningCandidate],
) -> dict[str, Any]:
    set_global_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "benchmark": "language_recurrent_screening_portfolio",
        "cache_path": str(cache_path),
        "seed": seed,
        "device": device,
        "sample_prompt": sample_prompt,
        "candidates": {},
        "scoreboard": {},
        "status": "running",
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for candidate in candidates:
        candidate_dir = output_dir / candidate.label
        start = perf_counter()
        config = DocAwareHeadToHeadConfig(
            cache_path=cache_path,
            output_dir=candidate_dir,
            target_tokens=candidate.target_tokens,
            seed=seed,
            sequence_length=candidate.sequence_length,
            model_filter="factorized_20m",
            device=device,
            sample_prompt=sample_prompt,
            sample_every_tokens=sample_every_tokens,
            sample_generation_tokens=sample_generation_tokens,
            partial_learning_rate=candidate.partial_learning_rate,
            partial_warmup_steps=candidate.partial_warmup_steps,
            partial_lr_schedule=candidate.partial_lr_schedule,
            partial_min_lr_scale=candidate.partial_min_lr_scale,
            partial_batch_size=candidate.partial_batch_size,
            partial_eval_batch_size=candidate.partial_eval_batch_size,
            factorized_embedding_dim=candidate.factorized_embedding_dim,
            factorized_hidden_dim=candidate.factorized_hidden_dim,
            factorized_memory_dim=candidate.factorized_memory_dim,
            factorized_untied_rank=candidate.factorized_untied_rank,
        )
        payload = run_docaware_head_to_head(config)
        report = payload["models"]["factorized_20m"]
        elapsed = perf_counter() - start
        summary["candidates"][candidate.label] = {
            "candidate": asdict(candidate),
            "report": report,
            "wall_clock_seconds": elapsed,
        }
        summary["scoreboard"][candidate.label] = {
            "best_val_loss": report["best_val_loss"],
            "final_val_loss": report["final_val_loss"],
            "parameter_count": report["parameter_count"],
            "train_tok_per_sec": report["train_tok_per_sec"],
            "pure_train_tok_per_sec": report["pure_train_tok_per_sec"],
            "tokens_to_target_loss": report["tokens_to_target_loss"],
            "seconds_to_target_loss": report["seconds_to_target_loss"],
            "sequence_length": candidate.sequence_length,
            "learning_rate": candidate.partial_learning_rate,
            "warmup_steps": candidate.partial_warmup_steps,
            "notes": candidate.notes,
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary["status"] = "completed"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a portfolio of shorter recurrent screening jobs.")
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--preset", choices=("default", "semifinal50m"), default="default")
    parser.add_argument("--sample-prompt", type=str, default="i went to the mall because")
    parser.add_argument("--sample-every-tokens", type=int, default=10_000_000)
    parser.add_argument("--sample-generation-tokens", type=int, default=40)
    args = parser.parse_args()

    payload = run_screening_portfolio(
        cache_path=args.cache_path,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
        sample_prompt=args.sample_prompt,
        sample_every_tokens=args.sample_every_tokens,
        sample_generation_tokens=args.sample_generation_tokens,
        candidates=_preset_candidates(args.preset),
    )
    print(json.dumps(payload["scoreboard"], indent=2))


if __name__ == "__main__":
    main()
