from __future__ import annotations

from pathlib import Path

import torch

from arc_tactic3.language_overnight_recurrent_sweep import (
    ModelDims,
    RunLogger,
    SyntheticStage,
    _flat_tokens_to_blocks,
    _select_finalists,
    _sequence_sweep_gate,
    _search_dims_for_target,
    _synthetic_config,
    build_candidate_model,
    build_synthetic_tasks,
    train_with_progress,
)
from arc_tactic3.language_realtext_microbench import TokenBlockDataset


def _random_dataset(num_sequences: int, sequence_length: int, vocab_size: int) -> TokenBlockDataset:
    blocks = torch.randint(0, vocab_size, (num_sequences, sequence_length + 1), dtype=torch.long)
    return TokenBlockDataset(blocks[:, :-1].contiguous(), blocks[:, 1:].contiguous())


def test_build_synthetic_tasks_shapes() -> None:
    tasks = build_synthetic_tasks(sequence_length=31, seed=13)
    assert len(tasks) == 4
    for task in tasks:
        assert task.train_dataset.input_ids.shape[1] == 31
        assert task.val_dataset.targets.shape[1] == 31
        assert task.answer_mask.shape == task.val_dataset.targets.shape


def test_train_with_progress_cpu_smoke(tmp_path: Path) -> None:
    vocab_size = 256
    sequence_length = 15
    train_dataset = _random_dataset(32, sequence_length, vocab_size)
    val_dataset = _random_dataset(8, sequence_length, vocab_size)
    dims = ModelDims(embedding_dim=32, hidden_dim=64, memory_dim=32, partial_untied_tokens=64)
    model = build_candidate_model(
        "partial_untied",
        dims=dims,
        vocab_size=vocab_size,
        sequence_length=sequence_length,
        train_dataset=train_dataset,
        dropout=0.1,
    )
    logger = RunLogger(tmp_path / "logs", total_tasks=1)
    config = _synthetic_config(
        type("Cfg", (), {
            "learning_rate": 2e-3,
            "weight_decay": 1e-4,
            "device": "cpu",
            "use_amp": False,
            "pin_memory": False,
            "use_fused_adamw": False,
            "optimizer_recipe": "default",
            "warmup_steps": 0,
            "lr_schedule": "none",
            "min_lr_scale": 1.0,
        })(),
        SyntheticStage(train_steps=2, eval_interval=1, batch_size=4, eval_batch_size=4, sequence_length=sequence_length),
        seed=13,
    )
    report = train_with_progress(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=None,
        model_name="partial_untied_smoke",
        config=config,
        logger=logger,
        task_id="smoke",
        label="cpu_smoke",
        sample_every_evals=999,
        eval_batch_limit=2,
    )
    assert report["parameter_count"] > 0
    assert report["final_val_loss"] > 0
    assert len(report["history"]) >= 2


def test_search_dims_returns_candidate() -> None:
    train_dataset = _random_dataset(16, 31, 256)
    dims = _search_dims_for_target(
        "learned_chunk_writer",
        target_params=2_000_000,
        vocab_size=256,
        sequence_length=31,
        train_dataset=train_dataset,
        dropout=0.1,
    )
    assert dims.embedding_dim >= 160
    assert dims.hidden_dim == dims.embedding_dim * 2


def test_flat_tokens_to_blocks_truncates_cleanly() -> None:
    tokens = torch.arange(20, dtype=torch.int32)
    blocks = _flat_tokens_to_blocks(tokens, block_size=6)
    assert blocks.shape == (3, 6)
    assert blocks[-1, -1].item() == 17


def test_select_finalists_requires_promoted_candidates_and_uses_long_hold() -> None:
    cheap = {
        "reports": {
            "partial_untied": {"final_val_loss": 10.0, "pure_train_tok_per_sec": 100.0},
            "local_global_partial_memory": {"final_val_loss": 8.0, "pure_train_tok_per_sec": 95.0},
            "chunk_mean_token_memory": {"final_val_loss": 8.5, "pure_train_tok_per_sec": 110.0},
            "learned_chunk_writer": {"final_val_loss": 8.6, "pure_train_tok_per_sec": 105.0},
        }
    }
    medium_a = {
        "reports": {
            "partial_untied": {"final_val_loss": 8.0, "pure_train_tok_per_sec": 100.0},
            "chunk_mean_token_memory": {"final_val_loss": 7.7, "pure_train_tok_per_sec": 101.0},
            "learned_chunk_writer": {"final_val_loss": 7.6, "pure_train_tok_per_sec": 100.0},
        }
    }
    medium_b = {
        "reports": {
            "partial_untied": {"final_val_loss": 8.0, "pure_train_tok_per_sec": 100.0},
            "chunk_mean_token_memory": {"final_val_loss": 7.8, "pure_train_tok_per_sec": 101.0},
            "learned_chunk_writer": {"final_val_loss": 7.7, "pure_train_tok_per_sec": 100.0},
        }
    }
    synthetic = {
        "tasks": {
            "t0": {
                "reports": {
                    "partial_untied": {"answer_token_accuracy": 1.0, "answer_sequence_accuracy": 1.0},
                    "chunk_mean_token_memory": {"answer_token_accuracy": 0.1, "answer_sequence_accuracy": 0.0},
                    "learned_chunk_writer": {"answer_token_accuracy": 0.2, "answer_sequence_accuracy": 0.1},
                }
            }
        }
    }
    long_hold = {
        "reports": {
            "partial_untied": {"final_val_loss": 7.0, "pure_train_tok_per_sec": 100.0},
            "chunk_mean_token_memory": {"final_val_loss": 7.4, "pure_train_tok_per_sec": 101.0},
        }
    }
    finalists = _select_finalists(cheap, [medium_a, medium_b], synthetic, long_hold)
    assert "local_global_partial_memory" not in finalists
    assert finalists[0] == "learned_chunk_writer"


def test_sequence_sweep_gate_blocks_consistent_loser() -> None:
    seq_sweep = {
        "champion_name": "partial_untied",
        "candidate_name": "learned_chunk_writer",
        "lengths": {
            "64": {"reports": {"partial_untied": {"final_val_loss": 7.0}, "learned_chunk_writer": {"final_val_loss": 7.1}}},
            "96": {"reports": {"partial_untied": {"final_val_loss": 6.9}, "learned_chunk_writer": {"final_val_loss": 7.0}}},
            "127": {"reports": {"partial_untied": {"final_val_loss": 6.8}, "learned_chunk_writer": {"final_val_loss": 6.95}}},
        },
    }
    gate = _sequence_sweep_gate(seq_sweep)
    assert gate["wins"] == 0
    assert gate["passed"] is False
