from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import torch

from arc_tactic3 import language_candidate_rollout_objective as candidate


def _make_cache(path: Path, *, vocab_size: int = 64, sequence_length: int = 7) -> None:
    block_size = sequence_length + 1
    train_tokens = torch.arange(block_size * 32, dtype=torch.long) % vocab_size
    val_tokens = torch.arange(block_size * 32, block_size * 40, dtype=torch.long) % vocab_size
    torch.save({"train_tokens": train_tokens, "val_tokens": val_tokens, "vocab_size": vocab_size}, path)


class _ShiftModel(torch.nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits = torch.full(
            (input_ids.size(0), input_ids.size(1), self.vocab_size),
            fill_value=-20.0,
            dtype=torch.float32,
            device=input_ids.device,
        )
        next_ids = (input_ids + 1) % self.vocab_size
        logits.scatter_(2, next_ids.unsqueeze(-1), 20.0)
        return logits


def test_rollout_continuation_tracks_continuation_targets() -> None:
    vocab_size = 32
    input_ids = torch.tensor([[3, 4, 5, 6, 7]], dtype=torch.long)
    targets = torch.tensor([[4, 5, 6, 7, 8]], dtype=torch.long)
    logits, rollout_targets, rollout_mask = candidate._rollout_continuation(
        _ShiftModel(vocab_size),
        input_ids=input_ids,
        targets=targets,
        prefix_length=3,
    )
    assert logits.shape == (1, 3, vocab_size)
    assert rollout_targets.tolist() == [[6, 7, 8]]
    assert torch.equal(rollout_mask, torch.ones_like(rollout_targets, dtype=logits.dtype))
    hits = candidate._sequence_exact_hits(logits, rollout_targets, rollout_mask)
    assert hits.tolist() == [1.0]


def test_run_rollout_objective_compare_smoke_cpu() -> None:
    temp_dir = Path(tempfile.mkdtemp(dir=Path.cwd()))
    cache_path = temp_dir / "cache.pt"
    _make_cache(cache_path)
    try:
        partial_config = candidate.RolloutObjectiveConfig(
            cache_path=cache_path,
            model_variant="partial_untied",
            device="cpu",
            use_amp=False,
            pin_memory=False,
            use_fused_adamw=False,
            sequence_length=7,
            train_blocks=16,
            val_blocks=4,
            batch_size=4,
            eval_batch_size=4,
            train_steps=2,
            eval_interval=1,
            recurrent_embedding_dim=16,
            recurrent_hidden_dim=24,
            recurrent_memory_dim=16,
            partial_untied_tokens=16,
            rollout_prefix_length=3,
        )
        partial_payload = candidate.run_rollout_objective_compare(
            partial_config,
            objective_modes=("ce_only", "ce_plus_both"),
        )
        champion_config = candidate.RolloutObjectiveConfig(
            cache_path=cache_path,
            model_variant="recurrent_champion",
            device="cpu",
            use_amp=False,
            pin_memory=False,
            use_fused_adamw=False,
            sequence_length=7,
            train_blocks=16,
            val_blocks=4,
            batch_size=4,
            eval_batch_size=4,
            train_steps=1,
            eval_interval=1,
            recurrent_embedding_dim=16,
            recurrent_hidden_dim=24,
            recurrent_memory_dim=16,
            rollout_prefix_length=3,
        )
        champion_payload = candidate.run_rollout_objective_compare(
            champion_config,
            objective_modes=("ce_only",),
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    assert partial_payload["benchmark"] == "language_candidate_rollout_objective"
    assert partial_payload["objective_modes"] == ["ce_only", "ce_plus_both"]
    assert set(partial_payload["results"]) == {"ce_only", "ce_plus_both"}
    for report in partial_payload["results"].values():
        assert report["parameter_count"] > 0
        assert report["final_val_loss"] >= 0.0
        assert report["final_val_rollout_loss"] >= 0.0
        assert report["train_tokens_seen"] > 0
        assert report["history"]
    assert champion_payload["config"]["model_variant"] == "recurrent_champion"
    assert set(champion_payload["results"]) == {"ce_only"}
