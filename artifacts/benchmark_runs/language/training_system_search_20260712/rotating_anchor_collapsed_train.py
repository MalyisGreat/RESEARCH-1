from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


COLLAPSED_TRAINER = Path(__file__).with_name("collapsed_token_recall_train.py")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


collapsed = load_module("collapsed_trainer_with_rotating_anchors", COLLAPSED_TRAINER)
trainer = collapsed.trainer
anchor_call_count = 0


def candidate_ids_with_targets_gpu(fixed_candidate_ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    target_ids = targets.detach().reshape(-1).unique(sorted=True)
    fixed_ids = fixed_candidate_ids.detach().to(targets.device, non_blocking=True)
    return torch.cat((fixed_ids, target_ids)).unique(sorted=True)


def rotating_anchor_loss(
    model,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    *,
    fixed_candidate_ids: torch.Tensor,
    config,
):
    global anchor_call_count
    candidate_ids = candidate_ids_with_targets_gpu(fixed_candidate_ids, targets)
    candidate_map = torch.full((config.vocab_size,), -1, dtype=torch.long, device=targets.device)
    candidate_map[candidate_ids] = torch.arange(candidate_ids.numel(), dtype=torch.long, device=targets.device)
    reduced_targets = candidate_map[targets.reshape(-1)].view_as(targets)
    if bool((reduced_targets < 0).any()):
        raise RuntimeError("candidate set missed a target token")
    hidden = model.factor_down(model.features(input_ids))
    candidate_weight = model.factor_up.weight.index_select(0, candidate_ids)
    candidate_bias = model.factor_up.bias.index_select(0, candidate_ids) if model.factor_up.bias is not None else None
    sampled_positions = torch.arange(targets.size(1), device=targets.device, dtype=torch.long)
    sampled_sum = trainer.model_cross_entropy_sum_chunked(
        model,
        hidden,
        input_ids,
        reduced_targets,
        candidate_ids,
        candidate_weight,
        candidate_bias,
        positions=sampled_positions,
        token_chunk_size=config.token_chunk_size,
    )
    sampled_loss = sampled_sum / targets.numel()

    offset = anchor_call_count % config.token_stride
    anchor_call_count += 1
    anchor_hidden = hidden[:, offset:: config.token_stride, :]
    anchor_targets = targets[:, offset:: config.token_stride]
    anchor_positions = torch.arange(offset, targets.size(1), config.token_stride, device=targets.device, dtype=torch.long)
    anchor_sum = trainer.model_cross_entropy_sum_chunked(
        model,
        anchor_hidden,
        input_ids,
        anchor_targets,
        torch.arange(config.vocab_size, device=targets.device, dtype=torch.long),
        model.factor_up.weight,
        model.factor_up.bias,
        positions=anchor_positions,
        token_chunk_size=config.token_chunk_size,
    )
    full_anchor_loss = anchor_sum / anchor_targets.numel()
    return 0.5 * (sampled_loss + full_anchor_loss), int(targets.numel()), int(candidate_ids.numel())


trainer.anchor_loss = rotating_anchor_loss


if __name__ == "__main__":
    trainer.train(trainer.parse_args())
