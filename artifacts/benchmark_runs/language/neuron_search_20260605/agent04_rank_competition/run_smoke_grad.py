from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ARTIFACT_DIR = Path(__file__).resolve().parent
TRAINER_PATH = ARTIFACT_DIR / "rank_competition_train.py"


def load_trainer():
    spec = importlib.util.spec_from_file_location("rank_competition_train", TRAINER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load trainer from {TRAINER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def finite_grad_stats(model: torch.nn.Module) -> dict[str, float | int | bool]:
    finite = True
    grad_norm_sq = 0.0
    grad_params = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        grad_params += 1
        finite = finite and bool(torch.isfinite(parameter.grad).all().item())
        grad_norm_sq += float(parameter.grad.detach().float().square().sum().item())
    return {
        "finite": finite,
        "grad_params": grad_params,
        "grad_norm": math.sqrt(grad_norm_sq),
    }


def run_model_check(module, block_type: str) -> dict[str, float | int | bool | str]:
    torch.manual_seed(123)
    config = module.TrainConfig(
        cache_path=ARTIFACT_DIR / "unused.pt",
        output_dir=ARTIFACT_DIR / "tests" / block_type,
        run_name=f"grad_{block_type}",
        vocab_size=512,
        sequence_length=31,
        batch_size=2,
        train_steps=1,
        val_blocks=1,
        embedding_dim=64,
        block_type=block_type,
        conv_layers=2,
        conv_rank=32,
        conv_kernel_size=7,
        memory_rank=16,
        landmark_stride=8,
        sampled_vocab_size=128,
        token_stride=4,
        token_chunk_size=128,
        full_eval_token_chunk_size=128,
    )
    model = module.CausalConvFactorizedLM(config)
    model.train()
    inputs = torch.randint(0, config.vocab_size, (config.batch_size, config.sequence_length), dtype=torch.long)
    targets = torch.randint(0, config.vocab_size, (config.batch_size, config.sequence_length), dtype=torch.long)
    hidden = model.factor_down(model.features(inputs))
    logits = model.factor_up(hidden)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    if not torch.isfinite(loss):
        raise RuntimeError(f"{block_type} produced non-finite loss")
    loss.backward()
    grad_stats = finite_grad_stats(model)
    return {
        "block_type": block_type,
        "loss": float(loss.item()),
        "parameter_count": int(module.count_parameters(model)),
        **grad_stats,
    }


def count_screen_params(module, block_type: str) -> int:
    config = module.TrainConfig(
        cache_path=ARTIFACT_DIR / "unused.pt",
        output_dir=ARTIFACT_DIR / "tests" / f"params_{block_type}",
        run_name=f"params_{block_type}",
        sequence_length=255,
        embedding_dim=192,
        block_type=block_type,
        conv_layers=2,
        conv_rank=96,
        conv_kernel_size=7,
        memory_rank=32,
        landmark_stride=64,
        sampled_vocab_size=4096,
        token_stride=4,
        token_chunk_size=512,
        full_eval_token_chunk_size=512,
    )
    return int(module.count_parameters(module.CausalConvFactorizedLM(config)))


def main() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    module = load_trainer()
    results = {
        "cuda_available": bool(torch.cuda.is_available()),
        "checks": [
            run_model_check(module, "multi_scale_lowrank_conv_memory"),
            run_model_check(module, "rank_competition_lowrank_conv_memory"),
            run_model_check(module, "energy_preserving_rank_competition_lowrank_conv_memory"),
        ],
        "screen_parameter_counts": {
            "multi_scale_lowrank_conv_memory": count_screen_params(module, "multi_scale_lowrank_conv_memory"),
            "rank_competition_lowrank_conv_memory": count_screen_params(
                module,
                "rank_competition_lowrank_conv_memory",
            ),
            "energy_preserving_rank_competition_lowrank_conv_memory": count_screen_params(
                module,
                "energy_preserving_rank_competition_lowrank_conv_memory",
            ),
        },
    }
    out_path = ARTIFACT_DIR / "tests" / "smoke_grad_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
