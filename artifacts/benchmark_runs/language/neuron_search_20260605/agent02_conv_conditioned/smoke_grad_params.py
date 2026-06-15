from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
TRAINER_PATH = ROOT / "standalone_longseq_anchor_train_conv_conditioned.py"


def load_trainer():
    spec = importlib.util.spec_from_file_location("conv_conditioned_trainer", TRAINER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {TRAINER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def check_grad(module, block_type: str) -> dict[str, float | int | bool | str]:
    torch.manual_seed(1234)
    config = module.TrainConfig(
        cache_path=Path("unused.pt"),
        output_dir=ROOT / "unused",
        run_name=f"grad_{block_type}",
        vocab_size=503,
        sequence_length=31,
        batch_size=2,
        train_steps=1,
        eval_interval=1,
        val_blocks=1,
        embedding_dim=64,
        block_type=block_type,
        conv_layers=2,
        conv_rank=32,
        conv_kernel_size=7,
        memory_rank=8,
        landmark_stride=8,
        sampled_vocab_size=128,
        token_stride=4,
        token_chunk_size=128,
        full_eval_token_chunk_size=128,
    )
    model = module.CausalConvFactorizedLM(config).cpu()
    input_ids = torch.randint(0, config.vocab_size, (config.batch_size, config.sequence_length), dtype=torch.long)
    targets = torch.randint(0, config.vocab_size, (config.batch_size, config.sequence_length), dtype=torch.long)
    hidden = model.factor_down(model.features(input_ids))
    logits = model.factor_up(hidden)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    loss.backward()
    grad_sq = 0.0
    tensors_seen = 0
    finite = True
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        tensors_seen += 1
        finite = finite and bool(torch.isfinite(parameter.grad).all().item())
        grad_sq += float(parameter.grad.float().square().sum().item())
    return {
        "block_type": block_type,
        "loss": float(loss.detach().item()),
        "parameter_count": int(module.count_parameters(model)),
        "grad_tensors": tensors_seen,
        "grad_norm": grad_sq ** 0.5,
        "finite_loss": bool(torch.isfinite(loss).item()),
        "finite_gradients": finite,
        "cuda_available": bool(torch.cuda.is_available()),
    }


def count_screen_params(module, block_type: str) -> int:
    config = module.TrainConfig(
        cache_path=Path("unused.pt"),
        output_dir=ROOT / "unused",
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
        learning_rate=0.0006,
        min_learning_rate=0.00001,
    )
    model = module.CausalConvFactorizedLM(config).cpu()
    return int(module.count_parameters(model))


def main() -> None:
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    module = load_trainer()
    baseline = "multi_scale_lowrank_conv_memory"
    candidate = "branch_disagreement_conditioned_lowrank_conv_memory"
    candidate_v2 = "branch_disagreement_gain_only_lowrank_conv_memory"
    result = {
        "trainer_path": str(TRAINER_PATH),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "checks": {
            baseline: check_grad(module, baseline),
            candidate: check_grad(module, candidate),
            candidate_v2: check_grad(module, candidate_v2),
        },
        "screen_parameter_counts": {
            baseline: count_screen_params(module, baseline),
            candidate: count_screen_params(module, candidate),
            candidate_v2: count_screen_params(module, candidate_v2),
        },
    }
    out_path = ROOT / "smoke_grad_params.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
