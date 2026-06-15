import importlib.util
import json
import os
import sys
from pathlib import Path

import torch


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

ARTIFACT_DIR = Path(__file__).resolve().parent
TRAINER_PATH = ARTIFACT_DIR / "adaptive_basis_train.py"


def load_trainer():
    spec = importlib.util.spec_from_file_location("adaptive_basis_train", TRAINER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {TRAINER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def finite_grad_summary(module: torch.nn.Module, loss: torch.Tensor) -> dict[str, float | int | bool]:
    loss.backward()
    total_params = 0
    grad_params = 0
    nonzero_grad_params = 0
    for parameter in module.parameters():
        if not parameter.requires_grad:
            continue
        total_params += parameter.numel()
        if parameter.grad is None:
            continue
        grad_params += parameter.numel()
        if not torch.isfinite(parameter.grad).all():
            return {
                "ok": False,
                "loss": float(loss.detach().item()),
                "total_params": total_params,
                "grad_params": grad_params,
                "nonzero_grad_params": nonzero_grad_params,
            }
        nonzero_grad_params += int((parameter.grad != 0).sum().item())
    return {
        "ok": True,
        "loss": float(loss.detach().item()),
        "total_params": total_params,
        "grad_params": grad_params,
        "nonzero_grad_params": nonzero_grad_params,
    }


def block_grad_check(trainer, block_cls) -> dict[str, float | int | bool]:
    torch.manual_seed(123)
    block = block_cls(
        dim=32,
        expansion=2,
        kernel_size=5,
        dilation=2,
        dropout=0.0,
        memory_rank=8,
        memory_kernel_size=9,
    )
    x = torch.randn(2, 17, 32)
    y = block(x)
    if y.shape != x.shape:
        raise RuntimeError(f"bad block output shape {tuple(y.shape)}")
    loss = y.square().mean()
    return finite_grad_summary(block, loss)


def model_grad_check(trainer, block_type: str) -> dict[str, float | int | bool]:
    torch.manual_seed(321)
    config = trainer.TrainConfig(
        cache_path=Path("unused.pt"),
        output_dir=Path("unused"),
        run_name=f"grad_{block_type}",
        vocab_size=128,
        sequence_length=31,
        train_steps=1,
        val_blocks=1,
        embedding_dim=32,
        block_type=block_type,
        conv_layers=2,
        conv_rank=16,
        conv_kernel_size=5,
        memory_rank=8,
        landmark_stride=9,
        sampled_vocab_size=32,
        token_stride=4,
        token_chunk_size=64,
        full_eval_token_chunk_size=64,
    )
    model = trainer.CausalConvFactorizedLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, config.sequence_length))
    targets = torch.randint(0, config.vocab_size, (2, config.sequence_length))
    fixed_candidate_ids = torch.arange(config.sampled_vocab_size)
    loss, token_count, candidate_size = trainer.anchor_loss(
        model,
        input_ids,
        targets,
        fixed_candidate_ids=fixed_candidate_ids,
        config=config,
    )
    summary = finite_grad_summary(model, loss)
    summary["token_count"] = token_count
    summary["candidate_size"] = candidate_size
    return summary


def mini_config(trainer, block_type: str):
    return trainer.TrainConfig(
        cache_path=Path("unused.pt"),
        output_dir=Path("unused"),
        run_name=f"params_{block_type}",
        sequence_length=255,
        train_steps=128,
        eval_interval=128,
        checkpoint_interval=0,
        milestone_checkpoint_interval=0,
        val_blocks=8,
        embedding_dim=192,
        block_type=block_type,
        conv_layers=2,
        conv_kernel_size=7,
        conv_rank=96,
        memory_rank=32,
        landmark_stride=64,
        sampled_vocab_size=4096,
        token_stride=4,
        token_chunk_size=512,
        full_eval_token_chunk_size=512,
        learning_rate=0.0006,
        min_learning_rate=0.00001,
    )


def main() -> None:
    trainer = load_trainer()
    baseline_type = "multi_scale_lowrank_conv_memory"
    candidate_type = "multi_scale_lowrank_conv_memory_adaptive_basis"
    candidate_v2_type = "multi_scale_lowrank_conv_memory_ffn_adaptive_basis"
    baseline_block = trainer.CausalMultiScaleLowRankConvMemoryBlock
    candidate_block = trainer.CausalMultiScaleLowRankConvMemoryAdaptiveBasisBlock
    candidate_v2_block = trainer.CausalMultiScaleLowRankConvMemoryFFNAdaptiveBasisBlock

    results = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_cuda_available": torch.cuda.is_available(),
        "baseline_block_grad": block_grad_check(trainer, baseline_block),
        "candidate_block_grad": block_grad_check(trainer, candidate_block),
        "candidate_v2_block_grad": block_grad_check(trainer, candidate_v2_block),
        "baseline_model_grad": model_grad_check(trainer, baseline_type),
        "candidate_model_grad": model_grad_check(trainer, candidate_type),
        "candidate_v2_model_grad": model_grad_check(trainer, candidate_v2_type),
    }
    baseline_model = trainer.CausalConvFactorizedLM(mini_config(trainer, baseline_type))
    candidate_model = trainer.CausalConvFactorizedLM(mini_config(trainer, candidate_type))
    candidate_v2_model = trainer.CausalConvFactorizedLM(mini_config(trainer, candidate_v2_type))
    results["baseline_mini_params"] = trainer.count_parameters(baseline_model)
    results["candidate_mini_params"] = trainer.count_parameters(candidate_model)
    results["candidate_v2_mini_params"] = trainer.count_parameters(candidate_v2_model)
    results["candidate_param_delta"] = results["candidate_mini_params"] - results["baseline_mini_params"]
    results["candidate_v2_param_delta"] = results["candidate_v2_mini_params"] - results["baseline_mini_params"]

    output_path = ARTIFACT_DIR / "smoke_grad_params.json"
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
