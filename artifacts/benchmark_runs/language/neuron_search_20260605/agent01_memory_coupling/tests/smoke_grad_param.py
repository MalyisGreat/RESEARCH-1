from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = ROOT / "standalone_longseq_anchor_train_memory_coupling.py"


def load_trainer():
    spec = importlib.util.spec_from_file_location("memory_coupling_trainer", TRAINER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load trainer from {TRAINER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def finite_gradient_check(module, block_type: str) -> dict[str, object]:
    torch.manual_seed(20260605)
    config = module.TrainConfig(
        cache_path=ROOT / "unused.pt",
        output_dir=ROOT / "unused",
        run_name=f"grad_{block_type}",
        vocab_size=257,
        sequence_length=15,
        embedding_dim=24,
        block_type=block_type,
        conv_layers=1,
        conv_kernel_size=5,
        conv_rank=16,
        memory_rank=8,
        landmark_stride=4,
    )
    block = module.make_mixer_block(config, 0).to("cpu")
    block.train()
    x = torch.randn(2, 15, 24, requires_grad=True)
    output = block(x)
    loss = output.square().mean() + 0.01 * output[:, -1].abs().mean()
    loss.backward()
    parameter_grads = [
        parameter.grad
        for parameter in block.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    finite_parameter_grads = all(bool(torch.isfinite(grad).all().item()) for grad in parameter_grads)
    nonzero_parameter_grads = sum(int(grad.abs().sum().item() > 0.0) for grad in parameter_grads)
    return {
        "block_type": block_type,
        "output_shape": list(output.shape),
        "loss": float(loss.item()),
        "finite_input_grad": bool(torch.isfinite(x.grad).all().item()) if x.grad is not None else False,
        "finite_parameter_grads": finite_parameter_grads,
        "nonzero_parameter_grads": nonzero_parameter_grads,
        "parameter_grad_tensors": len(parameter_grads),
    }


def mini_model_params(module, block_type: str) -> int:
    config = module.TrainConfig(
        cache_path=ROOT / "unused.pt",
        output_dir=ROOT / "unused",
        run_name=f"params_{block_type}",
        vocab_size=50_257,
        sequence_length=255,
        embedding_dim=192,
        block_type=block_type,
        conv_layers=2,
        conv_kernel_size=7,
        conv_rank=96,
        memory_rank=32,
        landmark_stride=64,
        sampled_vocab_size=4_096,
        token_stride=4,
        token_chunk_size=512,
        full_eval_token_chunk_size=512,
        learning_rate=0.0006,
        min_learning_rate=0.00001,
    )
    model = module.CausalConvFactorizedLM(config).to("cpu")
    return int(module.count_parameters(model))


def main() -> None:
    module = load_trainer()
    baseline = "multi_scale_lowrank_conv_memory"
    candidates = [
        "memory_selected_threshold_basis_lowrank_conv_memory",
        "memory_tiled_threshold_lowrank_conv_memory",
    ]
    block_types = [baseline, *candidates]
    result = {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "trainer_path": str(TRAINER_PATH),
        "gradient_checks": {block_type: finite_gradient_check(module, block_type) for block_type in block_types},
        "parameter_counts": {block_type: mini_model_params(module, block_type) for block_type in block_types},
    }
    out_path = ROOT / "tests" / "smoke_grad_param_result.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
