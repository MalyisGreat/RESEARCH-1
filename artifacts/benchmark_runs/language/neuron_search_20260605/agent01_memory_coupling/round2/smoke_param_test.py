import json
from pathlib import Path

import torch
import torch.nn.functional as F

import agent01_memory_coupling_trainer as trainer


def make_config(block_type: str) -> trainer.TrainConfig:
    return trainer.TrainConfig(
        cache_path=Path("unused.pt"),
        output_dir=Path("."),
        run_name=block_type,
        vocab_size=257,
        sequence_length=31,
        train_steps=1,
        eval_interval=1,
        checkpoint_interval=0,
        milestone_checkpoint_interval=0,
        val_blocks=1,
        embedding_dim=32,
        conv_layers=1,
        conv_kernel_size=7,
        conv_rank=16,
        memory_rank=8,
        landmark_stride=8,
        sampled_vocab_size=128,
        token_stride=4,
        token_chunk_size=64,
        full_eval_token_chunk_size=64,
        block_type=block_type,
    )


def run_case(block_type: str) -> dict:
    torch.manual_seed(123)
    config = make_config(block_type)
    model = trainer.CausalConvFactorizedLM(config)
    model.train()
    input_ids = torch.randint(0, config.vocab_size, (2, config.sequence_length), dtype=torch.long)
    targets = torch.randint(0, config.vocab_size, (2, config.sequence_length), dtype=torch.long)
    features = model.features(input_ids)
    logits = model.factor_up(model.factor_down(features))
    loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), targets.reshape(-1))
    loss.backward()
    grads = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    finite_grads = all(grad is not None and torch.isfinite(grad).all().item() for grad in grads)
    nonzero_grad_tensors = sum(int(grad is not None and grad.abs().sum().item() > 0.0) for grad in grads)
    return {
        "block_type": block_type,
        "parameter_count": trainer.count_parameters(model),
        "loss": float(loss.detach().item()),
        "finite_loss": bool(torch.isfinite(loss).item()),
        "finite_grads": bool(finite_grads),
        "nonzero_grad_tensors": nonzero_grad_tensors,
        "total_grad_tensors": len(grads),
    }


def main() -> None:
    results = {
        "baseline": run_case("multi_scale_lowrank_conv_memory"),
        "candidate": run_case("memory_threshold_basis_lowrank"),
    }
    Path("smoke_param_test.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
