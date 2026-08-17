from __future__ import annotations

import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hrm_text_stable_wave_train as experiment


def main() -> None:
    config = experiment.trainer.TrainConfig(
        cache_path=Path("unused"), output_dir=Path("unused"), run_name="smoke",
        vocab_size=128, sequence_length=32, train_steps=10,
        embedding_dim=32, conv_rank=16, memory_rank=8,
        attention_heads=4, landmark_stride=8,
    )
    model = experiment.StableHRMTextWaveLM(config)
    inputs = torch.randint(0, config.vocab_size, (2, config.sequence_length))
    logits = model.factor_up(model.factor_down(model.features(inputs)))
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, config.vocab_size), inputs.reshape(-1))
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(parameter.grad is not None for parameter in model.low_core.parameters())
    print(f"stable HRM smoke passed: loss={loss.item():.4f}, bp_steps={model.current_bp_steps()}")


if __name__ == "__main__":
    main()
