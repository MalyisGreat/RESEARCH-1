from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from torch import nn


ROOT = Path(__file__).parents[1]
TOKEN_RECALL_TRAINER = ROOT / "token_recall_search_20260616" / "token_recall_train.py"
FUSION_PROBE = Path(__file__).with_name("fused_wave10_probe.py")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


trainer = load_module("token_recall_train_collapsed_experiment", TOKEN_RECALL_TRAINER)
fusion = load_module("wave10_collapse_for_training", FUSION_PROBE)
OriginalWave10Block = trainer.CausalMultiScaleLowRankConvMemoryBlock


class CollapsedWave10DropIn(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        expansion: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        memory_rank: int,
        memory_kernel_size: int,
    ) -> None:
        super().__init__()
        original = OriginalWave10Block(
            dim=dim,
            expansion=expansion,
            kernel_size=kernel_size,
            dilation=dilation,
            dropout=dropout,
            memory_rank=memory_rank,
            memory_kernel_size=memory_kernel_size,
        )
        self.block = fusion.CollapsedWave10Block(original, dilation=dilation)

    def forward(self, x):
        return self.block(x)


trainer.CausalMultiScaleLowRankConvMemoryBlock = CollapsedWave10DropIn


if __name__ == "__main__":
    trainer.train(trainer.parse_args())
