from __future__ import annotations

import json
import os
import py_compile
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from linear_attention_memory import (
    CausalChunkedLinearAttentionMemory,
    CausalMultiScaleLowRankConvMemoryLinearAttentionBlock,
    count_parameters,
    linear_attention_cost,
)


def assert_all_finite(name: str, tensor: torch.Tensor | None) -> None:
    if tensor is None:
        raise AssertionError(f"{name} is None")
    if not bool(torch.isfinite(tensor).all()):
        raise AssertionError(f"{name} has non-finite values")


def check_module_backward(module: torch.nn.Module, x: torch.Tensor) -> dict[str, Any]:
    module.eval()
    x = x.detach().requires_grad_(True)
    y = module(x)
    if y.shape != x.shape:
        raise AssertionError(f"shape mismatch: got {tuple(y.shape)}, expected {tuple(x.shape)}")
    loss = y.square().mean()
    loss.backward()
    assert_all_finite("input_grad", x.grad)
    grad_norms: dict[str, float] = {}
    for name, parameter in module.named_parameters():
        assert_all_finite(f"grad:{name}", parameter.grad)
        grad_norms[name] = float(parameter.grad.norm().item())
    return {
        "loss": float(loss.item()),
        "input_grad_norm": float(x.grad.norm().item()),
        "parameter_grad_norms_sample": dict(list(grad_norms.items())[:8]),
    }


def check_causal_prefix(module: torch.nn.Module, x: torch.Tensor, prefix_tokens: int) -> float:
    module.eval()
    with torch.no_grad():
        y1 = module(x)
        x2 = x.clone()
        x2[:, prefix_tokens:, :] = x2[:, prefix_tokens:, :] + torch.randn_like(x2[:, prefix_tokens:, :]) * 3.0
        y2 = module(x2)
    return float((y1[:, :prefix_tokens, :] - y2[:, :prefix_tokens, :]).abs().max().item())


def check_directional_finite_difference(module: torch.nn.Module, x: torch.Tensor) -> dict[str, float]:
    module.eval()
    x = x.detach().requires_grad_(True)

    def objective(input_tensor: torch.Tensor) -> torch.Tensor:
        return module(input_tensor).square().mean()

    loss = objective(x)
    loss.backward()
    direction = torch.randn_like(x)
    direction = direction / direction.norm()
    analytical = torch.sum(x.grad * direction)
    eps = 1e-6
    with torch.no_grad():
        plus = objective(x + eps * direction)
        minus = objective(x - eps * direction)
    finite_difference = (plus - minus) / (2.0 * eps)
    relative_error = (analytical - finite_difference).abs() / finite_difference.abs().clamp_min(1e-12)
    return {
        "analytical_directional_grad": float(analytical.item()),
        "finite_difference_directional_grad": float(finite_difference.item()),
        "relative_error": float(relative_error.item()),
    }


def main() -> None:
    torch.manual_seed(20260616)
    torch.set_num_threads(1)
    script_dir = Path(__file__).resolve().parent

    primitive = CausalChunkedLinearAttentionMemory(
        dim=17,
        feature_rank=5,
        value_rank=7,
        chunk_size=4,
        dropout=0.0,
    ).double()
    primitive_x = torch.randn(2, 9, 17, dtype=torch.float64)
    primitive_backward = check_module_backward(primitive, primitive_x)
    primitive_causal_max_diff = check_causal_prefix(primitive, primitive_x, prefix_tokens=5)
    primitive_fd = check_directional_finite_difference(primitive, primitive_x)
    if primitive_causal_max_diff > 1e-10:
        raise AssertionError(f"primitive causality diff too high: {primitive_causal_max_diff}")
    if primitive_fd["relative_error"] > 1e-4:
        raise AssertionError(f"finite-difference relative error too high: {primitive_fd['relative_error']}")

    block = CausalMultiScaleLowRankConvMemoryLinearAttentionBlock(
        dim=17,
        expansion=2,
        kernel_size=3,
        dilation=2,
        dropout=0.0,
        memory_rank=5,
        memory_kernel_size=4,
        attention_feature_rank=4,
        attention_value_rank=6,
        attention_chunk_size=5,
    ).double()
    block_x = torch.randn(2, 11, 17, dtype=torch.float64)
    block_backward = check_module_backward(block, block_x)
    block_causal_max_diff = check_causal_prefix(block, block_x, prefix_tokens=6)
    if block_causal_max_diff > 1e-10:
        raise AssertionError(f"block causality diff too high: {block_causal_max_diff}")

    default_dim = 1831
    default_feature_rank = 32
    default_value_rank = 64
    default_chunk_size = 1024
    default_tokens = 10160
    default_primitive = CausalChunkedLinearAttentionMemory(
        dim=default_dim,
        feature_rank=default_feature_rank,
        value_rank=default_value_rank,
        chunk_size=default_chunk_size,
        dropout=0.0,
    )
    default_full_block = CausalMultiScaleLowRankConvMemoryLinearAttentionBlock(
        dim=default_dim,
        expansion=2,
        kernel_size=7,
        dilation=1,
        dropout=0.0,
        memory_rank=64,
        memory_kernel_size=128,
        attention_feature_rank=default_feature_rank,
        attention_value_rank=default_value_rank,
        attention_chunk_size=default_chunk_size,
    )
    py_compile.compile(str(script_dir / "linear_attention_memory.py"), doraise=True)
    py_compile.compile(str(script_dir / "test_smoke.py"), doraise=True)

    result = {
        "status": "passed",
        "device": "cpu",
        "torch_version": torch.__version__,
        "tests": {
            "syntax_import_smoke": {
                "module_imported": True,
                "py_compile": True,
            },
            "primitive_forward_backward_finite_gradients": primitive_backward,
            "primitive_causal_prefix_max_abs_diff": primitive_causal_max_diff,
            "primitive_directional_finite_difference": primitive_fd,
            "block_forward_backward_finite_gradients": block_backward,
            "block_causal_prefix_max_abs_diff": block_causal_max_diff,
        },
        "parameter_counts": {
            "small_primitive": count_parameters(primitive),
            "small_full_block": count_parameters(block),
            "default_dim1831_rank32_value64_primitive": count_parameters(default_primitive),
            "default_dim1831_memory64_full_block_with_linear_attention": count_parameters(default_full_block),
        },
        "cost_estimate_default_seq10160": linear_attention_cost(
            dim=default_dim,
            tokens=default_tokens,
            feature_rank=default_feature_rank,
            value_rank=default_value_rank,
            chunk_size=default_chunk_size,
        ),
    }
    (script_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
