from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from deep_wave_delta import DeepWaveDeltaConfig, DeepWaveDeltaLM, config_from_preset


def tiny_config(**overrides) -> DeepWaveDeltaConfig:
    values = {
        "vocab_size": 97,
        "dim": 32,
        "depth": 4,
        "ffn_expansion": 2,
        "output_rank": 16,
        "local_memory_rank": 8,
        "local_memory_kernel": 7,
        "conv_kernels": (3, 5),
        "recurrent_layers": 2,
        "recurrent_heads": 2,
        "key_dim": 8,
        "value_dim": 8,
        "min_half_life": 4.0,
        "max_half_life": 32.0,
        "dropout": 0.0,
        "use_fused_delta": False,
    }
    values.update(overrides)
    return DeepWaveDeltaConfig(**values)


def test_forward_backward_is_finite() -> None:
    torch.manual_seed(13)
    model = DeepWaveDeltaLM(tiny_config())
    tokens = torch.randint(0, 97, (2, 11))
    targets = torch.randint(0, 97, (2, 11))
    logits = model(tokens)
    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert logits.shape == (2, 11, 97)
    assert torch.isfinite(logits).all() and torch.isfinite(loss)
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


def test_model_is_causal() -> None:
    torch.manual_seed(13)
    model = DeepWaveDeltaLM(tiny_config()).eval()
    prefix = torch.randint(0, 97, (2, 7))
    left = torch.cat((prefix, torch.randint(0, 97, (2, 5))), dim=1)
    right = torch.cat((prefix, torch.randint(0, 97, (2, 5))), dim=1)
    with torch.no_grad():
        left_logits = model(left)[:, : prefix.size(1)]
        right_logits = model(right)[:, : prefix.size(1)]
    torch.testing.assert_close(left_logits, right_logits, atol=2e-5, rtol=2e-5)


def test_cached_inference_matches_parallel_sequence() -> None:
    torch.manual_seed(13)
    model = DeepWaveDeltaLM(tiny_config()).eval()
    tokens = torch.randint(0, 97, (2, 13))
    with torch.no_grad():
        parallel = model(tokens)
        cache = None
        pieces = []
        for position in range(tokens.size(1)):
            logits, cache = model.step(tokens[:, position], cache)
            pieces.append(logits)
        cached = torch.cat(pieces, dim=1)
    torch.testing.assert_close(parallel, cached, atol=4e-5, rtol=4e-5)


def test_state_and_parameter_reports_are_honest() -> None:
    model = DeepWaveDeltaLM(tiny_config())
    report = model.parameter_report()
    assert report["total"] == sum(parameter.numel() for parameter in model.parameters())
    assert report["recurrent_state_values"] == 2 * 2 * 8 * 8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FLA fused kernel requires CUDA")
def test_fused_delta_matches_reference() -> None:
    pytest.importorskip("fla")
    torch.manual_seed(13)
    config = tiny_config(use_fused_delta=True)
    # Keep parameters in fp32 so autocast exercises the production mixed-precision boundary.
    model = DeepWaveDeltaLM(config).cuda().eval()
    tokens = torch.randint(0, config.vocab_size, (2, 257), device="cuda")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        recurrent = next(block.recurrent for block in model.blocks if block.recurrent is not None)
        projected = recurrent._project(model.embedding(tokens))
        assert all(value.dtype == torch.bfloat16 for value in projected)
        fused = model(tokens)
        for block in model.blocks:
            if block.recurrent is not None:
                block.recurrent.use_fused = False
        reference = model(tokens)
    torch.testing.assert_close(fused.float(), reference.float(), atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("preset", ["10m", "100m", "350m"])
def test_presets_construct_and_report(preset: str, tmp_path: Path) -> None:
    config = config_from_preset(preset)
    model = DeepWaveDeltaLM(config)
    report = model.parameter_report()
    assert report["total"] > 0
    assert report["recurrent_state_values"] == (
        config.recurrent_layers * config.recurrent_heads * config.key_dim * config.value_dim
    )
    (tmp_path / f"{preset}.json").write_text(json.dumps(report), encoding="utf-8")
