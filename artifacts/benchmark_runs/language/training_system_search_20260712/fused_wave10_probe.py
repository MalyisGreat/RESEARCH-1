from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


TRAINER_PATH = Path(__file__).parents[1] / "standalone_longseq_anchor_train.py"


def load_trainer():
    spec = importlib.util.spec_from_file_location("wave10_base_for_fusion", TRAINER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {TRAINER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FusedDepthwiseBank(nn.Module):
    def __init__(self, layers: nn.ModuleList, left_paddings: list[int], dilation: int) -> None:
        super().__init__()
        if not layers:
            raise ValueError("at least one depthwise layer is required")
        self.dim = layers[0].in_channels
        self.branch_count = len(layers)
        self.dilation = dilation
        self.kernel_size = max(layer.kernel_size[0] for layer in layers)
        self.left_padding = (self.kernel_size - 1) * dilation
        weight = torch.zeros(self.dim * self.branch_count, 1, self.kernel_size)
        bias = torch.zeros(self.dim * self.branch_count)
        mask = torch.zeros_like(weight)
        with torch.no_grad():
            for channel in range(self.dim):
                for branch, layer in enumerate(layers):
                    kernel = layer.kernel_size[0]
                    output_channel = channel * self.branch_count + branch
                    weight[output_channel, 0, self.kernel_size - kernel :] = layer.weight[channel, 0]
                    mask[output_channel, 0, self.kernel_size - kernel :] = 1
                    if layer.bias is not None:
                        bias[output_channel] = layer.bias[channel]
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)
        self.register_buffer("weight_mask", mask, persistent=False)
        expected_paddings = [(layer.kernel_size[0] - 1) * dilation for layer in layers]
        if list(left_paddings) != expected_paddings:
            raise ValueError(f"unexpected paddings: {left_paddings} != {expected_paddings}")

    def forward(self, conv_input: torch.Tensor) -> torch.Tensor:
        output = F.conv1d(
            F.pad(conv_input, (self.left_padding, 0)),
            self.weight * self.weight_mask,
            self.bias,
            dilation=self.dilation,
            groups=self.dim,
        )
        batch, _, tokens = output.shape
        return output.view(batch, self.dim, self.branch_count, tokens).mean(dim=2).transpose(1, 2)


class CollapsedDepthwiseBank(nn.Module):
    """Exact functional collapse of an averaged bank of aligned depthwise convolutions."""

    def __init__(self, layers: nn.ModuleList, left_paddings: list[int], dilation: int) -> None:
        super().__init__()
        if not layers:
            raise ValueError("at least one depthwise layer is required")
        self.dim = layers[0].in_channels
        self.dilation = dilation
        self.kernel_size = max(layer.kernel_size[0] for layer in layers)
        self.left_padding = (self.kernel_size - 1) * dilation
        branch_count = len(layers)
        summed_weight = layers[0].weight.new_zeros(self.dim, 1, self.kernel_size)
        multiplicity = layers[0].weight.new_zeros(1, 1, self.kernel_size)
        biases = []
        with torch.no_grad():
            for layer in layers:
                kernel = layer.kernel_size[0]
                summed_weight[:, :, self.kernel_size - kernel :] += layer.weight
                multiplicity[:, :, self.kernel_size - kernel :] += 1
                if layer.bias is not None:
                    biases.append(layer.bias)
        raw_weight = summed_weight / multiplicity.clamp_min(1)
        self.weight = nn.Parameter(raw_weight)
        self.register_buffer("weight_scale", multiplicity / branch_count, persistent=False)
        self.bias = nn.Parameter(torch.stack(biases).mean(dim=0)) if biases else None
        expected_paddings = [(layer.kernel_size[0] - 1) * dilation for layer in layers]
        if list(left_paddings) != expected_paddings:
            raise ValueError(f"unexpected paddings: {left_paddings} != {expected_paddings}")

    def forward(self, conv_input: torch.Tensor) -> torch.Tensor:
        output = F.conv1d(
            F.pad(conv_input, (self.left_padding, 0)),
            self.weight * self.weight_scale,
            self.bias,
            dilation=self.dilation,
            groups=self.dim,
        )
        return output.transpose(1, 2)


class FusedWave10Block(nn.Module):
    def __init__(self, base: nn.Module, dilation: int) -> None:
        super().__init__()
        self.fused_depthwise = FusedDepthwiseBank(base.depthwise_layers, base.left_paddings, dilation)
        self.memory_left_padding = base.memory_left_padding
        self.conv_norm = copy.deepcopy(base.conv_norm)
        self.mix = copy.deepcopy(base.mix)
        self.memory_norm = copy.deepcopy(base.memory_norm)
        self.memory_down = copy.deepcopy(base.memory_down)
        self.memory_depthwise = copy.deepcopy(base.memory_depthwise)
        self.memory_up = copy.deepcopy(base.memory_up)
        self.ffn_norm = copy.deepcopy(base.ffn_norm)
        self.ffn_in = copy.deepcopy(base.ffn_in)
        self.ffn_out = copy.deepcopy(base.ffn_out)
        self.dropout = copy.deepcopy(base.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_input = self.conv_norm(x).transpose(1, 2)
        conv_output = self.fused_depthwise(conv_input)
        x = x + self.dropout(self.mix(F.relu(conv_output).square()))
        memory_input = self.memory_down(self.memory_norm(x)).transpose(1, 2)
        memory_output = self.memory_depthwise(F.pad(memory_input, (self.memory_left_padding, 0))).transpose(1, 2)
        x = x + self.dropout(self.memory_up(F.silu(memory_output)))
        hidden = F.relu(self.ffn_in(self.ffn_norm(x))).square()
        return x + self.dropout(self.ffn_out(hidden))


class CollapsedWave10Block(FusedWave10Block):
    def __init__(self, base: nn.Module, dilation: int) -> None:
        super().__init__(base, dilation)
        del self.fused_depthwise
        self.collapsed_depthwise = CollapsedDepthwiseBank(base.depthwise_layers, base.left_paddings, dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_input = self.conv_norm(x).transpose(1, 2)
        conv_output = self.collapsed_depthwise(conv_input)
        x = x + self.dropout(self.mix(F.relu(conv_output).square()))
        memory_input = self.memory_down(self.memory_norm(x)).transpose(1, 2)
        memory_output = self.memory_depthwise(F.pad(memory_input, (self.memory_left_padding, 0))).transpose(1, 2)
        x = x + self.dropout(self.memory_up(F.silu(memory_output)))
        hidden = F.relu(self.ffn_in(self.ffn_norm(x))).square()
        return x + self.dropout(self.ffn_out(hidden))


def mapped_fused_gradient(fused: FusedDepthwiseBank, branch: int, kernel: int) -> torch.Tensor:
    rows = []
    for channel in range(fused.dim):
        output_channel = channel * fused.branch_count + branch
        rows.append(fused.weight.grad[output_channel, 0, fused.kernel_size - kernel :])
    return torch.stack(rows).unsqueeze(1)


def equivalence_test(base_module, device: torch.device) -> dict[str, float]:
    torch.manual_seed(13)
    base = base_module.CausalMultiScaleLowRankConvMemoryBlock(
        dim=32,
        expansion=2,
        kernel_size=7,
        dilation=2,
        dropout=0.0,
        memory_rank=8,
        memory_kernel_size=16,
    ).to(device)
    fused = FusedWave10Block(base, dilation=2).to(device)
    base.train()
    fused.train()
    original_input = torch.randn(2, 127, 32, device=device, requires_grad=True)
    fused_input = original_input.detach().clone().requires_grad_(True)
    original_output = base(original_input)
    fused_output = fused(fused_input)
    original_loss = original_output.square().mean()
    fused_loss = fused_output.square().mean()
    original_loss.backward()
    fused_loss.backward()
    weight_diffs = []
    bias_diffs = []
    for branch, layer in enumerate(base.depthwise_layers):
        fused_grad = mapped_fused_gradient(fused.fused_depthwise, branch, layer.kernel_size[0])
        weight_diffs.append(float((layer.weight.grad - fused_grad).abs().max().item()))
        indices = torch.arange(fused.fused_depthwise.dim, device=device) * fused.fused_depthwise.branch_count + branch
        bias_diffs.append(float((layer.bias.grad - fused.fused_depthwise.bias.grad.index_select(0, indices)).abs().max().item()))
    return {
        "max_output_abs_diff": float((original_output - fused_output).abs().max().item()),
        "max_input_grad_abs_diff": float((original_input.grad - fused_input.grad).abs().max().item()),
        "max_weight_grad_abs_diff": max(weight_diffs),
        "max_bias_grad_abs_diff": max(bias_diffs),
    }


def collapsed_equivalence_test(base_module, device: torch.device) -> dict[str, float]:
    torch.manual_seed(13)
    base = base_module.CausalMultiScaleLowRankConvMemoryBlock(
        dim=32,
        expansion=2,
        kernel_size=7,
        dilation=2,
        dropout=0.0,
        memory_rank=8,
        memory_kernel_size=16,
    ).to(device)
    collapsed = CollapsedWave10Block(base, dilation=2).to(device)
    original_input = torch.randn(2, 127, 32, device=device, requires_grad=True)
    collapsed_input = original_input.detach().clone().requires_grad_(True)
    original_output = base(original_input)
    collapsed_output = collapsed(collapsed_input)
    original_output.square().mean().backward()
    collapsed_output.square().mean().backward()
    return {
        "max_output_abs_diff": float((original_output - collapsed_output).abs().max().item()),
        "max_input_grad_abs_diff": float((original_input.grad - collapsed_input.grad).abs().max().item()),
    }


def benchmark_block(base_module, device: torch.device, iterations: int) -> dict[str, object]:
    torch.manual_seed(13)
    base_cpu = base_module.CausalMultiScaleLowRankConvMemoryBlock(
        dim=896,
        expansion=2,
        kernel_size=7,
        dilation=1,
        dropout=0.0,
        memory_rank=64,
        memory_kernel_size=128,
    )
    fused_cpu = FusedWave10Block(base_cpu, dilation=1)
    sample = torch.randn(1, 10_160, 896, device=device)
    results: dict[str, object] = {}
    for name, module in (("original", base_cpu), ("fused", fused_cpu)):
        module = module.to(device).train()
        optimizer = torch.optim.AdamW(module.parameters(), lr=6e-4, fused=device.type == "cuda")
        scaler = torch.amp.GradScaler(device="cuda", enabled=device.type == "cuda")

        def step() -> float:
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                loss = module(sample).square().mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if device.type == "cuda":
                torch.cuda.synchronize()
            return time.perf_counter() - start

        step()
        times = [step() for _ in range(iterations)]
        results[name] = {
            "mean_step_ms": statistics.mean(times) * 1000,
            "median_step_ms": statistics.median(times) * 1000,
            "parameter_count": sum(parameter.numel() for parameter in module.parameters()),
        }
        del optimizer, scaler, module
        if device.type == "cuda":
            torch.cuda.empty_cache()
    original_ms = float(results["original"]["mean_step_ms"])
    fused_ms = float(results["fused"]["mean_step_ms"])
    results["fused_speed_ratio"] = original_ms / fused_ms
    return results


def fuse_model_blocks(model: nn.Module) -> nn.Module:
    for layer_index, block in enumerate(model.blocks):
        model.blocks[layer_index] = FusedWave10Block(block, dilation=2 ** (layer_index % 6))
    return model


def collapse_model_blocks(model: nn.Module) -> nn.Module:
    for layer_index, block in enumerate(model.blocks):
        model.blocks[layer_index] = CollapsedWave10Block(block, dilation=2 ** (layer_index % 6))
    return model


def benchmark_training_step(base_module, device: torch.device, iterations: int) -> dict[str, object]:
    config = base_module.TrainConfig(
        cache_path=Path("."),
        output_dir=Path("."),
        run_name="fused_probe",
        embedding_dim=896,
        block_type="multi_scale_lowrank_conv_memory",
        conv_layers=2,
        conv_rank=320,
        memory_rank=64,
        landmark_stride=128,
        sampled_vocab_size=16_384,
        token_stride=4,
        token_chunk_size=8_192,
        amp_dtype="fp16",
    )
    torch.manual_seed(13)
    original_cpu = base_module.CausalConvFactorizedLM(config)
    fused_cpu = fuse_model_blocks(copy.deepcopy(original_cpu))
    fixed_candidate_ids = torch.arange(config.sampled_vocab_size, dtype=torch.long)
    batch_inputs = torch.randint(0, 8_192, (1, config.sequence_length), device=device)
    batch_targets = torch.randint(0, 8_192, (1, config.sequence_length), device=device)
    results: dict[str, object] = {}
    for name, model in (("original", original_cpu), ("fused", fused_cpu)):
        model = model.to(device).train()
        parameters = list(model.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=6e-4, weight_decay=1e-4, fused=device.type == "cuda")
        scaler = torch.amp.GradScaler(device="cuda", enabled=device.type == "cuda")

        def step() -> tuple[float, float]:
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                loss, _, _ = base_module.anchor_loss(
                    model,
                    batch_inputs,
                    batch_targets,
                    fixed_candidate_ids=fixed_candidate_ids,
                    config=config,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            scaler.step(optimizer)
            scaler.update()
            if device.type == "cuda":
                torch.cuda.synchronize()
            return time.perf_counter() - start, float(loss.detach().item())

        step()
        samples = [step() for _ in range(iterations)]
        times = [sample[0] for sample in samples]
        results[name] = {
            "mean_step_ms": statistics.mean(times) * 1000,
            "median_step_ms": statistics.median(times) * 1000,
            "tokens_per_second": config.sequence_length / statistics.mean(times),
            "last_loss": samples[-1][1],
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        }
        del optimizer, scaler, model, parameters
        if device.type == "cuda":
            torch.cuda.empty_cache()
    original_ms = float(results["original"]["mean_step_ms"])
    fused_ms = float(results["fused"]["mean_step_ms"])
    results["fused_speed_ratio"] = original_ms / fused_ms
    return results


def benchmark_collapsed_training_step(base_module, device: torch.device, iterations: int) -> dict[str, object]:
    config = base_module.TrainConfig(
        cache_path=Path("."),
        output_dir=Path("."),
        run_name="collapsed_probe",
        embedding_dim=896,
        block_type="multi_scale_lowrank_conv_memory",
        conv_layers=2,
        conv_rank=320,
        memory_rank=64,
        landmark_stride=128,
        sampled_vocab_size=16_384,
        token_stride=4,
        token_chunk_size=8_192,
        amp_dtype="fp16",
    )
    torch.manual_seed(13)
    original_cpu = base_module.CausalConvFactorizedLM(config)
    collapsed_cpu = collapse_model_blocks(copy.deepcopy(original_cpu))
    fixed_candidate_ids = torch.arange(config.sampled_vocab_size, dtype=torch.long)
    batch_inputs = torch.randint(0, 8_192, (1, config.sequence_length), device=device)
    batch_targets = torch.randint(0, 8_192, (1, config.sequence_length), device=device)
    results: dict[str, object] = {}
    for name, model in (("original", original_cpu), ("collapsed", collapsed_cpu)):
        model = model.to(device).train()
        parameters = list(model.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=6e-4, weight_decay=1e-4, fused=device.type == "cuda")
        scaler = torch.amp.GradScaler(device="cuda", enabled=device.type == "cuda")

        def step() -> tuple[float, float]:
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                loss, _, _ = base_module.anchor_loss(
                    model,
                    batch_inputs,
                    batch_targets,
                    fixed_candidate_ids=fixed_candidate_ids,
                    config=config,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            scaler.step(optimizer)
            scaler.update()
            if device.type == "cuda":
                torch.cuda.synchronize()
            return time.perf_counter() - start, float(loss.detach().item())

        step()
        samples = [step() for _ in range(iterations)]
        times = [sample[0] for sample in samples]
        results[name] = {
            "mean_step_ms": statistics.mean(times) * 1000,
            "median_step_ms": statistics.median(times) * 1000,
            "tokens_per_second": config.sequence_length / statistics.mean(times),
            "last_loss": samples[-1][1],
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        }
        del optimizer, scaler, model, parameters
        if device.type == "cuda":
            torch.cuda.empty_cache()
    original_ms = float(results["original"]["mean_step_ms"])
    collapsed_ms = float(results["collapsed"]["mean_step_ms"])
    results["collapsed_speed_ratio"] = original_ms / collapsed_ms
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    base_module = load_trainer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "equivalence": equivalence_test(base_module, device),
        "collapsed_equivalence": collapsed_equivalence_test(base_module, device),
        "benchmark": benchmark_block(base_module, device, args.iterations),
        "full_training_benchmark": benchmark_training_step(base_module, device, args.iterations),
        "collapsed_training_benchmark": benchmark_collapsed_training_step(base_module, device, args.iterations),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
