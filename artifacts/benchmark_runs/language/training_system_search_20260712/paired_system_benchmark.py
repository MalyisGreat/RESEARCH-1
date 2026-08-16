from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]
TRAINER_PATH = ROOT / "token_recall_search_20260616" / "token_recall_train.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_config(module, *, candidate: bool):
    return module.TrainConfig(
        cache_path=Path("unused.pt"),
        output_dir=Path("unused"),
        run_name="candidate" if candidate else "reference",
        sequence_length=10_160,
        batch_size=1,
        train_steps=1,
        eval_interval=1,
        val_blocks=8,
        embedding_dim=512,
        block_type="multi_scale_lowrank_conv_memory",
        conv_layers=2,
        conv_rank=192,
        memory_rank=64,
        sampled_vocab_size=4_096 if candidate else 8_192,
        token_stride=24 if candidate else 4,
        token_chunk_size=20_000,
        full_eval_token_chunk_size=512,
        learning_rate=6e-4,
        min_learning_rate=1e-5,
        warmup_steps=64,
        weight_decay=1e-4,
        amp_dtype="fp16",
        recall_mode="factor_recall_gated_multiscale",
        recall_initial_scale=256.0,
    )


def benchmark(module, *, candidate: bool, iterations: int, device: torch.device) -> dict[str, object]:
    config = make_config(module, candidate=candidate)
    torch.manual_seed(13)
    model = module.CausalConvFactorizedLM(config).to(device).train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=6e-4, weight_decay=1e-4, fused=True)
    scaler = torch.amp.GradScaler(device="cuda", enabled=True)
    inputs = torch.randint(0, 8_192, (1, config.sequence_length), device=device)
    targets = torch.randint(0, 8_192, (1, config.sequence_length), device=device)
    candidate_ids = torch.arange(config.sampled_vocab_size, device=device)

    def step() -> float:
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss, _, _ = module.anchor_loss(
                model,
                inputs,
                targets,
                fixed_candidate_ids=candidate_ids,
                config=config,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize(device)
        return time.perf_counter() - started

    for _ in range(5):
        step()
    times = [step() for _ in range(iterations)]
    result = {
        "name": "candidate" if candidate else "reference",
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "mean_step_ms": statistics.fmean(times) * 1_000,
        "median_step_ms": statistics.median(times) * 1_000,
        "tokens_per_second": config.sequence_length / statistics.fmean(times),
        "times_ms": [value * 1_000 for value in times],
    }
    del model, parameters, optimizer, scaler, inputs, targets, candidate_ids
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    reference = load_module("paired_reference_trainer", TRAINER_PATH)
    os.environ["WAVE10_HEAD_MULTIPLIER"] = "1"
    os.environ["WAVE10_IDENTITY_HEAD"] = "1"
    os.environ["WAVE10_RESIDUAL_HEAD"] = "0"
    import sorted_induction_train as candidate

    runs = []
    for order in ((False, True), (True, False)):
        for is_candidate in order:
            runs.append(
                benchmark(
                    candidate.experiment.experiment.trainer if is_candidate else reference,
                    candidate=is_candidate,
                    iterations=args.iterations,
                    device=device,
                )
            )
    reference_means = [run["mean_step_ms"] for run in runs if run["name"] == "reference"]
    candidate_means = [run["mean_step_ms"] for run in runs if run["name"] == "candidate"]
    payload = {
        "device": torch.cuda.get_device_name(0),
        "runs": runs,
        "aggregate_reference_step_ms": statistics.fmean(reference_means),
        "aggregate_candidate_step_ms": statistics.fmean(candidate_means),
        "aggregate_speedup": statistics.fmean(reference_means) / statistics.fmean(candidate_means),
    }
    rendered = json.dumps(payload, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered, flush=True)


if __name__ == "__main__":
    main()
