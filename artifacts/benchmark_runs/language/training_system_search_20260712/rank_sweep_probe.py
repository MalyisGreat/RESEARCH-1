from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

import collapsed_token_recall_train as experiment


def benchmark_rank(base, rank: int, iterations: int, device: torch.device) -> dict[str, float | int]:
    config = base.TrainConfig(
        cache_path=Path("unused.pt"),
        output_dir=Path("unused"),
        run_name=f"rank_{rank}",
        sequence_length=10_160,
        batch_size=1,
        train_steps=1,
        eval_interval=1,
        val_blocks=8,
        embedding_dim=512,
        block_type="multi_scale_lowrank_conv_memory",
        conv_layers=2,
        conv_rank=rank,
        memory_rank=64,
        sampled_vocab_size=4_096,
        token_stride=8,
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
    torch.manual_seed(13)
    model = base.CausalConvFactorizedLM(config).to(device).train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=6e-4, weight_decay=1e-4, fused=True)
    scaler = torch.amp.GradScaler(device="cuda", enabled=True)
    inputs = torch.randint(0, 8_192, (1, config.sequence_length), device=device)
    targets = torch.randint(0, 8_192, (1, config.sequence_length), device=device)
    candidate_ids = torch.arange(config.sampled_vocab_size, device=device)

    def step() -> tuple[float, float]:
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss, _, _ = base.anchor_loss(
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
        return time.perf_counter() - started, float(loss.detach())

    for _ in range(3):
        step()
    samples = [step() for _ in range(iterations)]
    times = [sample[0] for sample in samples]
    result = {
        "rank": rank,
        "parameters": sum(parameter.numel() for parameter in parameters),
        "mean_step_ms": statistics.fmean(times) * 1_000,
        "median_step_ms": statistics.median(times) * 1_000,
        "tokens_per_second": config.sequence_length / statistics.fmean(times),
        "last_loss": samples[-1][1],
    }
    del model, optimizer, scaler, parameters, inputs, targets, candidate_ids
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = experiment.trainer
    device = torch.device("cuda")
    ranks = [96, 128, 160, 192, 224, 256, 320]
    payload = {
        "device": torch.cuda.get_device_name(0),
        "results": [benchmark_rank(base, rank, args.iterations, device) for rank in ranks],
    }
    rendered = json.dumps(payload, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered, flush=True)


if __name__ == "__main__":
    main()
