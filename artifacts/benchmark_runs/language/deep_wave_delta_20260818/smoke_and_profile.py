from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from deep_wave_delta import DeepWaveDeltaConfig, DeepWaveDeltaLM, config_from_preset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep Wave-Delta correctness and speed smoke test")
    parser.add_argument("--preset", choices=("10m", "100m", "350m"), default="10m")
    parser.add_argument("--vocab-size", type=int, default=50_257)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    config = config_from_preset(
        args.preset,
        vocab_size=args.vocab_size,
        use_fused_delta=device.type == "cuda",
    )
    torch.manual_seed(13)
    model = DeepWaveDeltaLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    losses: list[float] = []
    started = time.perf_counter()
    for _ in range(args.steps):
        tokens = torch.randint(0, config.vocab_size, (args.batch_size, args.sequence_length), device=device)
        targets = torch.roll(tokens, shifts=-1, dims=1)
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
            raise RuntimeError("non-finite loss or gradient")
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        losses.append(float(loss.detach()))
    elapsed = time.perf_counter() - started
    report = {
        "config": config.to_dict(),
        "parameters": model.parameter_report(),
        "device": str(device),
        "losses": losses,
        "finite": all(torch.isfinite(torch.tensor(losses))),
        "tokens_per_second": args.steps * args.batch_size * args.sequence_length / elapsed,
        "peak_allocated_mb": (
            torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None
        ),
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()

