from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch


HASH_SEEDS = (0x9E3779B1, 0x85EBCA77, 0xC2B2AE3D)


def hash_tokens(token_ids: torch.Tensor, table: int, buckets: int) -> tuple[torch.Tensor, torch.Tensor]:
    mixed = torch.bitwise_xor(token_ids, HASH_SEEDS[table])
    mixed = torch.bitwise_xor(mixed, torch.bitwise_right_shift(mixed, 16)) * 0x45D9F3B
    mixed = torch.bitwise_xor(mixed, torch.bitwise_right_shift(mixed, 16)) * 0x45D9F3B
    mixed = torch.bitwise_xor(mixed, torch.bitwise_right_shift(mixed, 16))
    bucket_ids = torch.remainder(mixed, buckets)
    signs = torch.ones_like(bucket_ids, dtype=torch.float32)
    return bucket_ids, signs


def build_sketches(input_ids: torch.Tensor, tables: int, buckets: int, dtype: torch.dtype) -> list[torch.Tensor]:
    batch, length = input_ids.shape
    sketches = []
    for table in range(tables):
        bucket_ids, signs = hash_tokens(input_ids, table, buckets)
        events = torch.zeros(batch, length, buckets, device=input_ids.device, dtype=dtype)
        events.scatter_(2, bucket_ids.unsqueeze(-1), 1.0)
        sketches.append(events.cumsum(dim=1))
    return sketches


def window_state(prefix: torch.Tensor, positions: torch.Tensor, window: int) -> torch.Tensor:
    current = prefix.index_select(1, positions)
    prior_positions = positions - window
    valid = prior_positions >= 0
    prior = prefix.index_select(1, prior_positions.clamp(min=0))
    return current - prior * valid.view(1, -1, 1).to(prefix.dtype)


def decode(
    sketches: list[torch.Tensor],
    candidate_ids: torch.Tensor,
    positions: torch.Tensor,
    window: int,
    buckets: int,
) -> torch.Tensor:
    estimates = []
    for table, prefix in enumerate(sketches):
        bucket_ids, signs = hash_tokens(candidate_ids, table, buckets)
        states = window_state(prefix, positions, window)
        gathered = states.index_select(2, bucket_ids)
        estimates.append(gathered)
    return torch.stack(estimates, dim=0).amin(dim=0)


def decode_direct(
    input_ids: torch.Tensor,
    candidate_ids: torch.Tensor,
    positions: torch.Tensor,
    window: int,
    tables: int,
    buckets: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    offsets = torch.arange(window, device=input_ids.device)
    source_positions = positions[:, None] - offsets[None, :]
    valid = source_positions >= 0
    safe_positions = source_positions.clamp(min=0)
    source_tokens = input_ids.index_select(1, safe_positions.reshape(-1)).view(input_ids.size(0), positions.numel(), window)
    estimate = None
    for table in range(tables):
        source_buckets, _ = hash_tokens(source_tokens, table, buckets)
        candidate_buckets, _ = hash_tokens(candidate_ids, table, buckets)
        seen = torch.zeros(input_ids.size(0), positions.numel(), buckets, device=input_ids.device, dtype=dtype)
        seen.scatter_(2, source_buckets, valid.view(1, positions.numel(), window).to(dtype))
        table_estimate = seen.index_select(2, candidate_buckets)
        estimate = table_estimate if estimate is None else torch.minimum(estimate, table_estimate)
    return estimate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--buckets", type=int, default=1_024)
    parser.add_argument("--tables", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--realistic", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(13)
    vocab_size = 50_257
    length = 10_160
    input_ids = torch.randint(vocab_size, (1, length), device=device)
    candidate_ids = torch.arange(vocab_size, device=device) if args.realistic else torch.randperm(vocab_size, device=device)[:4_096]
    positions = torch.arange(512, length, 12 if args.realistic else 151, device=device)

    if args.realistic:
        direct_times = []
        direct_estimates = None
        for _ in range(3):
            direct_estimates = decode_direct(
                input_ids, candidate_ids, positions, 512, args.tables, args.buckets, torch.float16
            )
        for _ in range(args.iterations):
            torch.cuda.synchronize()
            started = time.perf_counter()
            direct_estimates = decode_direct(
                input_ids, candidate_ids, positions, 512, args.tables, args.buckets, torch.float16
            )
            torch.cuda.synchronize()
            direct_times.append(time.perf_counter() - started)
        payload = {
            "device": torch.cuda.get_device_name(0),
            "tables": args.tables,
            "buckets": args.buckets,
            "positions": positions.numel(),
            "candidates": candidate_ids.numel(),
            "direct_build_decode_mean_ms": statistics.fmean(direct_times) * 1_000,
            "direct_build_decode_median_ms": statistics.median(direct_times) * 1_000,
            "output_mb": direct_estimates.numel() * direct_estimates.element_size() / 2**20,
            "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20,
            "finite": bool(torch.isfinite(direct_estimates).all()),
        }
        rendered = json.dumps(payload, indent=2)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered, flush=True)
        return

    torch.cuda.synchronize()
    started = time.perf_counter()
    sketches = build_sketches(input_ids, args.tables, args.buckets, torch.float16)
    torch.cuda.synchronize()
    build_ms = (time.perf_counter() - started) * 1_000

    for _ in range(3):
        decode(sketches, candidate_ids, positions, 512, args.buckets)
    times = []
    for _ in range(args.iterations):
        torch.cuda.synchronize()
        started = time.perf_counter()
        estimates = decode(sketches, candidate_ids, positions, 512, args.buckets)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - started)

    for _ in range(3):
        decode_direct(input_ids, candidate_ids, positions, 512, args.tables, args.buckets, torch.float16)
    direct_times = []
    for _ in range(args.iterations):
        torch.cuda.synchronize()
        started = time.perf_counter()
        direct_estimates = decode_direct(
            input_ids, candidate_ids, positions, 512, args.tables, args.buckets, torch.float16
        )
        torch.cuda.synchronize()
        direct_times.append(time.perf_counter() - started)

    source_positions = positions[:, None] - torch.arange(512, device=device)[None, :]
    source_tokens = input_ids[0].index_select(0, source_positions.reshape(-1)).view(positions.numel(), 512)
    truth = source_tokens.unsqueeze(-1).eq(candidate_ids.view(1, 1, -1)).any(dim=1)
    predicted = estimates[0] > 0.5
    direct_predicted = direct_estimates[0] > 0.5
    true_positive = int((predicted & truth).sum())
    false_positive = int((predicted & ~truth).sum())
    false_negative = int((~predicted & truth).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    payload = {
        "device": torch.cuda.get_device_name(0),
        "tables": args.tables,
        "buckets": args.buckets,
        "positions": positions.numel(),
        "candidates": candidate_ids.numel(),
        "build_ms": build_ms,
        "decode_mean_ms": statistics.fmean(times) * 1_000,
        "decode_median_ms": statistics.median(times) * 1_000,
        "direct_build_decode_mean_ms": statistics.fmean(direct_times) * 1_000,
        "direct_build_decode_median_ms": statistics.median(direct_times) * 1_000,
        "sketch_memory_mb": sum(t.numel() * t.element_size() for t in sketches) / 2**20,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": false_positive / max(int((~truth).sum()), 1),
        "direct_matches_dense": bool(torch.equal(direct_predicted, predicted)),
        "finite": bool(torch.isfinite(estimates).all()),
    }
    rendered = json.dumps(payload, indent=2)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered, flush=True)


if __name__ == "__main__":
    main()
