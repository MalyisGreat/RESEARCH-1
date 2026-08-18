from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from arc_tactic3.language_partial_untied_cluster import (
    PartialUntiedClusterConfig,
    ensure_fineweb_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a GPT-2 FineWeb-Edu token cache on the current host.")
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--train-tokens", type=int, default=100_014_723)
    parser.add_argument("--val-tokens", type=int, default=325_152)
    parser.add_argument("--sequence-length", type=int, default=10_160)
    parser.add_argument("--tokenization-batch-size", type=int, default=4_096)
    parser.add_argument("--dataset-name", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--tokenizer-name", default="gpt2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config = PartialUntiedClusterConfig(
        output_dir=args.cache_path.parent,
        cache_path=args.cache_path,
        dataset_name=args.dataset_name,
        tokenizer_name=args.tokenizer_name,
        total_tokens=args.train_tokens + args.val_tokens,
        train_tokens=args.train_tokens,
        val_tokens=args.val_tokens,
        sequence_length=args.sequence_length,
        tokenization_batch_size=args.tokenization_batch_size,
        device="cpu",
        use_amp=False,
        pin_memory=False,
        use_fused_adamw=False,
        cache_dataset_on_device=False,
    )
    train_dataset, val_dataset, vocab_size, cache_path = ensure_fineweb_cache(config, print_progress=True)
    result = {
        "cache_path": str(cache_path.resolve()),
        "train_blocks": len(train_dataset),
        "val_blocks": len(val_dataset),
        "train_tokens": args.train_tokens,
        "val_tokens": args.val_tokens,
        "vocab_size": vocab_size,
        "elapsed_seconds": time.perf_counter() - started,
    }
    result_path = cache_path.with_suffix(cache_path.suffix + ".build.json")
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
