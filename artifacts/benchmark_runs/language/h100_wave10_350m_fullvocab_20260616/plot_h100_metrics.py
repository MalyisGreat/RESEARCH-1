from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def to_float(value: str) -> float | None:
    if value is None or value == "" or value.lower() == "nan":
        return None
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot H100 Wave10 metrics.csv.")
    parser.add_argument("metrics_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or args.metrics_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    with args.metrics_csv.open("r", encoding="utf-8", newline="") as handle:
        rows.extend(csv.DictReader(handle))

    train_rows = [row for row in rows if row.get("event") == "train"]
    eval_rows = [row for row in rows if row.get("event") == "eval_full_vocab"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    if train_rows:
        ax.plot(
            [float(row["tokens_seen"]) / 1e9 for row in train_rows],
            [float(row["train_loss"]) for row in train_rows],
            alpha=0.45,
            linewidth=1.0,
            label="sampled+anchor train loss",
        )
    if eval_rows:
        ax.plot(
            [float(row["tokens_seen"]) / 1e9 for row in eval_rows],
            [float(row["val_loss_full_vocab"]) for row in eval_rows],
            marker="o",
            linewidth=2.0,
            label="full-vocab validation loss",
        )
    ax.set_title("Wave10 350M H100 Loss")
    ax.set_xlabel("Tokens seen (billions)")
    ax.set_ylabel("Loss")
    ax.legend()
    loss_png = output_dir / "loss_full_vocab.png"
    fig.tight_layout()
    fig.savefig(loss_png)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    if train_rows:
        ax.plot(
            [float(row["tokens_seen"]) / 1e9 for row in train_rows],
            [float(row["rolling_100_tok_per_sec"]) for row in train_rows],
            label="rolling 100-step tok/s",
        )
        ax.plot(
            [float(row["tokens_seen"]) / 1e9 for row in train_rows],
            [float(row["pure_train_tok_per_sec"]) for row in train_rows],
            label="cumulative pure train tok/s",
            alpha=0.75,
        )
    ax.set_title("Wave10 350M H100 Throughput")
    ax.set_xlabel("Tokens seen (billions)")
    ax.set_ylabel("Tokens/sec")
    ax.legend()
    throughput_png = output_dir / "throughput.png"
    fig.tight_layout()
    fig.savefig(throughput_png)
    plt.close(fig)

    print(f"wrote {loss_png}")
    print(f"wrote {throughput_png}")


if __name__ == "__main__":
    main()
