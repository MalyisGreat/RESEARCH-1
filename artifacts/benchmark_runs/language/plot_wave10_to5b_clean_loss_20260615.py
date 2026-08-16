from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


RUNS_ROOT = Path(r"E:\CODEXRESEARCH\house_compute_hub\runs")
OUT_DIR = Path(r"E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\plots_20260615")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_RE = re.compile(
    r"TRAIN step=(\d+)/492126 tokens=(\d+) loss=([0-9.]+) lr=([0-9.eE+-]+)(?: pure_tok_s=([0-9.]+))?"
)
EVAL_RE = re.compile(r"EVAL step=(\d+)/492126 tokens=(\d+) train=([0-9.]+) val=([0-9.]+)")

train_rows: list[dict[str, object]] = []
eval_rows: list[dict[str, object]] = []
logs_used: set[str] = set()

for path in RUNS_ROOT.rglob("*.log"):
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        continue
    if "wave10_3080_lowrank_conv_memory_76m" not in text and "492126" not in text:
        continue
    matched = False
    for line in text.splitlines():
        train_match = TRAIN_RE.search(line)
        if train_match:
            step, tokens, loss, lr, tok_s = train_match.groups()
            token_count = int(tokens)
            if token_count < 2_000_000_000:
                continue
            matched = True
            train_rows.append(
                {
                    "source": str(path),
                    "kind": "train",
                    "step": int(step),
                    "tokens_seen": token_count,
                    "tokens_b": token_count / 1e9,
                    "train_loss": float(loss),
                    "val_loss": "",
                    "lr": float(lr),
                    "tok_s": float(tok_s) if tok_s else "",
                }
            )
            continue
        eval_match = EVAL_RE.search(line)
        if eval_match:
            step, tokens, train_loss, val_loss = eval_match.groups()
            token_count = int(tokens)
            if token_count < 2_000_000_000:
                continue
            matched = True
            eval_rows.append(
                {
                    "source": str(path),
                    "kind": "eval",
                    "step": int(step),
                    "tokens_seen": token_count,
                    "tokens_b": token_count / 1e9,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "lr": "",
                    "tok_s": "",
                }
            )
    if matched:
        logs_used.add(str(path))


def keep_latest_by_token(rows: list[dict[str, object]], metric_key: str) -> list[dict[str, object]]:
    by_key: dict[tuple[int, object], dict[str, object]] = {}
    for row in sorted(rows, key=lambda r: (str(r["source"]), int(r["tokens_seen"]))):
        by_key[(int(row["tokens_seen"]), row[metric_key])] = row
    return sorted(by_key.values(), key=lambda r: int(r["tokens_seen"]))


train_rows = keep_latest_by_token(train_rows, "train_loss")
eval_rows = keep_latest_by_token(eval_rows, "val_loss")
all_rows = sorted(train_rows + eval_rows, key=lambda r: (int(r["tokens_seen"]), str(r["kind"])))

csv_path = OUT_DIR / "wave10_76m_to5b_clean_loss_so_far_20260615.csv"
with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["source", "kind", "step", "tokens_seen", "tokens_b", "train_loss", "val_loss", "lr", "tok_s"],
    )
    writer.writeheader()
    writer.writerows(all_rows)

plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(12, 6.5), dpi=150)

if train_rows:
    ax.plot(
        [float(row["tokens_b"]) for row in train_rows],
        [float(row["train_loss"]) for row in train_rows],
        color="#4C78A8",
        linewidth=1.0,
        alpha=0.42,
        label="train loss, logged every 500 steps",
    )

if eval_rows:
    ax.plot(
        [float(row["tokens_b"]) for row in eval_rows],
        [float(row["val_loss"]) for row in eval_rows],
        color="#E45756",
        marker="o",
        markersize=4,
        linewidth=2.0,
        label="validation loss",
    )
    ax.scatter(
        [float(row["tokens_b"]) for row in eval_rows],
        [float(row["train_loss"]) for row in eval_rows],
        color="#72B7B2",
        s=18,
        label="train loss at eval",
        zorder=3,
    )

for token_b, label in [(2.0, "2B"), (3.0, "3B"), (4.0, "4B"), (5.0, "5B target")]:
    ax.axvline(token_b, color="#999999", linewidth=0.8, linestyle="--", alpha=0.45)
    ax.text(token_b + 0.01, 5.02, label, rotation=90, va="top", ha="left", fontsize=8, color="#555555")

ax.set_title("Wave10 76M low-rank conv-memory loss, 2B to current")
ax.set_xlabel("Tokens seen (billions)")
ax.set_ylabel("Loss")
ax.set_xlim(1.98, 5.02)
ax.set_ylim(3.35, 5.05)
ax.legend(loc="upper right")
fig.subplots_adjust(left=0.07, right=0.99, bottom=0.1, top=0.93)

png_path = OUT_DIR / "wave10_76m_to5b_clean_loss_so_far_20260615.png"
fig.savefig(png_path)

latest_train = max(train_rows, key=lambda row: int(row["tokens_seen"])) if train_rows else None
latest_eval = max(eval_rows, key=lambda row: int(row["tokens_seen"])) if eval_rows else None
summary = {
    "png": str(png_path),
    "csv": str(csv_path),
    "train_points": len(train_rows),
    "eval_points": len(eval_rows),
    "latest_train": latest_train,
    "latest_eval": latest_eval,
    "logs_used": sorted(logs_used),
}
print(json.dumps(summary, indent=2))
