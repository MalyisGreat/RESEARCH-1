from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RUNS = {
    "Transformer control": "rope_transformer_seed13",
    "HRM full backprop": "rope_hrm_full_seed13",
    "HRM credit warmup": "rope_hrm_warmup_seed13",
}
COLORS = {"Transformer control": "#222222", "HRM full backprop": "#177245", "HRM credit warmup": "#b33a3a"}


def main() -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=160)
    for label, directory in RUNS.items():
        report = json.loads((ROOT / "prefixlm_runs" / directory / "result.json").read_text(encoding="utf-8"))
        history = report["history"]
        ax.plot(
            [point["response_tokens"] for point in history],
            [point["val_loss"] for point in history],
            marker="o",
            linewidth=2.2,
            label=label,
            color=COLORS[label],
        )
    ax.set_xlabel("Supervised response tokens")
    ax.set_ylabel("Full-vocabulary response validation loss")
    ax.set_title("Official-data PrefixLM recipe screen")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "prefixlm_recipe_curves.png")


if __name__ == "__main__":
    main()
