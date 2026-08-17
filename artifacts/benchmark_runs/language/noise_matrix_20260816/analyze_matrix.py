from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SEEDS = (13, 17, 23, 29, 31)
CONDITIONS = ("phrase23", "phrase234", "phrase23_semantic")
LABELS = {
    "phrase23": "Phrase 2+3",
    "phrase234": "Phrase 2+3+4",
    "phrase23_semantic": "Phrase 2+3 + semantic",
}


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def ci95(values: list[float]) -> float:
    return 2.776 * statistics.stdev(values) / math.sqrt(len(values))


def main() -> None:
    results = {
        condition: {
            seed: json.loads((ROOT / f"{condition}_seed{seed}" / "result.json").read_text(encoding="utf-8"))["report"]
            for seed in SEEDS
        }
        for condition in CONDITIONS
    }
    repetition = json.loads((ROOT / "repetition_matrix.json").read_text(encoding="utf-8"))["conditions"]

    rows = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            result = results[condition][seed]
            repeats = repetition[condition][str(seed)]["aggregate"]
            rows.append({
                "condition": condition,
                "seed": seed,
                "val_loss": result["final_val_loss"],
                "tok_per_sec": result["pure_train_tok_per_sec"],
                "peak_vram_mb": result["peak_vram_mb"],
                **{f"repeat_{order}": repeats[f"repeat_{order}"] for order in range(1, 5)},
            })
    with (ROOT / "matrix_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {"conditions": {}, "paired_vs_phrase23": {}}
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        summary["conditions"][condition] = {}
        for metric in ("val_loss", "tok_per_sec", "peak_vram_mb", "repeat_1", "repeat_2", "repeat_3", "repeat_4"):
            values = [float(row[metric]) for row in selected]
            mean, sd = mean_sd(values)
            summary["conditions"][condition][metric] = {"mean": mean, "sample_sd": sd}
    control = {row["seed"]: row for row in rows if row["condition"] == "phrase23"}
    for condition in CONDITIONS[1:]:
        selected = {row["seed"]: row for row in rows if row["condition"] == condition}
        summary["paired_vs_phrase23"][condition] = {}
        for metric in ("val_loss", "repeat_1", "repeat_2", "repeat_3", "repeat_4"):
            differences = [float(selected[seed][metric]) - float(control[seed][metric]) for seed in SEEDS]
            mean, sd = mean_sd(differences)
            summary["paired_vs_phrase23"][condition][metric] = {
                "differences": differences,
                "mean": mean,
                "sample_sd": sd,
                "ci95_half_width": ci95(differences),
            }
    (ROOT / "matrix_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)
    colors = {"phrase23": "#202020", "phrase234": "#177245", "phrase23_semantic": "#b33a3a"}
    for condition in CONDITIONS:
        histories = [results[condition][seed]["history"] for seed in SEEDS]
        tokens = [point["tokens_seen"] / 1e6 for point in histories[0]]
        means = [statistics.mean(history[index]["val_loss"] for history in histories) for index in range(len(tokens))]
        sds = [statistics.stdev(history[index]["val_loss"] for history in histories) for index in range(len(tokens))]
        ax.plot(tokens, means, marker="o", linewidth=2.2, label=LABELS[condition], color=colors[condition])
        ax.fill_between(tokens, [m - s for m, s in zip(means, sds)], [m + s for m, s in zip(means, sds)], color=colors[condition], alpha=0.12)
    ax.set_xlabel("Token exposures (millions)")
    ax.set_ylabel("Full-vocabulary validation loss")
    ax.set_title("Five-seed paired retrieval matrix")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "validation_curves.png")


if __name__ == "__main__":
    main()
