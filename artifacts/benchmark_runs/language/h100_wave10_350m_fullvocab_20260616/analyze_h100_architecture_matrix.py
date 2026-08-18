from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ARCHITECTURES = ("wave", "delta_gain", "delta_router")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_runs(root: Path) -> list[dict[str, Any]]:
    runs = []
    for path in sorted(root.glob("*_350m_100m_seed*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        report = payload["report"]
        architecture = payload.get("architecture") or payload.get("h100_args", {}).get("architecture")
        if architecture is None:
            architecture = path.parent.name.split("_350m_", 1)[0]
        runs.append(
            {
                "architecture": architecture,
                "seed": int(payload["config"]["seed"]),
                "path": str(path),
                **report,
            }
        )
    return runs


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[run["architecture"]].append(run)
    summary: dict[str, Any] = {}
    wave_by_seed = {run["seed"]: run for run in grouped.get("wave", [])}
    for architecture in ARCHITECTURES:
        arms = sorted(grouped.get(architecture, []), key=lambda row: row["seed"])
        if not arms:
            continue
        losses = [run["final_val_loss_full_vocab"] for run in arms]
        paired = [
            run["final_val_loss_full_vocab"] - wave_by_seed[run["seed"]]["final_val_loss_full_vocab"]
            for run in arms
            if run["seed"] in wave_by_seed
        ]
        summary[architecture] = {
            "runs": len(arms),
            "seeds": [run["seed"] for run in arms],
            "endpoint_loss_mean": statistics.mean(losses),
            "endpoint_loss_stdev": statistics.stdev(losses) if len(losses) > 1 else 0.0,
            "paired_delta_vs_wave_mean": statistics.mean(paired) if paired else None,
            "wins_vs_wave": sum(delta < 0 for delta in paired),
            "paired_comparisons": len(paired),
            "tok_per_sec_mean": statistics.mean(run["pure_train_tok_per_sec"] for run in arms),
            "wall_tok_per_sec_mean": statistics.mean(run["wall_tok_per_sec"] for run in arms),
            "peak_allocated_mb_mean": statistics.mean(run["peak_allocated_mb"] for run in arms),
            "parameter_count": arms[0]["parameter_count"],
        }
    return summary


def write_curves(runs: list[dict[str, Any]], path: Path) -> None:
    rows = []
    for run in runs:
        for point in run["history"]:
            rows.append(
                {
                    "architecture": run["architecture"],
                    "seed": run["seed"],
                    "step": int(point["step"]),
                    "tokens_seen": int(point["tokens_seen"]),
                    "val_loss_full_vocab": point["val_loss_full_vocab"],
                    "train_loss": point["train_loss"],
                }
            )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_plot(runs: list[dict[str, Any]], path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[run["architecture"]].append(run)
    colors = {"wave": "#111111", "delta_gain": "#0072B2", "delta_router": "#D55E00"}
    fig, axis = plt.subplots(figsize=(9, 5.5))
    for architecture in ARCHITECTURES:
        arms = grouped.get(architecture, [])
        if not arms:
            continue
        tokens = [point["tokens_seen"] for point in arms[0]["history"]]
        means = [statistics.mean(run["history"][i]["val_loss_full_vocab"] for run in arms) for i in range(len(tokens))]
        axis.plot([token / 1e6 for token in tokens], means, marker="o", linewidth=2.2, label=architecture, color=colors[architecture])
        for run in arms:
            axis.plot(
                [point["tokens_seen"] / 1e6 for point in run["history"]],
                [point["val_loss_full_vocab"] for point in run["history"]],
                linewidth=0.8,
                alpha=0.35,
                color=colors[architecture],
            )
    axis.set_xlabel("Training tokens (millions)")
    axis.set_ylabel("Full-vocabulary validation loss")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()
    output = args.output_dir or args.run_root
    output.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.run_root)
    if not runs:
        raise RuntimeError(f"no completed runs found under {args.run_root}")
    summary = aggregate(runs)
    ranking = sorted(summary, key=lambda architecture: summary[architecture]["endpoint_loss_mean"])
    payload = {"completed_runs": len(runs), "quality_ranking": ranking, "architectures": summary}
    (output / "ARCHITECTURE_MATRIX_SUMMARY.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_curves(runs, output / "architecture_validation_curves.csv")
    plotted = write_plot(runs, output / "architecture_validation_curves.png")
    lines = [
        "# H100 350M Architecture Matrix",
        "",
        f"Completed runs: {len(runs)}",
        "",
        "| Architecture | Seeds | Full-val loss | Paired vs Wave | Wins | Tok/s | Peak VRAM | Parameters |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for architecture in ranking:
        row = summary[architecture]
        paired = row["paired_delta_vs_wave_mean"]
        paired_text = f"{paired:+.6f}" if paired is not None else "n/a"
        lines.append(
            f"| {architecture} | {row['runs']} | {row['endpoint_loss_mean']:.6f} +/- {row['endpoint_loss_stdev']:.6f} "
            f"| {paired_text} | {row['wins_vs_wave']}/{row['paired_comparisons']} "
            f"| {row['tok_per_sec_mean']:,.0f} | {row['peak_allocated_mb_mean']:,.0f} MB "
            f"| {row['parameter_count']:,} |"
        )
    lines.extend(["", f"Quality ranking: {' > '.join(ranking)}", "", f"Curve image created: {plotted}", ""])
    (output / "ARCHITECTURE_MATRIX_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
