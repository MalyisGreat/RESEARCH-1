from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "figures"
DOCS = ROOT / "docs"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_dirs() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)


def _build_50m_curves(summary: dict) -> None:
    partial = _load_json(ARTIFACTS / "watch_runs" / "partial_untied_watch_50m_20260328" / "final.json")
    nano = _load_json(ARTIFACTS / "watch_runs" / "nanochat_watch_50m_20260328_retry2" / "final.json")

    partial_hist = partial["report"]["history"]
    nano_hist = nano["report"]["history"]

    plt.figure(figsize=(9, 5.5))
    plt.plot(
        [point["tokens_seen"] / 1_000_000 for point in partial_hist[1:]],
        [point["val_loss"] for point in partial_hist[1:]],
        marker="o",
        label="partial_untied",
    )
    plt.plot(
        [point["tokens_seen"] / 1_000_000 for point in nano_hist[1:]],
        [point["val_loss"] for point in nano_hist[1:]],
        marker="o",
        label="nanochat_small",
    )
    plt.xlabel("Training Tokens Seen (millions)")
    plt.ylabel("Validation Loss")
    plt.title("50M-Token FineWeb-Edu Run")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "loss_curve_50m.png", dpi=180)
    plt.close()

    summary["runs_50m"] = {
        "partial_untied": {
            "final_val_loss": partial["report"]["final_val_loss"],
            "train_tok_per_sec": partial["report"]["train_tok_per_sec"],
            "pure_train_tok_per_sec": partial["report"]["pure_train_tok_per_sec"],
            "peak_vram_mb": partial["report"]["peak_vram_mb"],
            "parameter_count": partial["report"]["parameter_count"],
        },
        "nanochat_small": {
            "final_val_loss": nano["report"]["final_val_loss"],
            "train_tok_per_sec": nano["report"]["train_tok_per_sec"],
            "pure_train_tok_per_sec": nano["report"]["pure_train_tok_per_sec"],
            "peak_vram_mb": nano["report"]["peak_vram_mb"],
            "parameter_count": nano["report"]["parameter_count"],
        },
    }


def _build_50m_tradeoff(summary: dict) -> None:
    data = summary["runs_50m"]
    labels = ["partial_untied", "nanochat_small"]
    loss = [data[label]["final_val_loss"] for label in labels]
    tok = [data[label]["train_tok_per_sec"] / 1000.0 for label in labels]
    vram = [data[label]["peak_vram_mb"] / 1024.0 for label in labels]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.3))
    axes[0].bar(labels, loss, color=["#2b6cb0", "#dd6b20"])
    axes[0].set_title("Final Val Loss")
    axes[1].bar(labels, tok, color=["#2b6cb0", "#dd6b20"])
    axes[1].set_title("Train Throughput (k tok/s)")
    axes[2].bar(labels, vram, color=["#2b6cb0", "#dd6b20"])
    axes[2].set_title("Peak VRAM (GB)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("50M-Token Head-to-Head")
    fig.tight_layout()
    plt.savefig(FIGURES / "tradeoff_50m.png", dpi=180)
    plt.close(fig)


def _build_fastlearn_scaling(summary: dict) -> None:
    scaling = _load_json(ARTIFACTS / "benchmark_runs" / "language" / "language_fastlearn_scaling_gpt2icl_hybrid_20260327.json")
    scales = scaling["scales"]
    fast = [scaling["results"][scale]["models"]["fast_gru"]["adaptation_auc_mean"] for scale in scales]
    gpt = [scaling["results"][scale]["models"]["gpt2_like"]["adaptation_auc_mean"] for scale in scales]
    seq_fast = [scaling["results"][scale]["models"]["fast_gru"]["shot8_sequence_accuracy"] for scale in scales]
    seq_gpt = [scaling["results"][scale]["models"]["gpt2_like"]["shot8_sequence_accuracy"] for scale in scales]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].plot(scales, fast, marker="o", label="fast_gru")
    axes[0].plot(scales, gpt, marker="o", label="gpt2_like")
    axes[0].set_title("Few-Shot Adaptation AUC")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(scales, seq_fast, marker="o", label="fast_gru")
    axes[1].plot(scales, seq_gpt, marker="o", label="gpt2_like")
    axes[1].set_title("Shot-8 Sequence Accuracy")
    axes[1].grid(alpha=0.25)
    fig.suptitle("Synthetic Fast-Learning Scaling")
    fig.tight_layout()
    plt.savefig(FIGURES / "fastlearn_scaling.png", dpi=180)
    plt.close(fig)

    summary["fastlearn_scaling"] = {
        scale: {
            "fast_gru_auc": scaling["results"][scale]["models"]["fast_gru"]["adaptation_auc_mean"],
            "gpt2_like_auc": scaling["results"][scale]["models"]["gpt2_like"]["adaptation_auc_mean"],
            "fast_gru_seq_acc": scaling["results"][scale]["models"]["fast_gru"]["shot8_sequence_accuracy"],
            "gpt2_like_seq_acc": scaling["results"][scale]["models"]["gpt2_like"]["shot8_sequence_accuracy"],
        }
        for scale in scales
    }


def _build_model_scatter(summary: dict) -> None:
    fair = _load_json(ARTIFACTS / "benchmark_runs" / "language" / "language_recurrent_nano_tricks_fair_20260327.json")
    actual = _load_json(ARTIFACTS / "benchmark_runs" / "language" / "language_nanochat_actual_compare_1p5x_moredata_20260327.json")

    chosen = {
        "recurrent_baseline": fair["summary"]["recurrent_baseline"],
        "recurrent_champion": fair["summary"]["recurrent_champion"],
        "partial_untied": fair["summary"]["partial_untied"],
        "factorized_untied": fair["summary"]["factorized_untied"],
        "nanochat_small": actual["summary"]["nanochat_small"],
    }

    normalized = {}
    for name, row in chosen.items():
        normalized[name] = {
            "mean_final_val_loss": row.get("mean_final_val_loss", row.get("avg_final_val_loss")),
            "mean_train_tok_per_sec": row.get("mean_train_tok_per_sec", row.get("avg_train_tok_per_sec")),
            "parameter_count": row["parameter_count"],
        }

    plt.figure(figsize=(8.8, 5.4))
    colors = {
        "recurrent_baseline": "#4a5568",
        "recurrent_champion": "#2b6cb0",
        "partial_untied": "#2f855a",
        "factorized_untied": "#805ad5",
        "nanochat_small": "#dd6b20",
    }
    for name, row in normalized.items():
        plt.scatter(
            row["mean_train_tok_per_sec"] / 1000.0,
            row["mean_final_val_loss"],
            s=max(row["parameter_count"] / 30000.0, 40.0),
            color=colors[name],
            alpha=0.85,
            label=name,
        )
        plt.text(row["mean_train_tok_per_sec"] / 1000.0 + 0.2, row["mean_final_val_loss"] + 0.005, name, fontsize=8)

    plt.xlabel("Train Throughput (k tok/s)")
    plt.ylabel("Mean Final Val Loss")
    plt.title("Quality / Throughput Tradeoff")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES / "quality_throughput_scatter.png", dpi=180)
    plt.close()

    summary["model_scatter"] = normalized


def _write_summary_files(summary: dict) -> None:
    (DOCS / "results_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rows = []
    for name, row in summary["runs_50m"].items():
        rows.append(
            {
                "experiment": "50m_head_to_head",
                "model": name,
                "final_val_loss": row["final_val_loss"],
                "train_tok_per_sec": row["train_tok_per_sec"],
                "pure_train_tok_per_sec": row["pure_train_tok_per_sec"],
                "peak_vram_mb": row["peak_vram_mb"],
                "parameter_count": row["parameter_count"],
            }
        )
    for scale, row in summary["fastlearn_scaling"].items():
        rows.append(
            {
                "experiment": f"fastlearn_{scale}",
                "model": "fast_gru",
                "final_val_loss": "",
                "train_tok_per_sec": "",
                "pure_train_tok_per_sec": "",
                "peak_vram_mb": "",
                "parameter_count": "",
                "adaptation_auc": row["fast_gru_auc"],
                "sequence_accuracy": row["fast_gru_seq_acc"],
            }
        )
        rows.append(
            {
                "experiment": f"fastlearn_{scale}",
                "model": "gpt2_like",
                "final_val_loss": "",
                "train_tok_per_sec": "",
                "pure_train_tok_per_sec": "",
                "peak_vram_mb": "",
                "parameter_count": "",
                "adaptation_auc": row["gpt2_like_auc"],
                "sequence_accuracy": row["gpt2_like_seq_acc"],
            }
        )

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with (DOCS / "results_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    _ensure_dirs()
    summary: dict[str, dict] = {}
    _build_50m_curves(summary)
    _build_50m_tradeoff(summary)
    _build_fastlearn_scaling(summary)
    _build_model_scatter(summary)
    _write_summary_files(summary)
    print("Generated figures and summary files.")


if __name__ == "__main__":
    main()
