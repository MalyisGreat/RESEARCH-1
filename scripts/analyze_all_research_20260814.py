from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


REPO_ROOT = Path(__file__).resolve().parents[1]
LANGUAGE_ROOT = REPO_ROOT / "artifacts" / "benchmark_runs" / "language"
WATCH_ROOT = REPO_ROOT / "artifacts" / "watch_runs"
HUB_ROOT = REPO_ROOT / "house_compute_hub" / "runs"
OUTPUT_DIR = LANGUAGE_ROOT / "research_analysis_20260814"
PARSER_PATH = LANGUAGE_ROOT / "plot_research_curves_20260614.py"

PALETTE = {
    "long_context_lm": "#2563eb",
    "short_lm": "#0f766e",
    "neuron_search": "#d97706",
    "synthetic_fastlearn": "#9333ea",
    "other": "#64748b",
}

DISPLAY_LABEL_OVERRIDES = {
    "20260604-022628-5c855b": "Dense anchor 160M continuation 2B to 5B",
    "20260603-181136-f3f589": "Dense anchor 160M initial run to 2B",
    "20260605-024617-ebeaa2": "Wave10 76M direct run to 2.20B",
    "20260614-235439-ba54d5": "Wave10 76M continuation 2.24B to 2.54B",
    "20260615-133335-509e45": "Wave10 76M continuation 2.90B to 3.05B",
    "20260615-141307-266267": "Wave10 76M continuation 3.10B to 4.01B",
}


def load_parser():
    spec = importlib.util.spec_from_file_location("research_curve_parser", PARSER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load parser: {PARSER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.OUTPUT_DIR = OUTPUT_DIR
    module.HUB_RUNS_ROOT = HUB_ROOT
    return module


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        try:
            value = float(value.replace(",", ""))
            return value if math.isfinite(value) else None
        except ValueError:
            return None
    return None


def regime(label: str, source: str) -> str:
    text = f"{label} {source}".lower()
    if any(key in text for key in ("neuron", "manual_self", "neuron_search")):
        return "neuron_search"
    if any(key in text for key in ("fastlearn", "gpt2icl", "adaptation_auc")):
        return "synthetic_fastlearn"
    if any(
        key in text
        for key in (
            "longseq",
            "wave",
            "lowrank",
            "low_rank",
            "anchor",
            "subquadratic",
            "token_recall",
            "training_system_search",
        )
    ):
        return "long_context_lm"
    if any(
        key in text
        for key in (
            "nanochat",
            "partial_untied",
            "recurrent",
            "realtext",
            "language_",
            "gpt2",
            "gru_only",
        )
    ):
        return "short_lm"
    return "other"


def parse_jsonl(parser) -> list[Any]:
    points: list[Any] = []
    point_type = parser.Point
    roots = [LANGUAGE_ROOT, WATCH_ROOT]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.jsonl"):
            if OUTPUT_DIR in path.parents:
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line in lines:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                val = number(payload.get("val_loss", payload.get("latest_val_loss")))
                tokens = number(payload.get("tokens_seen"))
                if val is None or tokens is None or tokens <= 0:
                    continue
                points.append(
                    point_type(
                        source=str(path),
                        label=str(payload.get("run_name") or path.parent.name),
                        tokens=tokens,
                        val_loss=val,
                        train_loss=number(payload.get("train_loss", payload.get("latest_train_loss"))),
                        step=number(payload.get("step")),
                        params=number(payload.get("parameter_count")),
                        kind="jsonl_metric",
                    )
                )
    return points


def parse_bom_json(parser, json_files: list[Path]) -> list[Any]:
    """Recover valid result files written with a UTF-8 BOM."""
    points: list[Any] = []
    for path in json_files:
        if OUTPUT_DIR in path.parents:
            continue
        try:
            raw = path.read_bytes()
            if not raw.startswith(b"\xef\xbb\xbf"):
                continue
            payload = json.loads(raw.decode("utf-8-sig"))
        except Exception:
            continue
        if isinstance(payload, dict):
            parser.parse_result_like_dict(points, path, payload)
    return points


def parse_csv(parser) -> list[Any]:
    points: list[Any] = []
    point_type = parser.Point
    roots = [LANGUAGE_ROOT, WATCH_ROOT]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.csv"):
            if OUTPUT_DIR in path.parents:
                continue
            try:
                with path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    fields = {field.strip().lower() for field in (reader.fieldnames or [])}
                    if not ({"val_loss", "validation_loss"} & fields):
                        continue
                    for row in reader:
                        val = number(row.get("val_loss") or row.get("validation_loss"))
                        tokens = number(row.get("tokens_seen") or row.get("tokens") or row.get("train_tokens_seen"))
                        step = number(row.get("step"))
                        if tokens is None and step is not None:
                            tokens = step
                        if val is None or tokens is None or tokens <= 0:
                            continue
                        row_source = str(row.get("source") or "").strip()
                        label = str(row.get("label") or row.get("variant") or path.stem)
                        if row_source:
                            source_path = Path(row_source)
                            source_id = source_path.parent.name or source_path.stem
                            label = f"{label} [{source_id}]"
                        points.append(
                            point_type(
                                source=row_source or str(path),
                                label=label,
                                tokens=tokens,
                                val_loss=val,
                                train_loss=number(row.get("train_loss")),
                                step=step,
                                params=number(row.get("params") or row.get("parameter_count")),
                                kind="csv_curve",
                            )
                        )
            except (OSError, UnicodeError):
                continue
    return points


def metadata_for_source(source: str) -> dict[str, float | None]:
    path = Path(source)
    result = {"train_tok_per_sec": None, "peak_vram_mb": None, "parameter_count": None}
    if path.suffix.lower() != ".json" or not path.exists():
        return result
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return result

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                lowered = key.lower()
                if lowered in {"train_tok_per_sec", "avg_train_tok_per_sec", "pure_train_tok_per_sec"} and result["train_tok_per_sec"] is None:
                    result["train_tok_per_sec"] = number(item)
                elif lowered in {"peak_vram_mb", "avg_peak_vram_mb"} and result["peak_vram_mb"] is None:
                    result["peak_vram_mb"] = number(item)
                elif lowered in {"parameter_count", "params"} and result["parameter_count"] is None:
                    result["parameter_count"] = number(item)
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(payload)
    return result


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_inventory() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in (LANGUAGE_ROOT, WATCH_ROOT):
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or OUTPUT_DIR in path.parents:
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            lower = str(path).lower()
            if any(token in lower for token in ("\\cache\\", "checkpoint", ".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".parquet")):
                category = "heavy_artifact"
            elif path.suffix.lower() in {".json", ".jsonl", ".csv", ".log", ".txt", ".md"}:
                category = "result_or_document"
            elif path.suffix.lower() in {".py", ".ps1", ".sh"}:
                category = "source_or_command"
            else:
                category = "other"
            rows.append({"path": str(path.relative_to(REPO_ROOT)), "bytes": size, "extension": path.suffix.lower(), "category": category})
    return sorted(rows, key=lambda row: row["path"])


def plot_all_curves(groups: dict[str, list[Any]], path: Path, ymax: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(16, 10), dpi=180)
    seen = set()
    for label, values in groups.items():
        if len(values) < 2:
            continue
        kind = regime(label, values[0].source)
        xs = [point.tokens / 1e9 for point in values if point.tokens > 0]
        ys = [point.val_loss for point in values if point.tokens > 0]
        if len(xs) < 2:
            continue
        color = PALETTE[kind]
        # Keep the raw view restrained, but make the clipped/readable view
        # legible when hundreds of curves overlap.
        opacity = 0.26 if ymax is not None else 0.18
        width = 0.95 if ymax is not None else 0.8
        ax.plot(xs, ys, color=color, linewidth=width, alpha=opacity)
        seen.add(kind)
    for kind, color in PALETTE.items():
        if kind in seen:
            ax.plot([], [], color=color, linewidth=2.8, label=kind.replace("_", " "))
    ax.set_xscale("log")
    ax.set_xlabel("training tokens (billions, log scale)")
    ax.set_ylabel("validation loss (lower is better)")
    ax.set_title("All observed validation-loss curves (sparse checkpoints)")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.45)
    if ymax is not None:
        ax.set_ylim(3.8, ymax)
        ax.text(0.01, 0.015, f"gaps = no recorded evaluation; view clipped at loss {ymax:g}", transform=ax.transAxes, fontsize=8, color="#475569")
    ax.legend(framealpha=0.92)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_major_curves(groups: dict[str, list[Any]], path: Path) -> None:
    candidates = [
        (label, values)
        for label, values in groups.items()
        if len(values) >= 2 and max(p.tokens for p in values) >= 1e8 and max(p.val_loss for p in values) <= 11.5
    ]
    candidates.sort(key=lambda item: max(p.tokens for p in item[1]), reverse=True)
    fig, ax = plt.subplots(figsize=(14, 8), dpi=180)
    selected = {
        label
        for label, values in sorted(candidates, key=lambda item: min(p.val_loss for p in item[1]))[:8]
    }
    for index, (label, values) in enumerate(candidates):
        xs = [p.tokens / 1e9 for p in values if p.tokens > 0]
        ys = [p.val_loss for p in values if p.tokens > 0]
        if len(xs) < 2:
            continue
        highlight = label in selected
        color = PALETTE[regime(label, values[0].source)] if highlight else "#94a3b8"
        line = ax.plot(xs, ys, color=color, linewidth=1.8 if highlight else 0.6, alpha=0.9 if highlight else 0.18, label=label if highlight else None)
    ax.set_xscale("log")
    ax.set_xlabel("training tokens (billions, log scale)")
    ax.set_ylabel("validation loss")
    ax.set_title("Major runs: validation loss versus training tokens")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.45)
    ax.set_ylim(3.8, 11.2)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), framealpha=0.92, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_clean_long_context_scaling(groups: dict[str, list[Any]], path: Path) -> None:
    """Publication-style view of the three comparable mature LM families."""
    fig, ax = plt.subplots(figsize=(14, 8), dpi=200)
    families = [
        {
            "name": "Wave10 76M low-rank conv-memory",
            "color": "#2563eb",
            "marker": "o",
            "linestyle": "-",
            "sources": [
                "20260605-024617-ebeaa2",
                "20260614-235439-ba54d5",
                "20260615-133335-509e45",
                "20260615-141307-266267",
            ],
            "labels": [],
        },
        {
            "name": "Dense anchor 160M",
            "color": "#d97706",
            "marker": "s",
            "linestyle": "--",
            "sources": ["20260603-181136-f3f589", "20260604-022628-5c855b"],
            "labels": [],
        },
        {
            "name": "Dense anchor 80M",
            "color": "#0f766e",
            "marker": "^",
            "linestyle": "-.",
            "sources": [],
            "labels": [
                "language_longseq_anchor16_80m_fresh1b_after2b_seq10160_seed13_20260603/causal_conv_mixer_sampled_vocab_anchor16",
            ],
        },
    ]

    plotted: dict[str, list[Any]] = {}
    for family in families:
        segments: list[list[Any]] = []
        for source_id in family["sources"]:
            matches = [values for values in groups.values() if source_id in str(values[0].source)]
            if matches:
                segments.append(max(matches, key=len))
        for label in family["labels"]:
            values = groups.get(label)
            if values:
                segments.append(values)

        family_points: list[Any] = []
        for index, values in enumerate(segments):
            observed = sorted((p for p in values if p.tokens >= 5e7 and p.val_loss <= 6.2), key=lambda p: p.tokens)
            if len(observed) < 2:
                continue
            ax.plot(
                [p.tokens / 1e9 for p in observed],
                [p.val_loss for p in observed],
                color=family["color"],
                linewidth=2.25,
                linestyle=family["linestyle"],
                marker=family["marker"],
                markersize=4.2,
                markerfacecolor="white",
                markeredgewidth=1.1,
                label=family["name"] if index == 0 else None,
                zorder=3,
            )
            family_points.extend(observed)
        plotted[family["name"]] = family_points

    wave_points = plotted.get("Wave10 76M low-rank conv-memory", [])
    if wave_points:
        best = min(wave_points, key=lambda point: point.val_loss)
        ax.annotate(
            f"best observed {best.val_loss:.3f}\n{best.tokens / 1e9:.2f}B tokens",
            xy=(best.tokens / 1e9, best.val_loss),
            xytext=(-112, 32),
            textcoords="offset points",
            fontsize=9,
            color="#1e3a8a",
            arrowprops={"arrowstyle": "-", "color": "#1e3a8a", "linewidth": 1.0},
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#bfdbfe"},
        )

    ax.set_xlim(0, 5.15)
    ax.set_ylim(4.0, 6.05)
    ax.set_xlabel("training tokens (billions)")
    ax.set_ylabel("validation loss (lower is better)")
    ax.set_title(
        "Long-context language-model scaling\nObserved validation checkpoints; resumed runs remain disconnected",
        loc="left",
        fontsize=16,
        pad=14,
    )
    ax.grid(True, axis="y", color="#d7dde5", alpha=0.8, linewidth=0.7)
    ax.grid(True, axis="x", color="#e5e7eb", alpha=0.45, linewidth=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    ax.text(
        0.0,
        -0.13,
        "Lines connect checkpoints only within the same recorded run. Blank intervals contain no evaluation measurement.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#475569",
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_experiment_atlas(groups: dict[str, list[Any]], output_dir: Path) -> list[dict[str, Any]]:
    """Render every meaningful multi-point curve as an opaque small multiple."""
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[tuple[str, str, list[Any]]] = []
    seen_fingerprints = set()
    for label, values in groups.items():
        observed = sorted(
            (point for point in values if point.tokens > 0 and 0 < point.val_loss < 20),
            key=lambda point: point.tokens,
        )
        if len(observed) < 3:
            continue
        fingerprint = tuple((round(point.tokens), round(point.val_loss, 6)) for point in observed)
        if fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(fingerprint)
        candidates.append((regime(label, observed[0].source), label, observed))

    regime_order = {"long_context_lm": 0, "short_lm": 1, "other": 2, "neuron_search": 3, "synthetic_fastlearn": 4}
    candidates.sort(
        key=lambda item: (
            regime_order.get(item[0], 9),
            -max(point.tokens for point in item[2]),
            min(point.val_loss for point in item[2]),
            item[1],
        )
    )

    index_rows: list[dict[str, Any]] = []
    page_size = 16
    page_count = math.ceil(len(candidates) / page_size)
    pdf_path = output_dir.parent / "experiment_curve_atlas.pdf"
    with PdfPages(pdf_path) as pdf:
        for page_index in range(page_count):
            page_items = candidates[page_index * page_size : (page_index + 1) * page_size]
            fig, axes = plt.subplots(4, 4, figsize=(18, 14), dpi=160)
            axes_flat = list(axes.flat)
            regimes_on_page = []
            for panel_index, (kind, label, values) in enumerate(page_items):
                ax = axes_flat[panel_index]
                regimes_on_page.append(kind.replace("_", " "))
                max_tokens = max(point.tokens for point in values)
                unit = 1e9 if max_tokens >= 1e8 else 1e6
                unit_label = "B" if unit == 1e9 else "M"
                xs = [point.tokens / unit for point in values]
                ys = [point.val_loss for point in values]
                color = PALETTE[kind]
                ax.plot(
                    xs,
                    ys,
                    color=color,
                    linewidth=2.0,
                    marker="o",
                    markersize=3.4,
                    markerfacecolor="white",
                    markeredgewidth=0.9,
                    alpha=1.0,
                )
                padding = max((max(ys) - min(ys)) * 0.14, 0.04)
                ax.set_ylim(min(ys) - padding, max(ys) + padding)
                ax.grid(True, axis="y", color="#d7dde5", linewidth=0.55)
                ax.grid(True, axis="x", color="#eef1f4", linewidth=0.4)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                readable_label = label.replace("_", " ").replace("/", " / ")
                for source_id, display_label in DISPLAY_LABEL_OVERRIDES.items():
                    if source_id in str(values[0].source):
                        readable_label = f"{display_label} [{source_id}]"
                        break
                short_label = textwrap.shorten(readable_label, width=68, placeholder="...")
                wrapped_label = textwrap.fill(short_label, width=34, max_lines=2, placeholder="...")
                ax.set_title(
                    f"#{page_index * page_size + panel_index + 1:03d}  {wrapped_label}",
                    loc="left",
                    fontsize=8.2,
                    pad=5,
                )
                ax.set_xlabel(f"tokens ({unit_label})", fontsize=7.5)
                ax.set_ylabel("val loss", fontsize=7.5)
                ax.tick_params(labelsize=7)
                ax.text(
                    0.99,
                    0.97,
                    f"final {ys[-1]:.3f}  |  best {min(ys):.3f}\nn={len(values)}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=7,
                    color="#334155",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 1.5},
                )
                index_rows.append(
                    {
                        "atlas_id": page_index * page_size + panel_index + 1,
                        "page": page_index + 1,
                        "panel": panel_index + 1,
                        "regime": kind,
                        "label": label,
                        "points": len(values),
                        "max_tokens": max_tokens,
                        "final_val_loss": ys[-1],
                        "best_val_loss": min(ys),
                        "source": values[0].source,
                    }
                )
            for ax in axes_flat[len(page_items) :]:
                ax.axis("off")
            page_regimes = ", ".join(dict.fromkeys(regimes_on_page))
            fig.suptitle(
                f"Experiment curve atlas — page {page_index + 1} of {page_count}\n{page_regimes}; opaque observed checkpoints",
                x=0.04,
                y=0.995,
                ha="left",
                fontsize=16,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.0, w_pad=1.6)
            page_path = output_dir / f"experiment_curve_atlas_page_{page_index + 1:02d}.png"
            fig.savefig(page_path, bbox_inches="tight", facecolor="white")
            pdf.savefig(fig, bbox_inches="tight", facecolor="white")
            plt.close(fig)
    return index_rows


def plot_opaque_experiment_overlay(groups: dict[str, list[Any]], path: Path) -> int:
    """Put every unique multi-checkpoint experiment on one full-opacity canvas."""
    candidates: list[tuple[str, str, list[Any]]] = []
    seen_fingerprints = set()
    for label, values in groups.items():
        observed = sorted(
            (point for point in values if point.tokens > 0 and 0 < point.val_loss < 20),
            key=lambda point: point.tokens,
        )
        if len(observed) < 3:
            continue
        fingerprint = tuple((round(point.tokens), round(point.val_loss, 6)) for point in observed)
        if fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(fingerprint)
        candidates.append((regime(label, observed[0].source), label, observed))

    candidates.sort(key=lambda item: max(point.tokens for point in item[2]))
    markers = {"long_context_lm": "o", "short_lm": "s", "other": "D", "neuron_search": "^", "synthetic_fastlearn": "v"}
    line_styles = ["-", "--", "-.", ":"]
    fig, ax = plt.subplots(figsize=(24, 14), dpi=200)
    for index, (kind, _label, values) in enumerate(candidates):
        xs = [point.tokens / 1e9 for point in values]
        ys = [point.val_loss for point in values]
        is_wave10 = any(source_id in str(values[0].source) for source_id in (
            "20260605-024617-ebeaa2",
            "20260614-235439-ba54d5",
            "20260615-133335-509e45",
            "20260615-141307-266267",
        ))
        ax.plot(
            xs,
            ys,
            color="#1d4ed8" if is_wave10 else PALETTE[kind],
            linewidth=2.7 if is_wave10 else 1.35,
            linestyle="-" if is_wave10 else line_styles[index % len(line_styles)],
            marker=markers.get(kind, "o"),
            markersize=3.0 if is_wave10 else 2.2,
            markerfacecolor="white",
            markeredgewidth=0.75,
            alpha=1.0,
            zorder=5 if is_wave10 else 2,
        )

    for kind in ("long_context_lm", "short_lm", "other"):
        if any(candidate[0] == kind for candidate in candidates):
            ax.plot(
                [],
                [],
                color=PALETTE[kind],
                linewidth=2.5,
                marker=markers[kind],
                markerfacecolor="white",
                label=kind.replace("_", " "),
            )
    ax.set_xscale("log")
    ax.set_xlim(8e-6, 5.3)
    ax.set_ylim(4.0, 16.2)
    ax.set_xlabel("training tokens (billions, log scale)", fontsize=12)
    ax.set_ylabel("validation loss (lower is better)", fontsize=12)
    ax.set_title(
        f"All {len(candidates)} multi-checkpoint experiments on one graph\n"
        "Fully opaque observed curves; colors indicate evaluation regime, not direct comparability",
        loc="left",
        fontsize=20,
        pad=16,
    )
    ax.grid(True, which="major", color="#cbd5e1", linewidth=0.7)
    ax.grid(True, which="minor", color="#e5e7eb", linewidth=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=11)
    ax.text(
        0.0,
        -0.08,
        "Each line is one unique checkpoint history with at least three evaluations. Exact duplicates removed; no smoothing or gap filling.",
        transform=ax.transAxes,
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return len(candidates)


def plot_final_scatter(rows: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 8), dpi=180)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["max_tokens"] and row["final_val_loss"] and row["final_val_loss"] < 20:
            grouped.setdefault(row["regime"], []).append(row)
    for kind, values in grouped.items():
        ax.scatter([r["max_tokens"] / 1e9 for r in values], [r["final_val_loss"] for r in values], s=20, alpha=0.45, color=PALETTE[kind], label=kind.replace("_", " "))
    ax.set_xscale("log")
    ax.set_xlabel("final training tokens (billions, log scale)")
    ax.set_ylabel("final validation loss")
    ax.set_title("Final validation loss for every extracted curve")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.45)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), framealpha=0.92, fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_params(rows: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 8), dpi=180)
    for kind, color in PALETTE.items():
        values = [r for r in rows if r["params"] and r["final_val_loss"] and r["final_val_loss"] < 20 and r["regime"] == kind]
        if not values:
            continue
        ax.scatter([r["params"] / 1e6 for r in values], [r["final_val_loss"] for r in values], s=22, alpha=0.45, color=color, label=kind.replace("_", " "))
    ax.set_xscale("log")
    ax.set_xlabel("parameter count (millions, log scale)")
    ax.set_ylabel("final validation loss")
    ax.set_title("Parameter scale versus final validation loss")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.45)
    ax.legend(framealpha=0.92)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_throughput(rows: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 8), dpi=180)
    for kind, color in PALETTE.items():
        values = [r for r in rows if r["throughput"] and r["final_val_loss"] and r["final_val_loss"] < 20 and r["regime"] == kind]
        if not values:
            continue
        ax.scatter([r["throughput"] / 1000 for r in values], [r["final_val_loss"] for r in values], s=22, alpha=0.45, color=color, label=kind.replace("_", " "))
    ax.set_xlabel("training throughput (thousand tokens/s)")
    ax.set_ylabel("final validation loss")
    ax.set_title("Training throughput versus final validation loss")
    ax.grid(True, alpha=0.25, linewidth=0.45)
    ax.legend(framealpha=0.92)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_frontier(rows: list[dict[str, Any]], path: Path) -> None:
    values = [r for r in rows if r["max_tokens"] and r["final_val_loss"] and r["final_val_loss"] < 20]
    values.sort(key=lambda row: row["max_tokens"])
    frontier: list[dict[str, Any]] = []
    best = float("inf")
    for row in values:
        if row["final_val_loss"] < best:
            best = row["final_val_loss"]
            frontier.append(row)
    fig, ax = plt.subplots(figsize=(13, 8), dpi=180)
    ax.scatter([r["max_tokens"] / 1e9 for r in values], [r["final_val_loss"] for r in values], s=12, alpha=0.18, color="#64748b")
    ax.plot([r["max_tokens"] / 1e9 for r in frontier], [r["final_val_loss"] for r in frontier], marker="o", color="#dc2626", linewidth=2.0, label="empirical best-so-far frontier")
    ax.set_xscale("log")
    ax.set_xlabel("final training tokens (billions, log scale)")
    ax.set_ylabel("final validation loss")
    ax.set_title("Empirical quality frontier across all extracted runs")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.45)
    ax.legend(framealpha=0.92)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def collect_fastlearn(json_files: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in json_files:
        if "fastlearn" not in path.name.lower() or path.name.lower() in {"summary.json", "state.json"}:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, dict):
            continue
        for scale, scale_payload in results.items():
            if not isinstance(scale_payload, dict):
                continue
            models = scale_payload.get("models")
            if not isinstance(models, dict):
                continue
            for model, metrics in models.items():
                if not isinstance(metrics, dict):
                    continue
                row = {"source": str(path.relative_to(REPO_ROOT)), "scale": scale, "model": model}
                for key in (
                    "parameter_count_mean",
                    "adaptation_auc_mean",
                    "autoregressive_adaptation_auc_mean",
                    "shot8_token_accuracy",
                    "shot8_autoregressive_token_accuracy",
                    "shot8_sequence_accuracy",
                    "shot8_autoregressive_sequence_accuracy",
                ):
                    row[key] = metrics.get(key)
                rows.append(row)
    return rows


def plot_fastlearn(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=180)
    for model, color in (("fast_gru", "#9333ea"), ("gpt2_like", "#64748b")):
        values = [row for row in rows if row["model"] == model and number(row.get("parameter_count_mean")) and number(row.get("adaptation_auc_mean"))]
        values.sort(key=lambda row: number(row["parameter_count_mean"]) or 0)
        if not values:
            continue
        axes[0].plot([number(row["parameter_count_mean"]) / 1000 for row in values], [number(row["adaptation_auc_mean"]) for row in values], marker="o", color=color, label=model)
        axes[1].plot([number(row["parameter_count_mean"]) / 1000 for row in values], [number(row.get("shot8_token_accuracy")) for row in values], marker="o", color=color, label=model)
    axes[0].set_title("Synthetic fast-learning adaptation AUC")
    axes[0].set_ylabel("adaptation AUC")
    axes[1].set_title("Synthetic fast-learning shot-8 token accuracy")
    axes[1].set_ylabel("token accuracy")
    for ax in axes:
        ax.set_xlabel("parameters (thousands)")
        ax.grid(True, alpha=0.25, linewidth=0.45)
        ax.legend(framealpha=0.92)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_neurons(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(13, 8), dpi=180)
    for steps, color in ((1024, "#d97706"), (2048, "#16a34a")):
        values = [row for row in rows if row.get("steps") == steps]
        if not values:
            continue
        ax.scatter([row["mean_speed_ratio"] for row in values], [row["mean_val_delta"] for row in values], s=[24 + 8 * min(row["n"], 6) for row in values], color=color, alpha=0.65, label=f"{steps} steps")
    ax.axhline(0, color="#334155", linewidth=0.8)
    ax.axvline(1, color="#334155", linewidth=0.8)
    ax.set_xlabel("speed ratio versus matched baseline")
    ax.set_ylabel("validation-loss delta versus baseline")
    ax.set_title("Neuron-search quality/speed tradeoff")
    ax.grid(True, alpha=0.25, linewidth=0.45)
    ax.legend(framealpha=0.92)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parser = load_parser()
    json_files = list(LANGUAGE_ROOT.rglob("*.json")) + list(WATCH_ROOT.rglob("*.json"))
    jsonl_files = list(LANGUAGE_ROOT.rglob("*.jsonl")) + list(WATCH_ROOT.rglob("*.jsonl"))
    csv_files = list(LANGUAGE_ROOT.rglob("*.csv")) + list(WATCH_ROOT.rglob("*.csv"))
    parse_errors = 0
    for path in json_files:
        if OUTPUT_DIR in path.parents:
            continue
        try:
            json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            parse_errors += 1

    raw_points = parser.parse_json_files() + parse_bom_json(parser, json_files) + parse_jsonl(parser) + parse_csv(parser) + parser.parse_log_files()
    points = parser.dedupe(raw_points)
    groups = parser.dedupe_groups_by_curve(parser.grouped(points))
    curve_rows = parser.final_rows(groups)

    rows: list[dict[str, Any]] = []
    for row in curve_rows:
        source = Path(row["source"])
        metadata = metadata_for_source(row["source"])
        params = row["params"] or metadata["parameter_count"]
        rows.append(
            {
                **row,
                "regime": regime(row["label"], row["source"]),
                "params": params,
                "throughput": metadata["train_tok_per_sec"],
                "peak_vram_mb": metadata["peak_vram_mb"],
                "source": str(source.relative_to(REPO_ROOT)) if source.is_relative_to(REPO_ROOT) else str(source),
            }
        )

    point_rows = []
    for point in sorted(points, key=lambda item: (item.label, item.tokens, item.source)):
        source = Path(point.source)
        point_rows.append(
            {
                "label": point.label,
                "regime": regime(point.label, point.source),
                "tokens": point.tokens,
                "val_loss": point.val_loss,
                "train_loss": point.train_loss,
                "step": point.step,
                "params": point.params,
                "kind": point.kind,
                "source": str(source.relative_to(REPO_ROOT)) if source.is_relative_to(REPO_ROOT) else str(source),
            }
        )

    write_csv(OUTPUT_DIR / "all_points.csv", point_rows, ["label", "regime", "tokens", "val_loss", "train_loss", "step", "params", "kind", "source"])
    write_csv(OUTPUT_DIR / "curve_summary.csv", rows, ["label", "regime", "points", "max_tokens", "final_val_loss", "best_val_loss", "best_tokens", "params", "throughput", "peak_vram_mb", "kind", "source"])

    neuron_rows = parser.parse_neuron_rows()
    write_csv(OUTPUT_DIR / "neuron_search_summary.csv", neuron_rows, list(neuron_rows[0].keys()) if neuron_rows else ["variant", "steps", "n", "wins", "mean_val_delta", "best_val_delta", "worst_val_delta", "mean_speed_ratio", "mean_param_delta"])
    fastlearn_rows = collect_fastlearn(json_files)
    write_csv(OUTPUT_DIR / "fastlearn_scaling.csv", fastlearn_rows, ["source", "scale", "model", "parameter_count_mean", "adaptation_auc_mean", "autoregressive_adaptation_auc_mean", "shot8_token_accuracy", "shot8_autoregressive_token_accuracy", "shot8_sequence_accuracy", "shot8_autoregressive_sequence_accuracy"])
    inventory = build_inventory()
    write_csv(OUTPUT_DIR / "artifact_inventory.csv", inventory, ["path", "bytes", "extension", "category"])

    quality = {
        "analysis_date": "2026-08-14",
        "json_files_seen": len(json_files),
        "jsonl_files_seen": len(jsonl_files),
        "csv_files_seen": len(csv_files),
        "json_parse_errors": parse_errors,
        "raw_points": len(raw_points),
        "deduped_points": len(points),
        "unique_curves": len(rows),
        "fastlearn_rows": len(fastlearn_rows),
        "neuron_rows": len(neuron_rows),
        "inventory_files": len(inventory),
        "inventory_bytes": sum(row["bytes"] for row in inventory),
        "inventory_heavy_bytes": sum(row["bytes"] for row in inventory if row["category"] == "heavy_artifact"),
        "regime_counts": {kind: sum(1 for row in rows if row["regime"] == kind) for kind in PALETTE},
        "notes": [
            "Curves are extracted from result JSON, JSONL metrics, CSV curves, and recognized log evaluations.",
            "Losses from different tokenizers, objectives, contexts, and validation sets are retained but classified; they are not treated as one causal scaling law.",
            "Large caches and checkpoints are intentionally not loaded; only metadata/result files are read.",
        ],
    }
    (OUTPUT_DIR / "quality_audit.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")

    plot_all_curves(groups, OUTPUT_DIR / "all_curves_validation_loss_vs_tokens.png")
    plot_all_curves(groups, OUTPUT_DIR / "all_curves_validation_loss_vs_tokens_clipped.png", ymax=12.0)
    plot_clean_long_context_scaling(groups, OUTPUT_DIR / "clean_long_context_scaling.png")
    opaque_overlay_count = plot_opaque_experiment_overlay(groups, OUTPUT_DIR / "all_experiments_opaque_overlay.png")
    atlas_rows = plot_experiment_atlas(groups, OUTPUT_DIR / "experiment_curve_atlas_pages")
    write_csv(
        OUTPUT_DIR / "experiment_curve_atlas_index.csv",
        atlas_rows,
        ["atlas_id", "page", "panel", "regime", "label", "points", "max_tokens", "final_val_loss", "best_val_loss", "source"],
    )
    plot_major_curves(groups, OUTPUT_DIR / "major_curves_validation_loss_vs_tokens.png")
    plot_final_scatter(rows, OUTPUT_DIR / "all_final_loss_vs_tokens.png")
    plot_params(rows, OUTPUT_DIR / "final_loss_vs_parameters.png")
    plot_throughput(rows, OUTPUT_DIR / "throughput_vs_final_loss.png")
    plot_frontier(rows, OUTPUT_DIR / "empirical_quality_frontier.png")
    plot_fastlearn(fastlearn_rows, OUTPUT_DIR / "fastlearn_scaling.png")
    plot_neurons(neuron_rows, OUTPUT_DIR / "neuron_search_tradeoff.png")

    ranked_candidates = sorted(
        [row for row in rows if row["regime"] == "long_context_lm" and row["max_tokens"] and row["max_tokens"] >= 1e8],
        key=lambda row: row["final_val_loss"] or 99,
    )
    long_rows = []
    seen_rank_rows = set()
    for row in ranked_candidates:
        key = (row["source"], round(row["max_tokens"]), round(row["final_val_loss"] or 99, 6))
        if key in seen_rank_rows:
            continue
        seen_rank_rows.add(key)
        long_rows.append(row)
        if len(long_rows) == 20:
            break
    report_lines = [
        "# Complete Research Analysis",
        "",
        "Generated 2026-08-14 from the repository's result JSON, JSONL metrics, CSV curves, and recognized logs.",
        "",
        "## Coverage",
        "",
        f"- {quality['json_files_seen']} JSON files, {quality['jsonl_files_seen']} JSONL files, and {quality['csv_files_seen']} CSV files were inspected.",
        f"- {quality['deduped_points']} deduplicated validation-loss points form {quality['unique_curves']} unique curves.",
        f"- {opaque_overlay_count} unique multi-checkpoint experiments are shown together in the fully opaque overlay.",
        f"- {len(atlas_rows)} distinct experiments with at least three checkpoints are rendered across {math.ceil(len(atlas_rows) / 16)} atlas pages.",
        f"- Regime counts: " + ", ".join(f"{kind}={count}" for kind, count in quality["regime_counts"].items()) + ".",
        "- Raw caches and checkpoints were not loaded; their result metadata remains represented through result files and manifests.",
        "",
        "## Quality Rules",
        "",
        "- All curves are preserved in `all_points.csv`; no curve was silently dropped because it used a different tokenizer or objective.",
        "- Regime colors are descriptive, not evidence that losses are directly comparable across regimes.",
        "- The causal scaling frontier is only a descriptive best-so-far envelope, not a fitted scaling law.",
        "- Some rows are derived checkpoint-history CSVs or previously extracted curve tables; they are retained for provenance but should not be counted as independent training runs.",
        "",
        "## Findings",
        "",
        "- The strongest long-context result is Wave10 76M low-rank conv-memory at about 4.0899 validation loss around 4.013B tokens. Its direct run reaches 4.1517 at 2.2B, followed by three separately recorded continuation segments through 4.013B.",
        "- A prior comparison CSV merged those Wave10 continuation segments with a different 160M dense-anchor run. The analysis now groups CSV rows by original run source; the apparent 4.09-to-4.58 zig-zag was a data-joining error, not model deterioration.",
        "- The 160M dense anchor reaches 4.5841 at 5B tokens and 4.6422 at 2B, so the larger dense line did not beat the best low-rank conv-memory checkpoint.",
        "- The earlier 8M local short-context comparison remains a separate regime: partial-untied is about 5.33 versus the NanoChat-inspired mini-port about 5.40 at 50M cached-token exposures.",
        "- Synthetic fast-learning screens favor fast_gru over the GPT2-like control at every tested small/medium/large scale, but those are adaptation metrics, not language-model validation loss.",
        "- Neuron-search deltas include strong one-seed rows and a repeated three-seed hidden-drop-square signal; they remain short screens and are not scale-cleared architecture wins.",
        "",
        "## Long-Run Ranking",
        "",
        "| Curve | Regime | Tokens | Final loss | Best loss | Params | Source |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in long_rows:
        report_lines.append(f"| `{row['label']}` | {row['regime']} | {row['max_tokens']:.0f} | {row['final_val_loss']:.4f} | {row['best_val_loss']:.4f} | {row['params'] or ''} | `{row['source']}` |")
    report_lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `clean_long_context_scaling.png`: primary, publication-style comparison of the three mature comparable LM families.",
            "- `all_experiments_opaque_overlay.png`: every unique multi-checkpoint experiment on one large, fully opaque graph.",
            "- `experiment_curve_atlas.pdf`: every unique experiment with at least three observed validation checkpoints, one opaque curve per panel.",
            "- `experiment_curve_atlas_pages/` and `experiment_curve_atlas_index.csv`: page images and source index for the atlas.",
            "- `all_curves_validation_loss_vs_tokens.png`: every extracted multi-point curve, colored by regime.",
            "- `all_curves_validation_loss_vs_tokens_clipped.png`: readable loss view clipped at 12; raw points remain in the CSV.",
            "- `major_curves_validation_loss_vs_tokens.png`: long-running curves with readable annotations.",
            "- `all_final_loss_vs_tokens.png`: final-loss comparison across every curve.",
            "- `final_loss_vs_parameters.png`: parameter-scale comparison where parameter metadata exists.",
            "- `throughput_vs_final_loss.png`: speed/quality tradeoff where throughput metadata exists.",
            "- `empirical_quality_frontier.png`: descriptive best-so-far frontier.",
            "- `fastlearn_scaling.png` and `fastlearn_scaling.csv`: synthetic adaptation scaling curves.",
            "- `neuron_search_tradeoff.png` and `neuron_search_summary.csv`: neuron-search quality/speed curves.",
            "- `artifact_inventory.csv`: every research artifact file with size and heavy-artifact classification; caches/checkpoints are inventoried but not loaded.",
            "- `all_points.csv`, `curve_summary.csv`, and `quality_audit.json`: inspectable source tables and QA metadata.",
        ]
    )
    (OUTPUT_DIR / "research_analysis.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(OUTPUT_DIR), **quality}, indent=2))


if __name__ == "__main__":
    main()
