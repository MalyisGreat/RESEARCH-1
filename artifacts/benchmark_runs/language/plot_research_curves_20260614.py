from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


LANGUAGE_ROOT = Path(r"E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language")
HUB_RUNS_ROOT = Path(r"E:\CODEXRESEARCH\house_compute_hub\runs")
OUTPUT_DIR = LANGUAGE_ROOT / "research_curves_20260614"


@dataclass
class Point:
    source: str
    label: str
    tokens: float
    val_loss: float
    train_loss: float | None
    step: float | None
    params: float | None
    kind: str


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def to_float(value: Any) -> float | None:
    if is_number(value):
        return float(value)
    if isinstance(value, str):
        try:
            parsed = float(value.replace(",", ""))
            return parsed if math.isfinite(parsed) else None
        except ValueError:
            return None
    return None


def compact_label(path: Path, fallback: str) -> str:
    parts = path.parts
    name = path.stem
    if "variant_results" in parts:
        idx = parts.index("variant_results")
        run = parts[idx - 1] if idx > 0 else path.parent.name
        return f"{run}/{name}"
    if "manual_self" in parts:
        idx = parts.index("manual_self")
        tail = "/".join(parts[idx + 1 : -1])
        return f"neuron_search/{tail}/{name}" if tail else f"neuron_search/{name}"
    if path.suffix.lower() == ".log":
        parent = path.parent.name
        return f"hub/{parent}/{path.stem}"
    return fallback or name


def label_from_json(path: Path, payload: Any, local_label: str | None = None) -> str:
    if local_label:
        return f"{compact_label(path, path.stem)}/{local_label}"
    if isinstance(payload, dict):
        run_name = payload.get("run_name")
        variant = payload.get("variant")
        benchmark = payload.get("benchmark")
        if isinstance(run_name, str):
            return run_name
        if isinstance(variant, str):
            return f"{compact_label(path, path.stem)}/{variant}"
        if isinstance(benchmark, str) and path.name == "result.json":
            return f"{compact_label(path, path.parent.name)}/{benchmark}"
    return compact_label(path, path.stem)


def config_sequence_length(context: dict[str, Any] | None) -> float | None:
    if not context:
        return None
    cfg = context.get("config") if isinstance(context.get("config"), dict) else context
    for key in ("sequence_length", "seq", "block_size"):
        value = to_float(cfg.get(key)) if isinstance(cfg, dict) else None
        if value and value > 0:
            return value
    return None


def params_from_context(context: dict[str, Any] | None) -> float | None:
    if not context:
        return None
    for key in ("parameter_count", "params", "candidate_params", "baseline_params"):
        value = to_float(context.get(key))
        if value:
            return value
    report = context.get("report") if isinstance(context.get("report"), dict) else None
    if report:
        return params_from_context(report)
    return None


def append_history_points(
    out: list[Point],
    *,
    path: Path,
    history: list[Any],
    context: dict[str, Any] | None,
    local_label: str | None,
    kind: str,
) -> None:
    seq_len = config_sequence_length(context)
    params = params_from_context(context)
    label = label_from_json(path, context or {}, local_label)
    for entry in history:
        if not isinstance(entry, dict):
            continue
        val = to_float(entry.get("val_loss"))
        if val is None:
            val = to_float(entry.get("validation_loss"))
        if val is None:
            continue
        tokens = to_float(entry.get("tokens_seen"))
        if tokens is None:
            tokens = to_float(entry.get("train_tokens_seen"))
        step = to_float(entry.get("step"))
        if tokens is None and step is not None and seq_len:
            tokens = step * seq_len
        if tokens is None:
            continue
        train = to_float(entry.get("train_loss"))
        out.append(
            Point(
                source=str(path),
                label=label,
                tokens=tokens,
                val_loss=val,
                train_loss=train,
                step=step,
                params=params,
                kind=kind,
            )
        )


def parse_result_like_dict(out: list[Point], path: Path, payload: dict[str, Any]) -> None:
    # Common root shape: {"results": {"variant": {"history": [...]}}}
    results = payload.get("results")
    if isinstance(results, dict):
        for variant, report in results.items():
            if isinstance(report, dict) and isinstance(report.get("history"), list):
                context = {**payload, **report}
                append_history_points(
                    out,
                    path=path,
                    history=report["history"],
                    context=context,
                    local_label=str(variant),
                    kind="json_history",
                )

    # Variant result shape: {"report": {"history": [...]}}
    report = payload.get("report")
    if isinstance(report, dict) and isinstance(report.get("history"), list):
        context = {**payload, **report}
        append_history_points(
            out,
            path=path,
            history=report["history"],
            context=context,
            local_label=payload.get("variant") if isinstance(payload.get("variant"), str) else None,
            kind="json_report_history",
        )

    if isinstance(payload.get("history"), list):
        append_history_points(
            out,
            path=path,
            history=payload["history"],
            context=payload,
            local_label=payload.get("variant") if isinstance(payload.get("variant"), str) else None,
            kind="json_history",
        )

    # Final-only result files from short screens.
    final_val = to_float(payload.get("final_val_loss"))
    final_tokens = to_float(payload.get("train_tokens_seen"))
    if final_tokens is None:
        final_tokens = to_float(payload.get("tokens_seen"))
    final_step = to_float(payload.get("step"))
    seq_len = config_sequence_length(payload)
    if final_tokens is None and final_step is not None and seq_len:
        final_tokens = final_step * seq_len
    if final_val is not None and final_tokens is not None:
        out.append(
            Point(
                source=str(path),
                label=label_from_json(path, payload),
                tokens=final_tokens,
                val_loss=final_val,
                train_loss=to_float(payload.get("final_train_loss")),
                step=final_step,
                params=params_from_context(payload),
                kind="json_final",
            )
        )


def parse_json_files() -> list[Point]:
    points: list[Point] = []
    for path in LANGUAGE_ROOT.rglob("*.json"):
        if OUTPUT_DIR in path.parents:
            continue
        lower_name = path.name.lower()
        if (
            lower_name == "state.json"
            or ".state.json" in lower_name
            or lower_name.endswith("_ranked_results.json")
            or lower_name.endswith("_screen_results.json")
            or lower_name.startswith("aggregate_results")
            or lower_name in {"design_notes.json", "grad_and_params.json"}
        ):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            parse_result_like_dict(points, path, payload)
        elif isinstance(payload, list):
            for index, item in enumerate(payload):
                if isinstance(item, dict):
                    parse_result_like_dict(points, path, {"results": {f"row_{index}": item}, **item})
    return points


START_RE = re.compile(r"START run=(?P<run>\S+).*?(?:params=(?P<params>[0-9,]+))?", re.IGNORECASE)
RUN_NAME_RE = re.compile(r"RUN_NAME=(?P<run>\S+)")
EVAL_RE = re.compile(
    r"EVAL step=(?P<step>\d+)/(?:\d+)\s+tokens=(?P<tokens>\d+)\s+train=(?P<train>[0-9.]+)\s+val=(?P<val>[0-9.]+)",
    re.IGNORECASE,
)


def parse_log_files() -> list[Point]:
    points: list[Point] = []
    roots = [HUB_RUNS_ROOT, LANGUAGE_ROOT]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.log"):
            if OUTPUT_DIR in path.parents:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            run = compact_label(path, path.stem)
            params: float | None = None
            for line in text.splitlines():
                start_match = START_RE.search(line)
                if start_match:
                    run = start_match.group("run")
                    param_text = start_match.group("params")
                    params = to_float(param_text) if param_text else params
                name_match = RUN_NAME_RE.search(line)
                if name_match:
                    run = name_match.group("run")
                eval_match = EVAL_RE.search(line)
                if eval_match:
                    points.append(
                        Point(
                            source=str(path),
                            label=run,
                            tokens=float(eval_match.group("tokens")),
                            val_loss=float(eval_match.group("val")),
                            train_loss=float(eval_match.group("train")),
                            step=float(eval_match.group("step")),
                            params=params,
                            kind="log_eval",
                        )
                    )
    return points


def dedupe(points: list[Point]) -> list[Point]:
    seen: set[tuple[str, str, int, str]] = set()
    clean: list[Point] = []
    for p in points:
        if not math.isfinite(p.tokens) or not math.isfinite(p.val_loss):
            continue
        if p.tokens < 0 or p.val_loss <= 0 or p.val_loss > 50:
            continue
        key = (p.source, p.label, int(round(p.tokens)), f"{p.val_loss:.8f}")
        if key in seen:
            continue
        seen.add(key)
        clean.append(p)
    return clean


def grouped(points: list[Point]) -> dict[str, list[Point]]:
    groups: dict[str, list[Point]] = {}
    for point in points:
        groups.setdefault(point.label, []).append(point)
    for values in groups.values():
        values.sort(key=lambda p: (p.tokens, p.val_loss))
    return groups


def dedupe_groups_by_curve(groups: dict[str, list[Point]]) -> dict[str, list[Point]]:
    signatures: dict[tuple[tuple[int, str], ...], str] = {}
    output: dict[str, list[Point]] = {}
    for label, values in sorted(groups.items(), key=lambda item: (len(item[0]), item[0])):
        by_tokens: dict[int, Point] = {}
        for point in values:
            by_tokens[int(round(point.tokens))] = point
        unique = sorted(by_tokens.values(), key=lambda p: p.tokens)
        signature = tuple((int(round(point.tokens)), f"{point.val_loss:.7f}") for point in unique)
        if signature in signatures:
            continue
        signatures[signature] = label
        output[label] = unique
    return output


def final_rows(groups: dict[str, list[Point]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, values in groups.items():
        if not values:
            continue
        by_tokens: dict[int, Point] = {}
        for point in values:
            by_tokens[int(round(point.tokens))] = point
        unique = sorted(by_tokens.values(), key=lambda p: p.tokens)
        final = unique[-1]
        best = min(unique, key=lambda p: p.val_loss)
        rows.append(
            {
                "label": label,
                "points": len(unique),
                "max_tokens": final.tokens,
                "final_val_loss": final.val_loss,
                "best_val_loss": best.val_loss,
                "best_tokens": best.tokens,
                "params": final.params,
                "kind": final.kind,
                "source": final.source,
            }
        )
    rows.sort(key=lambda row: (-(row["max_tokens"] or 0), row["final_val_loss"]))
    return rows


def is_long_lm_curve(label: str, values: list[Point]) -> bool:
    max_tokens = max(p.tokens for p in values)
    lower = label.lower()
    if max_tokens >= 100_000_000:
        return True
    return any(token in lower for token in ("2b", "3b", "5b", "600m", "300m", "longseq_anchor16"))


def is_highlight(label: str) -> bool:
    lower = label.lower()
    return any(
        key in lower
        for key in (
            "lowrank_conv_memory_76m",
            "low_rank_conv_memory_76m",
            "80m_2b_lr1e3",
            "80m_2b",
            "160m_5b",
            "160m_2b",
            "40m_600m",
            "40m_300m",
            "wave10_3080_lowrank",
        )
    )


def short_name(label: str, max_len: int = 74) -> str:
    label = label.replace("language_", "").replace("standalone_longseq_anchor_train_", "")
    if len(label) <= max_len:
        return label
    return "..." + label[-(max_len - 3) :]


def plot_curves(groups: dict[str, list[Point]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 9), dpi=180)
    long_items = [(label, values) for label, values in groups.items() if len(values) >= 2 and is_long_lm_curve(label, values)]
    short_items = [(label, values) for label, values in groups.items() if len(values) >= 2 and not is_long_lm_curve(label, values)]

    # Draw short probes as light context.
    for label, values in short_items:
        xs = [p.tokens / 1e9 for p in values if p.tokens > 0]
        ys = [p.val_loss for p in values if p.tokens > 0]
        if len(xs) < 2:
            continue
        ax.plot(xs, ys, color="#b9c0c9", linewidth=0.6, alpha=0.12)

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    highlight_labels: list[str] = []
    color_index = 0
    for label, values in sorted(long_items, key=lambda item: max(p.tokens for p in item[1])):
        xs = [p.tokens / 1e9 for p in values if p.tokens > 0]
        ys = [p.val_loss for p in values if p.tokens > 0]
        if len(xs) < 2:
            continue
        highlight = is_highlight(label)
        color = color_cycle[color_index % len(color_cycle)] if highlight else "#64748b"
        color_index += 1 if highlight else 0
        ax.plot(
            xs,
            ys,
            marker="o" if highlight else None,
            markersize=3.4 if highlight else 0,
            linewidth=2.3 if highlight else 0.9,
            alpha=0.98 if highlight else 0.28,
            color=color,
            label=short_name(label) if highlight else None,
            zorder=3 if highlight else 2,
        )
        if highlight:
            highlight_labels.append(label)
            ax.annotate(
                short_name(label, 44),
                xy=(xs[-1], ys[-1]),
                xytext=(5, 0),
                textcoords="offset points",
                fontsize=7.2,
                color=color,
                va="center",
            )

    ax.set_xscale("log")
    ax.set_xlabel("training tokens, billions (log scale)")
    ax.set_ylabel("validation loss (lower is better)")
    ax.set_title("Language research runs: validation loss vs training tokens")
    ax.grid(True, which="both", linewidth=0.45, alpha=0.28)
    ax.set_ylim(3.75, 11.25)
    ax.text(
        0.01,
        0.015,
        "Faint lines: extracted short/probe curves. Highlighted lines: major long/comparable LM runs. Mixed old probes may use different cache/val sizes.",
        transform=ax.transAxes,
        fontsize=8,
        color="#475569",
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper right", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_final_scatter(rows: list[dict[str, Any]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7.5), dpi=180)
    filtered = [row for row in rows if row["max_tokens"] and row["max_tokens"] > 0 and row["final_val_loss"] < 15]
    xs = [row["max_tokens"] / 1e9 for row in filtered]
    ys = [row["final_val_loss"] for row in filtered]
    colors = ["#ef4444" if is_highlight(row["label"]) else "#64748b" for row in filtered]
    sizes = [56 if is_highlight(row["label"]) else 16 for row in filtered]
    alphas = [0.95 if is_highlight(row["label"]) else 0.32 for row in filtered]
    for x, y, c, s, a, row in zip(xs, ys, colors, sizes, alphas, filtered):
        ax.scatter([x], [y], color=c, s=s, alpha=a)
        if is_highlight(row["label"]):
            ax.annotate(short_name(row["label"], 46), xy=(x, y), xytext=(5, 2), textcoords="offset points", fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("final training tokens, billions (log scale)")
    ax.set_ylabel("final validation loss (lower is better)")
    ax.set_title("Final point from every extracted language-loss run")
    ax.grid(True, which="both", linewidth=0.45, alpha=0.28)
    ax.set_ylim(3.75, 12.0)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_outputs(points: list[Point], rows: list[dict[str, Any]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "extracted_points.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["label", "tokens", "val_loss", "train_loss", "step", "params", "kind", "source"],
        )
        writer.writeheader()
        for point in sorted(points, key=lambda p: (p.label, p.tokens)):
            writer.writerow(
                {
                    "label": point.label,
                    "tokens": point.tokens,
                    "val_loss": point.val_loss,
                    "train_loss": point.train_loss,
                    "step": point.step,
                    "params": point.params,
                    "kind": point.kind,
                    "source": point.source,
                }
            )
    with (OUTPUT_DIR / "curve_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["label", "points", "max_tokens", "final_val_loss", "best_val_loss", "best_tokens", "params", "kind", "source"],
        )
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "point_count": len(points),
        "curve_count": len(rows),
        "top_by_long_final_loss": [
            row
            for row in sorted(
                [r for r in rows if r["max_tokens"] >= 100_000_000],
                key=lambda r: r["final_val_loss"],
            )[:25]
        ],
        "outputs": {
            "curves_png": str(OUTPUT_DIR / "all_language_val_loss_vs_tokens.png"),
            "final_scatter_png": str(OUTPUT_DIR / "final_points_val_loss_vs_tokens.png"),
            "points_csv": str(OUTPUT_DIR / "extracted_points.csv"),
            "summary_csv": str(OUTPUT_DIR / "curve_summary.csv"),
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    points = dedupe(parse_json_files() + parse_log_files())
    groups = dedupe_groups_by_curve(grouped(points))
    rows = final_rows(groups)
    write_outputs(points, rows)
    plot_curves(groups, OUTPUT_DIR / "all_language_val_loss_vs_tokens.png")
    plot_final_scatter(rows, OUTPUT_DIR / "final_points_val_loss_vs_tokens.png")
    print(f"POINTS={len(points)}")
    print(f"CURVES={len(rows)}")
    print(f"OUTPUT_DIR={OUTPUT_DIR}")
    for row in sorted([r for r in rows if r["max_tokens"] >= 100_000_000], key=lambda r: r["final_val_loss"])[:12]:
        print(
            f"TOP_LONG label={short_name(row['label'], 90)} "
            f"tokens={row['max_tokens']:.0f} final_val={row['final_val_loss']:.4f} best_val={row['best_val_loss']:.4f}"
        )


if __name__ == "__main__":
    main()
