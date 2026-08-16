from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO_ROOT / "artifacts" / "benchmark_runs" / "language" / "research_analysis_20260814"
OUTPUT_DIR = ANALYSIS_DIR / "html_report_20260815"
POINTS_CSV = ANALYSIS_DIR / "all_points.csv"
CURVES_CSV = ANALYSIS_DIR / "curve_summary.csv"
NEURON_CSV = ANALYSIS_DIR / "neuron_search_summary.csv"
FASTLEARN_CSV = ANALYSIS_DIR / "fastlearn_scaling.csv"
ATLAS_INDEX_CSV = ANALYSIS_DIR / "experiment_curve_atlas_index.csv"

MAJOR_SOURCE_NAMES = {
    "20260604-022628-5c855b": "Dense anchor 160M continuation 2B to 5B",
    "20260603-181136-f3f589": "Dense anchor 160M initial run to 2B",
    "20260605-024617-ebeaa2": "Wave10 76M direct run to 2.20B",
    "20260614-235439-ba54d5": "Wave10 76M continuation 2.24B to 2.54B",
    "20260615-133335-509e45": "Wave10 76M continuation 2.90B to 3.05B",
    "20260615-141307-266267": "Wave10 76M continuation 3.10B to 4.01B",
}

HORIZONS = [
    ("under_1m", "Under 1M tokens", 0, 1e6, "tokens_m", "training tokens (millions)"),
    ("1m_10m", "1M to 10M tokens", 1e6, 1e7, "tokens_m", "training tokens (millions)"),
    ("10m_50m", "10M to 50M tokens", 1e7, 5e7, "tokens_m", "training tokens (millions)"),
    ("50m_300m", "50M to 300M tokens", 5e7, 3e8, "tokens_m", "training tokens (millions)"),
    ("300m_1b", "300M to 1B tokens", 3e8, 1e9, "tokens_b", "training tokens (billions)"),
    ("1b_2_5b", "1B to 2.5B tokens", 1e9, 2.5e9, "tokens_b", "training tokens (billions)"),
    ("over_2_5b", "Over 2.5B tokens", 2.5e9, float("inf"), "tokens_b", "training tokens (billions)"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def compact_label(label: str, source: str, index: int) -> str:
    for source_id, display in MAJOR_SOURCE_NAMES.items():
        if source_id in source:
            return f"#{index:03d} {display}"
    text = label.lower()
    replacements = (
        ("language_longseq_", ""),
        ("language_", ""),
        ("/standalone_longseq_anchor_train", ""),
        ("/causal_conv_mixer_sampled_vocab_anchor16", ""),
        ("_20260603", ""),
        ("_20260604", ""),
        ("_20260605", ""),
        ("_20260617", ""),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    text = re.sub(r"[_/]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 34:
        text = text[:31].rstrip() + "..."
    return f"#{index:03d} {text}"


def architecture_family(label: str, source: str) -> str:
    text = f"{label} {source}".lower()
    if any(term in text for term in ("wave10", "lowrank", "low_rank")):
        return "Low-rank conv-memory"
    if any(term in text for term in ("sorted_induction", "token_recall", "exact_recency", "sparse_exact")):
        return "Token-addressable recall"
    if any(term in text for term in ("wave6", "multiscale", "multi_scale")):
        return "Multiscale convolution"
    if "partial_untied" in text or "partial untied" in text:
        return "Partial-untied recurrent"
    if "nanochat" in text:
        return "NanoChat-inspired mini-port"
    if any(term in text for term in ("fast_gru", "recurrent", "gru")):
        return "Recurrent / GRU"
    if any(term in text for term in ("factorized", "factor_recall")):
        return "Factorized mixer"
    if any(term in text for term in ("anchor", "dense", "conv_mixer", "wave2", "wave4")):
        return "Dense conv anchor"
    if "neuron" in text:
        return "Neuron-screen variant"
    return "Other experimental"


def compact_family(family: str) -> str:
    return {
        "Low-rank conv-memory": "LR",
        "Dense conv anchor": "DA",
        "Multiscale convolution": "MS",
        "Token-addressable recall": "TR",
        "Partial-untied recurrent": "PU",
        "NanoChat-inspired mini-port": "NC",
        "Recurrent / GRU": "RNN",
        "Factorized mixer": "FM",
        "Neuron-screen variant": "NS",
        "Other experimental": "O",
    }.get(family, family)


def infer_params_m(label: str, points: list[dict[str, Any]]) -> float | None:
    recorded = [point["params"] / 1e6 for point in points if point["params"]]
    if recorded:
        return recorded[0]
    text = label.lower()
    for pattern in (
        r"(?:anchor16|lowrank|conv_memory|dense|multiscale)[_ -]?(\d{1,3})m",
        r"(?:^|[_ /-])(350|160|90|80|76|51|40|38|20|10|8)m(?:[_ /-]|$)",
    ):
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    return None


def horizon_for(max_tokens: float) -> tuple[str, str, str, str]:
    for horizon_id, title, lower, upper, field, axis_title in HORIZONS:
        if lower <= max_tokens < upper:
            return horizon_id, title, field, axis_title
    raise AssertionError(max_tokens)


def source_ref(source: str) -> str:
    normalized = source.replace("\\", "/")
    for source_id in MAJOR_SOURCE_NAMES:
        if source_id in normalized:
            return source_id
    parts = [part for part in normalized.split("/") if part]
    return "/".join(parts[-3:]) if parts else "unknown"


def downsample_points(points: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    if len(points) <= limit:
        return points
    indexes = {round(index * (len(points) - 1) / (limit - 1)) for index in range(limit)}
    return [points[index] for index in sorted(indexes)]


def unique_curves() -> list[dict[str, Any]]:
    raw_points = read_csv(POINTS_CSV)
    atlas_by_label = {row["label"]: int(row["atlas_id"]) for row in read_csv(ATLAS_INDEX_CSV)}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_points:
        if row["label"] not in atlas_by_label:
            continue
        tokens = number(row.get("tokens"))
        val_loss = number(row.get("val_loss"))
        if not tokens or not val_loss or tokens <= 0 or not 0 < val_loss < 20:
            continue
        grouped[row["label"]].append(
            {
                "tokens": tokens,
                "val_loss": val_loss,
                "train_loss": number(row.get("train_loss")),
                "params": number(row.get("params")),
                "source": row.get("source") or "",
                "regime": row.get("regime") or "other",
            }
        )

    curves: list[dict[str, Any]] = []
    seen = set()
    candidates = []
    for label, points in grouped.items():
        points.sort(key=lambda point: point["tokens"])
        if len(points) < 3:
            continue
        fingerprint = tuple((round(point["tokens"]), round(point["val_loss"], 6)) for point in points)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        candidates.append((label, points))
    candidates.sort(key=lambda item: atlas_by_label[item[0]])

    for label, points in candidates:
        index = atlas_by_label[label]
        source = points[0]["source"]
        max_tokens = max(point["tokens"] for point in points)
        horizon_id, horizon_title, token_field, axis_title = horizon_for(max_tokens)
        curves.append(
            {
                "id": index,
                "label": label,
                "alias": compact_label(label, source, index),
                "source_ref": source_ref(source),
                "regime": points[0]["regime"],
                "family": architecture_family(label, source),
                "params_m": infer_params_m(label, points),
                "points": points,
                "point_count": len(points),
                "max_tokens": max_tokens,
                "final_loss": points[-1]["val_loss"],
                "best_loss": min(point["val_loss"] for point in points),
                "horizon_id": horizon_id,
                "horizon_title": horizon_title,
                "token_field": token_field,
                "axis_title": axis_title,
            }
        )
    return curves


def chart_source(source_id: str, label: str, path: str, description: str) -> dict[str, Any]:
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "description": description,
            "sql": f"SELECT * FROM read_csv_auto('{path}', header = true);",
            "executed_at": "2026-08-15T01:30:00-05:00",
            "tables_used": [path],
            "filters": ["finite validation loss", "positive token count", "exact curve fingerprints deduplicated"],
            "metric_definitions": [
                "training tokens: cumulative token exposures reported by the trainer or checkpoint history",
                "validation loss: recorded causal language-model validation loss; lower is better",
                "best loss: minimum recorded validation loss within one unique checkpoint history",
            ],
        },
    }


def markdown(block_id: str, body: str, source_id: str | None = None) -> dict[str, Any]:
    block: dict[str, Any] = {"id": block_id, "type": "markdown", "body": body, "layout": "full"}
    if source_id:
        block["sourceId"] = source_id
    return block


def build_artifact() -> dict[str, Any]:
    curves = unique_curves()
    by_horizon: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for curve in curves:
        by_horizon[curve["horizon_id"]].append(curve)

    sources = [
        chart_source(
            "curve_points",
            "Deduplicated experiment validation curves",
            "artifacts/benchmark_runs/language/research_analysis_20260814/all_points.csv",
            "Repository-wide curve extraction from result JSON, JSONL, CSV, and recognized logs.",
        ),
        chart_source(
            "curve_summary",
            "Experiment curve summary",
            "artifacts/benchmark_runs/language/research_analysis_20260814/curve_summary.csv",
            "Per-curve endpoints, minima, parameter metadata, throughput, and source provenance.",
        ),
        chart_source(
            "neuron_summary",
            "Neuron-search comparison summary",
            "artifacts/benchmark_runs/language/research_analysis_20260814/neuron_search_summary.csv",
            "Short matched neuron screens with validation-loss delta, speed ratio, seed count, and wins.",
        ),
        chart_source(
            "fastlearn_summary",
            "Synthetic fast-learning scaling summary",
            "artifacts/benchmark_runs/language/research_analysis_20260814/fastlearn_scaling.csv",
            "Synthetic adaptation metrics by model family and parameter scale.",
        ),
    ]

    datasets: dict[str, list[dict[str, Any]]] = {}
    charts: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    cards: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []

    long_curves = [curve for curve in curves if curve["regime"] == "long_context_lm"]
    best_mature = min((curve for curve in long_curves if curve["max_tokens"] >= 1e9), key=lambda curve: curve["best_loss"])
    max_curve = max(curves, key=lambda curve: curve["max_tokens"])
    datasets["headline"] = [
        {
            "curve_histories": len(curves),
            "long_context_histories": len(long_curves),
            "best_mature_loss": best_mature["best_loss"],
            "max_tokens_b": max_curve["max_tokens"] / 1e9,
        }
    ]
    cards.extend(
        [
            {
                "id": "card_curves",
                "dataset": "headline",
                "description": "Unique histories with at least three validation checkpoints after exact fingerprint deduplication.",
                "sourceId": "curve_summary",
                "metrics": [{"label": "Curve histories", "field": "curve_histories", "format": "number"}],
            },
            {
                "id": "card_long",
                "dataset": "headline",
                "description": "Histories classified as long-context language-model experiments.",
                "sourceId": "curve_summary",
                "metrics": [{"label": "Long-context histories", "field": "long_context_histories", "format": "number"}],
            },
            {
                "id": "card_best",
                "dataset": "headline",
                "description": "Best observed loss among histories reaching at least one billion tokens; not a confidence-adjusted estimate.",
                "sourceId": "curve_points",
                "metrics": [{"label": "Best mature loss", "field": "best_mature_loss", "format": "number"}],
            },
            {
                "id": "card_tokens",
                "dataset": "headline",
                "description": "Largest cumulative token count represented by a recorded validation history.",
                "sourceId": "curve_summary",
                "metrics": [{"label": "Maximum training", "field": "max_tokens_b", "format": "number"}],
            },
        ]
    )

    blocks.append(markdown("title", "# Research Scaling Atlas"))
    blocks.append(
        markdown(
            "technical_summary",
            "## Technical summary\n\n"
            "**Wave10 76M low-rank conv-memory is the strongest mature result in the recorded long-context evidence, reaching 4.090 validation loss at 4.01B tokens.** "
            "The dense 160M anchor reaches 4.584 at 5B, while the dense 80M history bottoms near 4.772 before ending around 4.806. "
            "This advantage is large enough to justify a controlled replication, but the repository does not yet contain a full-vocabulary, same-seed, same-token-budget scaling study that proves architecture-level superiority.\n\n"
            "The report separates 107 deduplicated checkpoint histories by maximum token exposure and architecture family. Short probes, neuron screens, synthetic adaptation tests, and mature language runs remain distinct because their losses and objectives are not interchangeable.",
            "curve_points",
        )
    )
    blocks.append({"id": "headline_metrics", "type": "metric-strip", "cardIds": [card["id"] for card in cards], "layout": "full"})

    # Curated mature comparison.
    mature_source_ids = set(MAJOR_SOURCE_NAMES)
    mature_rows = []
    for curve in curves:
        if not any(source_id in curve["source_ref"] for source_id in mature_source_ids) and not (
            curve["family"] == "Dense conv anchor" and curve["max_tokens"] >= 2e9
        ):
            continue
        for point in curve["points"]:
            if point["tokens"] < 5e7 or point["val_loss"] > 6.2:
                continue
            if any(source_id in curve["source_ref"] for source_id in (
                "20260605-024617-ebeaa2",
                "20260614-235439-ba54d5",
                "20260615-133335-509e45",
                "20260615-141307-266267",
            )):
                mature_series = "W10"
            elif any(source_id in curve["source_ref"] for source_id in (
                "20260603-181136-f3f589",
                "20260604-022628-5c855b",
            )):
                mature_series = "D160"
            else:
                mature_series = "D80"
            mature_rows.append(
                {
                    "tokens_b": point["tokens"] / 1e9,
                    "validation_loss": point["val_loss"],
                    "run": curve["alias"],
                    "series": mature_series,
                    "family": curve["family"],
                    "params_m": curve["params_m"],
                    "source_ref": curve["source_ref"],
                }
            )
    datasets["mature_scaling"] = mature_rows
    charts.append(
        {
            "id": "mature_scaling",
            "title": "Mature long-context validation curves",
            "subtitle": "Observed checkpoints from comparable sampled-vocabulary long-context runs; resume segments remain separate",
            "showDescription": True,
            "type": "line",
            "intent": "comparison",
            "question": "Which mature architecture family achieves the lowest recorded validation loss as token exposure grows?",
            "rationale": "A highlighted multi-series line chart preserves learning shape and keeps resumed histories identifiable.",
            "dataset": "mature_scaling",
            "sourceId": "curve_points",
            "encodings": {
                "x": {"field": "tokens_b", "type": "quantitative", "label": "training tokens", "unit": "B"},
                "y": {"field": "validation_loss", "type": "quantitative", "label": "validation loss"},
                "color": {"field": "series", "type": "nominal", "label": "architecture series"},
                "tooltip": [
                    {"field": "run", "type": "text", "label": "recorded run segment"},
                    {"field": "family", "type": "nominal", "label": "architecture family"},
                    {"field": "params_m", "type": "quantitative", "label": "parameters", "unit": "M"},
                    {"field": "source_ref", "type": "text", "label": "source reference"},
                ],
            },
            "xAxisTitle": "training tokens (billions)",
            "yAxisTitle": "validation loss",
            "layout": "full",
            "palette": {"kind": "categorical", "name": "research"},
            "legend": {"position": "right", "title": "architecture series"},
            "settings": {"showPoints": "always"},
            "surface": {"surface": "export", "interactiveLegend": True, "showControls": False, "viewMode": "visualization"},
        }
    )
    blocks.append(
        markdown(
            "mature_finding",
            "## Wave10 leads the mature sampled-vocabulary runs\n\n"
            "Wave10 improves rapidly through its direct 2.20B-token history and continues to 4.090 across three separately recorded resume segments. "
            "The dense 160M line continues to improve through 5B tokens but remains roughly 0.49 loss above Wave10's best observed checkpoint. "
            "The gap is promising, but it mixes architecture and training-system choices such as anchor stride and output handling; the next decisive test must hold those constant.",
            "curve_points",
        )
    )
    blocks.append({"id": "mature_chart_block", "type": "chart", "chartId": "mature_scaling", "layout": "full"})

    horizon_counts = Counter(curve["horizon_id"] for curve in curves)
    family_counts = Counter(curve["family"] for curve in curves)
    blocks.append(
        markdown(
            "horizon_intro",
            "## Token-horizon views separate screens from scaling evidence\n\n"
            "Each experiment appears in exactly one horizon according to its maximum recorded token exposure. Within a horizon, lines share a readable token unit and are grouped by architecture family in the legend. "
            "Legend codes: **LR** low-rank conv-memory, **DA** dense anchor, **MS** multiscale convolution, **TR** token recall, **PU** partial untied, **NC** NanoChat mini, **RNN** recurrent, **FM** factorized mixer, **NS** neuron screen, and **O** other. "
            "These charts are descriptive inventories: even inside one horizon, losses may use different context lengths, sampled-vocabulary settings, or validation sets.",
        )
    )

    for horizon_id, horizon_title, _lower, _upper, token_field, axis_title in HORIZONS:
        horizon_curves = by_horizon.get(horizon_id, [])
        if not horizon_curves:
            continue
        rows = []
        for curve in horizon_curves:
            for point in downsample_points(curve["points"]):
                rows.append(
                    {
                        "tokens_m": point["tokens"] / 1e6,
                        "tokens_b": point["tokens"] / 1e9,
                        "validation_loss": point["val_loss"],
                        "run": curve["alias"],
                        "family": curve["family"],
                        "family_short": compact_family(curve["family"]),
                        "regime": curve["regime"],
                        "max_tokens": curve["max_tokens"],
                        "best_loss": curve["best_loss"],
                        "point_count": curve["point_count"],
                        "source_ref": curve["source_ref"],
                    }
                )
        dataset_id = f"horizon_{horizon_id}"
        chart_id = f"chart_{horizon_id}"
        datasets[dataset_id] = rows
        best_curve = min(horizon_curves, key=lambda curve: curve["best_loss"])
        dominant_families = ", ".join(f"{name} ({count})" for name, count in Counter(curve["family"] for curve in horizon_curves).most_common(3))
        blocks.append(
            markdown(
                f"finding_{horizon_id}",
                f"## {horizon_title}: {len(horizon_curves)} distinct histories\n\n"
                f"The lowest recorded loss in this horizon is **{best_curve['best_loss']:.3f}** from `{best_curve['alias']}`. "
                f"The most represented families are {dominant_families}. "
                "Treat steep three-point lines as screens rather than established learning dynamics; longer histories carry more evidential weight.",
                "curve_points",
            )
        )
        charts.append(
            {
                "id": chart_id,
                "title": f"Validation curves: {horizon_title}",
                "subtitle": f"{len(horizon_curves)} unique histories; one line per deduplicated experiment",
                "showDescription": True,
                "type": "line",
                "intent": "trend",
                "question": f"How do architecture histories behave when training ends in the {horizon_title.lower()} band?",
                "rationale": "Horizon-specific lines prevent billion-token runs from compressing short screens into an unreadable corner.",
                "dataset": dataset_id,
                "sourceId": "curve_points",
                "encodings": {
                    "x": {"field": token_field, "type": "quantitative", "label": "training tokens"},
                    "y": {"field": "validation_loss", "type": "quantitative", "label": "validation loss"},
                    "color": {"field": "family_short", "type": "nominal", "label": "architecture family"},
                    "lineStyle": {"field": "regime", "type": "nominal", "label": "evaluation regime"},
                    "tooltip": [
                        {"field": "run", "type": "text", "label": "experiment"},
                        {"field": "family", "type": "nominal", "label": "architecture family"},
                        {"field": "regime", "type": "nominal", "label": "evaluation regime"},
                        {"field": "best_loss", "type": "quantitative", "label": "best recorded loss"},
                        {"field": "point_count", "type": "quantitative", "label": "checkpoints"},
                        {"field": "source_ref", "type": "text", "label": "source reference"},
                    ],
                },
                "xAxisTitle": axis_title,
                "yAxisTitle": "validation loss",
                "layout": "full",
                "palette": {"kind": "categorical", "name": "research"},
                "legend": {"position": "right", "title": "architecture family"},
                "settings": {"showPoints": "always"},
                "surface": {"surface": "export", "interactiveLegend": True, "showControls": False, "viewMode": "visualization"},
                "maxRows": 1200,
            }
        )
        blocks.append({"id": f"block_{chart_id}", "type": "chart", "chartId": chart_id, "layout": "full"})

    # Endpoint relationship and architecture table.
    endpoint_rows = []
    for curve in curves:
        endpoint_rows.append(
            {
                "run": curve["alias"],
                "family": curve["family"],
                "regime": curve["regime"],
                "log10_tokens": math.log10(curve["max_tokens"]),
                "max_tokens_b": curve["max_tokens"] / 1e9,
                "final_loss": curve["final_loss"],
                "best_loss": curve["best_loss"],
                "params_m": curve["params_m"],
                "checkpoints": curve["point_count"],
                "horizon": curve["horizon_title"],
                "source_ref": curve["source_ref"],
            }
        )
    datasets["endpoints"] = endpoint_rows
    charts.append(
        {
            "id": "endpoint_scatter",
            "title": "Final loss versus maximum token exposure",
            "subtitle": "Each point is one unique history; x-axis is log10(tokens), so one unit equals 10x more training",
            "showDescription": True,
            "type": "scatter",
            "intent": "relationship",
            "question": "Which architecture families occupy the strongest quality region at each training scale?",
            "rationale": "A scatter separates endpoint quality from within-run learning shape and exposes scale coverage.",
            "dataset": "endpoints",
            "sourceId": "curve_summary",
            "encodings": {
                "x": {"field": "log10_tokens", "type": "quantitative", "label": "log10 training tokens"},
                "y": {"field": "final_loss", "type": "quantitative", "label": "final validation loss"},
                "color": {"field": "regime", "type": "nominal", "label": "evaluation regime"},
                "size": {"field": "checkpoints", "type": "quantitative", "label": "recorded checkpoints"},
                "tooltip": [
                    {"field": "run", "type": "text", "label": "experiment"},
                    {"field": "max_tokens_b", "type": "quantitative", "label": "maximum tokens", "unit": "B"},
                    {"field": "best_loss", "type": "quantitative", "label": "best loss"},
                    {"field": "params_m", "type": "quantitative", "label": "parameters", "unit": "M"},
                    {"field": "horizon", "type": "nominal", "label": "token horizon"},
                    {"field": "regime", "type": "nominal", "label": "evaluation regime"},
                ],
            },
            "xAxisTitle": "log10(training tokens)",
            "yAxisTitle": "final validation loss",
            "layout": "full",
            "palette": {"kind": "categorical", "name": "research"},
            "legend": {"position": "right", "title": "evaluation regime"},
            "surface": {"surface": "export", "interactiveLegend": True, "showControls": False, "viewMode": "visualization"},
        }
    )
    blocks.append(
        markdown(
            "architecture_result",
            "## Architecture quality improves with scale, but the frontier is family-dependent\n\n"
            "The endpoint view shows two different effects: nearly every serious family improves as token exposure grows, while Wave10 occupies a distinctly lower-loss region than the dense anchors at billion-token scale. "
            "Points are colored by evaluation regime because losses should only be compared within compatible objectives and validation setups. Architecture family remains available in the evidence table and chart tooltip. Point size reflects checkpoint count, which helps distinguish sustained histories from sparse screens.",
            "curve_summary",
        )
    )
    blocks.append({"id": "endpoint_scatter_block", "type": "chart", "chartId": "endpoint_scatter", "layout": "full"})

    family_rows = []
    for family, count in family_counts.most_common():
        family_curves = [curve for curve in curves if curve["family"] == family]
        family_rows.append(
            {
                "family": compact_family(family),
                "histories": count,
                "best_loss": min(curve["best_loss"] for curve in family_curves),
                "max_tokens_b": max(curve["max_tokens"] for curve in family_curves) / 1e9,
                "mature_histories": sum(curve["max_tokens"] >= 1e9 for curve in family_curves),
            }
        )
    datasets["family_summary"] = family_rows
    tables.append(
        {
            "id": "family_summary_table",
            "title": "Architecture-family evidence coverage",
            "subtitle": "Counts and endpoint summaries across deduplicated histories; mixed regimes limit direct loss comparison",
            "showDescription": True,
            "dataset": "family_summary",
            "sourceId": "curve_summary",
            "defaultSort": {"field": "histories", "direction": "desc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "family", "label": "Architecture family", "type": "text"},
                {"field": "histories", "label": "Histories", "format": "number"},
                {"field": "mature_histories", "label": ">=1B histories", "format": "number"},
                {"field": "best_loss", "label": "Best loss", "format": "number"},
                {"field": "max_tokens_b", "label": "Max tokens (B)", "format": "number"},
            ],
        }
    )
    blocks.append({"id": "family_summary_table_block", "type": "table", "tableId": "family_summary_table", "layout": "full"})

    # Neuron search relationship.
    neuron_rows = []
    for row in read_csv(NEURON_CSV):
        speed = number(row.get("mean_speed_ratio"))
        delta = number(row.get("mean_val_delta"))
        seeds = number(row.get("n"))
        if speed is None or delta is None or seeds is None:
            continue
        neuron_rows.append(
            {
                "variant": row.get("variant") or row.get("design") or "unknown",
                "speed_ratio": speed,
                "val_delta": delta,
                "seeds": seeds,
                "steps": row.get("steps") or row.get("screen_steps") or "unknown",
                "wins": number(row.get("wins")) or 0,
                "decision": row.get("decision") or row.get("verdict") or "screen only",
            }
        )
    datasets["neuron_search"] = neuron_rows
    if neuron_rows:
        charts.append(
            {
                "id": "neuron_tradeoff",
                "title": "Neuron-screen quality and speed tradeoff",
                "subtitle": "Negative loss delta is better; speed ratio above 1 is faster than the matched baseline",
                "showDescription": True,
                "type": "scatter",
                "intent": "relationship",
                "question": "Which neuron variants improve short-screen validation loss without a prohibitive speed cost?",
                "rationale": "A scatter exposes whether quality improvements compensate for throughput changes and how many seeds support each point.",
                "dataset": "neuron_search",
                "sourceId": "neuron_summary",
                "encodings": {
                    "x": {"field": "speed_ratio", "type": "quantitative", "label": "speed ratio"},
                    "y": {"field": "val_delta", "type": "quantitative", "label": "validation-loss delta"},
                    "color": {"field": "steps", "type": "nominal", "label": "screen length"},
                    "size": {"field": "seeds", "type": "quantitative", "label": "seed count"},
                    "label": {"field": "variant", "type": "text", "label": "variant"},
                    "tooltip": [
                        {"field": "wins", "type": "quantitative", "label": "wins"},
                        {"field": "decision", "type": "text", "label": "decision"},
                    ],
                },
                "xAxisTitle": "speed ratio versus baseline",
                "yAxisTitle": "validation-loss delta versus baseline",
                "layout": "full",
                "palette": {"kind": "categorical", "name": "research"},
                "referenceLines": [
                    {"axis": "x", "value": 1, "label": "baseline speed", "color": "neutral", "lineStyle": "dashed"},
                    {"axis": "y", "value": 0, "label": "baseline loss", "color": "neutral", "lineStyle": "dashed"},
                ],
                "legend": {"position": "right", "title": "screen length"},
                "surface": {"surface": "export", "interactiveLegend": True, "showControls": False, "viewMode": "visualization"},
            }
        )
        blocks.append(
            markdown(
                "neuron_result",
                "## Neuron screens produced leads, not a scale-cleared replacement\n\n"
                "Several hidden-drop squared-neuron variants improve matched short-screen loss, and the repeated three-seed hidden-drop-square result is more credible than the one-seed outliers. "
                "However, these tests are too short to establish billion-token behavior. A candidate should advance only after a matched 50M- to 100M-token language run with identical validation and throughput accounting.",
                "neuron_summary",
            )
        )
        blocks.append({"id": "neuron_tradeoff_block", "type": "chart", "chartId": "neuron_tradeoff", "layout": "full"})

    # Synthetic fast-learning evidence.
    fastlearn_rows = []
    for row in read_csv(FASTLEARN_CSV):
        params = number(row.get("parameter_count_mean"))
        auc = number(row.get("adaptation_auc_mean"))
        if params is None or auc is None:
            continue
        fastlearn_rows.append(
            {
                "model": row.get("model") or row.get("model_type") or "unknown",
                "scale": row.get("scale") or row.get("size") or "unknown",
                "params_k": params / 1e3,
                "adaptation_auc": auc,
                "shot8_accuracy": number(row.get("shot8_token_accuracy")),
                "seed_count": number(row.get("seed_count")) or number(row.get("n")),
            }
        )
    datasets["fastlearn"] = fastlearn_rows
    if fastlearn_rows:
        charts.append(
            {
                "id": "fastlearn_scaling",
                "title": "Synthetic fast-learning adaptation by parameter count",
                "subtitle": "Adaptation AUC is a synthetic probe metric and is not language-model validation loss",
                "showDescription": True,
                "type": "line",
                "intent": "comparison",
                "question": "Which small recurrent or GPT-like design adapts fastest as parameter count increases?",
                "rationale": "A line comparison shows whether the synthetic adaptation advantage persists across parameter scales.",
                "dataset": "fastlearn",
                "sourceId": "fastlearn_summary",
                "encodings": {
                    "x": {"field": "params_k", "type": "quantitative", "label": "parameters", "unit": "k"},
                    "y": {"field": "adaptation_auc", "type": "quantitative", "label": "adaptation AUC"},
                    "color": {"field": "model", "type": "nominal", "label": "model"},
                    "lineStyle": {"field": "scale", "type": "nominal", "label": "scale"},
                    "tooltip": [
                        {"field": "shot8_accuracy", "type": "quantitative", "label": "8-shot token accuracy"},
                        {"field": "seed_count", "type": "quantitative", "label": "seed count"},
                    ],
                },
                "xAxisTitle": "parameter count (thousands)",
                "yAxisTitle": "adaptation AUC",
                "layout": "full",
                "palette": {"kind": "categorical", "name": "research"},
                "legend": {"position": "right", "title": "model"},
                "settings": {"showPoints": "always"},
                "surface": {"surface": "export", "interactiveLegend": True, "showControls": False, "viewMode": "visualization"},
            }
        )
        blocks.append(
            markdown(
                "fastlearn_result",
                "## Fast-GRU wins the synthetic adaptation probe, but that result does not transfer automatically\n\n"
                "Fast-GRU leads the GPT2-like control at every tested small, medium, and large synthetic scale. This is useful evidence that recurrent state can learn rapid adaptation behavior efficiently. "
                "It is not evidence that the same design will match Wave10 on natural-language validation loss, long-context recall, or generation quality; those require a language-training bridge experiment.",
                "fastlearn_summary",
            )
        )
        blocks.append({"id": "fastlearn_scaling_block", "type": "chart", "chartId": "fastlearn_scaling", "layout": "full"})

    # Coverage table by token horizon.
    horizon_table_rows = []
    for horizon_id, title, _lower, _upper, _field, _axis in HORIZONS:
        cohort = by_horizon.get(horizon_id, [])
        if not cohort:
            continue
        horizon_table_rows.append(
            {
                "horizon": title,
                "histories": len(cohort),
                "median_checkpoints": sorted(curve["point_count"] for curve in cohort)[len(cohort) // 2],
                "regime_mix": (
                    f"{sum(curve['regime'] == 'long_context_lm' for curve in cohort)} long / "
                    f"{sum(curve['regime'] == 'short_lm' for curve in cohort)} short"
                ),
                "comparability": "moderate" if horizon_id in {"300m_1b", "1b_2_5b", "over_2_5b"} else "low / mixed screens",
            }
        )
    datasets["horizon_coverage"] = horizon_table_rows
    tables.append(
        {
            "id": "horizon_coverage_table",
            "title": "Evidence coverage by training-token horizon",
            "subtitle": "The mature bands contain fewer but longer histories; short bands contain heterogeneous screens",
            "showDescription": True,
            "dataset": "horizon_coverage",
            "sourceId": "curve_summary",
            "defaultSort": {"field": "histories", "direction": "desc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "horizon", "label": "Token horizon", "type": "text"},
                {"field": "histories", "label": "Histories", "format": "number"},
                {"field": "median_checkpoints", "label": "Median pts", "format": "number"},
                {"field": "regime_mix", "label": "Regime mix", "type": "text"},
                {"field": "comparability", "label": "Evidence quality", "type": "text"},
            ],
        }
    )
    blocks.append(
        markdown(
            "scope_definitions",
            "## Scope and metric definitions\n\n"
            "**Unit of analysis.** One curve is a unique ordered sequence of `(tokens, validation loss)` checkpoints after exact fingerprint deduplication. Histories with fewer than three checkpoints are retained in the raw tables but excluded from trend charts.\n\n"
            "**Token horizon.** A history is assigned by its maximum recorded cumulative token exposure, not by intended run length.\n\n"
            "**Validation loss.** Lower is better within a consistent tokenizer, objective, context length, vocabulary treatment, and validation corpus. Values across different regimes are displayed for inventory but are not treated as a single scaling law.\n\n"
            "**Architecture family.** Families are inferred from experiment labels and source identifiers. Known legacy mislabeled comparison files are overridden using their original run IDs.",
        )
    )
    blocks.append({"id": "horizon_coverage_table_block", "type": "table", "tableId": "horizon_coverage_table", "layout": "full"})

    blocks.append(
        markdown(
            "methodology",
            "## Methodology and robustness checks\n\n"
            "The extractor scans recorded JSON, JSONL, CSV, and recognized trainer logs, then normalizes token counts and validation loss into one point table. CSV rows are grouped by their original run source so comparison files cannot merge different models into a fake curve. "
            "Exact checkpoint fingerprints remove duplicate copies exported through multiple reports. No loss smoothing, interpolation, or gap filling is applied.\n\n"
            "Robustness checks include source-level separation of the Wave10 and dense-160M continuations, exclusion of non-finite losses, explicit regime classification, and separate treatment of neuron and synthetic adaptation metrics. "
            "The report does not fit a parametric scaling law because the experiment grid is neither controlled nor statistically balanced.",
        )
    )
    blocks.append(
        markdown(
            "limitations",
            "## What the evidence cannot yet prove\n\n"
            "- Parameter count and throughput metadata are missing for several log-derived histories.\n"
            "- Many short screens have only three to six validation evaluations and one seed.\n"
            "- Sampled-vocabulary validation, anchor stride, identity-output changes, and tokenizer/context differences confound architecture comparisons.\n"
            "- Resume segments establish continuity of checkpoints, not proof that optimizer state and data order were perfectly controlled.\n"
            "- Generation quality, exact recall, code/number copying, and full-vocabulary rare-token behavior are not summarized by validation loss alone.\n"
            "- The data support descriptive ranking and experiment triage, not causal attribution or a universal scaling exponent.",
        )
    )
    blocks.append(
        markdown(
            "next_steps",
            "## The next decisive experiment is a controlled Wave10 replication\n\n"
            "1. Train Wave10 76M and the strongest dense 80M control from scratch with the same tokenizer, cache, sequence length, optimizer, seed, and full-vocabulary validation.\n"
            "2. Evaluate at fixed token milestones through at least 500M tokens before committing to a billion-token extension.\n"
            "3. Preserve throughput, peak VRAM, gradient norm, learning rate, sampled-vocabulary loss, full-vocabulary loss, and checkpoint samples in one result schema.\n"
            "4. If Wave10 retains a meaningful loss advantage, repeat at approximately 160M parameters and add a token-addressable recall ablation.\n"
            "5. Advance neuron variants only after a matched 50M- to 100M-token language screen; keep synthetic fast-learning work as a separate research track.",
        )
    )
    blocks.append(
        markdown(
            "further_questions",
            "## Further questions\n\n"
            "- Does Wave10's advantage survive full-vocabulary validation and identical anchor frequency?\n"
            "- How much of the gain comes from the low-rank conv-memory block versus the training-system changes bundled with it?\n"
            "- Does the architecture preserve names, numbers, delimiters, and repeated phrases at long context?\n"
            "- Which parameter scale gives the best information per GPU-dollar before a cloud-scale run?\n"
            "- Can the recurrent and synthetic fast-learning advantages be transferred into the long-context language objective without losing throughput?",
        )
    )

    # The packaged reader's resizable table surface has a fixed mobile minimum
    # width. Preserve both audit datasets in the artifact, but render their key
    # coverage facts as responsive prose so the portable report never leaks
    # horizontal overflow on narrow screens.
    family_summary_text = "\n".join(
        f"- **{row['family']}:** {row['histories']} histories, "
        f"{row['mature_histories']} at >=1B tokens, best loss {row['best_loss']:.3f}, "
        f"maximum {row['max_tokens_b']:.2f}B tokens."
        for row in family_rows
    )
    horizon_summary_text = "\n".join(
        f"- **{row['horizon']}:** {row['histories']} histories, median {row['median_checkpoints']} checkpoints, "
        f"{row['regime_mix']}; evidence quality: {row['comparability']}."
        for row in horizon_table_rows
    )
    for index, block in enumerate(blocks):
        if block.get("id") == "family_summary_table_block":
            blocks[index] = markdown(
                "family_summary_text",
                "## Architecture-family evidence coverage\n\n" + family_summary_text,
                "curve_summary",
            )
        elif block.get("id") == "horizon_coverage_table_block":
            blocks[index] = markdown(
                "horizon_coverage_text",
                "## Evidence coverage by training-token horizon\n\n" + horizon_summary_text,
                "curve_summary",
            )
    tables = []

    manifest = {
        "version": 1,
        "surface": "report",
        "title": "Research Scaling Atlas",
        "description": "Technical analysis of language-model architecture experiments split by training-token horizon and architecture family.",
        "generatedAt": "2026-08-15T01:30:00-05:00",
        "cards": cards,
        "charts": charts,
        "tables": tables,
        "sources": sources,
        "blocks": blocks,
    }
    snapshot = {
        "version": 1,
        "generatedAt": "2026-08-15T01:30:00-05:00",
        "status": "ready",
        "datasets": datasets,
        "accessIssues": [],
    }
    return {"surface": "report", "manifest": manifest, "snapshot": snapshot, "sources": sources}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = build_artifact()
    artifact_path = OUTPUT_DIR / "artifact.json"
    artifact_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=True), encoding="utf-8")
    chart_map = [
        {
            "section": block["id"],
            "chart_id": block.get("chartId"),
            "table_id": block.get("tableId"),
        }
        for block in artifact["manifest"]["blocks"]
        if block["type"] in {"chart", "table"}
    ]
    (OUTPUT_DIR / "chart_map.json").write_text(json.dumps(chart_map, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "artifact": str(artifact_path),
                "datasets": len(artifact["snapshot"]["datasets"]),
                "charts": len(artifact["manifest"]["charts"]),
                "tables": len(artifact["manifest"]["tables"]),
                "blocks": len(artifact["manifest"]["blocks"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
