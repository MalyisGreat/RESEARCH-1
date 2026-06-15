from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


ARTIFACT_ROOT = Path(__file__).resolve().parent
OUT_JSON = ARTIFACT_ROOT / "aggregate_results_20260605_current.json"
OUT_CSV = ARTIFACT_ROOT / "aggregate_results_20260605_current.csv"
OUT_MD = ARTIFACT_ROOT / "synthesis_20260605_current.md"
OUT_PLOT = ARTIFACT_ROOT / "loss_delta_1024_key_variants.png"


def load_result(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def metric(report: dict[str, Any], key: str) -> float | None:
    value = report.get(key)
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def pair_rows() -> list[dict[str, Any]]:
    baselines: dict[str, dict[str, Any]] = {}
    results: dict[str, dict[str, Any]] = {}
    for result_path in ARTIFACT_ROOT.glob("screen_*/result.json"):
        result = load_result(result_path)
        if not result:
            continue
        run_name = result_path.parent.name
        results[run_name] = result
        if run_name.endswith("_baseline"):
            baselines[run_name[: -len("_baseline")]] = result

    match_keys = (
        "cache_path",
        "seed",
        "sequence_length",
        "train_steps",
        "val_blocks",
        "embedding_dim",
        "conv_layers",
        "conv_rank",
        "conv_kernel_size",
        "memory_rank",
        "landmark_stride",
        "sampled_vocab_size",
        "token_stride",
        "token_chunk_size",
        "full_eval_token_chunk_size",
        "learning_rate",
        "min_learning_rate",
        "warmup_steps",
        "weight_decay",
        "amp_dtype",
    )

    rows: list[dict[str, Any]] = []
    for run_name, result in sorted(results.items()):
        if run_name.endswith("_baseline"):
            continue
        candidate_config = result["config"]
        matches: list[tuple[int, str, dict[str, Any]]] = []
        for prefix, baseline in baselines.items():
            if not run_name.startswith(f"{prefix}_"):
                continue
            baseline_config = baseline["config"]
            if all(candidate_config.get(key) == baseline_config.get(key) for key in match_keys):
                matches.append((len(prefix), prefix, baseline))
        if not matches:
            continue
        _, prefix, baseline = max(matches, key=lambda item: item[0])
        base_report = baseline["report"]
        base_params = int(base_report["parameter_count"])
        base_val = metric(base_report, "final_val_loss")
        base_train = metric(base_report, "final_train_loss")
        base_tps = metric(base_report, "pure_train_tok_per_sec")
        if base_val is None:
            continue
        variant = result["config"]["block_type"]
        report = result["report"]
        val = metric(report, "final_val_loss")
        train = metric(report, "final_train_loss")
        tps = metric(report, "pure_train_tok_per_sec")
        params = int(report["parameter_count"])
        row = {
            "group": prefix,
            "variant": variant,
            "seed": int(result["config"].get("seed", -1)),
            "steps": int(result["config"].get("train_steps", -1)),
            "sequence_length": int(result["config"].get("sequence_length", -1)),
            "val_blocks": int(result["config"].get("val_blocks", -1)),
            "embedding_dim": int(result["config"].get("embedding_dim", -1)),
            "conv_layers": int(result["config"].get("conv_layers", -1)),
            "sampled_vocab_size": int(result["config"].get("sampled_vocab_size", -1)),
            "cache_path": result["config"].get("cache_path", ""),
            "baseline_val_loss": base_val,
            "candidate_val_loss": val,
            "val_delta_vs_baseline": None if val is None else val - base_val,
            "baseline_train_loss": base_train,
            "candidate_train_loss": train,
            "train_delta_vs_baseline": None if train is None or base_train is None else train - base_train,
            "baseline_tok_per_sec": base_tps,
            "candidate_tok_per_sec": tps,
            "speed_ratio_vs_baseline": None if tps is None or not base_tps else tps / base_tps,
            "baseline_peak_vram_mb": metric(base_report, "peak_vram_mb"),
            "candidate_peak_vram_mb": metric(report, "peak_vram_mb"),
            "baseline_params": base_params,
            "candidate_params": params,
            "param_delta": params - base_params,
            "result_path": str(Path(result["config"]["output_dir"]) / "result.json"),
            "baseline_result_path": str(Path(baseline["config"]["output_dir"]) / "result.json"),
        }
        rows.append(row)
    rows.sort(
        key=lambda row: (
            row["sequence_length"],
            row["steps"],
            row["group"],
            math.inf if row["val_delta_vs_baseline"] is None else row["val_delta_vs_baseline"],
        )
    )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, int, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    seen_quality_keys: set[tuple[Any, ...]] = set()
    for row in rows:
        quality_key = (
            row["variant"],
            row["seed"],
            row["steps"],
            row["sequence_length"],
            row["val_blocks"],
            row["cache_path"],
            row["baseline_val_loss"],
            row["candidate_val_loss"],
        )
        if quality_key in seen_quality_keys:
            continue
        seen_quality_keys.add(quality_key)
        cache_kind = "real" if "finewebedu" in row["cache_path"].lower() else "synthetic"
        key = (row["variant"], row["steps"], row["sequence_length"], row["val_blocks"], cache_kind)
        buckets[key].append(row)

    summaries: list[dict[str, Any]] = []
    for (variant, steps, seq, val_blocks, cache_kind), items in sorted(buckets.items()):
        deltas = [float(item["val_delta_vs_baseline"]) for item in items if item["val_delta_vs_baseline"] is not None]
        speed = [float(item["speed_ratio_vs_baseline"]) for item in items if item["speed_ratio_vs_baseline"] is not None]
        if not deltas:
            continue
        summaries.append(
            {
                "variant": variant,
                "steps": steps,
                "sequence_length": seq,
                "val_blocks": val_blocks,
                "cache_kind": cache_kind,
                "n": len(deltas),
                "mean_val_delta": sum(deltas) / len(deltas),
                "best_val_delta": min(deltas),
                "worst_val_delta": max(deltas),
                "wins": sum(1 for delta in deltas if delta < 0),
                "mean_speed_ratio": None if not speed else sum(speed) / len(speed),
                "mean_param_delta": sum(int(item["param_delta"]) for item in items) / len(items),
                "groups": sorted(item["group"] for item in items),
            }
        )
    summaries.sort(key=lambda row: (row["cache_kind"] != "real", -row["sequence_length"], -row["steps"], -row["val_blocks"], row["mean_val_delta"]))
    return summaries


def write_csv(rows: list[dict[str, Any]]) -> None:
    if not rows:
        OUT_CSV.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt_float(value: Any, digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def write_markdown(rows: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> None:
    long_summaries = [
        item
        for item in summaries
        if item["cache_kind"] == "real" and item["sequence_length"] == 10160 and item["steps"] == 1024 and item["val_blocks"] == 8
    ]
    top_long = min(long_summaries, key=lambda item: item["mean_val_delta"]) if long_summaries else None
    lines: list[str] = [
        "# Manual Neuron Search Synthesis - Current 2026-06-05",
        "",
        f"Artifact root: `{ARTIFACT_ROOT}`",
        "",
        "## Status",
        "",
    ]
    if top_long:
        lines.extend(
            [
                "No 5B-token or 3080 scale-up run has been started. No candidate is cleared for scale-up yet; the current research leader is `{variant}`.".format(
                    variant=top_long["variant"]
                ),
                "",
                "`{variant}` is the current long-gate leader on real FineWebEdu seq10160 val8: "
                "n={n}, wins={wins}, mean val delta={mean}, best={best}, worst={worst}, "
                "mean speed ratio={speed}, mean param delta={params}. This is a real measured signal, "
                "but it is not scale-cleared because longer confirmation has exposed stability risk.".format(
                    variant=top_long["variant"],
                    n=top_long["n"],
                    wins=top_long["wins"],
                    mean=fmt_float(top_long["mean_val_delta"]),
                    best=fmt_float(top_long["best_val_delta"]),
                    worst=fmt_float(top_long["worst_val_delta"]),
                    speed=fmt_float(top_long["mean_speed_ratio"], 3),
                    params=fmt_float(top_long["mean_param_delta"], 1),
                ),
            ]
        )
        failed_confirmations = [
            row
            for row in rows
            if row["variant"] == top_long["variant"]
            and "finewebedu" in row["cache_path"].lower()
            and row["sequence_length"] == 10160
            and row["val_blocks"] == 8
            and row["steps"] > 1024
            and row["candidate_val_loss"] is None
        ]
        if failed_confirmations:
            lines.extend(
                [
                    "",
                    "Longer weak-seed confirmation for `{variant}` has at least one non-finite candidate loss; "
                    "that blocks 3080 or 5B-token scale-up until a stable second iteration beats the same controls.".format(
                        variant=top_long["variant"]
                    ),
                ]
            )
    else:
        lines.append("No real seq10160 1024-step gate has been completed yet.")
    lines.extend(
        [
        "",
        "## Ranked Aggregate Buckets",
        "",
        "| variant | cache | seq | val blocks | steps | n | wins | mean val delta | best | worst | mean speed | mean param delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    )
    for item in summaries:
        lines.append(
            "| {variant} | {cache_kind} | {sequence_length} | {val_blocks} | {steps} | {n} | {wins} | {mean} | {best} | {worst} | {speed} | {params} |".format(
                variant=item["variant"],
                cache_kind=item["cache_kind"],
                sequence_length=item["sequence_length"],
                val_blocks=item["val_blocks"],
                steps=item["steps"],
                n=item["n"],
                wins=item["wins"],
                mean=fmt_float(item["mean_val_delta"]),
                best=fmt_float(item["best_val_delta"]),
                worst=fmt_float(item["worst_val_delta"]),
                speed=fmt_float(item["mean_speed_ratio"], 3),
                params=fmt_float(item["mean_param_delta"], 1),
            )
        )

    long_rows = [
        row
        for row in rows
        if row["sequence_length"] == 10160 and row["steps"] == 1024 and row["val_blocks"] == 8 and "finewebedu" in row["cache_path"].lower()
    ]
    long_table_rows: list[dict[str, Any]] = []
    seen_long_keys: set[tuple[Any, ...]] = set()
    for row in long_rows:
        long_key = (
            row["variant"],
            row["seed"],
            row["baseline_val_loss"],
            row["candidate_val_loss"],
        )
        if long_key in seen_long_keys:
            continue
        seen_long_keys.add(long_key)
        long_table_rows.append(row)
    lines.extend(
        [
            "",
            "## 40M Seq10160 1024-Step Gate",
            "",
            "| seed | val blocks | variant | baseline val | candidate val | val delta | train delta | speed ratio | peak VRAM MB | param delta |",
            "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(long_table_rows, key=lambda item: (item["variant"], item["seed"])):
        lines.append(
            "| {seed} | {val_blocks} | {variant} | {base_val} | {cand_val} | {delta} | {train_delta} | {speed} | {vram} | {param_delta} |".format(
                seed=row["seed"],
                val_blocks=row["val_blocks"],
                variant=row["variant"],
                base_val=fmt_float(row["baseline_val_loss"]),
                cand_val=fmt_float(row["candidate_val_loss"]),
                delta=fmt_float(row["val_delta_vs_baseline"]),
                train_delta=fmt_float(row["train_delta_vs_baseline"]),
                speed=fmt_float(row["speed_ratio_vs_baseline"], 3),
                vram=fmt_float(row["candidate_peak_vram_mb"], 1),
                param_delta=row["param_delta"],
            )
        )

    failed_rows = [
        row
        for row in rows
        if row["candidate_val_loss"] is None or row["candidate_train_loss"] is None
    ]
    if failed_rows:
        lines.extend(
            [
                "",
                "## Failed Or Non-Finite Runs",
                "",
                "| group | seed | seq | val blocks | steps | variant | baseline val | candidate val | speed ratio | peak VRAM MB | result |",
                "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in sorted(failed_rows, key=lambda item: (item["sequence_length"], item["steps"], item["variant"], item["seed"])):
            lines.append(
                "| {group} | {seed} | {seq} | {val_blocks} | {steps} | {variant} | {base_val} | {cand_val} | {speed} | {vram} | `{result}` |".format(
                    group=row["group"],
                    seed=row["seed"],
                    seq=row["sequence_length"],
                    val_blocks=row["val_blocks"],
                    steps=row["steps"],
                    variant=row["variant"],
                    base_val=fmt_float(row["baseline_val_loss"]),
                    cand_val=fmt_float(row["candidate_val_loss"]),
                    speed=fmt_float(row["speed_ratio_vs_baseline"], 3),
                    vram=fmt_float(row["candidate_peak_vram_mb"], 1),
                    result=row["result_path"],
                )
            )

    lines.extend(
        [
            "",
            "## Decisions",
            "",
            "- Reject full phase-amplitude replacement for scale-up: real-text seeds were unstable.",
            "- Reject stable competition for scale-up: synthetic gains did not transfer to real FineWebEdu.",
            "- Reject cheap scalar memory gates for scale-up: strong 128-step wins did not survive the 1024-step gate.",
            "- Reject memory-square gain variants for scale-up: they are fast, but long 1024-step quality regressed on seed 13.",
            "- Reject memory-novelty variants for now: 128-step quality matched equivalent raw-memory gates but with lower speed.",
            "- Reject memory-conv agreement variants for scale-up: the centered agreement path duplicated the simpler centered residual result and was slower.",
            "- Reject boundary-local residual variants for scale-up: 128-step wins were weaker than centered residual and memory-centered gates.",
            "- Reject constrained memory-gate variants for scale-up: the apparent val4 win was a validation-block mismatch artifact; val8 verification on seed 13 lost to baseline for bounded and RMS gates.",
            "- Keep `rank_competition_neuron` as a robust cheap survivor: val8 1024-step seeds 13/17/29 were 3/3 wins, mean val delta -0.003895, mean speed ratio about 0.990, and only +2 params.",
            "- Demote `rank_competition_memory_centered_gate_neuron`: val8 1024-step seeds 13/17/29/31/43 were 4/5 wins, mean val delta -0.005945, best -0.014662, worst +0.003996, mean speed ratio about 0.869, and +135170 params.",
            "- Keep `rank_competition_memory_suppressed_centered_gate_neuron` as the 1024-step research leader, not a scale-up candidate: val8 1024-step seeds 13/17/29/31/43 were 5/5 wins, mean val delta -0.010564, best -0.019104, worst -0.000137, mean speed ratio about 0.838, and +135170 params, but the 2048-step weak-seed confirmation diverged to NaN.",
            "- Reject `rank_competition_memory_scalar_centered_gate_neuron` for scale-up: it was the best 128-step val8 cost ablation, but the 1024-step gate fell to 2/3 wins, mean val delta -0.000859, and a seed 13 regression.",
            "- Keep `rank_competition_centered_residual_neuron` as a cheap modest survivor: val8 1024-step seeds 13/17/29 were 3/3 wins, mean val delta -0.003985, but it is only slightly better than `rank_competition_neuron` and much slower.",
            "- Reject group and factorized memory-centered rank gates for now: 128-step screens were 3/3 wins but worse than the scalar/no-memory cost ablations at higher parameter or speed cost.",
            "- Reject `rank_competition_memory_small_centered_gate_neuron` and `rank_competition_memory_normed_centered_gate_neuron` for now: both targeted the seed 31 failure, but 1024-step seed 31 still regressed by +0.002827 and +0.003501 respectively.",
            "- Reject `rank_competition_memory_uncertainty_centered_gate_neuron` and `rank_competition_memory_within_group_centered_gate_neuron` for now: both failed the focused 1024-step seed 31 gate.",
            "- Reject the first stable-suppressed second iterations for scale-up: small-gain, bounded-small, and RMS-limited routes lost the 1024-step seed 31 gate; the bounded route barely won at 1024 but catastrophically regressed at 2048 with val delta +269.588360.",
            "- Reject `rank_competition_memory_suppressed_stopgrad_mask_centered_gate_neuron` for scale-up: detaching the suppression mask fixed the 2048 divergence mode and improved the seed 31 1024 margin to -0.000440, but 2048 validation still regressed by +0.008068.",
            "- Reject `rank_competition_memory_suppressed_aux_head_centered_gate_neuron`: the separate residual head diverged by the 1024-step gate, with val delta +24.916751 and +1183748 params.",
            "- Keep `rank_competition_mild_neuron` and `rank_competition_soft_neuron` as cheap 1024-only pressure-curve survivors, not scale-up candidates: seed 31 1024 deltas were -0.002389 and -0.001207 at +2 params and about 0.97 speed ratio, but 2048 deltas were +0.007532 and +0.003544.",
            "- Reject the softer pressure endpoints for scale-up: `rank_competition_ultrasoft_neuron` and `rank_competition_feather_neuron` stayed stable and near baseline at 2048, but still lost by +0.000931 and +0.000121 respectively.",
            "- Reject fixed-pressure rank competition for scale-up: freezing the feather pressure removed the learned inhibition drift and added 0 params, but the 2048 weak-seed confirmation still lost by +0.000111; fixed trace and dust were weaker already at 1024.",
            "- Do not start the 5B-token run or queue this for 3080 scale-up yet. This round found real 1024-step improvements and one stability fix for the suppressed-memory route, but no candidate has beaten the matched 2048 weak-seed baseline.",
            "- Next ablation: move away from adding signed memory-route capacity and test block-local regularizers or data-scale confirmations that can reduce the repeated lower-train/worse-val pattern without changing tokenizer, objective, or metric.",
            "",
            "## Commands Used",
            "",
            "Latest 3-seed 1024-step hybrid gate:",
            "",
            "```powershell",
            "$cache='E:\\CODEXRESEARCH\\RESEARCH-1\\artifacts\\benchmark_runs\\language\\neuron_search_20260605\\manual_self\\real_cache_finewebedu_sample_seq10160_train192_val8_gpt2.pt'",
            "$env:CUDA_VISIBLE_DEVICES='0'",
            "$env:MANUAL_SEARCH_CACHE=$cache",
            "$env:MANUAL_SEARCH_SEQUENCE_LENGTH='10160'",
            "$env:MANUAL_SEARCH_VAL_BLOCKS='8'",
            "$env:MANUAL_SEARCH_STEPS='1024'",
            "$env:MANUAL_SEARCH_TAG='real_seq10160_40m_1024'",
            "$env:MANUAL_SEARCH_BLOCKS='rank_competition_memory_centered_gate_neuron'",
            "$env:MANUAL_SEARCH_REUSE_EXISTING='1'",
            "$env:MANUAL_SEARCH_EMBEDDING_DIM='512'",
            "$env:MANUAL_SEARCH_CONV_LAYERS='2'",
            "$env:MANUAL_SEARCH_CONV_KERNEL_SIZE='7'",
            "$env:MANUAL_SEARCH_CONV_RANK='192'",
            "$env:MANUAL_SEARCH_MEMORY_RANK='64'",
            "$env:MANUAL_SEARCH_LANDMARK_STRIDE='128'",
            "$env:MANUAL_SEARCH_SAMPLED_VOCAB_SIZE='24576'",
            "$env:MANUAL_SEARCH_TOKEN_STRIDE='4'",
            "$env:MANUAL_SEARCH_TOKEN_CHUNK_SIZE='512'",
            "$env:MANUAL_SEARCH_FULL_EVAL_TOKEN_CHUNK_SIZE='1024'",
            "foreach($seed in 13,17,29){",
            "  $env:MANUAL_SEARCH_SEED=[string]$seed",
            "  python -u 'E:\\CODEXRESEARCH\\RESEARCH-1\\artifacts\\benchmark_runs\\language\\neuron_search_20260605\\manual_self\\manual_neuron_search.py'",
            "}",
            "```",
            "",
            "Aggregation and verification:",
            "",
            "```powershell",
            "python -u 'E:\\CODEXRESEARCH\\RESEARCH-1\\artifacts\\benchmark_runs\\language\\neuron_search_20260605\\manual_self\\aggregate_manual_results.py'",
            "python -B -m py_compile 'E:\\CODEXRESEARCH\\RESEARCH-1\\artifacts\\benchmark_runs\\language\\neuron_search_20260605\\manual_self\\manual_neuron_search.py' 'E:\\CODEXRESEARCH\\RESEARCH-1\\artifacts\\benchmark_runs\\language\\neuron_search_20260605\\manual_self\\aggregate_manual_results.py'",
            "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader",
            "```",
            "",
            "## Generated Files",
            "",
            f"- `{OUT_JSON}`",
            f"- `{OUT_CSV}`",
            f"- `{OUT_PLOT}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = pair_rows()
    summaries = summarize(rows)
    OUT_JSON.write_text(json.dumps({"rows": rows, "summaries": summaries}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(rows)
    write_markdown(rows, summaries)
    print(json.dumps({"rows": len(rows), "summaries": len(summaries), "markdown": str(OUT_MD)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
