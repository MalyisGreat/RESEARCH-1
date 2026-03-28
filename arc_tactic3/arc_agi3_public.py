from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import arc_agi
from arc_agi import OperationMode

from .arc_agi3_policy import ArcAgi3TACTICPublicAgent, PolicyTuning
from .progress import ProgressBar
from .qwen_arc_advisor import OllamaArcAdvisor


@dataclass(frozen=True, slots=True)
class RunResult:
    score: float
    total_levels_completed: int
    total_levels: int
    total_actions: int
    scorecard: dict[str, Any]
    diagnostics: dict[str, int]


def scorecard_to_dict(scorecard) -> dict[str, Any]:
    if isinstance(scorecard, dict):
        return scorecard
    if hasattr(scorecard, "model_dump"):
        return scorecard.model_dump()
    if hasattr(scorecard, "dict"):
        return scorecard.dict()
    return json.loads(str(scorecard))


def run_public_benchmark(
    *,
    limit: int | None = None,
    show_progress: bool = True,
    competition_mode: bool = False,
    qwen_advisor: bool = False,
    qwen_budget_per_level: int = 1,
    qwen_mechanic_advisor: bool = False,
    qwen_mechanic_budget_per_level: int = 1,
    tuning: PolicyTuning | None = None,
) -> RunResult:
    if competition_mode and limit is not None:
        raise ValueError("`limit` is not supported in competition mode because ARC scorecards remain full-suite.")
    operation_mode = OperationMode.COMPETITION if competition_mode else OperationMode.ONLINE
    arc = arc_agi.Arcade(operation_mode=operation_mode)
    environments = tuple(arc.get_environments())
    if limit is not None:
        environments = environments[:limit]
    agent = ArcAgi3TACTICPublicAgent(
        action_advisor=OllamaArcAdvisor() if qwen_advisor else None,
        action_advisor_budget_per_level=qwen_budget_per_level if qwen_advisor else 0,
        mechanic_advisor=OllamaArcAdvisor() if qwen_mechanic_advisor else None,
        mechanic_advisor_budget_per_level=qwen_mechanic_budget_per_level if qwen_mechanic_advisor else 0,
        tuning=tuning,
    )
    progress = ProgressBar("arc-agi3-public", len(environments)) if show_progress else None
    seen_ids: set[str] = set()

    for index, info in enumerate(environments, start=1):
        if info.game_id in seen_ids:
            raise RuntimeError(f"Duplicate environment in scorecard run: {info.game_id}")
        seen_ids.add(info.game_id)
        if progress is not None:
            progress.update(index - 1, detail=info.game_id)
        env = arc.make(info.game_id)
        agent.play_environment(
            env,
            env_id=info.game_id,
            baseline_actions=tuple(info.baseline_actions),
            show_progress=False,
        )
        if progress is not None:
            progress.update(index, detail=info.game_id)

    if progress is not None:
        progress.finish("scorecard finalizing")
    scorecard = arc.close_scorecard()
    payload = scorecard_to_dict(scorecard)
    return RunResult(
        score=float(payload["score"]),
        total_levels_completed=int(payload["total_levels_completed"]),
        total_levels=int(payload["total_levels"]),
        total_actions=int(payload["total_actions"]),
        scorecard=payload,
        diagnostics=agent.diagnostics.as_dict(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the lightweight TACTIC-style ARC-AGI-3 public benchmark.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on the number of public environments.")
    parser.add_argument("--no-progress", action="store_true", help="Disable CLI progress output.")
    parser.add_argument(
        "--competition-mode",
        action="store_true",
        help="Run the benchmark in ARC competition mode for stricter leaderboard-aligned behavior.",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Include policy diagnostic counters in the JSON output.",
    )
    parser.add_argument(
        "--qwen-advisor",
        action="store_true",
        help="Enable the local Ollama Qwen sidecar as an explicit advisor.",
    )
    parser.add_argument(
        "--qwen-budget-per-level",
        type=int,
        default=1,
        help="How many Qwen advisor calls to allow per level when --qwen-advisor is enabled.",
    )
    parser.add_argument(
        "--qwen-mechanic-advisor",
        action="store_true",
        help="Enable the local Ollama Qwen sidecar as a mechanic summarizer.",
    )
    parser.add_argument(
        "--qwen-mechanic-budget-per-level",
        type=int,
        default=1,
        help="How many Qwen mechanic-summary calls to allow per level when --qwen-mechanic-advisor is enabled.",
    )
    parser.add_argument(
        "--tuning-json",
        type=str,
        default=None,
        help="Optional path to a JSON file, or raw JSON object, with PolicyTuning overrides for sweep-friendly runs.",
    )
    args = parser.parse_args()
    tuning = None
    if args.tuning_json:
        raw_payload = args.tuning_json
        payload_text = Path(raw_payload).read_text() if Path(raw_payload).exists() else raw_payload
        tuning = PolicyTuning.from_dict(json.loads(payload_text))
    result = run_public_benchmark(
        limit=args.limit,
        show_progress=not args.no_progress,
        competition_mode=args.competition_mode,
        qwen_advisor=args.qwen_advisor,
        qwen_budget_per_level=args.qwen_budget_per_level,
        qwen_mechanic_advisor=args.qwen_mechanic_advisor,
        qwen_mechanic_budget_per_level=args.qwen_mechanic_budget_per_level,
        tuning=tuning,
    )
    payload = {
        "score": result.score,
        "total_levels_completed": result.total_levels_completed,
        "total_levels": result.total_levels,
        "total_actions": result.total_actions,
        "environments": result.scorecard["environments"],
    }
    if args.diagnostics:
        payload["diagnostics"] = result.diagnostics
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
