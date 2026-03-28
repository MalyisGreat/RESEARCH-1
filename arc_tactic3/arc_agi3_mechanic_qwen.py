from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

import arc_agi
from arc_agi import OperationMode

from .arc_agi3_policy import ArcAgi3TACTICPublicAgent
from .arc_agi3_public import scorecard_to_dict
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


def run_public_benchmark(
    *,
    show_progress: bool = True,
    competition_mode: bool = True,
    qwen_budget_per_level: int = 1,
) -> RunResult:
    operation_mode = OperationMode.COMPETITION if competition_mode else OperationMode.ONLINE
    arc = arc_agi.Arcade(operation_mode=operation_mode)
    environments = tuple(arc.get_environments())
    agent = ArcAgi3TACTICPublicAgent(
        mechanic_advisor=OllamaArcAdvisor(),
        mechanic_advisor_budget_per_level=qwen_budget_per_level,
    )
    progress = ProgressBar("arc-agi3-qwen-mechanic", len(environments)) if show_progress else None

    for index, info in enumerate(environments, start=1):
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
    payload = scorecard_to_dict(arc.close_scorecard())
    return RunResult(
        score=float(payload["score"]),
        total_levels_completed=int(payload["total_levels_completed"]),
        total_levels=int(payload["total_levels"]),
        total_actions=int(payload["total_actions"]),
        scorecard=payload,
        diagnostics=agent.diagnostics.as_dict(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the explicit Qwen mechanic-prior candidate on the ARC-AGI-3 public benchmark."
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable CLI progress output.")
    parser.add_argument(
        "--online-mode",
        action="store_true",
        help="Use online mode instead of competition mode.",
    )
    parser.add_argument(
        "--qwen-budget-per-level",
        type=int,
        default=1,
        help="How many mechanic-summary Qwen calls to allow per level.",
    )
    args = parser.parse_args()
    result = run_public_benchmark(
        show_progress=not args.no_progress,
        competition_mode=not args.online_mode,
        qwen_budget_per_level=args.qwen_budget_per_level,
    )
    print(
        json.dumps(
            {
                "score": result.score,
                "total_levels_completed": result.total_levels_completed,
                "total_levels": result.total_levels,
                "total_actions": result.total_actions,
                "environments": result.scorecard["environments"],
                "diagnostics": result.diagnostics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
