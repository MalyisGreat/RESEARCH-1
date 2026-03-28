from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import arc_agi

HARNESS_SRC = Path(__file__).resolve().parents[1] / "arc-agi-3-benchmarking" / "src"
if str(HARNESS_SRC) not in sys.path:
    sys.path.insert(0, str(HARNESS_SRC))

from arcagi3.arc3tester import ARC3Tester  # noqa: E402
from .arc_agi3_harness_agent import TACTICHarnessAgent
from .arc_agi3_policy import per_level_budget


def scorecard_to_dict(scorecard: Any) -> dict[str, Any]:
    if hasattr(scorecard, "model_dump"):
        return scorecard.model_dump()
    if isinstance(scorecard, dict):
        return scorecard
    return json.loads(str(scorecard))


def total_episode_budget(baseline_actions: tuple[int, ...]) -> int:
    if not baseline_actions:
        return 96
    return sum(per_level_budget(baseline_actions, idx) for idx in range(len(baseline_actions)))


def run_public_benchmark(*, limit: int | None = None, save_path: Path | None = None) -> dict[str, Any]:
    arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.ONLINE)
    environments = tuple(arcade.get_environments())
    if limit is not None:
        environments = environments[:limit]

    results: list[dict[str, Any]] = []
    total_score = 0.0
    total_levels_completed = 0
    total_actions = 0

    for info in environments:
        episode_budget = total_episode_budget(tuple(int(v) for v in info.baseline_actions))
        tester = ARC3Tester(
            config="heuristic-local",
            save_results_dir=None,
            overwrite_results=True,
            max_actions=episode_budget,
            max_episode_actions=episode_budget,
            num_plays=1,
            submit_scorecard=True,
            agent_class=TACTICHarnessAgent,
            agent_kwargs={},
        )
        result = tester.play_game(info.game_id)
        scorecard = scorecard_to_dict(result.scorecard_payload or {})
        total_score += float(scorecard.get("score", 0.0))
        total_levels_completed += int(scorecard.get("total_levels_completed", 0))
        total_actions += int(scorecard.get("total_actions", result.actions_taken))
        results.append(
            {
                "game_id": info.game_id,
                "title": info.title,
                "baseline_actions": list(info.baseline_actions),
                "episode_budget": episode_budget,
                "score": float(scorecard.get("score", 0.0)),
                "total_levels_completed": int(scorecard.get("total_levels_completed", 0)),
                "total_levels": int(scorecard.get("total_levels", 0)),
                "total_actions": int(scorecard.get("total_actions", result.actions_taken)),
                "card_id": result.card_id,
                "scorecard_url": result.scorecard_url,
                "final_state": result.final_state,
            }
        )

    summary = {
        "mode": "official-harness-online",
        "config": "heuristic-local",
        "agent": "tactic-public",
        "environments": len(results),
        "aggregate_score": total_score,
        "total_levels_completed": total_levels_completed,
        "total_actions": total_actions,
        "results": results,
    }
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the TACTIC ARC public benchmark through the official harness.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of public games to run.")
    parser.add_argument(
        "--save",
        type=Path,
        default=Path("E:/DEVNEW/arc_tactic3/benchmark_runs/arc_harness_online_baseline_20260326.json"),
        help="Where to save the benchmark summary JSON.",
    )
    args = parser.parse_args()
    summary = run_public_benchmark(limit=args.limit, save_path=args.save)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
