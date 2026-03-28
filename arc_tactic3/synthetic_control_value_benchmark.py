from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from arcengine import GameAction

from .arc_agi3_policy import (
    ActionCandidate,
    ArcAgi3TACTICPublicAgent,
    EnvironmentMemory,
    FrameComponent,
    LevelMemory,
    coarse_cell_for_coord,
    coarse_path_distance,
    extract_components,
)


@dataclass(frozen=True, slots=True)
class SyntheticCase:
    family: str
    frame: np.ndarray
    deltas: dict[str, tuple[int, int]]
    optimal_actions: tuple[str, ...]


def _actor_component(frame: np.ndarray) -> FrameComponent:
    return next(component for component in extract_components(frame) if component.color == 5)


def _target_component(frame: np.ndarray) -> FrameComponent:
    return next(component for component in extract_components(frame) if component.color == 7)


def _random_open_case(rng: random.Random) -> SyntheticCase:
    frame = np.zeros((64, 64), dtype=np.int8)
    actor_row = rng.randrange(8, 48, 8)
    actor_col = rng.choice((8, 16, 24, 32))
    target_row = actor_row
    target_choices = tuple(col for col in (actor_col + 16, actor_col + 24) if col <= 48)
    target_col = rng.choice(target_choices)
    frame[actor_row : actor_row + 8, actor_col : actor_col + 8] = 5
    frame[target_row : target_row + 8, target_col : target_col + 8] = 7
    deltas = {"ACTION1": (0, 8), "ACTION2": (8, 0), "ACTION3": (0, -8), "ACTION4": (-8, 0)}
    return SyntheticCase(family="direct_open", frame=frame, deltas=deltas, optimal_actions=("ACTION1",))


def _random_blocked_detour_case(rng: random.Random) -> SyntheticCase:
    frame = np.zeros((64, 64), dtype=np.int8)
    actor_row = rng.choice((8, 16))
    actor_col = 8
    target_row = actor_row
    target_col = 40
    wall_col = 24
    wall_top = 0
    wall_bottom = rng.choice((24, 32, 40))
    frame[actor_row : actor_row + 8, actor_col : actor_col + 8] = 5
    frame[target_row : target_row + 8, target_col : target_col + 8] = 7
    frame[wall_top:wall_bottom, wall_col : wall_col + 8] = 9
    deltas = {"ACTION1": (0, 8), "ACTION2": (24, 0), "ACTION3": (0, -8), "ACTION4": (-8, 0)}
    optimal_actions = ("ACTION2",)
    return SyntheticCase(family="blocked_detour", frame=frame, deltas=deltas, optimal_actions=optimal_actions)


def _random_near_target_case(rng: random.Random) -> SyntheticCase:
    frame = np.zeros((64, 64), dtype=np.int8)
    actor_row = rng.randrange(8, 40, 8)
    actor_col = rng.randrange(8, 40, 8)
    frame[actor_row : actor_row + 2, actor_col : actor_col + 2] = 5
    frame[actor_row : actor_row + 2, actor_col + 8 : actor_col + 10] = 7
    deltas = {"ACTION1": (0, 1), "ACTION2": (1, 0), "ACTION3": (0, -1), "ACTION4": (-1, 0)}
    return SyntheticCase(family="near_target_local", frame=frame, deltas=deltas, optimal_actions=("ACTION1",))


def generate_cases(*, per_family: int, seed: int) -> list[SyntheticCase]:
    rng = random.Random(seed)
    cases: list[SyntheticCase] = []
    for _ in range(per_family):
        cases.append(_random_open_case(rng))
        cases.append(_random_blocked_detour_case(rng))
        cases.append(_random_near_target_case(rng))
    return cases


def _legacy_score(
    frame: np.ndarray,
    actor_component: FrameComponent,
    target_anchor: tuple[int, int],
    delta: tuple[int, int],
    confidence: float,
) -> float:
    next_anchor = (actor_component.anchor[0] + delta[0], actor_component.anchor[1] + delta[1])
    if not (0 <= next_anchor[0] < frame.shape[0] and 0 <= next_anchor[1] < frame.shape[1]):
        return float("-inf")
    current_distance = abs(actor_component.anchor[0] - target_anchor[0]) + abs(actor_component.anchor[1] - target_anchor[1])
    next_distance = abs(next_anchor[0] - target_anchor[0]) + abs(next_anchor[1] - target_anchor[1])
    improvement = current_distance - next_distance
    score = 0.55 * confidence + 0.32
    score += 0.28 / (1.0 + next_distance)
    if improvement > 0:
        score += 0.62 * min(improvement, 3)
    elif improvement < 0:
        score -= 0.16 * min(-improvement, 3)
    return score


def _optimal_actions(case: SyntheticCase) -> tuple[str, ...]:
    if case.family != "blocked_detour":
        return case.optimal_actions
    actor = _actor_component(case.frame)
    target = _target_component(case.frame)
    actor_cell = coarse_cell_for_coord(actor.anchor)
    target_cell = coarse_cell_for_coord(target.anchor)
    blocked: set[tuple[int, int]] = set()
    for component in extract_components(case.frame):
        if component.color == 9:
            rows = range(component.bounds[0] // 8, component.bounds[2] // 8 + 1)
            cols = range(component.bounds[1] // 8, component.bounds[3] // 8 + 1)
            for row in rows:
                for col in cols:
                    blocked.add((row, col))
    best_distance = 99
    best: list[str] = []
    for action_name, delta in case.deltas.items():
        next_anchor = (actor.anchor[0] + delta[0], actor.anchor[1] + delta[1])
        if not (0 <= next_anchor[0] < 64 and 0 <= next_anchor[1] < 64):
            continue
        next_cell = coarse_cell_for_coord(next_anchor)
        distance = coarse_path_distance(next_cell, target_cell, blocked=frozenset(blocked - {next_cell, target_cell}))
        if distance < best_distance:
            best_distance = distance
            best = [action_name]
        elif distance == best_distance:
            best.append(action_name)
    return tuple(best)


def evaluate_cases(cases: list[SyntheticCase]) -> dict[str, object]:
    agent = ArcAgi3TACTICPublicAgent()
    family_rows: dict[str, dict[str, int]] = {}
    for case in cases:
        memory = EnvironmentMemory()
        level_memory = LevelMemory(step_index=6, last_changed_step=3)
        actor_component = _actor_component(case.frame)
        target_component = _target_component(case.frame)
        for action_name, delta in case.deltas.items():
            memory.record_motion(action_name, delta, actor_component.feature)
        for _ in range(3):
            memory.interaction_target_stat_for(target_component.feature).update(4.0, changed=True, level_gain=True)
        candidates = tuple(
            ActionCandidate(getattr(GameAction, action_name), None, (action_name, None), action_name)
            for action_name in case.deltas
        )
        interaction_targets = ((target_component.anchor, 1.0),)
        current_scores = {
            candidate.action.name: agent._control_map_candidate_score(
                candidate,
                case.frame,
                memory,
                level_memory,
                actor_component=actor_component,
                interaction_targets=interaction_targets,
            )
            for candidate in candidates
        }
        legacy_scores = {
            action_name: _legacy_score(case.frame, actor_component, target_component.anchor, delta, 1.0)
            for action_name, delta in case.deltas.items()
        }
        current_choice = max(current_scores.items(), key=lambda item: item[1])[0]
        legacy_choice = max(legacy_scores.items(), key=lambda item: item[1])[0]
        optimal_actions = _optimal_actions(case)
        row = family_rows.setdefault(
            case.family,
            {
                "cases": 0,
                "current_hits": 0,
                "legacy_hits": 0,
            },
        )
        row["cases"] += 1
        row["current_hits"] += int(current_choice in optimal_actions)
        row["legacy_hits"] += int(legacy_choice in optimal_actions)
    summary = {"families": {}, "total_cases": len(cases)}
    for family, row in family_rows.items():
        summary["families"][family] = {
            **row,
            "current_accuracy": row["current_hits"] / max(row["cases"], 1),
            "legacy_accuracy": row["legacy_hits"] / max(row["cases"], 1),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic control-value benchmark for ARC TACTIC.")
    parser.add_argument("--per-family", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cases = generate_cases(per_family=args.per_family, seed=args.seed)
    summary = evaluate_cases(cases)
    payload = {"seed": args.seed, "per_family": args.per_family, **summary}
    text = json.dumps(payload, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
