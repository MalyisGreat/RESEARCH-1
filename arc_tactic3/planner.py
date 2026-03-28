from __future__ import annotations

import heapq
from dataclasses import dataclass

from .core import Action, ClickAction, GameState, manhattan
from .dsl import is_solved, simulate_action
from .hypotheses import MechanicHypothesis


@dataclass(frozen=True, slots=True)
class PlannerResult:
    actions: tuple[Action, ...]
    expanded: int


def candidate_actions(state: GameState, hypothesis: MechanicHypothesis) -> tuple[Action, ...]:
    actions: list[Action] = [button for button, _ in hypothesis.movement_map]
    if hypothesis.click_mode == "switch":
        actions.extend(ClickAction(row, col) for row, col in sorted(state.switches))
    elif hypothesis.click_mode == "teleport":
        actions.extend(ClickAction(row, col) for row, col in state.portals)
    return tuple(actions)


def heuristic(state: GameState, family: str) -> int:
    if family == "push_box":
        if state.boxes & state.targets:
            return 0
        if not state.boxes or not state.targets:
            return 3
        box_target = min(manhattan(box, target) for box in state.boxes for target in state.targets)
        player_box = min(manhattan(state.player, box) for box in state.boxes)
        return box_target + player_box

    if not state.goals:
        return 0
    goal_dist = min(manhattan(state.player, goal) for goal in state.goals)
    if family == "key_goal" and not state.has_key and state.keys:
        goal_dist += min(manhattan(state.player, key) for key in state.keys)
    if family == "switch_goal" and not state.switch_active and state.switches:
        goal_dist += min(manhattan(state.player, switch) for switch in state.switches)
    return goal_dist


def plan_with_hypothesis(
    state: GameState,
    hypothesis: MechanicHypothesis,
    *,
    max_expansions: int = 4000,
    max_depth: int = 40,
) -> PlannerResult | None:
    queue: list[tuple[int, int, int, GameState, tuple[Action, ...]]] = []
    start_key = state.signature()
    best_cost = {start_key: 0}
    serial = 0
    heapq.heappush(queue, (heuristic(state, hypothesis.family), 0, serial, state, ()))
    expanded = 0
    actions = candidate_actions(state, hypothesis)

    while queue and expanded < max_expansions:
        _, cost, _, current, path = heapq.heappop(queue)
        expanded += 1
        if is_solved(current, hypothesis.family):
            return PlannerResult(actions=path, expanded=expanded)
        if cost >= max_depth:
            continue
        for action in actions:
            next_state = simulate_action(
                current,
                action,
                hypothesis.config(tuple(button for button, _ in hypothesis.movement_map)),
            )
            next_key = next_state.signature()
            next_cost = cost + 1
            if next_cost >= best_cost.get(next_key, 10**9):
                continue
            best_cost[next_key] = next_cost
            serial += 1
            heapq.heappush(
                queue,
                (
                    next_cost + heuristic(next_state, hypothesis.family),
                    next_cost,
                    serial,
                    next_state,
                    path + (action,),
                ),
            )
    return None
