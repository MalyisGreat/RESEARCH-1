from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState


ASCII_PALETTE = " .,:;ox%#@ABCDEFGHIJKLMN"


@dataclass(frozen=True, slots=True)
class ClickPoint:
    x: int
    y: int
    score: float
    source: str


@dataclass(frozen=True, slots=True)
class ActionChoice:
    action_id: int
    x: int | None = None
    y: int | None = None
    source: str = "button"

    def key(self) -> tuple[int, int | None, int | None]:
        return (self.action_id, self.x, self.y)

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {"action_id": self.action_id, "source": self.source}
        if self.x is not None and self.y is not None:
            payload["x"] = self.x
            payload["y"] = self.y
        return payload


@dataclass(slots=True)
class Stat:
    count: int = 0
    total_gain: float = 0.0
    best_gain: float = 0.0

    def update(self, gain: float) -> None:
        self.count += 1
        self.total_gain += gain
        if gain > self.best_gain:
            self.best_gain = gain

    @property
    def mean_gain(self) -> float:
        if self.count <= 0:
            return 0.0
        return self.total_gain / self.count


@dataclass(slots=True)
class RunResult:
    game_id: str
    steps: int
    levels_completed: int
    win_levels: int
    state: str
    action_trace: list[dict[str, object]] = field(default_factory=list)


@dataclass(slots=True)
class BenchmarkResult:
    games: list[RunResult]
    scorecard: dict[str, object]
    operation_mode: str
    step_budget: int
    game_count: int


def frame_array(obs) -> np.ndarray:
    arr = np.asarray(obs.frame, dtype=np.int16)
    if arr.ndim == 3:
        arr = arr[-1]
    return arr


def downsample_signature(frame: np.ndarray, size: int = 8) -> tuple[int, ...]:
    block_h = max(frame.shape[0] // size, 1)
    block_w = max(frame.shape[1] // size, 1)
    values: list[int] = []
    for row in range(0, frame.shape[0], block_h):
        if len(values) >= size * size:
            break
        for col in range(0, frame.shape[1], block_w):
            patch = frame[row : row + block_h, col : col + block_w]
            if patch.size == 0:
                values.append(0)
                continue
            unique, counts = np.unique(patch, return_counts=True)
            values.append(int(unique[np.argmax(counts)]))
            if len(values) >= size * size:
                break
    if len(values) < size * size:
        values.extend([0] * (size * size - len(values)))
    return tuple(values)


def state_signature(obs) -> tuple[object, ...]:
    frame = frame_array(obs)
    return (
        frame.tobytes(),
        downsample_signature(frame),
        tuple(int(value) for value in obs.available_actions),
        int(obs.levels_completed),
    )


def frame_change_ratio(prev_frame: np.ndarray, next_frame: np.ndarray) -> float:
    if prev_frame.shape != next_frame.shape:
        return 1.0
    return float(np.mean(prev_frame != next_frame))


def connected_components(frame: np.ndarray) -> list[tuple[int, list[tuple[int, int]]]]:
    height, width = frame.shape
    seen = np.zeros((height, width), dtype=bool)
    groups: list[tuple[int, list[tuple[int, int]]]] = []

    for row in range(height):
        for col in range(width):
            color = int(frame[row, col])
            if color == 0 or seen[row, col]:
                continue
            cells: list[tuple[int, int]] = []
            queue = deque([(row, col)])
            seen[row, col] = True
            while queue:
                current_row, current_col = queue.popleft()
                cells.append((current_row, current_col))
                for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nxt_row = current_row + d_row
                    nxt_col = current_col + d_col
                    if 0 <= nxt_row < height and 0 <= nxt_col < width and not seen[nxt_row, nxt_col]:
                        if int(frame[nxt_row, nxt_col]) == color:
                            seen[nxt_row, nxt_col] = True
                            queue.append((nxt_row, nxt_col))
            groups.append((color, cells))
    return groups


def candidate_click_points(frame: np.ndarray) -> list[ClickPoint]:
    height, width = frame.shape
    points: dict[tuple[int, int], ClickPoint] = {}

    def add_point(x: int, y: int, score: float, source: str) -> None:
        x = max(0, min(width - 1, int(x)))
        y = max(0, min(height - 1, int(y)))
        key = (x, y)
        existing = points.get(key)
        if existing is None or score > existing.score:
            points[key] = ClickPoint(x=x, y=y, score=score, source=source)

    comps = connected_components(frame)
    for color, cells in comps:
        rows = [row for row, _ in cells]
        cols = [col for _, col in cells]
        min_row, max_row = min(rows), max(rows)
        min_col, max_col = min(cols), max(cols)
        center_row = int(round(sum(rows) / len(rows)))
        center_col = int(round(sum(cols) / len(cols)))
        extent = (max_row - min_row + 1) * (max_col - min_col + 1)
        score = math.log2(len(cells) + 1) + min(extent / 64.0, 1.5) + color * 0.01
        add_point(center_col, center_row, score + 0.30, "component_center")
        add_point(min_col, min_row, score, "bbox_corner")
        add_point(max_col, min_row, score, "bbox_corner")
        add_point(min_col, max_row, score, "bbox_corner")
        add_point(max_col, max_row, score, "bbox_corner")
        add_point((min_col + max_col) // 2, min_row, score - 0.05, "bbox_mid")
        add_point((min_col + max_col) // 2, max_row, score - 0.05, "bbox_mid")
        add_point(min_col, (min_row + max_row) // 2, score - 0.05, "bbox_mid")
        add_point(max_col, (min_row + max_row) // 2, score - 0.05, "bbox_mid")

    for y in range(4, height, 8):
        for x in range(4, width, 8):
            patch = frame[max(0, y - 2) : min(height, y + 3), max(0, x - 2) : min(width, x + 3)]
            density = float(np.mean(patch != 0))
            if density > 0.04:
                add_point(x, y, 0.10 + density, "grid_probe")

    ranked = sorted(points.values(), key=lambda item: (-item.score, item.y, item.x))
    return ranked[:48]


class RealArcExplorer:
    def __init__(self, *, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._state_action_stats: dict[tuple[object, ...], dict[tuple[int, int | None, int | None], Stat]] = {}
        self._global_stats: dict[tuple[int, int | None, int | None], Stat] = {}
        self._last_action_key: tuple[int, int | None, int | None] | None = None

    def choose(self, obs, action_space: list[GameAction]) -> ActionChoice:
        signature = state_signature(obs)
        per_state = self._state_action_stats.setdefault(signature, {})
        frame = frame_array(obs)
        candidates: list[ActionChoice] = []

        for action in action_space:
            action_id = int(action.value)
            if action_id == int(GameAction.ACTION6.value):
                for click in candidate_click_points(frame):
                    candidates.append(
                        ActionChoice(
                            action_id=action_id,
                            x=click.x,
                            y=click.y,
                            source=click.source,
                        )
                    )
            else:
                candidates.append(ActionChoice(action_id=action_id))

        if not candidates:
            return ActionChoice(action_id=int(GameAction.RESET), source="reset")

        best_choice = candidates[0]
        best_score = float("-inf")
        for choice in candidates:
            key = choice.key()
            local = per_state.get(key, Stat())
            global_stat = self._global_stats.get(key, Stat())
            novelty = 1.0 / (1.0 + local.count)
            global_novelty = 1.0 / math.sqrt(1.0 + global_stat.count)
            impact = max(local.best_gain, local.mean_gain, global_stat.best_gain, global_stat.mean_gain)
            structure_bonus = 0.0
            if choice.source == "component_center":
                structure_bonus = 0.25
            elif choice.source == "bbox_mid":
                structure_bonus = 0.12
            elif choice.source == "grid_probe":
                structure_bonus = 0.05
            repeat_penalty = 0.35 if key == self._last_action_key else 0.0
            tie_break = self._rng.random() * 0.01
            score = 2.4 * novelty + 0.8 * global_novelty + 4.0 * impact + structure_bonus - repeat_penalty + tie_break
            if score > best_score:
                best_score = score
                best_choice = choice

        self._last_action_key = best_choice.key()
        return best_choice

    def update(self, prev_obs, action: ActionChoice, next_obs) -> None:
        prev_sig = state_signature(prev_obs)
        prev_frame = frame_array(prev_obs)
        next_frame = frame_array(next_obs)
        gain = frame_change_ratio(prev_frame, next_frame)
        gain += 1.5 * max(0, int(next_obs.levels_completed) - int(prev_obs.levels_completed))
        if next_obs.state == GameState.WIN:
            gain += 4.0
        if next_obs.state == GameState.GAME_OVER:
            gain -= 0.5

        state_stats = self._state_action_stats.setdefault(prev_sig, {})
        state_stats.setdefault(action.key(), Stat()).update(gain)
        self._global_stats.setdefault(action.key(), Stat()).update(gain)


def action_enum(action_id: int) -> GameAction:
    return GameAction.from_id(int(action_id))


def action_data(choice: ActionChoice) -> dict[str, int] | None:
    if choice.x is None or choice.y is None:
        return None
    return {"x": int(choice.x), "y": int(choice.y)}


def render_ascii(obs, *, last_action: ActionChoice | None, step_index: int) -> str:
    frame = frame_array(obs)
    rows: list[str] = []
    for row in frame:
        glyphs = []
        for value in row:
            index = int(value)
            if index < 0:
                index = 0
            if index >= len(ASCII_PALETTE):
                index = len(ASCII_PALETTE) - 1
            glyphs.append(ASCII_PALETTE[index])
        rows.append("".join(glyphs))

    header = [
        f"step={step_index} state={obs.state.name} levels={obs.levels_completed}/{obs.win_levels}",
        f"available={list(obs.available_actions)}",
        f"last_action={last_action.to_payload() if last_action is not None else None}",
    ]
    return "\n".join(header + rows)


def serialize_scorecard(raw_scorecard) -> dict[str, object]:
    if hasattr(raw_scorecard, "model_dump"):
        return raw_scorecard.model_dump()
    if hasattr(raw_scorecard, "dict"):
        return raw_scorecard.dict()
    if hasattr(raw_scorecard, "__dict__"):
        return dict(raw_scorecard.__dict__)
    return {"value": str(raw_scorecard)}


def ensure_games_cached(games: list[str]) -> None:
    bootstrap = Arcade(operation_mode=OperationMode.NORMAL)
    for game_id in games:
        env = bootstrap.make(game_id)
        if env is None:
            raise RuntimeError(f"failed to bootstrap environment {game_id}")
        obs = env.reset()
        if obs is None:
            raise RuntimeError(f"failed to bootstrap reset for {game_id}")


def run_game(
    arc: Arcade,
    game_id: str,
    *,
    step_budget: int,
    render: bool,
    delay: float,
    seed: int,
) -> RunResult:
    env = arc.make(game_id)
    if env is None:
        raise RuntimeError(f"failed to create environment {game_id}")

    obs = env.reset()
    if obs is None:
        raise RuntimeError(f"failed to reset environment {game_id}")

    explorer = RealArcExplorer(seed=seed)
    trace: list[dict[str, object]] = []
    last_action: ActionChoice | None = None

    if render:
        print(render_ascii(obs, last_action=last_action, step_index=0))
        time.sleep(delay)

    for step_index in range(1, step_budget + 1):
        choice = explorer.choose(obs, env.action_space)
        nxt = env.step(action_enum(choice.action_id), data=action_data(choice))
        if nxt is None:
            break
        explorer.update(obs, choice, nxt)
        trace.append(
            {
                "step": step_index,
                "action": choice.to_payload(),
                "state": nxt.state.name,
                "levels_completed": int(nxt.levels_completed),
            }
        )
        obs = nxt
        last_action = choice
        if render:
            sys.stdout.write("\x1b[2J\x1b[H")
            sys.stdout.write(render_ascii(obs, last_action=last_action, step_index=step_index))
            sys.stdout.write("\n")
            sys.stdout.flush()
            time.sleep(delay)
        if obs.state in (GameState.WIN, GameState.GAME_OVER):
            break

    return RunResult(
        game_id=game_id,
        steps=len(trace),
        levels_completed=int(obs.levels_completed),
        win_levels=int(obs.win_levels),
        state=obs.state.name,
        action_trace=trace,
    )


def public_game_ids() -> list[str]:
    discovery = Arcade(operation_mode=OperationMode.NORMAL)
    environments = discovery.get_environments()
    return [item.game_id.split("-", 1)[0] for item in environments]


def run_benchmark(
    games: list[str],
    *,
    step_budget: int,
    render: bool,
    delay: float,
    seed: int,
    force_offline: bool = True,
) -> BenchmarkResult:
    unique_games = list(dict.fromkeys(games))
    if force_offline:
        ensure_games_cached(unique_games)
        arc = Arcade(operation_mode=OperationMode.OFFLINE)
        mode_name = "OFFLINE"
    else:
        arc = Arcade(operation_mode=OperationMode.NORMAL)
        mode_name = "NORMAL"

    results = [
        run_game(
            arc,
            game_id=game_id,
            step_budget=step_budget,
            render=render,
            delay=delay,
            seed=seed,
        )
        for game_id in unique_games
    ]
    scorecard = serialize_scorecard(arc.get_scorecard())
    return BenchmarkResult(
        games=results,
        scorecard=scorecard,
        operation_mode=mode_name,
        step_budget=step_budget,
        game_count=len(unique_games),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a generic frame-based ARC-AGI-3 explorer.")
    parser.add_argument("--games", nargs="+", default=["ls20"], help="Game ids to run.")
    parser.add_argument("--all-public", action="store_true", help="Run against the full current public game list.")
    parser.add_argument("--steps", type=int, default=120, help="Max actions per game.")
    parser.add_argument("--render", action="store_true", help="Render in ASCII in the terminal.")
    parser.add_argument("--delay", type=float, default=0.08, help="Delay between rendered steps.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--mode",
        choices=("offline", "normal"),
        default="offline",
        help="Benchmark execution mode. Offline is the recommended serious benchmark path after cache bootstrap.",
    )
    parser.add_argument("--json-out", type=Path, default=None, help="Optional output path for results JSON.")
    args = parser.parse_args()

    games = public_game_ids() if args.all_public else args.games
    benchmark = run_benchmark(
        games,
        step_budget=args.steps,
        render=args.render,
        delay=args.delay,
        seed=args.seed,
        force_offline=args.mode == "offline",
    )

    payload = {
        "operation_mode": benchmark.operation_mode,
        "step_budget": benchmark.step_budget,
        "game_count": benchmark.game_count,
        "scorecard": benchmark.scorecard,
        "summary": {
            "score": benchmark.scorecard.get("score", 0.0),
            "total_environments_completed": benchmark.scorecard.get("total_environments_completed", 0),
            "total_environments": benchmark.scorecard.get("total_environments", benchmark.game_count),
            "total_levels_completed": benchmark.scorecard.get("total_levels_completed", 0),
            "total_levels": benchmark.scorecard.get("total_levels", 0),
            "total_actions": benchmark.scorecard.get("total_actions", 0),
        },
        "games": [
            {
                "game_id": result.game_id,
                "steps": result.steps,
                "levels_completed": result.levels_completed,
                "win_levels": result.win_levels,
                "state": result.state,
                "action_trace": result.action_trace,
            }
            for result in benchmark.games
        ],
    }
    text = json.dumps(payload, indent=2, default=str)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
