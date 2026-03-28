from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

from .core import (
    Action,
    CARDINALS,
    MOVE_BUTTONS,
    TILE_BOX,
    TILE_DOOR,
    TILE_EMPTY,
    TILE_GOAL,
    TILE_KEY,
    TILE_PLAYER,
    TILE_PORTAL,
    TILE_SWITCH,
    TILE_TARGET,
    TILE_WALL,
    ClickAction,
    Coord,
    Family,
    GameState,
    Grid,
    MechanicConfig,
    StepObservation,
    add_coords,
)


@dataclass(frozen=True, slots=True)
class EnvironmentCase:
    env_id: str
    family: Family
    config: MechanicConfig
    levels: tuple[GameState, ...]
    level_configs: tuple[MechanicConfig, ...] | None = None

    def config_for_level(self, level_index: int) -> MechanicConfig:
        if self.level_configs is None:
            return self.config
        return self.level_configs[level_index]


class HiddenMechanicEnvironment:
    def __init__(self, case: EnvironmentCase, *, reveal_affordances: bool = True) -> None:
        self.case = case
        self.reveal_affordances = reveal_affordances
        self.state: GameState | None = None
        self.level_index = 0
        self.current_config: MechanicConfig | None = None
        self.history: list[GameState] = []
        self.action_count = 0

    def reset(self, level_index: int) -> StepObservation:
        self.level_index = level_index
        self.state = self.case.levels[level_index]
        self.current_config = self.case.config_for_level(level_index)
        self.history = []
        self.action_count = 0
        return self._observe()

    def step(self, action: Action) -> StepObservation:
        if self.state is None:
            raise RuntimeError("reset() must be called before step().")
        assert self.current_config is not None
        if action == "undo" and self.current_config.allows_undo and self.history:
            self.state = self.history.pop()
            self.action_count += 1
            return self._observe()
        if action != "undo":
            self.history.append(self.state)
        self.state = simulate_action(self.state, action, self.current_config)
        self.action_count += 1
        return self._observe()

    def _observe(self) -> StepObservation:
        assert self.state is not None
        assert self.current_config is not None
        return StepObservation(
            frame=render_state(self.state),
            available_buttons=self.current_config.available_buttons,
            allows_click=bool(self.current_config.click_mode) if self.reveal_affordances else False,
            allows_undo=self.current_config.allows_undo if self.reveal_affordances else False,
            solved=is_solved(self.state, self.current_config.family),
            action_count=self.action_count,
        )


def in_bounds(state: GameState, coord: Coord) -> bool:
    return 0 <= coord[0] < state.height and 0 <= coord[1] < state.width


def paired_portal(portals: tuple[Coord, ...], coord: Coord) -> Coord:
    if len(portals) != 2:
        return coord
    first, second = portals
    return second if coord == first else first


def is_solved(state: GameState, family: Family) -> bool:
    if family == "push_box":
        return bool(state.boxes & state.targets)
    if family == "key_goal":
        return state.has_key and state.player in state.goals
    if family == "switch_goal":
        return state.switch_active and state.player in state.goals
    return state.player in state.goals


def simulate_action(state: GameState, action: Action, config: MechanicConfig) -> GameState:
    if isinstance(action, ClickAction):
        if config.click_mode == "switch" and action.coord in state.switches:
            return GameState(
                width=state.width,
                height=state.height,
                player=state.player,
                walls=state.walls,
                goals=state.goals,
                keys=state.keys,
                doors=frozenset(),
                boxes=state.boxes,
                targets=state.targets,
                switches=state.switches,
                portals=state.portals,
                has_key=state.has_key,
                switch_active=True,
            )
        if config.click_mode == "teleport" and action.coord in state.portals:
            return GameState(
                width=state.width,
                height=state.height,
                player=action.coord,
                walls=state.walls,
                goals=state.goals,
                keys=state.keys,
                doors=state.doors,
                boxes=state.boxes,
                targets=state.targets,
                switches=state.switches,
                portals=state.portals,
                has_key=state.has_key,
                switch_active=state.switch_active,
            )
        return state

    if action == "undo":
        return state

    delta = config.button_map().get(action)
    if delta is None:
        return state

    next_player = add_coords(state.player, delta)
    if not in_bounds(state, next_player):
        return state
    if next_player in state.walls or next_player in state.doors:
        return state

    boxes = set(state.boxes)
    if next_player in boxes:
        pushed = add_coords(next_player, delta)
        if not in_bounds(state, pushed):
            return state
        if pushed in state.walls or pushed in state.doors or pushed in boxes:
            return state
        boxes.remove(next_player)
        boxes.add(pushed)

    keys = set(state.keys)
    doors = set(state.doors)
    has_key = state.has_key
    switch_active = state.switch_active
    player = next_player

    if player in keys:
        keys.remove(player)
        has_key = True
        if config.family == "key_goal":
            doors.clear()
    if player in state.switches and config.family == "switch_goal":
        switch_active = True
        doors.clear()
    if config.family == "portal_goal" and player in state.portals:
        player = paired_portal(state.portals, player)

    return GameState(
        width=state.width,
        height=state.height,
        player=player,
        walls=state.walls,
        goals=state.goals,
        keys=frozenset(keys),
        doors=frozenset(doors),
        boxes=frozenset(boxes),
        targets=state.targets,
        switches=state.switches,
        portals=state.portals,
        has_key=has_key,
        switch_active=switch_active,
    )


def render_state(state: GameState) -> Grid:
    rows = [[TILE_EMPTY for _ in range(state.width)] for _ in range(state.height)]
    for row, col in state.walls:
        rows[row][col] = TILE_WALL
    for row, col in state.goals:
        rows[row][col] = TILE_GOAL
    for row, col in state.targets:
        rows[row][col] = TILE_TARGET
    for row, col in state.switches:
        rows[row][col] = TILE_SWITCH
    for row, col in state.portals:
        rows[row][col] = TILE_PORTAL
    for row, col in state.keys:
        rows[row][col] = TILE_KEY
    for row, col in state.doors:
        rows[row][col] = TILE_DOOR
    for row, col in state.boxes:
        rows[row][col] = TILE_BOX
    rows[state.player[0]][state.player[1]] = TILE_PLAYER
    return tuple(tuple(row) for row in rows)


def parse_ascii_level(layout: str) -> GameState:
    lines = [line.strip() for line in layout.strip("\n").splitlines()]
    height = len(lines)
    width = len(lines[0])
    walls: set[Coord] = set()
    goals: set[Coord] = set()
    keys: set[Coord] = set()
    doors: set[Coord] = set()
    boxes: set[Coord] = set()
    targets: set[Coord] = set()
    switches: set[Coord] = set()
    portals: list[Coord] = []
    player: Coord | None = None

    for row, line in enumerate(lines):
        for col, glyph in enumerate(line):
            coord = (row, col)
            if glyph == "#":
                walls.add(coord)
            elif glyph == "P":
                player = coord
            elif glyph == "G":
                goals.add(coord)
            elif glyph == "K":
                keys.add(coord)
            elif glyph == "D":
                doors.add(coord)
            elif glyph == "B":
                boxes.add(coord)
            elif glyph == "T":
                targets.add(coord)
            elif glyph == "S":
                switches.add(coord)
            elif glyph == "O":
                portals.append(coord)
    if player is None:
        raise ValueError("layout must contain a player")

    return GameState(
        width=width,
        height=height,
        player=player,
        walls=frozenset(walls),
        goals=frozenset(goals),
        keys=frozenset(keys),
        doors=frozenset(doors),
        boxes=frozenset(boxes),
        targets=frozenset(targets),
        switches=frozenset(switches),
        portals=tuple(portals),
    )


def build_hidden_mapping(seed: int) -> tuple[tuple[str, Coord | None], ...]:
    order = list(permutations(MOVE_BUTTONS, len(MOVE_BUTTONS)))[seed % 120]
    pairs: list[tuple[str, Coord | None]] = []
    for button, delta in zip(order[:4], CARDINALS, strict=True):
        pairs.append((button, delta))
    pairs.append((order[4], None))
    return tuple(sorted(pairs))


def make_case(
    env_id: str,
    family: Family,
    layouts: tuple[str, ...],
    seed: int,
    *,
    click_mode: str | None = None,
    allows_undo: bool = False,
) -> EnvironmentCase:
    return EnvironmentCase(
        env_id=env_id,
        family=family,
        config=MechanicConfig(
            family=family,
            available_buttons=MOVE_BUTTONS,
            movement_map=build_hidden_mapping(seed),
            click_mode=click_mode,
            allows_undo=allows_undo,
        ),
        levels=tuple(parse_ascii_level(layout) for layout in layouts),
    )


def build_benchmark_suite() -> tuple[EnvironmentCase, ...]:
    return (
        make_case(
            "reach_maze",
            "reach_goal",
            (
                """
                #######
                #P....#
                #.....#
                #....G#
                #######
                """,
                """
                ########
                #P.....#
                #.####.#
                #....#.#
                #.##...#
                #....#G#
                ########
                """,
                """
                #########
                #P......#
                #.#####.#
                #.#...#.#
                #.#.#.#.#
                #...#...#
                #.#####G#
                #########
                """,
                """
                ##########
                #P.......#
                #.######.#
                #.....#..#
                #.###.#.##
                #.#...#..#
                #.#.####.#
                #...#...G#
                ##########
                """,
                """
                ###########
                #P........#
                #.#######.#
                #.#.....#.#
                #.#.###.#.#
                #...#.#...#
                ###.#.#.###
                #.....#..G#
                ###########
                """,
            ),
            seed=3,
        ),
        make_case(
            "key_doors",
            "key_goal",
            (
                """
                #######
                #P.KDG#
                #.....#
                #######
                """,
                """
                ########
                #P..#..#
                #.#.#K##
                #.#...D#
                #.###.G#
                #......#
                ########
                """,
                """
                #########
                #P....#.#
                #.###.#K#
                #...#...#
                ###.###D#
                #.......#
                #.#####G#
                #########
                """,
                """
                ##########
                #P..#....#
                #.#.#.##.#
                #.#...#K.#
                #.#####..#
                #.....##D#
                #.######G#
                #........#
                ##########
                """,
                """
                ###########
                #P..#.....#
                #.#.#.###.#
                #.#...#...#
                #.#####.#K#
                #.....#.#.#
                #.###.#.#D#
                #...#...#G#
                ###########
                """,
            ),
            seed=17,
            allows_undo=True,
        ),
        make_case(
            "switch_paths",
            "switch_goal",
            (
                """
                #######
                #P.SDG#
                #.....#
                #######
                """,
                """
                ########
                #P.....#
                #.####S#
                #....#D#
                #.##...#
                #....#G#
                ########
                """,
                """
                #########
                #P....#.#
                #.###.#S#
                #...#...#
                ###.###D#
                #.......#
                #.#####G#
                #########
                """,
                """
                ##########
                #P..#....#
                #.#.#.##.#
                #.#...#S.#
                #.#####..#
                #.....##D#
                #.######G#
                #........#
                ##########
                """,
                """
                ###########
                #P..#.....#
                #.#.#.###.#
                #.#...#...#
                #.#####.#S#
                #.....#.#.#
                #.###.#.#D#
                #...#...#G#
                ###########
                """,
            ),
            seed=29,
        ),
        make_case(
            "switch_click",
            "switch_goal",
            (
                """
                #######
                #P..DG#
                #..S..#
                #######
                """,
                """
                ########
                #P..#..#
                #.#.#S##
                #.#...D#
                #.###.G#
                #......#
                ########
                """,
                """
                #########
                #P....#.#
                #.###.#S#
                #...#...#
                ###.###D#
                #.......#
                #.#####G#
                #########
                """,
                """
                ##########
                #P..#....#
                #.#.#.##.#
                #.#...#S.#
                #.#####..#
                #.....##D#
                #.######G#
                #........#
                ##########
                """,
                """
                ###########
                #P..#.....#
                #.#.#.###.#
                #.#...#...#
                #.#####.#S#
                #.....#.#.#
                #.###.#.#D#
                #...#...#G#
                ###########
                """,
            ),
            seed=41,
            click_mode="switch",
        ),
        make_case(
            "push_blocks",
            "push_box",
            (
                """
                #######
                #PBT..#
                #.....#
                #######
                """,
                """
                ########
                #P.....#
                #.B....#
                #......#
                #....T.#
                ########
                """,
                """
                #########
                #P......#
                #..###..#
                #..B....#
                #..###..#
                #....T..#
                #.......#
                #########
                """,
                """
                ##########
                #P.......#
                #..####..#
                #..B.....#
                #..#..#..#
                #..#..#T.#
                #........#
                ##########
                """,
                """
                ###########
                #P........#
                #..#####..#
                #..B......#
                #..#...#..#
                #..#...#T.#
                #.........#
                ###########
                """,
            ),
            seed=53,
        ),
        make_case(
            "portal_routes",
            "portal_goal",
            (
                """
                #######
                #PO..G#
                #..O..#
                #######
                """,
                """
                ########
                #P..#O.#
                #.#.##.#
                #.#...O#
                #.###..#
                #.....G#
                ########
                """,
                """
                #########
                #P....#.#
                #.###.#O#
                #...#...#
                ###.###.#
                #O......#
                #.#####G#
                #########
                """,
                """
                ##########
                #P..#....#
                #.#.#.##.#
                #.#...#O.#
                #.#####..#
                #.....##.#
                #.######G#
                #O.......#
                ##########
                """,
                """
                ###########
                #P..#.....#
                #.#.#.###.#
                #.#...#...#
                #.#####.#O#
                #.....#.#.#
                #.###.#.#.#
                #O..#...#G#
                ###########
                """,
            ),
            seed=67,
        ),
    )


def reorder_available_buttons(case: EnvironmentCase, order: tuple[str, ...]) -> EnvironmentCase:
    if set(order) != set(case.config.available_buttons):
        raise ValueError("order must contain the same button labels")
    def remap_config(config: MechanicConfig) -> MechanicConfig:
        return MechanicConfig(
            family=config.family,
            available_buttons=order,
            movement_map=config.movement_map,
            click_mode=config.click_mode,
            allows_undo=config.allows_undo,
        )
    return EnvironmentCase(
        env_id=f"{case.env_id}:reordered",
        family=case.family,
        config=remap_config(case.config),
        levels=case.levels,
        level_configs=tuple(remap_config(config) for config in case.level_configs) if case.level_configs else None,
    )


def rename_button_labels(case: EnvironmentCase, labels: tuple[str, ...]) -> EnvironmentCase:
    old_buttons = case.config.available_buttons
    if len(labels) != len(old_buttons):
        raise ValueError("labels must match button count")
    mapping = dict(zip(old_buttons, labels, strict=True))
    def rename_config(config: MechanicConfig) -> MechanicConfig:
        return MechanicConfig(
            family=config.family,
            available_buttons=labels,
            movement_map=tuple(sorted((mapping[button], delta) for button, delta in config.movement_map)),
            click_mode=config.click_mode,
            allows_undo=config.allows_undo,
        )
    return EnvironmentCase(
        env_id=f"{case.env_id}:renamed",
        family=case.family,
        config=rename_config(case.config),
        levels=case.levels,
        level_configs=tuple(rename_config(config) for config in case.level_configs) if case.level_configs else None,
    )
