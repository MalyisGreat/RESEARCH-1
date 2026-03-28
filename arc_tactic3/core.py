from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias


Coord: TypeAlias = tuple[int, int]
Grid: TypeAlias = tuple[tuple[int, ...], ...]
Family: TypeAlias = Literal[
    "reach_goal",
    "key_goal",
    "switch_goal",
    "push_box",
    "portal_goal",
]

MOVE_BUTTONS: tuple[str, ...] = ("alpha", "beta", "gamma", "delta", "epsilon")
UNDO_BUTTON = "undo"
CARDINALS: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))

TILE_EMPTY = 0
TILE_WALL = 1
TILE_PLAYER = 2
TILE_GOAL = 3
TILE_KEY = 4
TILE_DOOR = 5
TILE_BOX = 6
TILE_TARGET = 7
TILE_SWITCH = 8
TILE_PORTAL = 9

COLOR_TO_KIND = {
    TILE_WALL: "wall",
    TILE_PLAYER: "player",
    TILE_GOAL: "goal",
    TILE_KEY: "key",
    TILE_DOOR: "door",
    TILE_BOX: "box",
    TILE_TARGET: "target",
    TILE_SWITCH: "switch",
    TILE_PORTAL: "portal",
}

CLICKABLE_KINDS = {"switch", "portal", "goal", "key", "box", "player"}


@dataclass(frozen=True, slots=True)
class ClickAction:
    row: int
    col: int

    @property
    def coord(self) -> Coord:
        return (self.row, self.col)

    def __str__(self) -> str:
        return f"click({self.row},{self.col})"


Action: TypeAlias = str | ClickAction


@dataclass(frozen=True, slots=True)
class StaticScene:
    width: int
    height: int
    walls: frozenset[Coord]
    goals: frozenset[Coord]
    targets: frozenset[Coord]
    switches: frozenset[Coord]
    portals: tuple[Coord, ...]


@dataclass(frozen=True, slots=True)
class GameState:
    width: int
    height: int
    player: Coord
    walls: frozenset[Coord]
    goals: frozenset[Coord]
    keys: frozenset[Coord]
    doors: frozenset[Coord]
    boxes: frozenset[Coord]
    targets: frozenset[Coord]
    switches: frozenset[Coord]
    portals: tuple[Coord, ...]
    has_key: bool = False
    switch_active: bool = False

    def static_scene(self) -> StaticScene:
        return StaticScene(
            width=self.width,
            height=self.height,
            walls=self.walls,
            goals=self.goals,
            targets=self.targets,
            switches=self.switches,
            portals=self.portals,
        )

    def signature(self) -> tuple[object, ...]:
        return (
            self.player,
            tuple(sorted(self.keys)),
            tuple(sorted(self.doors)),
            tuple(sorted(self.boxes)),
            self.has_key,
            self.switch_active,
        )


@dataclass(frozen=True, slots=True)
class MechanicConfig:
    family: Family
    available_buttons: tuple[str, ...]
    movement_map: tuple[tuple[str, Coord | None], ...]
    click_mode: str | None = None
    allows_undo: bool = False

    def button_map(self) -> dict[str, Coord | None]:
        return dict(self.movement_map)


@dataclass(frozen=True, slots=True)
class StepObservation:
    frame: Grid
    available_buttons: tuple[str, ...]
    allows_click: bool
    allows_undo: bool
    solved: bool
    action_count: int


def add_coords(lhs: Coord, rhs: Coord) -> Coord:
    return (lhs[0] + rhs[0], lhs[1] + rhs[1])


def manhattan(lhs: Coord, rhs: Coord) -> int:
    return abs(lhs[0] - rhs[0]) + abs(lhs[1] - rhs[1])
