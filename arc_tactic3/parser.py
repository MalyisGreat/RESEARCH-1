from __future__ import annotations

from dataclasses import dataclass

from .core import CLICKABLE_KINDS, COLOR_TO_KIND, Coord, GameState, Grid, StaticScene


@dataclass(frozen=True, slots=True)
class ParsedObject:
    kind: str
    color: int
    cells: tuple[Coord, ...]
    anchor: Coord


@dataclass(frozen=True, slots=True)
class ParsedFrame:
    width: int
    height: int
    objects: tuple[ParsedObject, ...]

    def by_kind(self, kind: str) -> tuple[ParsedObject, ...]:
        return tuple(obj for obj in self.objects if obj.kind == kind)

    def clickable_targets(self) -> tuple[Coord, ...]:
        return tuple(obj.anchor for obj in self.objects if obj.kind in CLICKABLE_KINDS)

    def non_wall_anchors(self) -> tuple[Coord, ...]:
        return tuple(obj.anchor for obj in self.objects if obj.kind != "wall")


@dataclass(frozen=True, slots=True)
class TrackedObject:
    track_id: int
    kind: str
    anchor: Coord
    cells: tuple[Coord, ...]


class ObjectTracker:
    def __init__(self) -> None:
        self._next_id = 1
        self._last_by_kind: dict[str, list[TrackedObject]] = {}

    def update(self, parsed: ParsedFrame) -> tuple[TrackedObject, ...]:
        tracked: list[TrackedObject] = []
        next_by_kind: dict[str, list[TrackedObject]] = {}
        for kind in {obj.kind for obj in parsed.objects}:
            current = [obj for obj in parsed.objects if obj.kind == kind]
            previous = self._last_by_kind.get(kind, []).copy()
            current_tracked: list[TrackedObject] = []
            for obj in current:
                match = None
                if previous:
                    match = min(
                        previous,
                        key=lambda item: abs(item.anchor[0] - obj.anchor[0])
                        + abs(item.anchor[1] - obj.anchor[1]),
                    )
                    previous.remove(match)
                track_id = match.track_id if match is not None else self._allocate_id()
                current_tracked.append(
                    TrackedObject(track_id=track_id, kind=kind, anchor=obj.anchor, cells=obj.cells)
                )
            tracked.extend(current_tracked)
            next_by_kind[kind] = current_tracked
        self._last_by_kind = next_by_kind
        return tuple(sorted(tracked, key=lambda item: (item.kind, item.track_id)))

    def _allocate_id(self) -> int:
        track_id = self._next_id
        self._next_id += 1
        return track_id


def parse_frame(frame: Grid) -> ParsedFrame:
    height = len(frame)
    width = len(frame[0]) if frame else 0
    seen: set[Coord] = set()
    objects: list[ParsedObject] = []

    for row in range(height):
        for col in range(width):
            coord = (row, col)
            color = frame[row][col]
            if color == 0 or coord in seen or color not in COLOR_TO_KIND:
                continue
            stack = [coord]
            cells: list[Coord] = []
            while stack:
                current = stack.pop()
                if current in seen:
                    continue
                current_row, current_col = current
                if frame[current_row][current_col] != color:
                    continue
                seen.add(current)
                cells.append(current)
                for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    neighbor = (current_row + delta_row, current_col + delta_col)
                    if 0 <= neighbor[0] < height and 0 <= neighbor[1] < width and neighbor not in seen:
                        stack.append(neighbor)
            cells_tuple = tuple(sorted(cells))
            objects.append(
                ParsedObject(
                    kind=COLOR_TO_KIND[color],
                    color=color,
                    cells=cells_tuple,
                    anchor=cells_tuple[len(cells_tuple) // 2],
                )
            )
    return ParsedFrame(width=width, height=height, objects=tuple(objects))


def derive_static_scene(parsed: ParsedFrame) -> StaticScene:
    def cells(kind: str) -> frozenset[Coord]:
        return frozenset(cell for obj in parsed.by_kind(kind) for cell in obj.cells)

    return StaticScene(
        width=parsed.width,
        height=parsed.height,
        walls=cells("wall"),
        goals=cells("goal"),
        targets=cells("target"),
        switches=cells("switch"),
        portals=tuple(sorted(cells("portal"))),
    )


def build_state(
    parsed: ParsedFrame,
    static_scene: StaticScene | None = None,
    *,
    prior_state: GameState | None = None,
) -> GameState:
    def cells(kind: str) -> frozenset[Coord]:
        return frozenset(cell for obj in parsed.by_kind(kind) for cell in obj.cells)

    player_objects = parsed.by_kind("player")
    if len(player_objects) != 1:
        raise ValueError("exactly one player must be visible")

    player = player_objects[0].anchor
    walls = static_scene.walls if static_scene is not None else cells("wall")
    goals = static_scene.goals if static_scene is not None else cells("goal")
    targets = static_scene.targets if static_scene is not None else cells("target")
    switches = static_scene.switches if static_scene is not None else cells("switch")
    portals = static_scene.portals if static_scene is not None else tuple(sorted(cells("portal")))
    keys = cells("key")
    doors = cells("door")
    boxes = cells("box")

    has_key = prior_state.has_key if prior_state is not None else False
    if prior_state is not None and prior_state.keys and not keys:
        has_key = True

    switch_active = prior_state.switch_active if prior_state is not None else False
    if prior_state is not None and prior_state.doors and not doors and bool(switches):
        switch_active = True

    return GameState(
        width=parsed.width,
        height=parsed.height,
        player=player,
        walls=walls,
        goals=goals,
        keys=keys,
        doors=doors,
        boxes=boxes,
        targets=targets,
        switches=switches,
        portals=portals,
        has_key=has_key,
        switch_active=switch_active,
    )
