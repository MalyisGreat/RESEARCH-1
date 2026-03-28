from __future__ import annotations

import sys
import time
from dataclasses import dataclass


@dataclass
class ProgressBar:
    label: str
    total: int
    width: int = 24
    stream = sys.stderr

    def __post_init__(self) -> None:
        self.current = 0
        self.started_at = time.time()
        self._render("")

    def update(self, current: int, detail: str = "") -> None:
        self.current = max(0, min(current, self.total))
        self._render(detail)

    def advance(self, step: int = 1, detail: str = "") -> None:
        self.update(self.current + step, detail)

    def finish(self, detail: str = "") -> None:
        self.current = self.total
        self._render(detail)
        print(file=self.stream, flush=True)

    def _render(self, detail: str) -> None:
        total = max(self.total, 1)
        ratio = self.current / total
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.time() - self.started_at
        message = (
            f"\r{self.label} [{bar}] {self.current}/{self.total} "
            f"{ratio * 100:5.1f}% elapsed={elapsed:5.1f}s"
        )
        if detail:
            message += f" {detail}"
        print(message, end="", file=self.stream, flush=True)


class Spinner:
    def __init__(self, label: str, *, stream=None) -> None:
        self.label = label
        self.stream = stream or sys.stderr
        self.frames = "|/-\\"
        self.index = 0
        self.started_at = time.time()

    def tick(self, detail: str = "") -> None:
        frame = self.frames[self.index % len(self.frames)]
        self.index += 1
        elapsed = time.time() - self.started_at
        message = f"\r{self.label} {frame} elapsed={elapsed:5.1f}s"
        if detail:
            message += f" {detail}"
        print(message, end="", file=self.stream, flush=True)

    def finish(self, detail: str = "") -> None:
        elapsed = time.time() - self.started_at
        message = f"\r{self.label} done elapsed={elapsed:5.1f}s"
        if detail:
            message += f" {detail}"
        print(message, file=self.stream, flush=True)
