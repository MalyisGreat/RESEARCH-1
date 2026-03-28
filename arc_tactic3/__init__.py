"""TACTIC-3 toy benchmark and solver package."""

from .agents import (
    FrontierGraphAgent,
    RandomAgent,
    TACTICAgent,
    TACTICNoPlannerAgent,
    TACTICStrictAgent,
    TACTICNoTransferAgent,
)
from .dsl import EnvironmentCase, HiddenMechanicEnvironment, build_benchmark_suite
from .prior import LocalQwenPrior

__all__ = [
    "EnvironmentCase",
    "FrontierGraphAgent",
    "HiddenMechanicEnvironment",
    "RandomAgent",
    "TACTICAgent",
    "TACTICNoPlannerAgent",
    "TACTICStrictAgent",
    "TACTICNoTransferAgent",
    "LocalQwenPrior",
    "build_benchmark_suite",
]
