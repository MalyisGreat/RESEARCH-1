from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Callable

from .agents import (
    FrontierGraphAgent,
    LevelOutcome,
    RandomAgent,
    TACTICAgent,
    TACTICNoPlannerAgent,
    TACTICNoTransferAgent,
    TACTICStrictAgent,
)
from .dsl import EnvironmentCase, build_benchmark_suite, rename_button_labels, reorder_available_buttons
from .oracle import solve_with_oracle
from .progress import ProgressBar
from .protocol import BenchmarkProtocol, BenchmarkSplit, build_protocol
from .prior import LocalQwenPrior


LEVEL_WEIGHTS = (1, 2, 3, 4, 5)


@dataclass(frozen=True, slots=True)
class CaseReport:
    env_id: str
    weighted_score: float
    level_efficiencies: tuple[float, ...]
    oracle_steps: tuple[int, ...]
    transfer_gain: float
    mean_confidence: float
    mean_hypothesis_count: float


@dataclass(frozen=True, slots=True)
class AgentReport:
    name: str
    score: float
    per_level: tuple[float, ...]
    transfer_gain: float
    confidence_alignment_error: float
    robustness: dict[str, float]
    cases: tuple[CaseReport, ...]


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    reports: tuple[AgentReport, ...]

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        for report in self.reports:
            payload[report.name] = {
                "score": round(report.score, 3),
                "per_level": [round(value, 3) for value in report.per_level],
                "transfer_gain": round(report.transfer_gain, 3),
                "confidence_alignment_error": round(report.confidence_alignment_error, 3),
                "robustness": {key: round(value, 3) for key, value in sorted(report.robustness.items())},
            }
        return payload


@dataclass(frozen=True, slots=True)
class SplitReport:
    name: str
    score: float
    transfer_gain: float


@dataclass(frozen=True, slots=True)
class ProtocolAgentReport:
    name: str
    splits: tuple[SplitReport, ...]

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        for split in self.splits:
            payload[split.name] = {
                "score": round(split.score, 3),
                "transfer_gain": round(split.transfer_gain, 3),
            }
        return payload


@dataclass(frozen=True, slots=True)
class ProtocolBenchmarkReport:
    reports: tuple[ProtocolAgentReport, ...]

    def to_dict(self) -> dict[str, object]:
        return {report.name: report.to_dict() for report in self.reports}


def optimal_action_count(case: EnvironmentCase, level_index: int) -> int:
    oracle = solve_with_oracle(case, level_index, max_depth=80)
    if not oracle.solved:
        raise RuntimeError(f"no oracle solution for {case.env_id} level {level_index + 1}")
    return oracle.steps


def efficiency(outcome: LevelOutcome, optimal_actions: int) -> float:
    if not outcome.solved:
        return 0.0
    return optimal_actions / max(outcome.action_count, optimal_actions)


def evaluate_suite(
    agent,
    suite: tuple[EnvironmentCase, ...],
    *,
    progress_label: str | None = None,
) -> tuple[float, tuple[float, ...], tuple[CaseReport, ...]]:
    case_reports: list[CaseReport] = []
    per_level = [0.0 for _ in LEVEL_WEIGHTS]
    case_scores: list[float] = []
    progress = ProgressBar(progress_label, len(suite)) if progress_label else None

    for case_index, case in enumerate(suite, start=1):
        if progress is not None:
            progress.update(case_index - 1, detail=case.env_id)
        outcomes = agent.solve_case(case, step_limit=80)
        level_efficiencies: list[float] = []
        total_weight = float(sum(LEVEL_WEIGHTS))
        weighted_score = 0.0
        oracle_steps: list[int] = []
        confidence_values: list[float] = []
        hypothesis_counts: list[float] = []
        for level_index, outcome in enumerate(outcomes):
            optimal = optimal_action_count(case, level_index)
            value = efficiency(outcome, optimal)
            level_efficiencies.append(value)
            per_level[level_index] += value
            weighted_score += LEVEL_WEIGHTS[level_index] * value
            oracle_steps.append(optimal)
            confidence_values.append(outcome.top_confidence)
            hypothesis_counts.append(float(outcome.hypothesis_count))
        weighted_score /= total_weight
        case_scores.append(weighted_score)
        later = level_efficiencies[1:] if len(level_efficiencies) > 1 else level_efficiencies
        transfer_gain = (sum(later) / max(len(later), 1)) - level_efficiencies[0]
        case_reports.append(
            CaseReport(
                env_id=case.env_id,
                weighted_score=weighted_score,
                level_efficiencies=tuple(level_efficiencies),
                oracle_steps=tuple(oracle_steps),
                transfer_gain=transfer_gain,
                mean_confidence=sum(confidence_values) / max(len(confidence_values), 1),
                mean_hypothesis_count=sum(hypothesis_counts) / max(len(hypothesis_counts), 1),
            )
        )
        if progress is not None:
            progress.update(case_index, detail=case.env_id)

    count = len(suite)
    if progress is not None:
        progress.finish("suite complete")
    return sum(case_scores) / count, tuple(value / count for value in per_level), tuple(case_reports)


def confidence_alignment_error(cases: tuple[CaseReport, ...]) -> float:
    if not cases:
        return 0.0
    return sum(abs(case.mean_confidence - case.weighted_score) for case in cases) / len(cases)


def build_robustness_variants(suite: tuple[EnvironmentCase, ...]) -> dict[str, tuple[EnvironmentCase, ...]]:
    shuffled = tuple(
        reorder_available_buttons(case, tuple(reversed(case.config.available_buttons)))
        for case in suite
    )
    renamed = tuple(
        rename_button_labels(case, tuple(f"cmd_{index}" for index, _ in enumerate(case.config.available_buttons)))
        for case in suite
    )
    return {
        "button_order_reversed": shuffled,
        "button_labels_renamed": renamed,
    }


def evaluate_agent(agent_factory: Callable[[], object], *, show_progress: bool = False) -> AgentReport:
    suite = build_benchmark_suite()
    agent = agent_factory()
    score, per_level, cases = evaluate_suite(
        agent,
        suite,
        progress_label=f"{agent.name}:regression" if show_progress else None,
    )
    base_transfer_gain = sum(case.transfer_gain for case in cases) / max(len(cases), 1)
    robustness_scores: dict[str, float] = {}
    for name, variant_suite in build_robustness_variants(suite).items():
        variant_agent = agent_factory()
        variant_score, _, _ = evaluate_suite(
            variant_agent,
            variant_suite,
            progress_label=f"{agent.name}:{name}" if show_progress else None,
        )
        robustness_scores[name] = variant_score
    return AgentReport(
        name=agent.name,
        score=score,
        per_level=per_level,
        transfer_gain=base_transfer_gain,
        confidence_alignment_error=confidence_alignment_error(cases),
        robustness=robustness_scores,
        cases=cases,
    )


def evaluate_split(
    agent_factory: Callable[[], object],
    split: BenchmarkSplit,
    *,
    show_progress: bool = False,
) -> SplitReport:
    agent = agent_factory()
    score, _, cases = evaluate_suite(
        agent,
        split.cases,
        progress_label=f"{agent.name}:{split.name}" if show_progress else None,
    )
    transfer_gain = sum(case.transfer_gain for case in cases) / max(len(cases), 1)
    return SplitReport(name=split.name, score=score, transfer_gain=transfer_gain)


def evaluate_protocol_agent(
    agent_factory: Callable[[], object],
    protocol: BenchmarkProtocol | None = None,
    *,
    show_progress: bool = False,
) -> ProtocolAgentReport:
    active_protocol = protocol or build_protocol()
    agent = agent_factory()
    split_reports = []
    # Recreate the agent per split so split results are independent and cannot leak memory.
    for split in active_protocol.splits():
        split_reports.append(evaluate_split(agent_factory, split, show_progress=show_progress))
    return ProtocolAgentReport(name=agent.name, splits=tuple(split_reports))


def run_benchmark(*, show_progress: bool = False) -> BenchmarkReport:
    factories = default_agent_factories()
    return BenchmarkReport(reports=tuple(evaluate_agent(factory, show_progress=show_progress) for factory in factories))


def run_protocol_benchmark(
    protocol: BenchmarkProtocol | None = None,
    *,
    show_progress: bool = False,
) -> ProtocolBenchmarkReport:
    factories = default_agent_factories()
    return ProtocolBenchmarkReport(
        reports=tuple(evaluate_protocol_agent(factory, protocol, show_progress=show_progress) for factory in factories)
    )


def default_agent_factories(
    *,
    strict: bool = False,
    use_qwen_prior: bool = False,
    show_progress: bool = False,
) -> tuple[Callable[[], object], ...]:
    prior_factory = (lambda: LocalQwenPrior(show_progress=show_progress)) if use_qwen_prior else None

    def tactic_factory():
        if strict:
            return TACTICStrictAgent(proposal_prior=prior_factory() if prior_factory else None)
        return TACTICAgent(proposal_prior=prior_factory() if prior_factory else None)

    return (
        tactic_factory,
        TACTICNoTransferAgent,
        TACTICNoPlannerAgent,
        FrontierGraphAgent,
        lambda: RandomAgent(seed=7),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TACTIC toy benchmarks.")
    parser.add_argument(
        "--protocol",
        action="store_true",
        help="Run the generated split protocol instead of the small regression benchmark.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Run TACTIC in strict observation mode for the main evaluated agent.",
    )
    parser.add_argument(
        "--qwen-prior",
        action="store_true",
        help="Use the local Qwen GGUF as the proposal prior for the main TACTIC agent.",
    )
    parser.add_argument(
        "--replicas-per-case",
        type=int,
        default=2,
        help="Number of generated replicas per base case when --protocol is set.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable CLI progress bars and Qwen inference heartbeats.",
    )
    args = parser.parse_args()
    show_progress = not args.no_progress

    if args.protocol:
        protocol = build_protocol(replicas_per_case=args.replicas_per_case)
        if args.strict or args.qwen_prior:
            factories = default_agent_factories(
                strict=args.strict,
                use_qwen_prior=args.qwen_prior,
                show_progress=show_progress,
            )
            report = ProtocolBenchmarkReport(
                reports=tuple(
                    evaluate_protocol_agent(factory, protocol, show_progress=show_progress)
                    for factory in factories
                )
            ).to_dict()
        else:
            report = run_protocol_benchmark(protocol, show_progress=show_progress).to_dict()
    else:
        if args.strict or args.qwen_prior:
            factories = default_agent_factories(
                strict=args.strict,
                use_qwen_prior=args.qwen_prior,
                show_progress=show_progress,
            )
            report = BenchmarkReport(
                reports=tuple(evaluate_agent(factory, show_progress=show_progress) for factory in factories)
            ).to_dict()
        else:
            report = run_benchmark(show_progress=show_progress).to_dict()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
