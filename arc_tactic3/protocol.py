from __future__ import annotations

from dataclasses import dataclass

from .core import MechanicConfig
from .dsl import EnvironmentCase, build_benchmark_suite, build_hidden_mapping, rename_button_labels, reorder_available_buttons


@dataclass(frozen=True, slots=True)
class BenchmarkSplit:
    name: str
    cases: tuple[EnvironmentCase, ...]


@dataclass(frozen=True, slots=True)
class BenchmarkProtocol:
    train: BenchmarkSplit
    val: BenchmarkSplit
    test_iid: BenchmarkSplit
    test_transfer: BenchmarkSplit
    test_remapped: BenchmarkSplit
    test_ood: BenchmarkSplit

    def splits(self) -> tuple[BenchmarkSplit, ...]:
        return (
            self.train,
            self.val,
            self.test_iid,
            self.test_transfer,
            self.test_remapped,
            self.test_ood,
        )


def clone_case_with_seed(
    case: EnvironmentCase,
    *,
    env_suffix: str,
    seed: int,
    remap_each_level: bool = False,
) -> EnvironmentCase:
    def make_config(config_seed: int) -> MechanicConfig:
        return MechanicConfig(
            family=case.family,
            available_buttons=case.config.available_buttons,
            movement_map=build_hidden_mapping(config_seed),
            click_mode=case.config.click_mode,
            allows_undo=case.config.allows_undo,
        )

    config = make_config(seed)
    level_configs = None
    if remap_each_level:
        level_configs = tuple(make_config(seed + level_index * 17) for level_index in range(len(case.levels)))
        config = level_configs[0]

    return EnvironmentCase(
        env_id=f"{case.env_id}:{env_suffix}:{seed}",
        family=case.family,
        config=config,
        levels=case.levels,
        level_configs=level_configs,
    )


def build_protocol(*, replicas_per_case: int = 2) -> BenchmarkProtocol:
    base_suite = build_benchmark_suite()

    def make_split(name: str, base_seed: int, *, remap_each_level: bool = False) -> BenchmarkSplit:
        cases: list[EnvironmentCase] = []
        for replica_index in range(replicas_per_case):
            for case_index, case in enumerate(base_suite):
                seed = base_seed + replica_index * 101 + case_index * 7
                cases.append(
                    clone_case_with_seed(
                        case,
                        env_suffix=name,
                        seed=seed,
                        remap_each_level=remap_each_level,
                    )
                )
        return BenchmarkSplit(name=name, cases=tuple(cases))

    train = make_split("train", 100)
    val = make_split("val", 200)
    test_iid = make_split("test_iid", 300)
    test_transfer = make_split("test_transfer", 400)
    test_remapped = make_split("test_remapped", 500, remap_each_level=True)

    ood_base = make_split("test_ood", 600)
    ood_cases = tuple(
        rename_button_labels(
            reorder_available_buttons(case, tuple(reversed(case.config.available_buttons))),
            tuple(f"ood_{index}" for index, _ in enumerate(case.config.available_buttons)),
        )
        for case in ood_base.cases
    )
    test_ood = BenchmarkSplit(name="test_ood", cases=ood_cases)

    return BenchmarkProtocol(
        train=train,
        val=val,
        test_iid=test_iid,
        test_transfer=test_transfer,
        test_remapped=test_remapped,
        test_ood=test_ood,
    )


def protocol_manifest(protocol: BenchmarkProtocol) -> dict[str, list[str]]:
    return {
        split.name: [case.env_id for case in split.cases]
        for split in protocol.splits()
    }
