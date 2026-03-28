from types import SimpleNamespace

from arc_agi import OperationMode

from arc_tactic3 import arc_agi3_public


class FakeEnv:
    def __init__(self, game_id: str) -> None:
        self.game_id = game_id
        self.action_space = []


class FakeArcade:
    last_operation_mode = None

    def __init__(self, operation_mode=OperationMode.NORMAL, **_kwargs) -> None:
        FakeArcade.last_operation_mode = operation_mode
        self._environments = [
            SimpleNamespace(game_id="env-a", baseline_actions=(4,)),
            SimpleNamespace(game_id="env-b", baseline_actions=(5,)),
        ]

    def get_environments(self):
        return self._environments

    def make(self, game_id: str):
        return FakeEnv(game_id)

    def close_scorecard(self):
        return {
            "score": 0.5,
            "total_levels_completed": 2,
            "total_levels": 4,
            "total_actions": 10,
            "environments": [],
        }


class DuplicateArcade(FakeArcade):
    def __init__(self, operation_mode=OperationMode.NORMAL, **kwargs) -> None:
        super().__init__(operation_mode=operation_mode, **kwargs)
        self._environments = [
            SimpleNamespace(game_id="env-a", baseline_actions=(4,)),
            SimpleNamespace(game_id="env-a", baseline_actions=(5,)),
        ]


def test_run_public_benchmark_uses_competition_mode(monkeypatch) -> None:
    monkeypatch.setattr(arc_agi3_public.arc_agi, "Arcade", FakeArcade)

    played: list[str] = []

    def fake_play(self, env, **kwargs):
        played.append(kwargs["env_id"])

    monkeypatch.setattr(arc_agi3_public.ArcAgi3TACTICPublicAgent, "play_environment", fake_play)

    result = arc_agi3_public.run_public_benchmark(
        show_progress=False,
        competition_mode=True,
    )

    assert FakeArcade.last_operation_mode == OperationMode.COMPETITION
    assert played == ["env-a", "env-b"]
    assert result.score == 0.5


def test_run_public_benchmark_rejects_duplicate_environment_ids(monkeypatch) -> None:
    monkeypatch.setattr(arc_agi3_public.arc_agi, "Arcade", DuplicateArcade)
    monkeypatch.setattr(
        arc_agi3_public.ArcAgi3TACTICPublicAgent,
        "play_environment",
        lambda self, env, **kwargs: None,
    )

    try:
        arc_agi3_public.run_public_benchmark(show_progress=False)
    except RuntimeError as exc:
        assert "Duplicate environment" in str(exc)
    else:
        raise AssertionError("Expected duplicate-environment protection to raise")


def test_run_public_benchmark_rejects_limit_in_competition_mode(monkeypatch) -> None:
    monkeypatch.setattr(arc_agi3_public.arc_agi, "Arcade", FakeArcade)

    try:
        arc_agi3_public.run_public_benchmark(limit=1, show_progress=False, competition_mode=True)
    except ValueError as exc:
        assert "limit" in str(exc)
    else:
        raise AssertionError("Expected competition-mode limit protection to raise")
