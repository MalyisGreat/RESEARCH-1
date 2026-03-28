import numpy as np

from arc_tactic3.arc_agi3_harness_agent import TACTICHarnessAgent, _available_engine_actions

from arcagi3.utils.context import FrameState, GameProgress, SessionContext


class DummyGameClient:
    pass


def test_available_engine_actions_maps_numeric_action_ids() -> None:
    actions = _available_engine_actions(["1", 6, "7"])
    assert [action.name for action in actions] == ["ACTION1", "ACTION6", "ACTION7"]


def test_harness_agent_emits_valid_game_step(monkeypatch) -> None:
    monkeypatch.setattr(
        "arc_tactic3.arc_agi3_harness_agent._get_baseline_actions",
        lambda game_id: (6, 13, 31),
    )
    context = SessionContext(
        checkpoint_id="test-card",
        frames=FrameState(frame_grids=(np.zeros((64, 64), dtype=np.int8),)),
        game=GameProgress(
            game_id="ls20-016295f7601e",
            guid="guid-1",
            current_score=0,
            current_state="IN_PROGRESS",
            available_actions=("1", "6"),
        ),
    )
    agent = TACTICHarnessAgent(
        config="heuristic-local",
        game_client=DummyGameClient(),
        card_id="test-card",
    )
    step = agent.step(context)
    assert step.action["action"] in {"ACTION1", "ACTION6"}
