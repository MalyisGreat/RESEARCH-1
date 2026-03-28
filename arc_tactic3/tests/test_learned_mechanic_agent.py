from __future__ import annotations

from types import SimpleNamespace

from arc_tactic3.core import ClickAction, GameState
from arc_tactic3.hypotheses import MechanicPosterior
from arc_tactic3.learned_mechanic_agent import (
    AffordanceTransferSummary,
    LearnedMechanicAgent,
    LearnedProposalPrior,
    TransitionSummary,
    TrainConfig,
    derive_empirical_click_hint,
    generate_trace_samples,
    summarize_affordances_from_transitions,
    train_mechanic_model,
    update_empirical_click_votes,
)
from arc_tactic3.protocol import build_protocol


def test_generate_trace_samples_builds_shared_tokens() -> None:
    protocol = build_protocol(replicas_per_case=1)
    samples = generate_trace_samples(
        protocol.train.cases[:1],
        traces_per_level=1,
        rollout_steps=2,
        seed=123,
    )
    assert samples
    sample = samples[0]
    assert sample.family_index >= 0
    assert len(sample.direction_indices) == 5
    assert len(sample.effect_indices) == 5
    assert len(sample.action_mask) == 7
    assert "<bos>" in sample.tokens
    assert any(token.startswith("obj:player") for token in sample.tokens)
    assert any(token.startswith("txt:") for token in sample.tokens)


def test_train_mechanic_model_smoke_and_agent_runs() -> None:
    protocol = build_protocol(replicas_per_case=1)
    train_samples = generate_trace_samples(
        protocol.train.cases[:2],
        traces_per_level=1,
        rollout_steps=2,
        seed=5,
    )
    val_samples = generate_trace_samples(
        protocol.val.cases[:2],
        traces_per_level=1,
        rollout_steps=2,
        seed=6,
    )
    model, tokenizer, metrics = train_mechanic_model(
        train_samples,
        val_samples,
        config=TrainConfig(
            epochs=1,
            batch_size=8,
            embedding_dim=16,
            hidden_dim=32,
            train_traces_per_level=1,
            eval_traces_per_level=1,
            rollout_steps=2,
            num_layers=2,
            pooling="hybrid",
            teacher_rollin_probability=0.5,
            device="cpu",
        ),
    )
    assert "train_metrics" in metrics
    assert "val_metrics" in metrics
    assert "action_accuracy" in metrics["train_metrics"]
    assert "effect_accuracy" in metrics["val_metrics"]
    agent = LearnedMechanicAgent(
        model,
        tokenizer,
        device="cpu",
        explore_budget=2,
        confidence_threshold=0.5,
        use_effect_for_planning=True,
        use_empirical_controls=True,
        use_empirical_click_hint=True,
        use_empirical_greedy_actions=True,
    )
    outcomes = agent.solve_case(protocol.val.cases[0], step_limit=12)
    assert len(outcomes) == len(protocol.val.cases[0].levels)

    gated_agent = LearnedMechanicAgent(
        model,
        tokenizer,
        device="cpu",
        symbolic_plan_commit_steps=3,
        symbolic_plan_commit_confidence=0.5,
        symbolic_plan_commit_uncertainty_ceiling=0.6,
    )
    retained = gated_agent._retain_symbolic_plan(
        ("up", "left"),
        SimpleNamespace(available_buttons=("up",), top=(None, 0.9), uncertainty=0.2),
        SimpleNamespace(available_buttons=("up",)),
    )
    assert retained == ("up", "left")
    dropped = gated_agent._retain_symbolic_plan(
        ("up", "left"),
        SimpleNamespace(available_buttons=("up",), top=(None, 0.4), uncertainty=0.9),
        SimpleNamespace(available_buttons=("up",)),
    )
    assert dropped == ()
    transfer_agent = LearnedMechanicAgent(
        model,
        tokenizer,
        device="cpu",
        symbolic_transfer=True,
        symbolic_transfer_confidence_floor=0.7,
        symbolic_transfer_uncertainty_ceiling=0.4,
        symbolic_transfer_requires_solved=True,
    )
    assert transfer_agent._should_store_symbolic_prior(
        SimpleNamespace(uncertainty=0.2),
        confidence=0.9,
        solved=True,
    )
    assert not transfer_agent._should_store_symbolic_prior(
        SimpleNamespace(uncertainty=0.5),
        confidence=0.9,
        solved=True,
    )
    assert not transfer_agent._should_store_symbolic_prior(
        SimpleNamespace(uncertainty=0.2),
        confidence=0.6,
        solved=True,
    )
    assert not transfer_agent._should_store_symbolic_prior(
        SimpleNamespace(uncertainty=0.2),
        confidence=0.9,
        solved=False,
    )


def test_symbolic_posterior_mode_runs() -> None:
    protocol = build_protocol(replicas_per_case=1)
    train_samples = generate_trace_samples(
        protocol.train.cases[:2],
        traces_per_level=1,
        rollout_steps=2,
        seed=15,
    )
    val_samples = generate_trace_samples(
        protocol.val.cases[:2],
        traces_per_level=1,
        rollout_steps=2,
        seed=16,
    )
    model, tokenizer, _metrics = train_mechanic_model(
        train_samples,
        val_samples,
        config=TrainConfig(
            epochs=1,
            batch_size=8,
            embedding_dim=16,
            hidden_dim=32,
            train_traces_per_level=1,
            eval_traces_per_level=1,
            rollout_steps=2,
            seed=101,
            device="cpu",
        ),
    )
    agent = LearnedMechanicAgent(
        model,
        tokenizer,
        device="cpu",
        explore_budget=2,
        confidence_threshold=0.5,
        use_symbolic_posterior=True,
        symbolic_transfer=True,
        symbolic_direction_prior=False,
        symbolic_reprioritize=True,
        symbolic_reprioritize_uncertainty=0.0,
        symbolic_prior_power=1.5,
        symbolic_plan_uncertainty_ceiling=10.0,
        symbolic_plan_commit_steps=3,
        symbolic_plan_commit_confidence=0.5,
        symbolic_plan_commit_uncertainty_ceiling=0.6,
    )
    outcomes = agent.solve_case(protocol.val.cases[0], step_limit=12)
    assert len(outcomes) == len(protocol.val.cases[0].levels)


def test_learned_proposal_prior_applies_component_weights() -> None:
    class _FakeAgent:
        def predict_tokens(self, _tokens):
            return SimpleNamespace(
                family_probs=[0.8, 0.2, 0.1],
                click_probs=[0.6, 0.4],
                direction_probs=[[0.7, 0.3]],
            )

    prior = LearnedProposalPrior(
        _FakeAgent(),
        floor=0.0,
        power=1.0,
        family_weight=1.25,
        click_weight=0.2,
        direction_weight=0.5,
    )
    state = SimpleNamespace(
        height=5,
        width=5,
        player=(1, 1),
        goals=frozenset(),
        keys=frozenset(),
        doors=frozenset(),
        boxes=frozenset(),
        targets=frozenset(),
        switches=frozenset(),
        portals=frozenset(),
        has_key=False,
        switch_active=False,
    )
    prediction = prior.predict(
        state,
        available_buttons=("up",),
        allows_click=True,
        allows_undo=True,
    )
    assert prediction.family_scores
    assert max(prediction.family_scores.values()) > 0.8
    assert prediction.click_mode_scores
    assert max(prediction.click_mode_scores.values()) < 0.2
    assert max(prediction.button_direction_scores["up"].values()) < 0.5


def test_symbolic_transfer_modes_can_drop_direction_detail() -> None:
    state = GameState(
        width=5,
        height=5,
        player=(1, 1),
        walls=frozenset(),
        goals=frozenset({(4, 4)}),
        keys=frozenset(),
        doors=frozenset(),
        boxes=frozenset(),
        targets=frozenset(),
        switches=frozenset({(2, 2)}),
        portals=(),
        has_key=False,
        switch_active=False,
    )
    posterior = MechanicPosterior(
        ("up", "down", "left", "right", "stay"),
        state,
        allows_click=True,
        allows_undo=True,
    )
    hypotheses = list(posterior.weights)
    same_group = [h for h in hypotheses if h.family == "switch_goal" and h.click_mode is None][:2]
    click_group = [h for h in hypotheses if h.family == "switch_goal" and h.click_mode == "switch"][:1]
    assert len(same_group) == 2
    assert len(click_group) == 1
    posterior.weights = {
        same_group[0]: 0.6,
        same_group[1]: 0.2,
        click_group[0]: 0.2,
    }
    posterior.normalize()

    full_prior = posterior.transfer_prior(state, mode="full")
    family_click_prior = posterior.transfer_prior(state, mode="family_click")
    family_only_prior = posterior.transfer_prior(state, mode="family_only")
    family_click_summary = posterior.transfer_summary(state, mode="family_click")
    family_only_summary = posterior.transfer_summary(state, mode="family_only")

    assert full_prior[same_group[0]] > full_prior[same_group[1]]
    assert family_click_prior[same_group[0]] == family_click_prior[same_group[1]]
    assert family_only_prior[same_group[0]] == family_only_prior[same_group[1]]
    assert family_only_prior[same_group[0]] == family_only_prior[click_group[0]]
    assert family_click_summary.button_direction_scores == {}
    assert family_click_summary.click_mode_scores
    assert family_only_summary.click_mode_scores == {}
    assert family_click_summary.family_scores["switch_goal"] > 0


def test_affordance_summary_uses_state_changes_without_direction_transfer() -> None:
    summary = summarize_affordances_from_transitions(
        [
            TransitionSummary(
                action_kind="click",
                action_slot=5,
                clicked_kind="switch",
                player_move=(0, 0),
                keys_delta=0,
                doors_delta=0,
                boxes_delta=0,
                solved_delta=0,
                frame_changed=True,
                has_key_changed=False,
                switch_changed=True,
                boxes_moved=False,
            ),
            TransitionSummary(
                action_kind="move",
                action_slot=0,
                clicked_kind=None,
                player_move=(0, 1),
                keys_delta=-1,
                doors_delta=0,
                boxes_delta=0,
                solved_delta=0,
                frame_changed=True,
                has_key_changed=True,
                switch_changed=False,
                boxes_moved=False,
            ),
        ]
    )
    prior = summary.to_prior_prediction()
    assert prior.button_direction_scores == {}
    assert summary.family_scores["switch_goal"] > 0
    assert summary.family_scores["key_goal"] > 0
    assert summary.click_mode_scores["switch"] > 0
    assert summary.clicked_kind_scores["switch"] > 0
    assert summary.state_change_scores["switch_change"] > 0
    assert summary.state_change_scores["key_progress"] > 0
    click_only = summary.to_prior_prediction(mode="click_only")
    family_only = summary.to_prior_prediction(mode="family_only")
    none_prior = summary.to_prior_prediction(mode="none")
    assert click_only.family_scores == {}
    assert click_only.click_mode_scores
    assert family_only.click_mode_scores == {}
    assert family_only.family_scores
    assert none_prior.family_scores == {}
    assert none_prior.click_mode_scores == {}


def test_affordance_bonus_prefers_useful_click_kind() -> None:
    protocol = build_protocol(replicas_per_case=1)
    train_samples = generate_trace_samples(
        protocol.train.cases[:1],
        traces_per_level=1,
        rollout_steps=1,
        seed=21,
    )
    val_samples = generate_trace_samples(
        protocol.val.cases[:1],
        traces_per_level=1,
        rollout_steps=1,
        seed=22,
    )
    model, tokenizer, _metrics = train_mechanic_model(
        train_samples,
        val_samples,
        config=TrainConfig(
            epochs=1,
            batch_size=4,
            embedding_dim=8,
            hidden_dim=16,
            train_traces_per_level=1,
            eval_traces_per_level=1,
            rollout_steps=1,
            seed=123,
            device="cpu",
        ),
    )
    agent = LearnedMechanicAgent(
        model,
        tokenizer,
        device="cpu",
        use_symbolic_posterior=True,
        symbolic_affordance_transfer=True,
        symbolic_affordance_bonus_weight=1.0,
    )
    parsed = SimpleNamespace(
        objects=(
            SimpleNamespace(kind="switch", anchor=(2, 2)),
            SimpleNamespace(kind="portal", anchor=(3, 3)),
        )
    )
    state = SimpleNamespace(keys=frozenset(), switches=frozenset({(2, 2)}), boxes=frozenset())
    summary = AffordanceTransferSummary(
        family_scores={"switch_goal": 1.0},
        click_mode_scores={"switch": 1.0},
        action_kind_scores={"click": 1.0},
        clicked_kind_scores={"switch": 1.0},
        state_change_scores={"switch_change": 1.0},
    )
    switch_bonus = agent._affordance_action_bonus(  # noqa: SLF001
        ClickAction(2, 2),
        parsed,
        state,
        summary,
    )
    portal_bonus = agent._affordance_action_bonus(  # noqa: SLF001
        ClickAction(3, 3),
        parsed,
        state,
        summary,
    )
    move_bonus = agent._affordance_action_bonus(  # noqa: SLF001
        "up",
        parsed,
        state,
        summary,
    )
    assert switch_bonus > portal_bonus
    assert switch_bonus > move_bonus


def test_train_mechanic_model_is_reproducible_for_same_seed() -> None:
    protocol = build_protocol(replicas_per_case=1)
    train_samples = generate_trace_samples(
        protocol.train.cases[:1],
        traces_per_level=1,
        rollout_steps=2,
        seed=11,
    )
    val_samples = generate_trace_samples(
        protocol.val.cases[:1],
        traces_per_level=1,
        rollout_steps=2,
        seed=12,
    )
    config = TrainConfig(
        epochs=1,
        batch_size=8,
        embedding_dim=16,
        hidden_dim=32,
        train_traces_per_level=1,
        eval_traces_per_level=1,
        rollout_steps=2,
        seed=99,
        device="cpu",
    )
    _model_a, _tokenizer_a, metrics_a = train_mechanic_model(train_samples, val_samples, config=config)
    _model_b, _tokenizer_b, metrics_b = train_mechanic_model(train_samples, val_samples, config=config)
    assert metrics_a == metrics_b


def test_empirical_click_votes_require_useful_effect_and_min_votes() -> None:
    votes: dict[str, int] = {}
    update_empirical_click_votes(votes, clicked_kind="switch", frame_changed=False, player_move=(0, 0), solved_delta=0)
    assert votes == {}

    update_empirical_click_votes(votes, clicked_kind="switch", frame_changed=True, player_move=(0, 0), solved_delta=0)
    assert derive_empirical_click_hint(votes, min_votes=1) == "switch"
    assert derive_empirical_click_hint(votes, min_votes=2) is None

    update_empirical_click_votes(votes, clicked_kind="switch", frame_changed=False, player_move=(0, 1), solved_delta=0)
    assert derive_empirical_click_hint(votes, min_votes=2) == "switch"
