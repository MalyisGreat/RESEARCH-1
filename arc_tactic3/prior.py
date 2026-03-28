from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Callable, Protocol

from .core import Coord, Family, GameState
from .progress import Spinner


DIRECTION_NAMES = {
    (-1, 0): "up",
    (1, 0): "down",
    (0, -1): "left",
    (0, 1): "right",
    None: "none",
}
NAME_TO_DIRECTION = {name: direction for direction, name in DIRECTION_NAMES.items()}
FAMILY_ORDER: tuple[Family, ...] = (
    "reach_goal",
    "key_goal",
    "switch_goal",
    "push_box",
    "portal_goal",
)
CLICK_ORDER: tuple[str | None, ...] = (None, "switch", "teleport")


@dataclass(frozen=True, slots=True)
class PriorPrediction:
    family_scores: dict[Family, float] = field(default_factory=dict)
    click_mode_scores: dict[str | None, float] = field(default_factory=dict)
    button_direction_scores: dict[str, dict[Coord | None, float]] = field(default_factory=dict)


class MechanicProposalPrior(Protocol):
    def predict(
        self,
        state: GameState,
        *,
        available_buttons: tuple[str, ...],
        allows_click: bool,
        allows_undo: bool,
    ) -> PriorPrediction: ...


def summarize_state_for_model(
    state: GameState,
    *,
    available_buttons: tuple[str, ...],
    allows_click: bool,
    allows_undo: bool,
) -> str:
    def fmt(coords: frozenset[Coord] | tuple[Coord, ...]) -> str:
        values = sorted(coords)
        return ", ".join(f"({row},{col})" for row, col in values) if values else "none"

    return "\n".join(
        [
            f"grid: {state.height}x{state.width}",
            f"buttons: {', '.join(available_buttons)}",
            f"click_enabled: {allows_click}",
            f"undo_enabled: {allows_undo}",
            f"player: {state.player}",
            f"goals: {fmt(state.goals)}",
            f"keys: {fmt(state.keys)}",
            f"doors: {fmt(state.doors)}",
            f"boxes: {fmt(state.boxes)}",
            f"targets: {fmt(state.targets)}",
            f"switches: {fmt(state.switches)}",
            f"portals: {fmt(state.portals)}",
            f"has_key: {state.has_key}",
            f"switch_active: {state.switch_active}",
        ]
    )


def build_prior_prompt(
    state: GameState,
    *,
    available_buttons: tuple[str, ...],
    allows_click: bool,
    allows_undo: bool,
) -> str:
    summary = summarize_state_for_model(
        state,
        available_buttons=available_buttons,
        allows_click=allows_click,
        allows_undo=allows_undo,
    )
    buttons_block = ", ".join(available_buttons)
    return f"""You are proposing priors for an interactive grid puzzle.
Return JSON only with this schema:
{{
  "family_scores": {{"reach_goal": float, "key_goal": float, "switch_goal": float, "push_box": float, "portal_goal": float}},
  "click_mode_scores": {{"none": float, "switch": float, "teleport": float}},
  "button_direction_scores": {{
    "{available_buttons[0]}": {{"up": float, "down": float, "left": float, "right": float, "none": float}}
  }}
}}
Use non-negative scores. Unknowns should be low but non-zero.
Buttons available: {buttons_block}
State summary:
{summary}
"""


def build_compact_prior_prompt(
    state: GameState,
    *,
    available_buttons: tuple[str, ...],
    allows_click: bool,
    allows_undo: bool,
) -> str:
    summary = summarize_state_for_model(
        state,
        available_buttons=available_buttons,
        allows_click=allows_click,
        allows_undo=allows_undo,
    )
    direction_schema = ", ".join(f'"{button}":"up|down|left|right|none"' for button in available_buttons)
    return f"""Predict the hidden mechanic from one game frame.
/no_think
Return exactly one line of minified JSON.
The first character must be {{ and the last must be }}.
Do not use markdown, code fences, or explanations.
Use this exact schema and key order:
{{"family":"reach_goal|key_goal|switch_goal|push_box|portal_goal","click":"none|switch|teleport","dirs":{{{direction_schema}}}}}
Choose one valid value for every field.
State:
{summary}
JSON:
"""


def extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char == "{":
                depth += 1
                continue
            if char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
        start = text.find("{", start + 1)
    return None


def normalize_label(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().strip("`").strip().strip('"').strip().lower()
    return normalized or None


def build_compact_prior_grammar(available_buttons: tuple[str, ...]) -> str:
    def literal(text: str) -> str:
        return json.dumps(text)

    family_rule = " | ".join(
        literal(name)
        for name in ("reach_goal", "key_goal", "switch_goal", "push_box", "portal_goal")
    )
    click_rule = " | ".join(literal(name) for name in ("none", "switch", "teleport"))
    direction_rule = " | ".join(literal(name) for name in ("up", "down", "left", "right", "none"))
    dir_fields: list[str] = []
    for index, button in enumerate(available_buttons):
        if index:
            dir_fields.append('ws "," ws')
        button_literal = literal(f'"{button}"')
        dir_fields.append(f'{button_literal} ws ":" ws direction')
    dir_body = " ".join(dir_fields)
    family_key = literal('"family"')
    click_key = literal('"click"')
    dirs_key = literal('"dirs"')
    return (
        f'root ::= ws "{{" ws {family_key} ws ":" ws family ws "," ws '
        f'{click_key} ws ":" ws click ws "," ws {dirs_key} ws ":" ws '
        f'"{{" ws {dir_body} ws "}}" ws "}}" ws\n'
        f"family ::= {family_rule}\n"
        f"click ::= {click_rule}\n"
        f"direction ::= {direction_rule}\n"
        'ws ::= [ \\t\\n\\r]*\n'
    )


def build_coded_prior_prompt(
    state: GameState,
    *,
    available_buttons: tuple[str, ...],
    allows_click: bool,
    allows_undo: bool,
) -> str:
    summary = summarize_state_for_model(
        state,
        available_buttons=available_buttons,
        allows_click=allows_click,
        allows_undo=allows_undo,
    )
    return f"""Predict the hidden mechanic from one game frame.
/no_think
Return only integer codes separated by single spaces.
Format:
<family_code> <click_code>
Family codes: 0=reach_goal 1=key_goal 2=switch_goal 3=push_box 4=portal_goal
Click codes: 0=none 1=switch 2=teleport
Choose the most plausible family and click behavior only.
Do not guess the button-to-direction mapping from labels.
State:
{summary}
Codes:
"""


def build_coded_prior_grammar(available_buttons: tuple[str, ...]) -> str:
    _ = available_buttons
    return (
        'root ::= family " " click\n'
        'family ::= "0" | "1" | "2" | "3" | "4"\n'
        'click ::= "0" | "1" | "2"\n'
    )


def parse_prior_json(text: str, *, available_buttons: tuple[str, ...]) -> PriorPrediction:
    payload_text = extract_first_json_object(text)
    if payload_text is None:
        return PriorPrediction()
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        return PriorPrediction()

    family_scores: dict[Family, float] = {}
    for family in ("reach_goal", "key_goal", "switch_goal", "push_box", "portal_goal"):
        value = payload.get("family_scores", {}).get(family)
        if isinstance(value, (int, float)) and value >= 0:
            family_scores[family] = float(value)

    click_mode_scores: dict[str | None, float] = {}
    for key, value in payload.get("click_mode_scores", {}).items():
        normalized_key = None if key == "none" else key
        if normalized_key in {None, "switch", "teleport"} and isinstance(value, (int, float)) and value >= 0:
            click_mode_scores[normalized_key] = float(value)

    button_direction_scores: dict[str, dict[Coord | None, float]] = {}
    raw_button_scores = payload.get("button_direction_scores", {})
    for button in available_buttons:
        per_button: dict[Coord | None, float] = {}
        if isinstance(raw_button_scores, dict):
            for direction_name, value in raw_button_scores.get(button, {}).items():
                direction = NAME_TO_DIRECTION.get(direction_name)
                if direction_name in NAME_TO_DIRECTION and isinstance(value, (int, float)) and value >= 0:
                    per_button[direction] = float(value)
        if per_button:
            button_direction_scores[button] = per_button

    return PriorPrediction(
        family_scores=family_scores,
        click_mode_scores=click_mode_scores,
        button_direction_scores=button_direction_scores,
    )


def parse_compact_prior_json(text: str, *, available_buttons: tuple[str, ...]) -> PriorPrediction:
    payload_text = extract_first_json_object(text)
    if payload_text is None:
        return PriorPrediction()
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        return PriorPrediction()

    family = normalize_label(payload.get("family"))
    click = normalize_label(payload.get("click")) or "none"
    family_scores: dict[Family, float] = {}
    if family in {"reach_goal", "key_goal", "switch_goal", "push_box", "portal_goal"}:
        family_scores[family] = 4.0

    click_mode_scores: dict[str | None, float] = {}
    normalized_click = None if click == "none" else click
    if normalized_click in {None, "switch", "teleport"}:
        click_mode_scores[normalized_click] = 4.0

    button_direction_scores: dict[str, dict[Coord | None, float]] = {}
    raw_dirs = payload.get("dirs")
    if not isinstance(raw_dirs, dict):
        raw_dirs = payload.get("directions")
    if not isinstance(raw_dirs, dict):
        raw_dirs = payload.get("button_directions")
    if isinstance(raw_dirs, dict):
        for button in available_buttons:
            direction_name = normalize_label(raw_dirs.get(button))
            direction = NAME_TO_DIRECTION.get(direction_name)
            if direction_name in NAME_TO_DIRECTION:
                button_direction_scores[button] = {direction: 4.0}

    return PriorPrediction(
        family_scores=family_scores,
        click_mode_scores=click_mode_scores,
        button_direction_scores=button_direction_scores,
    )


def parse_coded_prior(text: str, *, available_buttons: tuple[str, ...]) -> PriorPrediction:
    tokens = text.strip().split()
    _ = available_buttons
    expected = 2
    if len(tokens) < expected:
        return PriorPrediction()
    try:
        values = [int(token) for token in tokens[:expected]]
    except ValueError:
        return PriorPrediction()

    family_scores: dict[Family, float] = {}
    family_index = values[0]
    if 0 <= family_index < len(FAMILY_ORDER):
        family_scores[FAMILY_ORDER[family_index]] = 4.0

    click_mode_scores: dict[str | None, float] = {}
    click_index = values[1]
    if 0 <= click_index < len(CLICK_ORDER):
        click_mode_scores[CLICK_ORDER[click_index]] = 4.0

    return PriorPrediction(
        family_scores=family_scores,
        click_mode_scores=click_mode_scores,
        button_direction_scores={},
    )


class CompletionModelPrior:
    def __init__(self, complete: Callable[[str], str]) -> None:
        self._complete = complete

    def predict(
        self,
        state: GameState,
        *,
        available_buttons: tuple[str, ...],
        allows_click: bool,
        allows_undo: bool,
    ) -> PriorPrediction:
        prompt = build_prior_prompt(
            state,
            available_buttons=available_buttons,
            allows_click=allows_click,
            allows_undo=allows_undo,
        )
        return parse_prior_json(self._complete(prompt), available_buttons=available_buttons)


class StaticPrior:
    def __init__(self, prediction: PriorPrediction) -> None:
        self.prediction = prediction

    def predict(
        self,
        state: GameState,
        *,
        available_buttons: tuple[str, ...],
        allows_click: bool,
        allows_undo: bool,
    ) -> PriorPrediction:
        return self.prediction


class LlamaCppCompletion:
    def __init__(
        self,
        model_path: str,
        *,
        n_ctx: int = 1024,
        max_tokens: int = 256,
        temperature: float = 0.0,
        n_gpu_layers: int = -1,
        flash_attn: bool = True,
        show_progress: bool = False,
        verbose: bool = False,
    ) -> None:
        self.model_path = str(Path(model_path).expanduser().resolve())
        self.n_ctx = n_ctx
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.n_gpu_layers = n_gpu_layers
        self.flash_attn = flash_attn
        self.show_progress = show_progress
        self.verbose = verbose
        self._model = None
        self._lock = Lock()

    def __call__(self, prompt: str) -> str:
        return self.complete(prompt)

    def complete(self, prompt: str, *, stop: list[str] | None = None, grammar_text: str | None = None) -> str:
        spinner = Spinner("qwen-prior") if self.show_progress else None
        if spinner is not None:
            spinner.tick("loading model")
        model = self._ensure_model()
        grammar = None
        if grammar_text:
            from llama_cpp import LlamaGrammar

            grammar = LlamaGrammar.from_string(grammar_text)
        active_stop = stop or ["\n", "\r", "</think>", "```", "<|im_end|>"]
        try:
            if spinner is not None:
                spinner.tick("streaming tokens")
            chunks = model(
                prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=1.0,
                top_k=1,
                repeat_penalty=1.0,
                seed=7,
                stop=active_stop,
                grammar=grammar,
                stream=True,
            )
            parts: list[str] = []
            tokens = 0
            for chunk in chunks:
                text = chunk["choices"][0].get("text", "")
                if text:
                    parts.append(text)
                    tokens += 1
                    if spinner is not None and (tokens == 1 or tokens % 8 == 0):
                        spinner.tick(f"tokens={tokens}")
            text = "".join(parts)
            if spinner is not None:
                spinner.finish(f"tokens={tokens}")
            return text
        except Exception:
            if spinner is not None:
                spinner.tick("fallback completion")
            chunks = model(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=1.0,
                top_k=1,
                repeat_penalty=1.0,
                seed=7,
                stop=active_stop,
                grammar=None,
                stream=True,
            )
            parts = []
            tokens = 0
            for chunk in chunks:
                text = chunk["choices"][0].get("text", "")
                if text:
                    parts.append(text)
                    tokens += 1
                    if spinner is not None and (tokens == 1 or tokens % 8 == 0):
                        spinner.tick(f"tokens={tokens}")
            text = "".join(parts)
            if spinner is not None:
                spinner.finish(f"tokens={tokens}")
            return text

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            from llama_cpp import Llama

            threads = max(1, (os.cpu_count() or 4) - 1)
            self._model = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=threads,
                n_threads_batch=threads,
                n_gpu_layers=self.n_gpu_layers,
                flash_attn=self.flash_attn,
                offload_kqv=True,
                chat_format="chatml",
                verbose=self.verbose,
            )
            return self._model


def default_local_qwen_model_path() -> str:
    candidates = (
        Path("models/qwen3-0.6b-instruct-q4_k_m/Qwen3-0.6B-Q4_K_M.gguf"),
        Path("agentic_health_triage/models/Qwen3-0.6B-Q4_K_M.gguf"),
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    raise FileNotFoundError("No local Qwen GGUF model found in expected locations.")


class LocalQwenPrior(CompletionModelPrior):
    def __init__(self, model_path: str | None = None, *, show_progress: bool = False) -> None:
        self._complete = LlamaCppCompletion(
            model_path or default_local_qwen_model_path(),
            max_tokens=24,
            show_progress=show_progress,
        )

    def predict(
        self,
        state: GameState,
        *,
        available_buttons: tuple[str, ...],
        allows_click: bool,
        allows_undo: bool,
    ) -> PriorPrediction:
        prompt = build_coded_prior_prompt(
            state,
            available_buttons=available_buttons,
            allows_click=allows_click,
            allows_undo=allows_undo,
        )
        grammar = build_coded_prior_grammar(available_buttons)
        response = self._complete.complete(prompt, grammar_text=grammar)
        return parse_coded_prior(response, available_buttons=available_buttons)
