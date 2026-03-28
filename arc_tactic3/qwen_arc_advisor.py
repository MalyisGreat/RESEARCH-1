from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class OllamaArcAdvisor:
    model: str = "qwen3.5:0.8b"
    endpoint: str = "http://127.0.0.1:11434/api/generate"
    timeout_s: float = 20.0
    num_predict: int = 256

    def _complete(self, prompt: str) -> dict | None:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": self.num_predict,
            },
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
            return None

    @staticmethod
    def _payload_text(payload: dict | None) -> str:
        if payload is None:
            return ""
        parts: list[str] = []
        for field in ("response", "thinking"):
            value = payload.get(field, "")
            if isinstance(value, str) and value:
                parts.append(value)
        return "\n".join(parts)

    def choose_option(self, prompt: str, allowed_options: tuple[str, ...]) -> str | None:
        payload = self._complete(prompt)
        if payload is None:
            return None
        patterns = {
            option: re.compile(rf"\b{re.escape(option)}\b", re.IGNORECASE)
            for option in allowed_options
        }
        text = self._payload_text(payload)
        for option, pattern in patterns.items():
            if pattern.search(text):
                return option
        return None

    def choose_action(self, prompt: str, allowed_actions: tuple[str, ...]) -> str | None:
        return self.choose_option(prompt, allowed_actions)

    def summarize_mechanic(
        self,
        prompt: str,
        *,
        allowed_modes: tuple[str, ...],
        allowed_goals: tuple[str, ...],
        allowed_focuses: tuple[str, ...],
    ) -> dict[str, Any] | None:
        payload = self._complete(prompt)
        if payload is None:
            return None
        text = self._payload_text(payload)
        if not text:
            return None

        parsed = self._parse_structured_summary(
            text,
            allowed_modes=allowed_modes,
            allowed_goals=allowed_goals,
            allowed_focuses=allowed_focuses,
        )
        if parsed is not None:
            return parsed

        upper = text.upper()
        mode = next((option for option in allowed_modes if re.search(rf"\b{re.escape(option)}\b", upper)), None)
        goal = next((option for option in allowed_goals if re.search(rf"\b{re.escape(option)}\b", upper)), None)
        focus = next((option for option in allowed_focuses if re.search(rf"\b{re.escape(option)}\b", upper)), None)
        confidence_match = re.search(r"\b(?:CONFIDENCE|CONF)\s*[:=]\s*([01](?:\.\d+)?)", upper)
        confidence = None
        if confidence_match is not None:
            try:
                confidence = float(confidence_match.group(1))
            except ValueError:
                confidence = None
        if mode is None and goal is None and focus is None and confidence is None:
            return None
        return {
            "mode": mode or "UNKNOWN",
            "goal": goal or "UNKNOWN",
            "focus": focus or "UNKNOWN",
            "confidence": max(0.0, min(confidence if confidence is not None else 0.35, 1.0)),
            "raw_text": text,
        }

    def _parse_structured_summary(
        self,
        text: str,
        *,
        allowed_modes: tuple[str, ...],
        allowed_goals: tuple[str, ...],
        allowed_focuses: tuple[str, ...],
    ) -> dict[str, Any] | None:
        block_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if block_match is None:
            return None
        try:
            parsed = json.loads(block_match.group(0))
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        mode = str(parsed.get("mode", "UNKNOWN")).upper()
        goal = str(parsed.get("goal", "UNKNOWN")).upper()
        focus = str(parsed.get("focus", "UNKNOWN")).upper()
        if mode not in allowed_modes:
            mode = "UNKNOWN"
        if goal not in allowed_goals:
            goal = "UNKNOWN"
        if focus not in allowed_focuses:
            focus = "UNKNOWN"
        confidence_value = parsed.get("confidence", 0.35)
        try:
            confidence = float(confidence_value)
        except (TypeError, ValueError):
            confidence = 0.35
        return {
            "mode": mode,
            "goal": goal,
            "focus": focus,
            "confidence": max(0.0, min(confidence, 1.0)),
            "raw_text": text,
        }
