"""Strict validation for LLM output.

THE RULE THIS MODULE ENFORCES
-----------------------------
A language model may do exactly two things to a signal:

  1. attach a written explanation, and
  2. veto it -- turn LONG or SHORT into NO_TRADE.

It may NOT change direction (LONG <-> SHORT), it may NOT turn NO_TRADE into a
trade, and it may NOT touch position size, stop loss or take profit. Those
come from ``app/strategies`` and ``app/risk`` and are pure arithmetic.

The veto is deliberately ONE-WAY. A model that can only remove trades can, at
worst, make the system trade less. A model that could add or flip trades
would be a decision-maker, and calibrating a language model's judgement
against real money is exactly what this project is built not to do.

Validation is intentionally paranoid: unknown fields are rejected, not
ignored. An unexpected key means the model is answering a different question
than the one we asked, and silently dropping it would hide that.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Bound the free-text field so a runaway generation cannot bloat the database.
MAX_REASONING_CHARS = 1200
MAX_RISK_FLAGS = 8
MAX_RISK_FLAG_CHARS = 120

ALLOWED_KEYS = {"action", "reasoning", "risk_flags", "confidence"}
REQUIRED_KEYS = {"action", "reasoning"}


class LLMAction(str, Enum):
    """The only two verdicts a model is allowed to return."""

    CONFIRM = "CONFIRM"  # keep the rule-based signal exactly as it is
    VETO = "VETO"        # downgrade the rule-based signal to NO_TRADE


class ValidationError(ValueError):
    """Raised when a model response does not satisfy the contract."""


@dataclass(frozen=True)
class LLMVerdict:
    """A validated model response. Constructing one proves it passed."""

    action: LLMAction
    reasoning: str
    risk_flags: tuple[str, ...] = ()
    # The model's own stated confidence. Recorded but NEVER used as the
    # trade's confidence score -- that is produced by the calibration model
    # in app/confidence, from historical outcomes. A model's self-report is
    # not evidence about the world.
    self_reported_confidence: float | None = None
    raw_response: str = ""

    @property
    def vetoed(self) -> bool:
        return self.action is LLMAction.VETO

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "reasoning": self.reasoning,
            "risk_flags": list(self.risk_flags),
            "self_reported_confidence": self.self_reported_confidence,
        }


# The schema handed to the model in the prompt. Kept next to the validator so
# the two cannot drift apart.
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["action", "reasoning"],
    "properties": {
        "action": {"type": "string", "enum": ["CONFIRM", "VETO"]},
        "reasoning": {"type": "string", "maxLength": MAX_REASONING_CHARS},
        "risk_flags": {
            "type": "array",
            "maxItems": MAX_RISK_FLAGS,
            "items": {"type": "string", "maxLength": MAX_RISK_FLAG_CHARS},
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}

SCHEMA_TEXT = json.dumps(RESPONSE_SCHEMA, indent=2)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> dict[str, Any]:
    """Parse the first JSON object in ``text``.

    Small local models routinely wrap JSON in prose or a markdown fence. We
    tolerate that wrapper -- it is a formatting quirk, not a semantic error --
    but nothing beyond it. If the payload itself is malformed, that is a
    failure and the caller falls back to the rule-based signal.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValidationError("empty response")

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped).strip()

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = _JSON_BLOCK.search(stripped)
        if not match:
            raise ValidationError(f"no JSON object found in response: {text[:120]!r}")
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValidationError(f"malformed JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValidationError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed


def validate_response(text: str) -> LLMVerdict:
    """Parse and validate a raw model response, or raise ValidationError.

    Every rejection names what was wrong, because "the model failed" with no
    detail makes the bake-off's validity metric impossible to act on.
    """
    payload = extract_json(text)

    unknown = set(payload) - ALLOWED_KEYS
    if unknown:
        raise ValidationError(f"unexpected field(s): {sorted(unknown)}")

    missing = REQUIRED_KEYS - set(payload)
    if missing:
        raise ValidationError(f"missing required field(s): {sorted(missing)}")

    # --- action ---
    raw_action = payload["action"]
    if not isinstance(raw_action, str):
        raise ValidationError(f"action must be a string, got {type(raw_action).__name__}")
    try:
        action = LLMAction(raw_action.strip().upper())
    except ValueError:
        raise ValidationError(
            f"action must be CONFIRM or VETO, got {raw_action!r}. "
            "A model cannot propose a direction here."
        ) from None

    # --- reasoning ---
    reasoning = payload["reasoning"]
    if not isinstance(reasoning, str):
        raise ValidationError(
            f"reasoning must be a string, got {type(reasoning).__name__}"
        )
    reasoning = reasoning.strip()
    if not reasoning:
        raise ValidationError("reasoning must not be empty")
    if len(reasoning) > MAX_REASONING_CHARS:
        raise ValidationError(
            f"reasoning is {len(reasoning)} chars, max {MAX_REASONING_CHARS}"
        )

    # --- risk_flags ---
    raw_flags = payload.get("risk_flags", [])
    if not isinstance(raw_flags, list):
        raise ValidationError(
            f"risk_flags must be an array, got {type(raw_flags).__name__}"
        )
    if len(raw_flags) > MAX_RISK_FLAGS:
        raise ValidationError(f"risk_flags has {len(raw_flags)} items, max {MAX_RISK_FLAGS}")
    flags: list[str] = []
    for flag in raw_flags:
        if not isinstance(flag, str):
            raise ValidationError("risk_flags entries must be strings")
        flag = flag.strip()
        if len(flag) > MAX_RISK_FLAG_CHARS:
            raise ValidationError(
                f"risk flag is {len(flag)} chars, max {MAX_RISK_FLAG_CHARS}"
            )
        if flag:
            flags.append(flag)

    # --- confidence (optional, recorded only) ---
    confidence = payload.get("confidence")
    if confidence is not None:
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValidationError("confidence must be a number")
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValidationError(f"confidence must be in [0, 1], got {confidence}")

    return LLMVerdict(
        action=action,
        reasoning=reasoning,
        risk_flags=tuple(flags),
        self_reported_confidence=confidence,
        raw_response=text,
    )
