"""Model provider adapters.

Each adapter turns a prompt string into a raw response string, and knows
whether it can run at all on this machine. Nothing here parses or trusts the
output -- that is ``schema.validate_response``'s job, and keeping the two
apart is what lets the bake-off measure "how often does this model produce
valid output" as a real number.

CREDENTIALS
-----------
No key is ever hard-coded, committed, or written to the database. Hosted
adapters read a key from the environment at call time and report themselves
unavailable when it is absent. These are LLM keys, not exchange keys: they
cannot move funds. The project's rule that no exchange credential exists
anywhere in this codebase is untouched -- there is still no venue key, and
still no order-placement path.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 60.0


class ProviderError(RuntimeError):
    """The provider could not be reached or returned an error."""


@dataclass(frozen=True)
class ProviderInfo:
    """What a provider is, and what it costs.

    Prices are USD per 1M tokens as published at the time of writing
    (2026-09). LLM pricing moves; ``model_bakeoff.py`` prints these back so a
    stale number is visible in the report rather than silently believed.
    See README "Open Decisions" (OD-7).
    """

    name: str
    kind: str  # "hosted" | "local" | "baseline"
    model_id: str
    input_usd_per_mtok: float = 0.0
    output_usd_per_mtok: float = 0.0
    notes: str = ""

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_usd_per_mtok / 1_000_000
            + output_tokens * self.output_usd_per_mtok / 1_000_000
        )


class Provider:
    """Base adapter. Subclasses implement ``complete`` and ``available``."""

    info: ProviderInfo

    def available(self) -> tuple[bool, str]:
        """Return (can_run, why_not)."""
        raise NotImplementedError

    def complete(self, prompt: str) -> str:
        """Return the model's raw text response."""
        raise NotImplementedError

    # Rough token estimate; only used for cost reporting, never for billing.
    @staticmethod
    def estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)


# --------------------------------------------------------------------------
# 1. Claude Haiku 4.5 (hosted)
# --------------------------------------------------------------------------

class ClaudeProvider(Provider):
    """Anthropic Messages API.

    Candidate because structured-output adherence is the property that
    matters most here (the response must be valid JSON every time), and the
    project already targets Claude elsewhere.
    """

    def __init__(self, model_id: str = "claude-haiku-4-5-20251001") -> None:
        self.info = ProviderInfo(
            name="claude-haiku-4.5",
            kind="hosted",
            model_id=model_id,
            input_usd_per_mtok=1.0,
            output_usd_per_mtok=5.0,
            notes="200K context; strongest structured-output adherence of the candidates",
        )

    def available(self) -> tuple[bool, str]:
        if not os.getenv("ANTHROPIC_API_KEY"):
            return False, "ANTHROPIC_API_KEY is not set"
        return True, ""

    def complete(self, prompt: str) -> str:
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ProviderError("ANTHROPIC_API_KEY is not set")

        body = json.dumps(
            {
                "model": self.info.model_id,
                "max_tokens": 700,
                "temperature": 0.0,  # determinism matters more than variety
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode()

        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "content-type": "application/json",
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=DEFAULT_TIMEOUT) as response:
                payload = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ProviderError(f"Anthropic request failed: {exc}") from exc

        blocks = payload.get("content", [])
        return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")


# --------------------------------------------------------------------------
# 2. GPT-5 mini (hosted)
# --------------------------------------------------------------------------

class OpenAIProvider(Provider):
    """OpenAI Responses/Chat API.

    Candidate purely as a second, cheaper vendor: relying on one provider for
    a component that runs continuously is a availability risk, not just a
    cost one.
    """

    def __init__(self, model_id: str = "gpt-5-mini") -> None:
        self.info = ProviderInfo(
            name="gpt-5-mini",
            kind="hosted",
            model_id=model_id,
            input_usd_per_mtok=0.25,
            output_usd_per_mtok=2.0,
            notes="272K context; cheapest hosted candidate, different vendor",
        )

    def available(self) -> tuple[bool, str]:
        if not os.getenv("OPENAI_API_KEY"):
            return False, "OPENAI_API_KEY is not set"
        return True, ""

    def complete(self, prompt: str) -> str:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ProviderError("OPENAI_API_KEY is not set")

        body = json.dumps(
            {
                "model": self.info.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
            }
        ).encode()

        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={
                "content-type": "application/json",
                "authorization": f"Bearer {key}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=DEFAULT_TIMEOUT) as response:
                payload = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ProviderError(f"OpenAI request failed: {exc}") from exc

        choices = payload.get("choices", [])
        if not choices:
            raise ProviderError(f"no choices in response: {payload}")
        return choices[0].get("message", {}).get("content", "")


# --------------------------------------------------------------------------
# 3. A small local model via Ollama
# --------------------------------------------------------------------------

class OllamaProvider(Provider):
    """Any local model served by Ollama.

    Candidate because the running cost is genuinely zero and there is no
    network dependency, which matters for something meant to run continuously.
    Defaults to qwen3:1.7b -- small enough for CPU-only inference.
    """

    def __init__(
        self,
        model_id: str = "qwen3:1.7b",
        host: str = "http://localhost:11434",
    ) -> None:
        self.host = host.rstrip("/")
        self.info = ProviderInfo(
            name=f"ollama/{model_id}",
            kind="local",
            model_id=model_id,
            input_usd_per_mtok=0.0,
            output_usd_per_mtok=0.0,
            notes="runs on CPU; zero marginal cost and no network dependency",
        )

    def available(self) -> tuple[bool, str]:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=3.0) as response:
                tags = json.loads(response.read())
        except Exception as exc:  # noqa: BLE001 - any failure means unavailable
            return False, f"Ollama not reachable at {self.host} ({exc})"

        installed = {m.get("name", "") for m in tags.get("models", [])}
        if not installed:
            return False, "Ollama is running but has no models pulled"
        if self.info.model_id not in installed and not any(
            name.split(":")[0] == self.info.model_id.split(":")[0] for name in installed
        ):
            return False, (
                f"model {self.info.model_id!r} not pulled "
                f"(available: {sorted(installed)})"
            )
        return True, ""

    def complete(self, prompt: str) -> str:
        body = json.dumps(
            {
                "model": self.info.model_id,
                "prompt": prompt,
                "stream": False,
                "format": "json",  # ask Ollama to constrain output to JSON
                "options": {"temperature": 0.0},
            }
        ).encode()

        request = urllib.request.Request(
            f"{self.host}/api/generate",
            data=body,
            headers={"content-type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=DEFAULT_TIMEOUT) as response:
                payload = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ProviderError(f"Ollama request failed: {exc}") from exc
        return payload.get("response", "")


# --------------------------------------------------------------------------
# 4. Deterministic baseline (always available)
# --------------------------------------------------------------------------

class NullVetoBaseline(Provider):
    """Never vetoes. The control arm of the bake-off.

    This is not a filler entry. Without it, a model that vetoes 30% of trades
    cannot be judged: you cannot tell whether it removed losers or simply
    removed trades. The baseline is what "no LLM at all" scores, and any
    candidate that does not beat it is not worth its latency, let alone its
    price.
    """

    def __init__(self) -> None:
        self.info = ProviderInfo(
            name="baseline-no-veto",
            kind="baseline",
            model_id="none",
            notes="control arm: the rule-based system with no LLM layer",
        )

    def available(self) -> tuple[bool, str]:
        return True, ""

    def complete(self, prompt: str) -> str:
        return json.dumps(
            {
                "action": "CONFIRM",
                "reasoning": "Baseline control: no model consulted, "
                "the rule-based signal stands unchanged.",
                "risk_flags": [],
            }
        )


class ScriptedProvider(Provider):
    """Returns canned responses in order. For tests only -- never networked."""

    def __init__(self, responses: list[str], name: str = "scripted", latency: float = 0.0):
        self.responses = list(responses)
        self.calls: list[str] = []
        self.latency = latency
        self.info = ProviderInfo(name=name, kind="baseline", model_id="scripted")

    def available(self) -> tuple[bool, str]:
        return True, ""

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        if self.latency:
            time.sleep(self.latency)
        if not self.responses:
            raise ProviderError("ScriptedProvider ran out of responses")
        return self.responses.pop(0)


def build_candidates() -> list[Provider]:
    """The four candidates plus the control, in bake-off order."""
    return [
        ClaudeProvider(),
        OpenAIProvider(),
        OllamaProvider(),
        NullVetoBaseline(),
    ]
