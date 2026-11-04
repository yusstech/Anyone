"""Model providers. A provider turns (system, messages, evidence) into raw segments.

Providers never decide what is shown — ``validate.validate_segments`` does.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Protocol

from .manifest import ModelSettings
from .store import Evidence

SEGMENTS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["quote", "extrapolation", "no_evidence"]},
                    "text": {"type": "string"},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["kind", "text", "evidence_ids"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["segments"],
    "additionalProperties": False,
}


class ProviderError(RuntimeError):
    """The model could not be reached or misbehaved. Message is safe to show users."""


class MalformedOutput(ProviderError):
    pass


@dataclass
class GenerationRequest:
    system: str
    messages: list[dict]
    evidence: dict[str, Evidence]  # label (E1..) -> evidence


@dataclass
class GenerationResult:
    segments: object
    model: str
    usage: dict = field(default_factory=dict)
    refused: bool = False
    refusal_category: str | None = None


@dataclass
class Completion:
    text: str
    data: object
    model: str
    usage: dict = field(default_factory=dict)
    refused: bool = False


class Provider(Protocol):
    name: str

    def generate(self, request: GenerationRequest) -> GenerationResult: ...


# ----------------------------------------------------------------- Anthropic

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, settings: ModelSettings, client=None, max_tokens: int = 16000):
        self.settings = settings
        self.max_tokens = max_tokens
        self._client = client

    @property
    def client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()  # credentials from env / `ant auth login` profile
        return self._client

    def _create(self, **kwargs):
        """Single place that talks to the API and maps errors to safe, actionable messages."""
        import anthropic

        try:
            if self.settings.refusal_fallback:
                return self.client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **kwargs)
            return self.client.messages.create(**kwargs)
        except anthropic.AuthenticationError as exc:
            raise ProviderError("Anthropic rejected the credentials. Set ANTHROPIC_API_KEY (or run `ant auth login`).") from exc
        except anthropic.PermissionDeniedError as exc:
            raise ProviderError(f"Your API key cannot use model {self.settings.name!r}.") from exc
        except anthropic.NotFoundError as exc:
            raise ProviderError(f"Model {self.settings.name!r} was not found. Check model.name in persona.yaml.") from exc
        except anthropic.RateLimitError as exc:
            raise ProviderError("Rate limited by the Anthropic API. Wait a moment and retry.") from exc
        except anthropic.BadRequestError as exc:
            raise ProviderError(f"The Anthropic API rejected the request: {exc.message}") from exc
        except anthropic.APIStatusError as exc:
            raise ProviderError(f"Anthropic API error ({exc.status_code}). Retry later.") from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderError("Could not reach the Anthropic API. Check your network connection.") from exc
        except anthropic.AnthropicError as exc:  # e.g. no credentials configured at all
            if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
                raise ProviderError("No Anthropic credentials found. Set ANTHROPIC_API_KEY (or run `ant auth login`), "
                                    "or try offline mode with --provider extractive.") from exc
            raise ProviderError(f"Anthropic client error: {type(exc).__name__}") from exc

    def complete(self, system: str, user: str, schema: dict | None = None) -> Completion:
        """Plain single-turn completion (used by baselines and the eval judge)."""
        output_config: dict = {"effort": self.settings.effort}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        response = self._create(model=self.settings.name, max_tokens=self.max_tokens, system=system,
                                messages=[{"role": "user", "content": user}], output_config=output_config)
        usage = _usage_dict(getattr(response, "usage", None))
        if response.stop_reason == "refusal":
            return Completion(text="", data=None, model=response.model, usage=usage, refused=True)
        if response.stop_reason == "max_tokens":
            raise MalformedOutput("The model ran out of output tokens before finishing.")
        text = next((b.text for b in response.content if getattr(b, "type", None) == "text"), "")
        data = None
        if schema is not None:
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise MalformedOutput("The model returned invalid JSON.") from exc
        return Completion(text=text, data=data, model=response.model, usage=usage)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        # No prompt caching: every request ends in a unique evidence+question block and the
        # system prompt is below the minimum cacheable size, so cache writes would never be read.
        response = self._create(
            model=self.settings.name,
            max_tokens=self.max_tokens,
            system=request.system,
            messages=request.messages,
            output_config={
                "effort": self.settings.effort,
                "format": {"type": "json_schema", "schema": SEGMENTS_SCHEMA},
            },
        )
        usage = _usage_dict(getattr(response, "usage", None))
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            return GenerationResult(segments=[], model=response.model, usage=usage, refused=True,
                                    refusal_category=getattr(details, "category", None))
        if response.stop_reason == "max_tokens":
            raise MalformedOutput("The model ran out of output tokens before finishing.")
        text = next((b.text for b in response.content if getattr(b, "type", None) == "text"), None)
        if text is None:
            raise MalformedOutput("The model returned no text.")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise MalformedOutput("The model returned invalid JSON.") from exc
        return GenerationResult(segments=data.get("segments") if isinstance(data, dict) else None,
                                model=response.model, usage=usage)


def _usage_dict(usage) -> dict:
    if usage is None:
        return {}
    keys = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    return {k: getattr(usage, k) for k in keys if getattr(usage, k, None) is not None}


# ---------------------------------------------------------------- Extractive

_SENTENCES = re.compile(r"[^.!?]+[.!?]+")


class ExtractiveProvider:
    """Offline, no-LLM mode: returns the most relevant passages as verbatim quotes.

    Useful for trying CiteSelf without an API key and as an evaluation baseline.
    """
    name = "extractive"

    def __init__(self, max_quotes: int = 2, max_sentences: int = 2):
        self.max_quotes = max_quotes
        self.max_sentences = max_sentences

    def generate(self, request: GenerationRequest) -> GenerationResult:
        segments: list[dict] = []
        for label, ev in list(request.evidence.items())[: self.max_quotes]:
            sentences = [s.strip() for s in _SENTENCES.findall(ev.text)][: self.max_sentences]
            quote = " ".join(sentences) if sentences else ev.text.strip()
            # Keep it a contiguous span of the source so it always verifies.
            if quote not in ev.text:
                quote = sentences[0] if sentences and sentences[0] in ev.text else ev.text.strip()
            segments.append({"kind": "quote", "text": quote, "evidence_ids": [label]})
        segments.append({"kind": "no_evidence", "evidence_ids": [],
                         "text": "Offline mode: these are the closest passages on record; no interpretation was generated."})
        return GenerationResult(segments=segments, model="extractive")


def make_provider(settings: ModelSettings, override: str | None = None) -> Provider:
    kind = override or settings.provider
    if kind == "extractive":
        return ExtractiveProvider()
    if kind == "anthropic":
        return AnthropicProvider(settings)
    raise ValueError(f"Unknown provider {kind!r}")
