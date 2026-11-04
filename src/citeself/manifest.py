"""persona.yaml — the owner's consent record, boundaries and settings."""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

MANIFEST_NAME = "persona.yaml"
DATA_DIR = ".citeself"
DEFAULT_MODEL = "claude-opus-5-5"


class ManifestError(ValueError):
    pass


class ConsentError(ManifestError):
    pass


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PersonaInfo(_Strict):
    name: str = Field(min_length=1, max_length=120)
    bio: str = Field(default="", max_length=2000)


class Consent(_Strict):
    """Self-attestation by the person being simulated.

    CiteSelf only supports simulating yourself. This record cannot prove
    identity; hosted deployments must add their own verification.
    """
    self_attested: bool
    attested_by: str = Field(min_length=1)
    attested_at: str
    statement: str = Field(min_length=1)


class Boundaries(_Strict):
    # Hard guard: questions containing any of these words/phrases are answered
    # with a boundary notice and never sent to the model.
    blocked_keywords: list[str] = Field(default_factory=list)
    # Soft guard: passed to the model as topics it must decline to extrapolate on.
    blocked_topics: list[str] = Field(default_factory=list)
    allow_extrapolation: bool = True

    @field_validator("blocked_keywords", "blocked_topics")
    @classmethod
    def _strip(cls, v: list[str]) -> list[str]:
        return [s.strip() for s in v if s and s.strip()]


class Privacy(_Strict):
    # Names of other people to replace with "[redacted]" at ingest time.
    redact_names: list[str] = Field(default_factory=list)


class ModelSettings(_Strict):
    provider: Literal["anthropic", "extractive"] = "anthropic"
    name: str = DEFAULT_MODEL
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    refusal_fallback: bool = True


class RetrievalSettings(_Strict):
    top_k: int = Field(default=6, ge=1, le=20)
    min_quote_words: int = Field(default=3, ge=1, le=50)


class Manifest(_Strict):
    citeself: Literal[1] = 1
    persona: PersonaInfo
    consent: Consent
    boundaries: Boundaries = Field(default_factory=Boundaries)
    privacy: Privacy = Field(default_factory=Privacy)
    model: ModelSettings = Field(default_factory=ModelSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)

    def require_consent(self) -> None:
        if not self.consent.self_attested:
            raise ConsentError(
                "consent.self_attested is false. CiteSelf only simulates the person who owns the "
                "material and has attested to it in persona.yaml."
            )

    @property
    def disclosure(self) -> str:
        n = self.persona.name
        return (f"AI simulation of {n}, built from {n}'s own published material. "
                f"This is not {n}. Quotes are verbatim and cited; everything else is labeled extrapolation.")


def load_manifest(root: Path) -> Manifest:
    path = root / MANIFEST_NAME
    if not path.exists():
        raise ManifestError(f"No {MANIFEST_NAME} in {root}. Run `citeself init` first.")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return Manifest.model_validate(data)
    except yaml.YAMLError as exc:
        raise ManifestError(f"{path} is not valid YAML: {exc}") from exc
    except ValidationError as exc:
        raise ManifestError(f"{path} is invalid:\n{exc}") from exc


def find_root(start: Path) -> Path:
    """Walk up from ``start`` to the nearest directory containing persona.yaml."""
    start = start.resolve()
    for d in (start, *start.parents):
        if (d / MANIFEST_NAME).exists():
            return d
    raise ManifestError(f"No {MANIFEST_NAME} found in {start} or its parents. Run `citeself init`.")


def write_new_manifest(root: Path, name: str) -> Path:
    path = root / MANIFEST_NAME
    if path.exists():
        raise ManifestError(f"{path} already exists")
    today = _dt.date.today().isoformat()
    manifest = Manifest(
        persona=PersonaInfo(name=name),
        consent=Consent(
            self_attested=True,
            attested_by=name,
            attested_at=today,
            statement=(f"I am {name}. I am creating an AI simulation of myself from material I wrote or "
                       "have the right to use, and I can revoke it at any time."),
        ),
    )
    header = (
        "# CiteSelf persona manifest. You may only build a simulation of YOURSELF.\n"
        "# Docs: https://github.com/yusstech/PersonaLab\n"
    )
    path.write_text(header + yaml.safe_dump(manifest.model_dump(), sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path
