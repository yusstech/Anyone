"""persona.yaml: who is simulated, on what basis, and within which boundaries."""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

MANIFEST_NAME = "persona.yaml"
DATA_DIR = ".anyone"
LEGACY_DATA_DIR = ".citeself"  # CiteSelf-era name; migrated in place by data_dir()
INDEX_NAME = "index.db"
DEFAULT_MODEL = "claude-opus-5-5"

SubjectKind = Literal["public_figure", "private_individual", "self"]
SUBJECT_KINDS: tuple[str, ...] = ("public_figure", "private_individual", "self")

# Never simulated for a private individual, whatever persona.yaml says. Merged into
# boundaries.blocked_topics on load, so removing them from the file has no effect.
PRIVATE_BLOCKED_TOPICS: tuple[str, ...] = ("health", "family", "religion", "sexuality", "location")


class ManifestError(ValueError):
    pass


class ConsentError(ManifestError):
    pass


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Subject(_Strict):
    """The person being simulated.

    kind:
      public_figure       someone with a public role, simulated from public sources
      private_individual  anyone else; professional public material only, stricter boundaries
      self                you, from your own material ("verified self" mode, needs ``consent``)
    """
    kind: SubjectKind
    name: str = Field(min_length=1, max_length=120)
    bio: str = Field(default="", max_length=2000)


class Consent(_Strict):
    """Self-attestation by the person being simulated. Only valid for ``subject.kind: self``.

    This record cannot prove identity; hosted deployments must add their own verification.
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
    # Replace email addresses, phone numbers and street addresses at ingest time.
    # Can only be turned off for subject.kind: self.
    redact_pii: bool = True


_HANDLE = r"^[A-Za-z0-9_]{1,15}$"


class Research(_Strict):
    """Inputs for `anyone new` / `anyone deepen`. Only meaningful for third-party kinds."""
    # Reference links from the user (their site, X profile, a talk page). Pages on these hosts
    # count as the subject's own words; an x.com/<handle> link confirms that handle.
    links: list[str] = Field(default_factory=list)
    # Disambiguation, e.g. "CEO of Acme, Lagos". Sent to the search model, never quoted.
    hint: str = Field(default="", max_length=500)
    # Confirmed X accounts: posts by these handles count as the subject's own words.
    x_handles: list[str] = Field(default_factory=list)

    @field_validator("links")
    @classmethod
    def _links(cls, v: list[str]) -> list[str]:
        out = []
        for link in (s.strip() for s in v):
            if not link:
                continue
            if not link.lower().startswith(("http://", "https://")):
                link = "https://" + link
            out.append(link)
        return out

    @field_validator("x_handles")
    @classmethod
    def _handles(cls, v: list[str]) -> list[str]:
        import re

        out: list[str] = []
        for h in (s.strip().lstrip("@") for s in v):
            if not re.match(_HANDLE, h):
                raise ValueError(f"not a valid X handle: {h!r}")
            if h.lower() not in {o.lower() for o in out}:
                out.append(h)
        return out


class ModelSettings(_Strict):
    provider: Literal["anthropic", "extractive"] = "anthropic"
    name: str = DEFAULT_MODEL
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    refusal_fallback: bool = True


class RetrievalSettings(_Strict):
    top_k: int = Field(default=6, ge=1, le=20)
    min_quote_words: int = Field(default=3, ge=1, le=50)


class Manifest(_Strict):
    anyone: Literal[2] = 2
    subject: Subject
    consent: Consent | None = None
    boundaries: Boundaries = Field(default_factory=Boundaries)
    privacy: Privacy = Field(default_factory=Privacy)
    research: Research = Field(default_factory=Research)
    model: ModelSettings = Field(default_factory=ModelSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)

    @model_validator(mode="after")
    def _rules_for_kind(self) -> "Manifest":
        kind = self.subject.kind
        if kind == "self":
            if self.consent is None:
                raise ValueError("subject.kind 'self' requires a consent block (your self-attestation)")
            return self
        if self.consent is not None:
            raise ValueError(f"consent is only valid for subject.kind 'self'; a {kind} cannot self-attest through "
                             "someone else's manifest. Remove the consent block.")
        if not self.privacy.redact_pii:
            raise ValueError(f"privacy.redact_pii cannot be turned off for subject.kind '{kind}'")
        if kind == "private_individual":
            have = {t.lower() for t in self.boundaries.blocked_topics}
            self.boundaries.blocked_topics += [t for t in PRIVATE_BLOCKED_TOPICS if t not in have]
        return self

    @property
    def is_self(self) -> bool:
        return self.subject.kind == "self"

    def require_consent(self) -> None:
        """Self mode needs a positive self-attestation. Other kinds have no consent to check;
        their disclosure says so on every answer."""
        if self.is_self and not (self.consent and self.consent.self_attested):
            raise ConsentError(
                "consent.self_attested is false. Self mode only simulates the person who owns the "
                "material and has attested to it in persona.yaml."
            )

    @property
    def source_basis(self) -> str:
        """What the simulation is built from, for prompts and tool descriptions."""
        n = self.subject.name
        return {"self": f"{n}'s own published material",
                "public_figure": f"public sources by and about {n}",
                "private_individual": f"public professional sources by and about {n}"}[self.subject.kind]

    @property
    def disclosure(self) -> str:
        n = self.subject.name
        tail = f"This is not {n}. Quotes are verbatim and cited; everything else is labeled extrapolation."
        if self.subject.kind == "self":
            return f"AI simulation of {n}, built from {n}'s own published material. {tail}"
        if self.subject.kind == "public_figure":
            return f"AI simulation of {n} based on public sources; {n} has not endorsed this. {tail}"
        return (f"AI simulation of {n} based on public professional sources; {n} has not endorsed this and "
                f"has not consented to being simulated. {tail}")

    @property
    def boundary_notice(self) -> str:
        if self.is_self:
            return f"{self.subject.name} has asked this simulation not to discuss that topic."
        return "This simulation does not discuss that topic."


def _upgrade(data: dict) -> dict:
    """Read a CiteSelf v1 manifest (self-simulation only) as v2 ``subject.kind: self``."""
    if "citeself" not in data:
        return data
    if data.get("citeself") != 1:
        raise ManifestError(f"Unknown manifest version citeself: {data.get('citeself')!r}")
    data = dict(data)
    data.pop("citeself")
    persona = data.pop("persona", None)
    if not isinstance(persona, dict):
        raise ManifestError("v1 manifest has no persona block")
    return {"anyone": 2, "subject": {"kind": "self", **persona}, **data}


def save_research(root: Path, research: Research) -> None:
    """Update only the research block of persona.yaml, keeping the user's other edits and comments
    above the YAML body."""
    path = root / MANIFEST_NAME
    raw = path.read_text(encoding="utf-8")
    import itertools

    header = "".join(itertools.takewhile(lambda line: line.startswith("#"), raw.splitlines(keepends=True)))
    data = yaml.safe_load(raw) or {}
    data["research"] = research.model_dump()
    Manifest.model_validate(_upgrade(data))
    path.write_text(header + yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def load_manifest(root: Path) -> Manifest:
    path = root / MANIFEST_NAME
    if not path.exists():
        raise ManifestError(f"No {MANIFEST_NAME} in {root}. Run `anyone init` first.")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ManifestError(f"{path} must be a YAML mapping")
        return Manifest.model_validate(_upgrade(data))
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
    raise ManifestError(f"No {MANIFEST_NAME} found in {start} or its parents. Run `anyone init`.")


def data_dir(root: Path) -> Path:
    """The persona's derived-data directory. A CiteSelf-era ``.citeself/`` is renamed in place."""
    new, old = root / DATA_DIR, root / LEGACY_DATA_DIR
    if not new.exists() and old.is_dir():
        old.rename(new)
    return new


def index_path(root: Path) -> Path:
    return data_dir(root) / INDEX_NAME


_HEADERS = {
    "self": ("# Anyone persona manifest, verified self mode: a simulation of YOURSELF,\n"
             "# from material you wrote or have the right to use.\n"),
    "public_figure": ("# Anyone persona manifest: a simulation of a public figure, from public sources.\n"
                      "# Every answer says the person has not endorsed it.\n"),
    "private_individual": ("# Anyone persona manifest: a simulation of a private individual, from public\n"
                           "# professional material only. The person has not consented. The blocked topics\n"
                           "# below are always enforced and email, phone and address redaction cannot be disabled.\n"),
}


def write_new_manifest(root: Path, name: str, kind: str = "self", research: Research | None = None) -> Path:
    path = root / MANIFEST_NAME
    if path.exists():
        raise ManifestError(f"{path} already exists")
    if kind not in SUBJECT_KINDS:
        raise ManifestError(f"kind must be one of {SUBJECT_KINDS}")
    consent = None
    if kind == "self":
        consent = Consent(
            self_attested=True,
            attested_by=name,
            attested_at=_dt.date.today().isoformat(),
            statement=(f"I am {name}. I am creating an AI simulation of myself from material I wrote or "
                       "have the right to use, and I can revoke it at any time."),
        )
    manifest = Manifest(subject=Subject(kind=kind, name=name), consent=consent, research=research or Research())
    header = _HEADERS[kind] + "# Docs: https://github.com/yusstech/Anyone\n"
    body = yaml.safe_dump(manifest.model_dump(exclude_none=True), sort_keys=False, allow_unicode=True)
    path.write_text(header + body, encoding="utf-8")
    return path
