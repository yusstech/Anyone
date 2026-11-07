"""persona.yaml v2: subject kinds, consent rules, disclosures and CiteSelf-era compatibility."""
import json

import pytest
import yaml

from anyone_ai.cli import main
from anyone_ai.engine import Persona, build_system_prompt
from anyone_ai.manifest import (DATA_DIR, LEGACY_DATA_DIR, PRIVATE_BLOCKED_TOPICS, ManifestError, index_path,
                                load_manifest)
from anyone_ai.providers import ExtractiveProvider
from anyone_ai.store import Store

from .conftest import FIXTURES


def write(root, data):
    root.mkdir(parents=True, exist_ok=True)
    (root / "persona.yaml").write_text(yaml.safe_dump(data))
    return root


def third_party(kind, **extra):
    return {"anyone": 2, "subject": {"kind": kind, "name": "Dana Reyes"}, **extra}


CONSENT = {"self_attested": True, "attested_by": "Dana Reyes", "attested_at": "2026-11-07", "statement": "It's me."}


def test_disclosure_per_kind(tmp_path):
    pub = load_manifest(write(tmp_path / "pub", third_party("public_figure")))
    assert "based on public sources; Dana Reyes has not endorsed this" in pub.disclosure
    assert "not Dana Reyes" in pub.disclosure and "consented" not in pub.disclosure

    priv = load_manifest(write(tmp_path / "priv", third_party("private_individual")))
    assert "has not consented to being simulated" in priv.disclosure
    assert "professional" in priv.disclosure

    me = load_manifest(write(tmp_path / "me", {**third_party("self"), "consent": CONSENT}))
    assert "built from Dana Reyes's own published material" in me.disclosure


def test_consent_only_for_self(tmp_path):
    with pytest.raises(ManifestError, match="requires a consent block"):
        load_manifest(write(tmp_path / "a", third_party("self")))
    for kind in ("public_figure", "private_individual"):
        with pytest.raises(ManifestError, match="only valid for subject.kind 'self'"):
            load_manifest(write(tmp_path / kind, {**third_party(kind), "consent": CONSENT}))


def test_unknown_kind_and_v1_keys_rejected(tmp_path):
    with pytest.raises(ManifestError):
        load_manifest(write(tmp_path / "a", third_party("celebrity")))
    with pytest.raises(ManifestError):  # v2 uses subject:, not persona:
        load_manifest(write(tmp_path / "b", {"anyone": 2, "persona": {"name": "X"}, "consent": CONSENT}))


def test_private_individual_topics_cannot_be_removed(tmp_path):
    m = load_manifest(write(tmp_path / "p", third_party("private_individual", boundaries={"blocked_topics": ["Health", "salary"]})))
    topics = m.boundaries.blocked_topics
    assert topics[:2] == ["Health", "salary"]
    assert {t.lower() for t in topics} == {"salary", *PRIVATE_BLOCKED_TOPICS} and len(topics) == 6


def test_pii_redaction_locked_on_for_third_parties(tmp_path):
    for kind in ("public_figure", "private_individual"):
        with pytest.raises(ManifestError, match="redact_pii cannot be turned off"):
            load_manifest(write(tmp_path / kind, third_party(kind, privacy={"redact_pii": False})))
    me = load_manifest(write(tmp_path / "me", {**third_party("self"), "consent": CONSENT, "privacy": {"redact_pii": False}}))
    assert me.privacy.redact_pii is False


def test_prompt_and_boundary_notice_follow_kind(tmp_path):
    priv = load_manifest(write(tmp_path / "p", third_party("private_individual", boundaries={"blocked_keywords": ["salary"]})))
    prompt = build_system_prompt(priv)
    assert "public professional sources by and about Dana Reyes" in prompt
    assert "private individual who has not consented" in prompt
    assert "This simulation must not cover these topics" in prompt and "location" in prompt
    assert "has asked" not in prompt

    pub = build_system_prompt(load_manifest(write(tmp_path / "f", third_party("public_figure"))))
    assert "private individual" not in pub and "public sources by and about Dana Reyes" in pub

    persona = Persona(tmp_path / "p", provider=ExtractiveProvider())
    try:
        a = persona.ask("What is her salary?")
    finally:
        persona.close()
    assert a.status == "boundary" and a.segments[0].text == "This simulation does not discuss that topic."


def test_citeself_v1_manifest_and_data_dir_still_work(persona_dir):
    """A CiteSelf persona (citeself: 1 + .citeself/) loads as kind self; its index is moved to .anyone/."""
    path = persona_dir / "persona.yaml"
    data = yaml.safe_load(path.read_text())
    data.pop("anyone")
    v1 = {"citeself": 1, "persona": data.pop("subject"), **data}
    v1["persona"].pop("kind")
    assert set(v1) == {"citeself", "persona", "consent", "boundaries", "privacy", "model"}
    path.write_text(yaml.safe_dump(v1))
    (persona_dir / DATA_DIR).rename(persona_dir / LEGACY_DATA_DIR)

    m = load_manifest(persona_dir)
    assert m.subject.kind == "self" and m.subject.name == "Avery Moreno" and m.consent.self_attested
    assert m.privacy.redact_pii is True

    p = Persona(persona_dir, provider=ExtractiveProvider())
    try:
        assert p.ask("release quality and rollback").status == "ok"
    finally:
        p.close()
    assert (persona_dir / DATA_DIR / "index.db").exists() and not (persona_dir / LEGACY_DATA_DIR).exists()


def test_index_path_prefers_existing_new_dir(tmp_path):
    (tmp_path / DATA_DIR).mkdir()
    (tmp_path / LEGACY_DATA_DIR).mkdir()
    assert index_path(tmp_path) == tmp_path / DATA_DIR / "index.db"
    assert (tmp_path / LEGACY_DATA_DIR).exists()  # never merged or deleted silently


def test_cli_init_third_party(tmp_path, capsys, monkeypatch):
    pub = tmp_path / "pub"
    assert main(["-C", str(pub), "init", "--name", "Dana Reyes", "--kind", "public_figure", "--yes"]) == 0
    written = yaml.safe_load((pub / "persona.yaml").read_text())
    assert written["anyone"] == 2 and written["subject"]["kind"] == "public_figure" and "consent" not in written

    priv = tmp_path / "priv"
    assert main(["-C", str(priv), "init", "--name", "Sam Okoro", "--kind", "private_individual", "--yes"]) == 0
    text = (priv / "persona.yaml").read_text()
    assert "professional material only" in text
    assert set(PRIVATE_BLOCKED_TOPICS) <= set(yaml.safe_load(text)["boundaries"]["blocked_topics"])

    # Wrong confirmation flag for the kind is an error, and declining creates nothing.
    assert main(["-C", str(tmp_path / "x"), "init", "--name", "A", "--kind", "public_figure", "--i-am-this-person"]) == 2
    assert main(["-C", str(tmp_path / "y"), "init", "--name", "A", "--yes"]) == 2
    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert main(["-C", str(tmp_path / "z"), "init", "--name", "A", "--kind", "private_individual"]) == 1
    assert not (tmp_path / "z").exists()


def test_cli_third_party_ingest_requires_explicit_authorship(tmp_path, capsys):
    root = tmp_path / "pub"
    assert main(["-C", str(root), "init", "--name", "Dana Reyes", "--kind", "public_figure", "--yes"]) == 0
    src = str(FIXTURES / "avery" / "sources")
    # stdin is not a TTY under pytest: never assume someone else wrote it.
    assert main(["-C", str(root), "ingest", src]) == 2
    assert "--authorship" in capsys.readouterr().err
    assert main(["-C", str(root), "ingest", src, "--authorship", "by_me"]) == 0
    capsys.readouterr()
    assert main(["-C", str(root), "ask", "release quality and rollback", "--provider", "extractive", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert "Dana Reyes has not endorsed this" in out["disclosure"]
    with Store(index_path(root)) as store:
        assert store.stats()["sources"] == 4
