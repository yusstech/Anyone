import csv
import json

import pytest
import yaml

from anyone_ai.engine import Persona
from anyone_ai.evaluate import (EVAL_FILE, init_eval, load_eval_set, planned_calls, run_eval, score_ratings,
                               write_outputs)
from anyone_ai.manifest import ManifestError
from anyone_ai.providers import Completion, ExtractiveProvider


class FakeLLM:
    """Baselines answer generically; the judge scores 'rollback' answers high, everything else low."""

    def __init__(self):
        self.calls = []

    def complete(self, system, user, schema=None):
        self.calls.append((system, user, schema))
        if schema is None:
            return Completion(text="They would probably ship fast and fix later.", data=None, model="fake")
        candidate = user.split("Candidate answer:\n", 1)[1]
        good = "rollback" in candidate.lower()
        return Completion(text="", data={"alignment": 5 if good else 2, "contradiction": not good,
                                         "rationale": "x"}, model="fake")


ITEMS = {"items": [
    {"id": "q1", "question": "How do you think about release quality and rollback?",
     "my_answer": "Severity of defects and a safe rollback decide it.", "covered": True,
     "my_answer_retest": "It depends on defect severity and having a rollback."},
    {"id": "q2", "question": "What is your favourite xylophone brand?", "my_answer": "I have none.", "covered": False},
]}


@pytest.fixture
def eval_dir(persona_dir):
    (persona_dir / EVAL_FILE).write_text(yaml.safe_dump(ITEMS))
    return persona_dir


def test_template_is_rejected_until_edited(tmp_path):
    init_eval(tmp_path)
    with pytest.raises(ManifestError, match="template"):
        load_eval_set(tmp_path)
    with pytest.raises(ManifestError):
        init_eval(tmp_path)


def test_duplicate_ids_rejected(tmp_path):
    (tmp_path / EVAL_FILE).write_text(yaml.safe_dump({"items": [ITEMS["items"][0], ITEMS["items"][0]]}))
    with pytest.raises(ManifestError, match="unique"):
        load_eval_set(tmp_path)


def test_run_eval_end_to_end(eval_dir):
    items = load_eval_set(eval_dir)
    llm = FakeLLM()
    persona = Persona(eval_dir, provider=ExtractiveProvider())
    run = run_eval(persona, llm, items)
    # fixture has a bio, so all three arms run. The eval LLM sees 2 baselines x 2 items + judge 3 arms x 2 items
    # + 1 ceiling; the plan also counts Anyone's own generation calls (upper bound: it may skip the model).
    assert run["arms"] == ["anyone", "bio_only", "generic"]
    assert len(llm.calls) == 2 * 2 + 3 * 2 + 1
    assert planned_calls(items, run["arms"], True) == len(llm.calls) + 2
    # the judge never learns which arm it is scoring
    assert not any("anyone" in c[1] or "bio_only" in c[1] for c in llm.calls if c[2] is not None)

    s = run["summary"]
    assert s["arms"]["anyone"]["alignment_covered"] == 5 and s["arms"]["generic"]["alignment_covered"] == 2
    assert s["ceiling_alignment"] == 5 and s["arms"]["anyone"]["normalized"] == 1.0
    assert s["anyone"]["abstention_on_uncovered"] == 1.0 and s["anyone"]["false_abstention_on_covered"] == 0.0
    assert s["anyone"]["proposed_quote_precision"] == 1.0

    out = write_outputs(eval_dir, run, seed=7)
    assert {p.name for p in out.iterdir()} == {"results.json", "report.md", "rating_sheet.csv", "rating_key.json"}
    report = (out / "report.md").read_text()
    assert "| anyone | 5" in report
    with (out / "rating_sheet.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6 and all(r["response"] in "XYZ" for r in rows)
    assert not any(arm in json.dumps(rows) for arm in ("anyone", "bio_only", "generic"))  # sheet is blind


def test_no_bio_drops_bio_arm(eval_dir):
    path = eval_dir / "persona.yaml"
    data = yaml.safe_load(path.read_text())
    data["subject"]["bio"] = ""
    path.write_text(yaml.safe_dump(data))
    run = run_eval(Persona(eval_dir, provider=ExtractiveProvider()), FakeLLM(), load_eval_set(eval_dir), use_judge=False)
    assert run["arms"] == ["anyone", "generic"] and run["summary"]["arms"]["anyone"]["alignment_covered"] is None


def test_score_blind_ratings(eval_dir):
    run = run_eval(Persona(eval_dir, provider=ExtractiveProvider()), FakeLLM(), load_eval_set(eval_dir), use_judge=False)
    out = write_outputs(eval_dir, run, seed=1)
    key = json.loads((out / "rating_key.json").read_text())
    sheet = out / "rating_sheet.csv"
    with sheet.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:  # owner rates the anyone answer 5, others 2
        r["rating_1_to_5"] = "5" if key[r["item_id"]][r["response"]] == "anyone" else "2"
    with sheet.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    result = score_ratings(out)
    assert result["items_compared"] == 2
    assert result["arms"]["anyone"] == {"mean_rating": 5.0, "ratings": 2, "win_rate": 1.0}
    assert result["arms"]["generic"]["win_rate"] == 0.0

    rows[0]["rating_1_to_5"] = "9"
    with sheet.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(ValueError, match="1-5"):
        score_ratings(out)


def test_unanswered_questions_are_listed(tmp_path):
    (tmp_path / EVAL_FILE).write_text(yaml.safe_dump({"items": [
        {"id": "a", "question": "Q?", "my_answer": "Mine."}, {"id": "b", "question": "Q2?", "my_answer": "  "},
        {"id": "c", "question": "Q3?"}]}))
    with pytest.raises(ManifestError, match="2 question\\(s\\) first: b, c"):
        load_eval_set(tmp_path)
