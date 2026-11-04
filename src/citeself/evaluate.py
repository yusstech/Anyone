"""citeself eval — measure how well the simulation matches *you*.

Ground truth is the owner's own answers in ``eval.yaml``, written without looking
at the AI. Three arms answer every question:

  citeself  the full pipeline (retrieval + validated, labeled segments)
  bio_only  a generic model given only the name and bio from persona.yaml
  generic   a generic model given only the name

A blind LLM judge (never told which arm wrote what) scores each answer's stance
alignment with the owner's answer. Because LLM judges are imperfect, the run
also writes a shuffled, anonymized rating sheet for the owner to score by hand;
``citeself eval score`` turns that into per-arm results.

Method follows Park et al. 2024 ("Generative Agent Simulations of 1,000 People"):
when the owner also gives a later re-answer, the owner's own consistency is the
ceiling and arm scores are reported normalized against it.
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import random
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .engine import Answer, Persona
from .manifest import DATA_DIR, ManifestError
from .providers import Completion

EVAL_FILE = "eval.yaml"
ARMS = ("citeself", "bio_only", "generic")

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "alignment": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "contradiction": {"type": "boolean"},
        "rationale": {"type": "string"},
    },
    "required": ["alignment", "contradiction", "rationale"],
    "additionalProperties": False,
}

JUDGE_SYSTEM = (
    "You compare a candidate answer with a person's own answer to the same question. Score how well the candidate "
    "captures the person's actual position and reasoning, on a 1-5 scale: 5 = same stance and same key reasons; "
    "4 = same stance, some reasons missing; 3 = partly aligned or too vague to tell; 2 = mostly misaligned; "
    "1 = opposite stance or entirely off-topic. A candidate that declines to answer scores 1 or 2 unless the "
    "person's own answer also declines. Ignore writing style, length, labels such as [quote] or [extrapolation], "
    "and whether the candidate speaks in first or third person. Set contradiction=true only if the candidate asserts "
    "something the person's answer clearly contradicts. Keep the rationale to one sentence."
)


class EvalItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    my_answer: str = Field(min_length=1)
    covered: bool = True                  # do you believe your ingested material addresses this?
    my_answer_retest: str | None = None   # optional: answer again later, without looking, to set the ceiling


class EvalSet(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[EvalItem] = Field(min_length=1)


class LLM(Protocol):
    def complete(self, system: str, user: str, schema: dict | None = None) -> Completion: ...


EVAL_TEMPLATE = """\
# CiteSelf evaluation set. Answer every question YOURSELF, in your own words,
# BEFORE running `citeself eval run`, and without looking at the AI's answers.
# Keep this file private: it is your ground truth.
#
#   covered: true  -> you believe your ingested material addresses this question
#   covered: false -> it doesn't; a good simulation should say it has no evidence
#   my_answer_retest (optional) -> answer again a week or more later, without
#     looking at your first answer. Your own consistency becomes the ceiling score.
#
# Aim for 15-30 questions. Mix topics you've written about with ones you haven't.
items:
  - id: q1
    question: "Replace with a question someone might ask you"
    my_answer: "Your honest answer, 1-4 sentences"
    covered: true
  - id: q2
    question: "A question your writing never touches"
    my_answer: "Your honest answer"
    covered: false
"""


def init_eval(root: Path) -> Path:
    path = root / EVAL_FILE
    if path.exists():
        raise ManifestError(f"{path} already exists")
    path.write_text(EVAL_TEMPLATE, encoding="utf-8")
    return path


def load_eval_set(root: Path) -> list[EvalItem]:
    path = root / EVAL_FILE
    if not path.exists():
        raise ManifestError(f"No {EVAL_FILE} in {root}. Run `citeself eval init` and write your answers first.")
    try:
        items = EvalSet.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {}).items
    except (yaml.YAMLError, ValidationError) as exc:
        raise ManifestError(f"{path} is invalid:\n{exc}") from exc
    ids = [i.id for i in items]
    if len(ids) != len(set(ids)):
        raise ManifestError(f"{path}: item ids must be unique")
    if any(i.question.startswith("Replace with") for i in items):
        raise ManifestError(f"{path} still contains the template example. Write your own questions and answers.")
    return items


def planned_calls(items: list[EvalItem], arms: list[str], judge: bool) -> int:
    per_item = len(arms) * (2 if judge else 1)
    return len(items) * per_item + (sum(1 for i in items if i.my_answer_retest) if judge else 0)


# --------------------------------------------------------------------- arms

def citeself_text(answer: Answer) -> str:
    return "\n".join(f"[{s.kind}] {s.text}" for s in answer.segments)


def baseline_answer(llm: LLM, name: str, bio: str, question: str) -> Completion:
    about = f"Here is all you know about {name}: {bio}" if bio else f"You know nothing about {name} beyond their name."
    system = ("You estimate how a specific person would answer a question. " + about +
              " Give your best estimate in 2-4 sentences, in the third person.")
    return llm.complete(system, f"How would {name} answer this question?\n\n{question}")


def judge(llm: LLM, question: str, reference: str, candidate: str) -> dict:
    user = (f"Question:\n{question}\n\nThe person's own answer:\n{reference}\n\n"
            f"Candidate answer:\n{candidate}")
    c = llm.complete(JUDGE_SYSTEM, user, JUDGE_SCHEMA)
    if c.refused or not isinstance(c.data, dict):
        return {"alignment": None, "contradiction": None, "rationale": "judge unavailable"}
    return {k: c.data.get(k) for k in ("alignment", "contradiction", "rationale")}


# ---------------------------------------------------------------------- run

@dataclass
class ItemResult:
    id: str
    question: str
    covered: bool
    answers: dict = field(default_factory=dict)      # arm -> text
    judgments: dict = field(default_factory=dict)    # arm -> {alignment, contradiction, rationale}
    citeself: dict = field(default_factory=dict)     # status + validation counts
    ceiling: dict | None = None


def _abstained(a: Answer) -> bool:
    return all(s.kind == "no_evidence" for s in a.segments)


def run_eval(persona: Persona, llm: LLM, items: list[EvalItem], arms: list[str] | None = None,
             use_judge: bool = True, progress=None) -> dict:
    arms = list(arms or ARMS)
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise ValueError(f"Unknown arms: {sorted(unknown)}")
    m = persona.manifest
    if "bio_only" in arms and not m.persona.bio:
        arms.remove("bio_only")  # identical to generic without a bio
    results: list[ItemResult] = []
    for n, item in enumerate(items, start=1):
        if progress:
            progress(n, len(items), item.id)
        r = ItemResult(item.id, item.question, item.covered)
        if "citeself" in arms:
            a = persona.ask(item.question)
            r.answers["citeself"] = citeself_text(a)
            v = a.validation
            r.citeself = {"status": a.status, "abstained": _abstained(a), "quotes_verified": v.quotes_verified,
                          "quotes_demoted": v.quotes_demoted, "removed": len(v.removed),
                          "segments": [s.kind for s in a.segments]}
        for arm in ("bio_only", "generic"):
            if arm in arms:
                bio = m.persona.bio if arm == "bio_only" else ""
                c = baseline_answer(llm, m.persona.name, bio, item.question)
                r.answers[arm] = c.text if not c.refused else "[declined]"
        if use_judge:
            for arm, text in r.answers.items():
                r.judgments[arm] = judge(llm, item.question, item.my_answer, text)
            if item.my_answer_retest:
                r.ceiling = judge(llm, item.question, item.my_answer, item.my_answer_retest)
        results.append(r)
    return {"persona": m.persona.name, "model": m.model.name, "arms": arms, "judge": use_judge,
            "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "items": [asdict(r) for r in results], "summary": summarize(results, arms)}


def _mean(xs: list[float]) -> float | None:
    return round(statistics.mean(xs), 3) if xs else None


def summarize(results: list[ItemResult], arms: list[str]) -> dict:
    covered = [r for r in results if r.covered]
    uncovered = [r for r in results if not r.covered]
    out: dict = {"items": len(results), "covered": len(covered), "uncovered": len(uncovered), "arms": {}}
    ceiling_scores = [r.ceiling["alignment"] for r in results if r.ceiling and r.ceiling.get("alignment")]
    ceiling = _mean(ceiling_scores)
    out["ceiling_alignment"] = ceiling
    for arm in arms:
        scores = [r.judgments[arm]["alignment"] for r in covered
                  if arm in r.judgments and r.judgments[arm].get("alignment")]
        contra = [r.judgments[arm]["contradiction"] for r in results
                  if arm in r.judgments and r.judgments[arm].get("contradiction") is not None]
        mean = _mean(scores)
        out["arms"][arm] = {
            "alignment_covered": mean,
            "normalized": round(mean / ceiling, 3) if mean and ceiling else None,
            "contradiction_rate": _mean([float(c) for c in contra]),
        }
    if "citeself" in arms:
        cs = [r.citeself for r in results]
        verified = sum(c["quotes_verified"] for c in cs)
        demoted = sum(c["quotes_demoted"] for c in cs)
        out["citeself"] = {
            # Of the quotes the model proposed, how many survived verbatim checking.
            "proposed_quote_precision": round(verified / (verified + demoted), 3) if verified + demoted else None,
            "quotes_shown": verified,
            "quotes_demoted": demoted,
            "segments_removed": sum(c["removed"] for c in cs),
            # Desired: abstain when your material doesn't cover it, answer when it does.
            "abstention_on_uncovered": _mean([float(r.citeself["abstained"]) for r in uncovered]),
            "false_abstention_on_covered": _mean([float(r.citeself["abstained"]) for r in covered]),
        }
    return out


# ------------------------------------------------------------ blind rating

LABELS = "XYZ"


def write_outputs(root: Path, run: dict, seed: int | None = None) -> Path:
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = root / DATA_DIR / "eval" / stamp
    out.mkdir(parents=True, exist_ok=False)
    (out / "results.json").write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "report.md").write_text(render_report(run), encoding="utf-8")

    rng = random.Random(seed)
    key: dict = {}
    with (out / "rating_sheet.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "question", "response", "text", "rating_1_to_5"])
        for item in run["items"]:
            arms = list(item["answers"])
            rng.shuffle(arms)
            key[item["id"]] = {}
            for label, arm in zip(LABELS, arms):
                key[item["id"]][label] = arm
                w.writerow([item["id"], item["question"], label, item["answers"][arm], ""])
    (out / "rating_key.json").write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
    return out


def score_ratings(run_dir: Path) -> dict:
    key = json.loads((run_dir / "rating_key.json").read_text(encoding="utf-8"))
    per_arm: dict[str, list[int]] = {}
    wins: dict[str, float] = {}
    rated_items = 0
    by_item: dict[str, dict[str, int]] = {}
    with (run_dir / "rating_sheet.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = (row.get("rating_1_to_5") or "").strip()
            if not raw:
                continue
            try:
                rating = int(raw)
            except ValueError as exc:
                raise ValueError(f"item {row['item_id']} response {row['response']}: rating must be 1-5") from exc
            if not 1 <= rating <= 5:
                raise ValueError(f"item {row['item_id']} response {row['response']}: rating must be 1-5")
            arm = key.get(row["item_id"], {}).get(row["response"])
            if arm is None:
                raise ValueError(f"Unknown item/response {row['item_id']}/{row['response']} (sheet edited?)")
            per_arm.setdefault(arm, []).append(rating)
            by_item.setdefault(row["item_id"], {})[arm] = rating
    for ratings in by_item.values():
        if len(ratings) < 2:
            continue
        rated_items += 1
        best = max(ratings.values())
        winners = [a for a, v in ratings.items() if v == best]
        for a in winners:
            wins[a] = wins.get(a, 0) + 1 / len(winners)  # ties share the win
    return {
        "items_compared": rated_items,
        "arms": {arm: {"mean_rating": _mean([float(x) for x in xs]), "ratings": len(xs),
                       "win_rate": round(wins.get(arm, 0) / rated_items, 3) if rated_items else None}
                 for arm, xs in sorted(per_arm.items())},
    }


def render_report(run: dict) -> str:
    s = run["summary"]
    lines = [f"# CiteSelf evaluation: {run['persona']}", "",
             f"{run['created_at']} · model `{run['model']}` · {s['items']} questions "
             f"({s['covered']} covered by your material, {s['uncovered']} not)", ""]
    if run["judge"]:
        lines += ["## Alignment with your own answers (LLM judge, covered questions)", "",
                  "| Arm | Alignment (1-5) | Normalized to your consistency | Contradiction rate |", "|---|---|---|---|"]
        for arm, a in s["arms"].items():
            lines.append(f"| {arm} | {a['alignment_covered']} | {a['normalized'] if a['normalized'] is not None else 'n/a'} "
                         f"| {a['contradiction_rate']} |")
        lines += ["", f"Your own consistency (re-answers vs first answers): {s['ceiling_alignment'] or 'n/a (add my_answer_retest)'}", ""]
    if "citeself" in s:
        c = s["citeself"]
        lines += ["## Grounding (CiteSelf only)", "",
                  f"- Proposed quotes that were verbatim: {c['proposed_quote_precision']} "
                  f"({c['quotes_shown']} shown, {c['quotes_demoted']} relabeled as extrapolation)",
                  f"- Unsupported segments removed: {c['segments_removed']}",
                  f"- Said \"no evidence\" when your material doesn't cover it: {c['abstention_on_uncovered']}",
                  f"- Said \"no evidence\" when it does (false abstention): {c['false_abstention_on_covered']}", ""]
    lines += ["## Caveats", "",
              "- The judge is an LLM and can be wrong. Your blind ratings (`rating_sheet.csv`) are the stronger signal.",
              "- Small question sets give noisy averages. Treat differences under ~0.5 points as inconclusive.",
              "- The `generic` and `bio_only` arms always answer; CiteSelf may decline. Compare abstention numbers too.", ""]
    return "\n".join(lines)
