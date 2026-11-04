from citeself.store import Evidence
from citeself.validate import FALLBACK_NO_EVIDENCE, validate_segments

EV = {
    "E1": Evidence("c1", "s1", "Talk", "2026-03-21", "p", "I do not want teams to hide every risk until launch day. "
                   "Release quality depends on the severity of defects and whether we have a safe rollback."),
    "E2": Evidence("c2", "s2", "Memo", None, "p", "I prefer a written decision memo that lays out assumptions."),
}


def seg(kind, text, ids=()):
    return {"kind": kind, "text": text, "evidence_ids": list(ids)}


def test_verbatim_quote_is_verified():
    out, rep = validate_segments([seg("quote", "Release quality depends on the severity of defects", ["E1"])], EV)
    assert out[0].kind == "quote" and out[0].verified and rep.quotes_verified == 1


def test_typographic_differences_and_wrapping_quotes_still_verify():
    out, _ = validate_segments([seg("quote", "“I do not want teams to hide  every risk”", ["E1"])], EV)
    assert out[0].verified and out[0].text == "I do not want teams to hide  every risk"


def test_fabricated_quote_is_demoted_never_shown_as_quote():
    out, rep = validate_segments([seg("quote", "I always ship on Fridays no matter what", ["E1"])], EV)
    assert out[0].kind == "extrapolation" and not out[0].verified
    assert "not found verbatim" in out[0].note and rep.quotes_demoted == 1


def test_paraphrase_with_one_word_changed_is_demoted():
    out, _ = validate_segments([seg("quote", "Release quality depends on the severity of bugs", ["E1"])], EV)
    assert out[0].kind == "extrapolation"


def test_quote_from_wrong_source_is_demoted():
    out, _ = validate_segments([seg("quote", "I prefer a written decision memo", ["E1"])], EV)
    assert out[0].kind == "extrapolation"


def test_short_quote_is_demoted():
    out, _ = validate_segments([seg("quote", "safe rollback", ["E1"])], EV, min_quote_words=3)
    assert out[0].kind == "extrapolation" and "minimum length" in out[0].note


def test_citations_narrowed_to_sources_containing_quote():
    out, _ = validate_segments([seg("quote", "I prefer a written decision memo", ["E1", "E2"])], EV)
    assert out[0].evidence_ids == ["E2"]


def test_fabricated_citation_stripped_and_unsupported_extrapolation_removed():
    out, rep = validate_segments([seg("extrapolation", "They would likely approve.", ["E9"])], EV)
    assert rep.stripped_citations and rep.removed
    assert out[0].kind == "no_evidence" and out[0].text == FALLBACK_NO_EVIDENCE


def test_extrapolation_with_valid_citation_kept():
    out, _ = validate_segments([seg("extrapolation", "They would want a rollback plan.", ["E1", "E1", "E7"])], EV)
    assert out[0].kind == "extrapolation" and out[0].evidence_ids == ["E1"]


def test_no_evidence_never_carries_citations():
    out, _ = validate_segments([seg("no_evidence", "Nothing on record.", ["E1"])], EV)
    assert out[0].evidence_ids == []


def test_extrapolation_disabled_by_owner():
    out, rep = validate_segments([seg("extrapolation", "x y z", ["E1"]),
                                  seg("quote", "Release quality depends on the severity of defects", ["E1"])],
                                 EV, allow_extrapolation=False)
    assert [s.kind for s in out] == ["quote"] and rep.removed


def test_garbage_inputs():
    for raw in (None, "text", 5, [], [1, "x", {"kind": "rant", "text": "x"}, {"kind": "quote", "text": "  "}]):
        out, _ = validate_segments(raw, EV)
        assert len(out) == 1 and out[0].kind == "no_evidence"


def test_quote_from_ai_assisted_source_is_demoted():
    ev = {"E1": Evidence("c9", "s9", "Draft", None, "p", "We put customers first in every decision we make.",
                         authorship="ai_assisted")}
    out, rep = validate_segments([seg("quote", "We put customers first in every decision", ["E1"])], ev)
    assert out[0].kind == "extrapolation" and "AI-assisted" in out[0].note and rep.quotes_demoted == 1
