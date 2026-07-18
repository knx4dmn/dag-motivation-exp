"""CPU-only tests for grammar: clause parser, EBNF build/escape, conformance (concept +
property forms), and exemplar synthesis + parse-under-own-grammar."""
from __future__ import annotations

import pytest

from motivation_exp import grammar as gr
from motivation_exp.datagen import Item


ENTITIES = ["Wren", "Sprocket"]
CONCEPTS = ["tumpus", "wumpus", "impus"]
PROPERTIES = ["slow", "shy"]


# --------------------------------------------------------------------------------------
# clause parser
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("sent,kind,subj,pred,is_prop,neg", [
    ("Wren is a tumpus.", "fact", "Wren", "tumpus", False, False),
    ("Wren is an impus.", "fact", "Wren", "impus", False, False),
    ("Wren is not slow.", "fact", "Wren", "slow", True, True),
    ("Every tumpus is a wumpus.", "rule", "tumpus", "wumpus", False, False),
    ("Each tumpus is not a wumpus.", "rule", "tumpus", "wumpus", False, True),
    ("Each yumpus is shy.", "rule", "yumpus", "shy", True, False),
    ("Tumpuses are wumpuses.", "rule", "tumpus", "wumpus", False, False),
    ("Tumpuses are not slow.", "rule", "tumpus", "slow", True, True),
])
def test_parse_clause_forms(sent, kind, subj, pred, is_prop, neg):
    c = gr.parse_clause(sent)
    assert c is not None
    assert (c.kind, c.subject, c.pred, c.is_property, c.negated) == (kind, subj, pred, is_prop, neg)


def test_parse_clause_rejects_nonclause():
    assert gr.parse_clause("The answer is True.") is None
    assert gr.parse_clause("nonsense here") is None


# --------------------------------------------------------------------------------------
# render_fact (used by checker MP synthesis)
# --------------------------------------------------------------------------------------
def test_render_fact_article_and_property():
    assert gr.render_fact("Wren", "tumpus", False, False) == "Wren is a tumpus."
    assert gr.render_fact("Wren", "impus", False, False) == "Wren is an impus."
    assert gr.render_fact("Wren", "slow", True, True) == "Wren is not slow."


def test_negate_and_swap_helpers():
    assert gr.negate_sentence("Wren is slow.") == "Wren is not slow."
    assert gr.negate_sentence("Wren is not a tumpus.") == "Wren is a tumpus."
    assert gr.swap_concept("Wren is a tumpus.", "wumpus") == "Wren is a wumpus."
    assert gr.swap_concept("Wren is slow.", "wumpus") is None  # property clause -> no concept swap


def test_paraphrase_parses_equal_for_guard():
    # the checker's guard tolerates paraphrase because both parse to the same Clause
    assert gr.parse_clause("Tumpuses are wumpuses.") == gr.parse_clause("Every tumpus is a wumpus.")


def test_strip_connectives_recovers_parse():
    # parse_clause fails on connective-prefixed steps; strip_connectives recovers them
    for phrasing in ["So Wren is a wumpus.", "Therefore, Wren is a wumpus.", "Thus Wren is a wumpus.",
                     "We know that Wren is a wumpus.", "Wren is therefore a wumpus.",
                     "Wren is a wumpus, so it is not slow."]:
        assert gr.parse_clause(phrasing) is None                       # raw: fails
        assert gr.parse_clause(gr.strip_connectives(phrasing)) == gr.parse_clause("Wren is a wumpus.")
    assert gr.strip_connectives("Wren is a wumpus.") == "Wren is a wumpus."  # canonical unchanged


# every canonical clause form, incl. negation / plural / property
_CANONICAL = [
    "Wren is a tumpus.", "Wren is an impus.", "Wren is not a tumpus.",
    "Wren is slow.", "Wren is not slow.",
    "Every tumpus is a wumpus.", "Each tumpus is not a wumpus.",
    "Every tumpus is shy.", "Each tumpus is not shy.",
    "Tumpuses are wumpuses.", "Tumpuses are not slow.",
]


def test_strip_connectives_is_clause_invariant_on_canonical_forms():
    """strip_connectives must NOT change the parsed Clause of an already-canonical step -- in
    particular it must never touch polarity (negated) or the predicate."""
    for s in _CANONICAL:
        assert gr.parse_clause(gr.strip_connectives(s)) == gr.parse_clause(s), s


def test_strip_connectives_never_removes_negation():
    # a connective in front of a NEGATED step: strip the connective, keep 'not' and the predicate
    for s in _CANONICAL:
        c = gr.parse_clause(s)
        for lead in ("So ", "Therefore, ", "Thus ", "We know that "):
            stripped = gr.parse_clause(gr.strip_connectives(lead + s[0].lower() + s[1:]))
            assert stripped == c, (lead, s)          # same subject/pred/is_property/NEGATED
    # explicit: negation survives
    assert gr.parse_clause(gr.strip_connectives("So Sally is not slow.")) == gr.parse_clause("Sally is not slow.")
    assert "not" in gr.strip_connectives("Therefore, Sally is not slow.")


def test_strip_connectives_does_not_over_strip_concepts_starting_like_connectives():
    # a concept/name that merely STARTS with connective letters must not be truncated
    assert gr.strip_connectives("Sopus is a wumpus.") == "Sopus is a wumpus."   # 'So' not a prefix token
    assert gr.strip_connectives("Thenpus is a wumpus.") == "Thenpus is a wumpus."


# --------------------------------------------------------------------------------------
# EBNF build
# --------------------------------------------------------------------------------------
def test_build_item_ebnf_covers_concepts_properties_plurals():
    ebnf = gr.build_item_ebnf(ENTITIES, CONCEPTS, PROPERTIES)
    assert "root ::= step+ answer" in ebnf
    assert '("True" | "False")' in ebnf
    assert '"Wren"' in ebnf
    assert '"tumpus"' in ebnf and '"tumpuses"' in ebnf and '"Tumpuses"' in ebnf  # sing + plural forms
    assert '"slow"' in ebnf and "property ::=" in ebnf


def test_build_item_ebnf_no_entities_drops_name_rule():
    ebnf = gr.build_item_ebnf([], CONCEPTS, PROPERTIES)
    assert "name ::=" not in ebnf
    assert "root ::= step+ answer" in ebnf


def test_build_item_ebnf_escapes_metacharacters():
    ebnf = gr.build_item_ebnf(['Wr"en'], ["tum\\pus"], [])
    assert '"Wr\\"en"' in ebnf
    assert '"tum\\\\pus"' in ebnf


def test_build_item_ebnf_empty_raises():
    with pytest.raises(ValueError):
        gr.build_item_ebnf([], [], [])


# --------------------------------------------------------------------------------------
# conformance
# --------------------------------------------------------------------------------------
def test_output_conforms_accepts_concept_and_property_steps():
    text = ("Wren is a tumpus.\nEvery tumpus is a wumpus.\nWren is a wumpus.\n"
            "Wumpuses are not slow.\nWren is not slow.\nThe answer is True.")
    assert gr.output_conforms(text, ENTITIES, CONCEPTS + ["wumpus"], PROPERTIES)


def test_output_conforms_rejects_out_of_vocab():
    assert not gr.output_conforms("Wren is a zzzpus.\nThe answer is True.", ENTITIES, CONCEPTS, PROPERTIES)
    assert not gr.output_conforms("Wren is fast.\nThe answer is True.", ENTITIES, CONCEPTS, PROPERTIES)  # 'fast' not in props


def test_output_conforms_rejects_missing_or_bad_answer():
    assert not gr.output_conforms("Wren is a tumpus.", ENTITIES, CONCEPTS, PROPERTIES)
    assert not gr.output_conforms("Wren is a tumpus.\nThe answer is Maybe.", ENTITIES, CONCEPTS, PROPERTIES)


# --------------------------------------------------------------------------------------
# exemplar synthesis
# --------------------------------------------------------------------------------------
def _exemplar_item():
    # realistic PrOntoQA morphology: concepts end in -us, plural -uses
    return Item(
        item_id="ex0", base_id="ex0",
        context=["Alph is a florpus.", "Every florpus is a glorpus.", "Glorpuses are not shy."],
        question="Alph is not shy.",
        gold=True,
        entities=["Alph"],
        concepts=["florpus", "glorpus"],
        properties=["shy"],
        gold_steps=["Alph is a florpus.", "Alph is a glorpus.", "Glorpuses are not shy.", "Alph is not shy."],
        n_hops=4,
    )


def test_synthesized_exemplar_conforms():
    ex = _exemplar_item()
    cot = gr.synthesize_exemplar_cot(ex)
    assert gr.output_conforms(cot, ex.entities, ex.concepts, ex.properties)
    assert cot.strip().endswith("The answer is True.")


def test_exemplar_raises_on_nonconforming_step():
    ex = _exemplar_item()
    ex.gold_steps = ["Alph flarbles the glorp."]  # not a valid clause
    with pytest.raises(ValueError):
        gr.synthesize_exemplar_cot(ex)
