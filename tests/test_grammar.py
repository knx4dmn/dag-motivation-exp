"""CPU-only tests for grammar: template derivations, EBNF shape/escaping, conformance,
exemplar synthesis + parse-under-own-grammar, and exemplar-not-in-any-bucket (R2 #2)."""
from __future__ import annotations

import pytest

from motivation_exp import grammar as gr
from motivation_exp import datagen as dg
from motivation_exp.datagen import Item


ENTITIES = ["Wren", "Sprocket"]
CONCEPTS = ["tumpus", "wumpus", "impus"]  # impus is vowel-initial -> "an impus"


# --------------------------------------------------------------------------------------
# template derivations stay mutually consistent (single source of truth)
# --------------------------------------------------------------------------------------
def test_render_template_article_agreement():
    assert gr.render_template("entity_pos", ["Wren", "tumpus"]) == "Wren is a tumpus."
    assert gr.render_template("entity_pos", ["Wren", "impus"]) == "Wren is an impus."
    assert gr.render_template("rule_pos", ["tumpus", "wumpus"]) == "Every tumpus is a wumpus."
    assert gr.render_template("rule_neg", ["tumpus", "impus"]) == "Every tumpus is not an impus."


def test_rendered_templates_match_their_own_regex():
    for tid in gr.STEP_TEMPLATES:
        # build a filler list of the right length using in-vocab values
        n_slots = len(gr._slot_indices(gr.STEP_TEMPLATES[tid]))
        fillers = []
        for i in gr._slot_indices(gr.STEP_TEMPLATES[tid]):
            fillers.append("Wren" if gr.STEP_TEMPLATES[tid][i] == gr.NAME else "tumpus")
        sentence = gr.render_template(tid, fillers[:n_slots])
        assert gr._TEMPLATE_REGEXES[tid].match(sentence), (tid, sentence)


# --------------------------------------------------------------------------------------
# EBNF build
# --------------------------------------------------------------------------------------
def test_build_item_ebnf_shape_and_vocab():
    ebnf = gr.build_item_ebnf(ENTITIES, CONCEPTS)
    assert "root ::= step+ answer" in ebnf
    assert '("True" | "False")' in ebnf
    assert '"Wren"' in ebnf and '"Sprocket"' in ebnf
    assert '"tumpus"' in ebnf and '"impus"' in ebnf
    assert "name ::=" in ebnf and "concept ::=" in ebnf


def test_build_item_ebnf_drops_name_templates_without_entities():
    ebnf = gr.build_item_ebnf([], CONCEPTS)
    # no name terminal rule and no NAME-based step should appear
    assert "name ::=" not in ebnf
    assert "root ::= step+ answer" in ebnf


def test_build_item_ebnf_escapes_metacharacters():
    ebnf = gr.build_item_ebnf(['Wr"en'], ["tum\\pus"])
    assert '"Wr\\"en"' in ebnf
    assert '"tum\\\\pus"' in ebnf


def test_build_item_ebnf_empty_concepts_raises():
    with pytest.raises(ValueError):
        gr.build_item_ebnf(ENTITIES, [])


# --------------------------------------------------------------------------------------
# conformance mirrors the grammar
# --------------------------------------------------------------------------------------
def test_output_conforms_accepts_valid_output():
    text = "Wren is a tumpus.\nWren is a wumpus.\nThe answer is True."
    assert gr.output_conforms(text, ENTITIES, CONCEPTS)


def test_output_conforms_rejects_out_of_vocab_concept():
    text = "Wren is a zzzpus.\nThe answer is True."  # zzzpus not in vocab
    assert not gr.output_conforms(text, ENTITIES, CONCEPTS)


def test_output_conforms_rejects_missing_answer_line():
    text = "Wren is a tumpus.\nWren is a wumpus."
    assert not gr.output_conforms(text, ENTITIES, CONCEPTS)


def test_output_conforms_rejects_bad_answer_token():
    text = "Wren is a tumpus.\nThe answer is Maybe."
    assert not gr.output_conforms(text, ENTITIES, CONCEPTS)


# --------------------------------------------------------------------------------------
# exemplar synthesis
# --------------------------------------------------------------------------------------
def _exemplar_item() -> Item:
    # independent ontology, its own vocabulary; gold_steps in templatic form
    return Item(
        item_id="ex0", base_id="ex0",
        context=["Alph is a florp.", "Every florp is a glorp."],
        question="Alph is a glorp.",
        gold=True,
        entities=["Alph"],
        concepts=["florp", "glorp"],
        gold_steps=["Alph is a florp.", "Alph is a glorp."],
        n_hops=2,
    )


def test_synthesized_exemplar_conforms_to_its_own_grammar():
    ex = _exemplar_item()
    cot = gr.synthesize_exemplar_cot(ex)
    # parses to completion under a conformance check driven by the SAME templates
    assert gr.output_conforms(cot, ex.entities, ex.concepts)
    assert cot.strip().endswith("The answer is True.")


def test_exemplar_normalizes_upstream_phrasing():
    # upstream uses "an" and odd spacing; normalization must canonicalize to grammar form
    ex = _exemplar_item()
    ex.gold_steps = ["Alph is an florp.", "Alph is a glorp."]
    cot = gr.synthesize_exemplar_cot(ex)
    assert "Alph is a florp." in cot  # "an florp" -> "a florp"
    assert gr.output_conforms(cot, ex.entities, ex.concepts)


def test_exemplar_base_id_absent_from_all_buckets(tmp_path):
    """R2 #2: the exemplar item must never appear in any bucket."""
    def wc(t: str) -> int:
        return len(t.split())

    # 15 base items -> reserve 1 exemplar + 10 calib, bucket the rest.
    # Vocabulary is letters-only (as real PrOntoQA is): NAME=[A-Z][a-zA-Z]*, CONCEPT=[a-z][a-zA-Z]*.
    import string
    bases = []
    for i in range(15):
        suf = string.ascii_lowercase[i]  # a..o, unique per item
        ent, c1, c2 = f"Ent{suf.upper()}", f"florp{suf}", f"glorp{suf}"
        raw = {
            "context": [f"{ent} is a {c1}.", f"Every {c1} is a {c2}."],
            "query": f"{ent} is a {c2}.",
            "answer": "True",
            "chain_of_thought": [f"{ent} is a {c1}.", f"{ent} is a {c2}."],
        }
        bases.append(dg.raw_prontoqa_adapter(raw, item_id=f"i{i}", base_id=f"b{i}"))

    bucket_items, calib, exemplar = dg.reserve_splits(bases, n_calib=10, n_exemplar=1)
    pool = [f"Dorp{string.ascii_uppercase[j % 26]} is a quz{string.ascii_lowercase[j % 26]}." for j in range(50)]
    buckets = dg.bucket_to_token_targets(bucket_items, wc, pool, targets=[30, 50])

    exemplar_ids = {e.base_id for e in exemplar}
    for b, its in buckets.items():
        for it in its:
            assert it.base_id not in exemplar_ids
    # and the exemplar's CoT still parses
    assert gr.synthesize_exemplar_cot(exemplar[0])
