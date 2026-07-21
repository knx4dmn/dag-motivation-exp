"""CPU-only tests for the format-only calculation grammar (no XGrammar needed).

Checks the EBNF is well-formed-ish (expected rules present) and that the pure-Python conformance
mirror ``calc_output_conforms`` accepts bare calculation output and rejects prose / malformed output.
"""
from __future__ import annotations

from motivation_exp.gsm8k_grammar import build_calc_ebnf, calc_output_conforms
from motivation_exp.gsm8k_runner import EXEMPLAR_COT_BARE


def test_ebnf_has_expected_rules():
    ebnf = build_calc_ebnf()
    for rule in ("root ::=", "line ::=", "expr ::=", "op ::=", "number ::=", "answer ::="):
        assert rule in ebnf
    assert '"#### "' in ebnf          # answer prefix
    assert "[0-9]+" in ebnf           # free-digit operands/results (not a context-number list)


def test_bare_exemplar_conforms():
    # the symbolic arm's few-shot exemplar must be valid under its own grammar (no phrasing drift)
    assert calc_output_conforms(EXEMPLAR_COT_BARE)


def test_conforms_accepts_bare_calculations():
    assert calc_output_conforms("48 / 2 = 24\n48 + 24 = 72\n#### 72")
    assert calc_output_conforms("3 * 4 = 12\n#### 12")
    assert calc_output_conforms("1 + 2 + 3 = 6\n#### 6")     # chained operators


def test_conforms_rejects_prose_and_malformed():
    assert not calc_output_conforms("In May she sold 48 / 2 = 24 clips.\n#### 24")  # prose line
    assert not calc_output_conforms("48 / 2 = 24\n72")        # no #### answer
    assert not calc_output_conforms("5 = 5\n#### 5")          # no operator in expr
    assert not calc_output_conforms("#### 72")                # no calculation line
    assert not calc_output_conforms("48 / 2 = 24")            # answer line missing
