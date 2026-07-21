"""CPU-only tests for the G0 core: number extraction, post-hoc step verdicts, attributability."""
from __future__ import annotations

from motivation_exp.g0_gsm8k import extract_number_answer, g0_item, g0_aggregate
from motivation_exp.gsm8k_datagen import GSMItem


def _item():
    return GSMItem(
        item_id="gsm0", base_id="gsm0",
        context=["Natalia sold 48 clips in April.", "She sold half as many in May.",
                 "Natalia has 99 marbles at home."],   # last = distractor (relevance 0)
        relevance=[1, 1, 0], question="How many clips did she sell altogether?",
        gold="72", gold_steps=["48/2 = 24", "48+24 = 72"], problem_quantities=["48"],
        names=["Natalia"], gold_context=["Natalia sold 48 clips in April.", "She sold half as many in May.",
                                         "Natalia has 99 marbles at home."], bucket=512,
    )


def test_extract_number_answer():
    assert extract_number_answer("... #### 72") == "72"
    assert extract_number_answer("So the answer is 72.") == "72"
    assert extract_number_answer("we get 24 then 72 total") == "72"   # last number fallback
    assert extract_number_answer("no numbers here") is None


def test_g0_item_correct_all_steps_pass():
    cot = "In May she sold 48 / 2 = 24 clips.\nAltogether she sold 48 + 24 = 72 clips.\n#### 72"
    r = g0_item(_item(), cot)
    assert r["n_checkable"] == 2 and r["n_pass"] == 2
    assert r["correct"] is True and r["has_reject"] is False


def test_g0_item_distractor_use_is_catchable():
    cot = "She sold 99 / 3 = 33 clips.\n#### 33"     # 99 is the distractor quantity
    r = g0_item(_item(), cot)
    assert r["correct"] is False and r["has_reject"] is True
    assert r["first_reject"][1] == "provenance_distractor"


def test_g0_item_checker_invisible_wrong_plan():
    cot = "She sold 48 * 2 = 96 clips.\n#### 96"       # clean arithmetic, valid provenance, wrong plan
    r = g0_item(_item(), cot)
    assert r["n_pass"] == 1 and r["correct"] is False and r["has_reject"] is False


def test_g0_aggregate_step_pass_and_catchable():
    it = _item()
    rows = [
        g0_item(it, "In May she sold 48 / 2 = 24 clips.\n48 + 24 = 72\n#### 72"),  # correct, 2 pass
        g0_item(it, "She sold 99 / 3 = 33 clips.\n#### 33"),                        # wrong, catchable
        g0_item(it, "She sold 48 * 2 = 96 clips.\n#### 96"),                        # wrong, invisible
    ]
    agg = g0_aggregate(rows)[512]
    assert agg["n_items"] == 3 and agg["n_wrong"] == 2
    assert agg["step_pass"] == 3 / 4                    # (2 + 0 + 1) / (2 + 1 + 1)
    assert agg["catchable_frac"] == 1 / 2               # 1 of 2 wrong items had a rejected step
