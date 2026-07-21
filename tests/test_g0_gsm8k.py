"""CPU-only tests for the G0 core: number extraction, post-hoc step verdicts, attributability."""
from __future__ import annotations

from motivation_exp.g0_gsm8k import extract_number_answer, g0_item, g0_aggregate, g0_verdict
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


def test_g0_aggregate_decomposition_parse_pooled_and_verdict():
    it = _item()
    rows = [
        g0_item(it, "In May she sold 48 / 2 = 24 clips.\n48 + 24 = 72\n#### 72"),  # correct, 2 pass
        g0_item(it, "She sold 99 / 3 = 33 clips.\n#### 33"),                        # wrong, distractor
        g0_item(it, "She sold 48 * 2 = 96 clips.\n#### 96"),                        # wrong, invisible
    ]
    agg = g0_aggregate(rows)
    a = agg[512]
    assert a["step_pass"] == 3 / 4 and a["catchable_frac"] == 1 / 2
    assert a["parse_rate"] == 1.0                       # every item has >=1 calculation
    assert (a["fail_missing"], a["fail_distractor"], a["fail_arith"]) == (0, 1, 0)
    assert agg["_pooled"]["catchable_frac"] == 1 / 2
    # healthy mix (distractor dominates missing) + catchable >= 1/3 -> PROCEED
    assert g0_verdict(agg) == "PROCEED"


def test_g0_verdict_stop_a_when_missing_dominates():
    it = _item()
    rows = [g0_item(it, "She sold 17 * 5 = 85 clips.\n#### 85")   # 17 and 85 both hallucinated
            for _ in range(3)]
    agg = g0_aggregate(rows)
    assert agg["_pooled"]["fail_missing"] > 0
    assert g0_verdict(agg) == "STOP_A"                  # missing class dominates -> model mismatch


def test_posthoc_no_cascade_from_arithmetic_failure():
    # step1 fails arithmetic (48/2 != 25); step2 uses the model's derived 25 -> must NOT be 'missing'
    it = _item()
    r = g0_item(it, "In May she sold 48 / 2 = 25 clips.\nThen 25 + 48 = 73 clips.\n#### 73")
    v = r["verdicts"]
    assert v[0]["accepted"] is False and v[0]["failed"] == "arithmetic"   # charged only its own error
    assert v[1]["accepted"] is True                                       # 25 recorded -> no cascade


def test_g0_verdict_fix_prompting_on_low_parse():
    it = _item()
    rows = [g0_item(it, "Let me think about this problem carefully. The answer is 5.")   # no calc line
            for _ in range(3)]
    agg = g0_aggregate(rows)
    assert agg[512]["parse_rate"] == 0.0
    assert g0_verdict(agg) == "FIX_PROMPTING"
