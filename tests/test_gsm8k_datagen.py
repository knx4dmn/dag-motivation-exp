"""CPU-only tests for GSM8K datagen: parsing, safe rational eval, distractors, bucketing, validation."""
from __future__ import annotations

from fractions import Fraction

import pytest

from motivation_exp import gsm8k_datagen as g


def wc(t: str) -> int:
    return len(t.split())


# a realistic GSM8K-format pair (calculator annotations + #### answer)
Q = ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
     "How many clips did Natalia sell altogether in April and May?")
A = ("In April, Natalia sold 48 clips.\n"
     "In May, she sold 48/2 = <<48/2=24>>24 clips.\n"
     "Altogether she sold 48+24 = <<48+24=72>>72 clips.\n#### 72")


# --------------------------------------------------------------------------------------
# numbers + arithmetic
# --------------------------------------------------------------------------------------
def test_safe_eval_exact_rational():
    assert g.safe_eval("48/2") == Fraction(24)
    assert g.safe_eval("1/3") == Fraction(1, 3)
    assert g.safe_eval("2 + 3*4") == Fraction(14)
    assert g.safe_eval("(10-4)/2") == Fraction(3)
    assert g.safe_eval("3.5*2") == Fraction(7)


def test_safe_eval_rejects_non_arithmetic():
    for bad in ["__import__('os')", "a+b", "2**3", "len([1])"]:
        with pytest.raises((ValueError, SyntaxError)):
            g.safe_eval(bad)


def test_number_extraction_and_normalization():
    assert g.extract_numbers("She had 1,000 apples and $5 and 3.5 kg.") == ["1000", "5", "3.5"]
    assert g.to_fraction("1,000") == Fraction(1000)
    assert g.to_fraction("$5") == Fraction(5)


# --------------------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------------------
def test_parse_gsm8k_item():
    it = g.parse_gsm8k_item(Q, A, item_id="gsm0")
    assert it.gold == "72"
    assert it.gold_steps == ["48/2 = 24", "48+24 = 72"]
    assert it.question.endswith("?") and "How many" in it.question
    assert "48" in it.problem_quantities
    assert "Natalia" in it.names
    assert it.relevance == [1] * len(it.context)         # all-original before bucketing
    assert it.gold_context == it.context


def test_parse_missing_answer_raises():
    with pytest.raises(ValueError):
        g.parse_gsm8k_item("Q?", "no final answer here", item_id="bad")


# --------------------------------------------------------------------------------------
# distractors
# --------------------------------------------------------------------------------------
def test_distractors_seeded_unique_use_item_names():
    it = g.parse_gsm8k_item(Q, A, item_id="gsm0")
    d1 = g.generate_distractors(it, 30, seed=0)
    d2 = g.generate_distractors(it, 30, seed=0)
    assert d1 == d2                                       # seeded reproducible
    assert len(set(d1)) == len(d1)                        # no dup within item
    assert it.names == ["Natalia"]                        # months excluded from the name pool
    assert all(any(nm in s for nm in it.names) for s in d1)   # names from the item's entity pool
    assert all(s.endswith(".") for s in d1)
    # amendment 2: no distractor number collides with the item's problem quantities (or whitelist)
    from fractions import Fraction
    prob = {Fraction(q) for q in it.problem_quantities}
    for s in d1:
        for num in g.extract_numbers(s):
            assert Fraction(num) not in prob and not g.is_whitelist_constant(num)


# --------------------------------------------------------------------------------------
# bucketing + validation
# --------------------------------------------------------------------------------------
def test_bucketing_hits_target_tracks_relevance_and_validates():
    it = g.parse_gsm8k_item(Q, A, item_id="gsm0")
    b = g.bucket_gsm_item(it, bucket=120, count_tokens=wc, seed=1)
    assert b.bucket == 120 and 114 <= b.token_count <= 126
    assert len(b.relevance) == len(b.context)
    # original problem sentences preserved with relevance 1; distractors relevance 0
    gold_set = set(b.gold_context)
    for s, r in zip(b.context, b.relevance):
        assert (s in gold_set) == (r == 1)
    assert sum(b.relevance) == len(b.gold_context)        # all originals present
    assert 0 in b.relevance                               # distractors injected
    ok, why = g.validate_gsm_item(b, wc)
    assert ok, why
    # question stays last
    assert g._assemble(b.context, b.question).rstrip().endswith(b.question)


def test_validate_flags_arithmetic_and_tolerance():
    it = g.parse_gsm8k_item(Q, A, item_id="gsm0")
    b = g.bucket_gsm_item(it, bucket=120, count_tokens=wc, seed=1)
    b.gold_steps = ["48/2 = 25"]                          # wrong arithmetic
    ok, why = g.validate_gsm_item(b, wc)
    assert not ok and "arithmetic" in why
    b2 = g.bucket_gsm_item(it, bucket=120, count_tokens=wc, seed=1)
    b2.token_count = 10
    ok2, why2 = g.validate_gsm_item(b2, wc)
    assert not ok2 and "outside" in why2


def test_bucket_targets_and_io(tmp_path):
    items = [g.parse_gsm8k_item(Q, A, item_id=f"gsm{i}", base_id=f"b{i}") for i in range(3)]
    buckets = g.bucket_gsm_to_targets(items, wc, targets=[80, 160], base_seed=0)
    assert set(buckets) == {80, 160} and len(buckets[80]) == 3
    p = tmp_path / "items.jsonl"
    for it in buckets[160]:
        g.append_gsm_item(str(p), it)
    loaded = g.load_gsm_items(str(p))
    assert len(loaded) == 3 and loaded[0].gold == "72"
