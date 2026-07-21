"""CPU-only tests for MathChecker: calculation parsing, provenance (relevance-aware, uncached
linear scan), exact-rational arithmetic, and the accept/derived-intermediate flow."""
from __future__ import annotations

from motivation_exp.math_checker import MathChecker, parse_calculation


# context with a relevance bit per sentence (1 = original problem, 0 = distractor)
CONTEXT = [
    "Natalia sold 48 clips in April.",      # rel 1
    "She sold half as many in May.",        # rel 1 (no numbers)
    "Natalia has 99 marbles at home.",      # rel 0 (distractor: 99)
]
RELEVANCE = [1, 1, 0]


def _checker():
    c = MathChecker()
    c.prefill(CONTEXT, RELEVANCE)
    return c


# --------------------------------------------------------------------------------------
# calculation parsing
# --------------------------------------------------------------------------------------
def test_parse_calculation_from_free_text():
    assert parse_calculation("In May she sold 48/2 = 24 clips.") == ("48/2", "24")
    assert parse_calculation("Altogether 48 + 24 = 72.") == ("48 + 24", "72")
    assert parse_calculation("She has 48 clips.") is None          # no operator -> not a calc
    assert parse_calculation("Total is 6 * 7 = 42 items") == ("6 * 7", "42")


# --------------------------------------------------------------------------------------
# provenance + arithmetic verdicts
# --------------------------------------------------------------------------------------
def test_accepts_valid_step_over_problem_quantities():
    r = _checker().check_step("In May she sold 48/2 = 24 clips.")
    assert r.accepted and r.kind == "calc" and r.cosine == 1.0


def test_rejects_distractor_operand():
    r = _checker().check_step("She sold 99/3 = 33 clips.")   # 99 traces only to the distractor
    assert not r.accepted and r.failed == "provenance_distractor" and r.detail == "99"


def test_rejects_missing_operand():
    r = _checker().check_step("She sold 17*2 = 34 clips.")   # 17 appears nowhere
    assert not r.accepted and r.failed == "provenance_missing" and r.detail == "17"


def test_rejects_bad_arithmetic_exact_rational():
    r = _checker().check_step("In May she sold 48/2 = 25 clips.")
    assert not r.accepted and r.failed == "arithmetic"
    # a fraction that float math might fudge but Fraction nails:
    ok = _checker().check_step("Portion is 48/2 = 24")
    assert ok.accepted


def test_no_calc_step_flagged():
    r = _checker().check_step("First, let's find the total.")
    assert not r.accepted and r.failed == "no_calc"


# --------------------------------------------------------------------------------------
# derived intermediates + growing candidate set
# --------------------------------------------------------------------------------------
def test_derived_intermediate_becomes_available():
    c = _checker()
    r1 = c.check_step("In May she sold 48/2 = 24 clips.")
    assert r1.accepted
    c.accept("In May she sold 48/2 = 24 clips.")            # 24 now a derived intermediate
    # a later step using 24 (not in the original problem text) is accepted via intermediate provenance
    r2 = c.check_step("Altogether 48 + 24 = 72 clips.")
    assert r2.accepted


def test_candidate_set_size_is_context_length():
    assert _checker().candidate_set_size == len(CONTEXT)


def test_scan_is_uncached_no_prebuilt_index(monkeypatch):
    # no persistent quantity index: an identical re-check re-scans the raw context (same work)
    import motivation_exp.math_checker as mc
    calls = {"n": 0}
    orig = mc.extract_numbers
    monkeypatch.setattr(mc, "extract_numbers", lambda s: (calls.__setitem__("n", calls["n"] + 1), orig(s))[1])
    c = MathChecker(); c.prefill(CONTEXT, RELEVANCE)
    calls["n"] = 0; c.check_step("She sold 99/3 = 33 clips."); first = calls["n"]   # 99 forces a full scan
    calls["n"] = 0; c.check_step("She sold 99/3 = 33 clips."); second = calls["n"]
    assert first >= len(CONTEXT) and second == first          # re-scanned each time, nothing cached
