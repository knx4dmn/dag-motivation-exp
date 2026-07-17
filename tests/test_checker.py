"""CPU-only tests for the semantic checker with a deterministic stub embedder.

Stub embeds a sentence as an L2-normalized hashed term-frequency vector: identical sentences
-> cosine 1.0. Drives restate / concept-MP / property-MP / reject branches without weights.
"""
from __future__ import annotations

import numpy as np
import pytest

from motivation_exp.checker import SemanticChecker


DIM = 512


def _stub_encode(texts):
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in t.lower().replace(".", " ").split():
            out[i, hash(w) % DIM] += 1.0
        n = np.linalg.norm(out[i])
        if n > 0:
            out[i] /= n
    return out


CONTEXT = [
    "Wren is a tumpus.",
    "Every tumpus is a wumpus.",
    "Wumpuses are not slow.",   # concept -> property rule (negated)
]


def _checker(tr=0.95, tm=0.95):
    c = SemanticChecker(_stub_encode, tau_restate=tr, tau_mp=tm)
    c.prefill(CONTEXT)
    return c


def test_prefill_parses_rules_and_frontier():
    c = _checker()
    assert ("tumpus", "wumpus", False, False) in c._rules      # concept rule
    assert ("wumpus", "slow", True, True) in c._rules          # property rule, negated
    assert ("Wren", "tumpus") in c._frontier
    assert c.candidate_set_size == 3


def test_restate_exact_context_accepts():
    c = _checker()
    res = c.check_step("Wren is a tumpus.")
    assert res.accepted and res.kind == "restate" and res.cosine == pytest.approx(1.0, abs=1e-5)


def test_concept_modus_ponens_then_property_chain():
    c = _checker()
    r1 = c.check_step("Wren is a wumpus.")     # (Wren is a tumpus) + (tumpus -> wumpus)
    assert r1.accepted and r1.kind == "mp"
    c.accept("Wren is a wumpus.")
    assert c.candidate_set_size == 4
    r2 = c.check_step("Wren is not slow.")     # (Wren is a wumpus) + (wumpus -> not slow)
    assert r2.accepted and r2.kind == "mp"


def test_garbage_rejected():
    c = _checker()
    assert not c.check_step("Wren is a zormph.").accepted


def test_synthesize_expected_concept_and_property():
    c = _checker()
    assert "Wren is a wumpus." in c._synthesize_expected()


def test_uncalibrated_tau_raises():
    with pytest.raises(ValueError):
        SemanticChecker(_stub_encode, tau_restate=None, tau_mp=0.8)


def test_candidate_set_growth_used_for_restate():
    c = _checker()
    c.accept("Numo is a jompus.")
    res = c.check_step("Numo is a jompus.")
    assert res.accepted and res.kind == "restate"


# --------------------------------------------------------------------------------------
# predicate/polarity guard (bge-small negation/concept blindness)
# --------------------------------------------------------------------------------------
def test_guard_blocks_negation_flip_but_allows_exact():
    ctx = ["Wren is not slow."]
    on = SemanticChecker(_stub_encode, 0.8, 0.8, predicate_guard=True); on.prefill(ctx)
    off = SemanticChecker(_stub_encode, 0.8, 0.8, predicate_guard=False); off.prefill(ctx)
    flip = "Wren is slow."  # opposite polarity, cosine ~0.87 to the context sentence
    assert not on.check_step(flip).accepted            # guard rejects the polarity flip
    r_off = off.check_step(flip)
    assert r_off.accepted and r_off.kind == "restate"  # WITHOUT guard: wrongly accepted
    assert on.check_step("Wren is not slow.").accepted  # exact restate still passes the guard


def test_guard_blocks_concept_swap():
    ctx = ["Wren is a tumpus."]   # no rule -> no MP path either
    on = SemanticChecker(_stub_encode, 0.7, 0.7, predicate_guard=True); on.prefill(ctx)
    assert not on.check_step("Wren is a wumpus.").accepted  # swapped concept -> guard blocks
    assert on.check_step("Wren is a tumpus.").accepted      # exact -> accepted


def test_guard_scans_all_above_tau_not_just_argmax():
    """A negation outscores the true paraphrase (bge-small can't separate them); the guard must
    still accept the valid step by scanning ALL above-tau candidates, not only the top-1."""
    S = "Tumpuses are wumpuses."          # step: rule(tumpus, wumpus, +)
    N = "Tumpuses are not wumpuses."      # negation: parse-UNEQUAL, HIGHER cosine (0.95)
    P = "Every tumpus is a wumpus."       # paraphrase: parse-EQUAL to S, LOWER cosine (0.88)
    vecs = {
        S: np.array([1.0, 0.0, 0.0], np.float32),
        N: np.array([0.95, np.sqrt(1 - 0.95**2), 0.0], np.float32),   # cos(S,N)=0.95
        P: np.array([0.88, 0.0, np.sqrt(1 - 0.88**2)], np.float32),   # cos(S,P)=0.88
    }
    enc = lambda texts: np.stack([vecs[t] for t in texts])

    c = SemanticChecker(enc, tau_restate=0.85, tau_mp=0.85, predicate_guard=True)
    c.prefill([N, P])                      # both above tau; argmax is the negation N
    r = c.check_step(S)
    assert r.accepted and r.kind == "restate"
    assert r.matched == P                  # picked the parse-equal paraphrase, not the argmax negation
    assert r.cosine == pytest.approx(0.88, abs=1e-4)


def test_full_candidate_set_similarity_runs_every_step():
    # the associative match is over the full growing candidate set (Panel B mechanism)
    c = _checker()
    n0 = c.candidate_set_size
    c.accept("Wren is a wumpus.")
    assert c.candidate_set_size == n0 + 1
    # a step that exactly restates the newly-added accepted step is retrieved from the grown set
    assert c.check_step("Wren is a wumpus.").accepted
