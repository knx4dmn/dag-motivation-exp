"""CPU-only tests for the semantic checker, using a deterministic stub embedder.

The stub embeds a sentence as an L2-normalized hashed term-frequency vector: identical
sentences -> cosine 1.0, sentences sharing k/n words -> cosine ~k/n. This lets us drive
the restate / modus-ponens / reject branches without any model weights.
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
    "Every wumpus is a vumpus.",
]


def _checker(tau_restate=0.95, tau_mp=0.95):
    c = SemanticChecker(_stub_encode, tau_restate=tau_restate, tau_mp=tau_mp)
    c.prefill(CONTEXT)
    return c


def test_prefill_parses_rules_and_frontier():
    c = _checker()
    assert ("tumpus", "wumpus", False) in c._rules
    assert ("wumpus", "vumpus", False) in c._rules
    assert ("Wren", "tumpus", False) in c._frontier
    assert c.candidate_set_size == 3


def test_restate_exact_context_sentence_accepts():
    c = _checker()
    res = c.check_step("Wren is a tumpus.")
    assert res.accepted and res.kind == "restate"
    assert res.cosine == pytest.approx(1.0, abs=1e-5)


def test_modus_ponens_step_accepts_and_chains():
    c = _checker()
    # not a context sentence, but MP: (Wren is a tumpus) + (tumpus -> wumpus)
    r1 = c.check_step("Wren is a wumpus.")
    assert r1.accepted and r1.kind == "mp"
    c.accept("Wren is a wumpus.")
    assert c.candidate_set_size == 4  # candidate set grows with accepted steps
    # next hop chains off the newly accepted fact: (Wren is a wumpus) + (wumpus -> vumpus)
    r2 = c.check_step("Wren is a vumpus.")
    assert r2.accepted and r2.kind == "mp"


def test_garbage_step_rejected():
    c = _checker()
    res = c.check_step("Wren is a zormph.")
    assert not res.accepted and res.kind is None


def test_synthesize_expected_applies_rules_to_frontier():
    c = _checker()
    expected = c._synthesize_expected()
    assert "Wren is a wumpus." in expected  # tumpus -> wumpus applied to Wren


def test_uncalibrated_tau_raises():
    with pytest.raises(ValueError):
        SemanticChecker(_stub_encode, tau_restate=None, tau_mp=0.8)


def test_candidate_set_used_for_restate_after_accept():
    c = _checker(tau_restate=0.95, tau_mp=0.95)
    c.accept("Numo is a jompus.")  # now an accepted derived step in the candidate set
    res = c.check_step("Numo is a jompus.")  # restates the accepted step
    assert res.accepted and res.kind == "restate"
