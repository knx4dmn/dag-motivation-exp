"""Tests for frontier-guided resampling (variant 2): grammar builder + the guided decode branch.

The XGrammar-based `make_guided_step` runs only on Colab (needs a tokenizer/model); here we drive
`semantic_decode`'s guided branch with a STUB guided_step to verify wiring, |E|/rank logging, and
the fallbacks. Default `frontier_guided=False` is covered by the rest of the suite staying green.
"""
from __future__ import annotations

from motivation_exp import grammar as gr
from motivation_exp.runner import DecodeCfg, semantic_decode
from motivation_exp.checker import CheckResult


# --------------------------------------------------------------------------------------
# grammar.build_frontier_ebnf
# --------------------------------------------------------------------------------------
def test_build_frontier_ebnf_shape_and_escaping():
    ebnf = gr.build_frontier_ebnf(["Wren is a wumpus.", "Wren is not slow."])
    assert ebnf.startswith("root ::= (")
    assert '"Wren is a wumpus."' in ebnf and '"Wren is not slow."' in ebnf
    assert ebnf.rstrip().endswith('"\\n"')


def test_build_frontier_ebnf_empty_raises():
    import pytest
    with pytest.raises(ValueError):
        gr.build_frontier_ebnf([])


# --------------------------------------------------------------------------------------
# stub backend + guided_step to drive the guided branch
# --------------------------------------------------------------------------------------
class StubBackend:
    def __init__(self, prompt_len):
        self._len = prompt_len
    def seq_len(self):
        return self._len
    def decode_step(self, token_id):
        self._len += 1
        return {"seq": self._len}
    def crop(self, length):
        self._len = length


class Sampler:  # attempt-0 free decode emits [1, 2] -> "word." (a boundary)
    def __init__(self):
        self.seq = [1, 2]; self.i = 0
    def __call__(self, logits, temperature, banned_first):
        tok = self.seq[self.i % len(self.seq)]; self.i += 1
        return tok


def _decode(ids):
    return " ".join({1: "word", 2: ".", 7: "guided"}.get(t, "?") for t in ids).replace(" .", ".")


class CheckerStub:
    def __init__(self, E):
        self._E = E; self.candidate_set_size = 5; self.accepted = []
    def check_step(self, text):
        return CheckResult(False, 0.1, None, None)     # reject blind attempts -> force guided/fallback
    def expected_steps(self):
        return list(self._E)
    def accept(self, text):
        self.accepted.append(text)


def _stub_guided(chosen_text):
    def guided_step(E, backend, logits, all_ids):
        all_ids.append(7)
        nxt = backend.decode_step(7)
        return [7], nxt, {"ok": True, "e_size": len(E), "choice_rank": 2,
                          "chosen": chosen_text, "n_forward": 1}
    return guided_step


def test_guided_branch_resolves_reject_and_logs_E_and_rank():
    backend = StubBackend(prompt_len=3)
    checker = CheckerStub(E=["Wren is a wumpus.", "Wren is a vumpus."])   # |E| = 2
    cfg = DecodeCfg(eot_id=9, max_new_tokens=1, per_step_cap=48, frontier_guided=True)
    gen, stats = semantic_decode(backend, {"seq": 3}, [10, 11, 12], _decode, checker,
                                 Sampler(), cfg, clock=lambda: 0.0,
                                 guided_step=_stub_guided("Wren is a wumpus."))
    assert stats.n_rejects == 1                 # attempt-0 free step was rejected
    assert stats.n_frontier_guided == 1         # resolved by the guided retry
    assert stats.e_sizes == [2] and stats.choice_ranks == [2]
    assert checker.accepted == ["Wren is a wumpus."]   # the derivable step was accepted
    assert gen == [7]


def test_guided_empty_E_falls_back_to_blind_then_forced_accept():
    backend = StubBackend(prompt_len=3)
    checker = CheckerStub(E=[])                  # frontier gives nothing -> blind fallback
    cfg = DecodeCfg(eot_id=9, max_new_tokens=1, per_step_cap=48, frontier_guided=True, max_retries=2)
    gen, stats = semantic_decode(backend, {"seq": 3}, [10, 11, 12], _decode, checker,
                                 Sampler(), cfg, clock=lambda: 0.0,
                                 guided_step=_stub_guided("unused"))
    assert stats.n_frontier_guided == 0
    assert stats.n_frontier_empty == cfg.max_retries    # E empty checked on each retry
    assert stats.n_forced_accepts == 1                  # blind never accepted -> forced accept
