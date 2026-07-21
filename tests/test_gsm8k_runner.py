"""CPU-only tests for the GSM8K math decode core + resume + correctness (no torch).

A stub backend records crop/decode calls so we can assert the KV-cache rollback index math (crop to
s-1, re-feed the boundary token) for math lines, plus the GSM8K-specific behaviors: prose lines pass
through without a check, a distractor calc is rejected and resampled, and the ``#### N`` line
terminates.
"""
from __future__ import annotations

from motivation_exp.gsm8k_runner import (
    answer_correct,
    math_semantic_decode,
    _has_line_boundary,
    _is_hash_answer,
)
from motivation_exp.math_checker import MathCheckResult
from motivation_exp.runner import DecodeCfg, _load_completed_keys, append_row


# token -> surface fragment; decode concatenates (newline token 2 ends a line)
_TOK = {1: "48 / 2 = 24", 2: "\n", 3: "Let me compute the total", 5: "99 / 3 = 33", 7: "#### 72"}


def _decode(ids):
    return "".join(_TOK.get(t, "?") for t in ids)


class StubBackend:
    def __init__(self, prompt_len: int):
        self._len = prompt_len
        self.crop_calls: list[int] = []
        self.refed_tokens: list[int] = []
        self._just_cropped = False

    def seq_len(self) -> int:
        return self._len

    def decode_step(self, token_id: int):
        if self._just_cropped:
            self.refed_tokens.append(token_id)
            self._just_cropped = False
        self._len += 1
        return {"fed": token_id, "seq": self._len}

    def crop(self, length: int):
        self.crop_calls.append(length)
        self._len = length
        self._just_cropped = True


class ScriptedSampler:
    def __init__(self, scripts):
        self.scripts, self.attempt, self.pos = scripts, 0, 0

    def __call__(self, logits, temperature, banned_first):
        seq = self.scripts[min(self.attempt, len(self.scripts) - 1)]
        tok = seq[self.pos]
        self.pos += 1
        if self.pos >= len(seq):
            self.pos, self.attempt = 0, self.attempt + 1
        return tok


class MathCheckerStub:
    """Rejects any line whose stripped text is in ``reject`` (as a distractor); accepts the rest."""

    def __init__(self, reject=()):
        self.candidate_set_size = 3
        self.accepted: list[str] = []
        self.checked: list[str] = []
        self._reject = set(reject)

    def check_step(self, text):
        self.checked.append(text.strip())
        if text.strip() in self._reject:
            return MathCheckResult(False, "provenance_distractor", "99", cosine=0.1)
        return MathCheckResult(True, None, None, cosine=1.0, kind="calc")

    def accept(self, text):
        self.accepted.append(text.strip())


def _cfg():
    return DecodeCfg(eot_id=99, max_new_tokens=12, per_step_cap=48)


# --------------------------------------------------------------------------------------
# boundary / terminal predicates
# --------------------------------------------------------------------------------------
def test_line_boundary_and_hash_answer():
    assert _has_line_boundary("48 / 2 = 24\n")
    assert not _has_line_boundary("48 / 2 = 24")
    assert _is_hash_answer("#### 72")
    assert not _is_hash_answer("48 + 24 = 72")


# --------------------------------------------------------------------------------------
# rollback index math on a distractor reject
# --------------------------------------------------------------------------------------
def test_distractor_reject_crops_to_s_minus_1_and_refeeds_boundary():
    prompt = [10, 11, 12]                      # s = 3
    backend = StubBackend(len(prompt))
    # line1 attempt0: [5,2] = "99 / 3 = 33" (distractor -> reject); attempt1: [1,2] = valid; then [7,2] answer
    sampler = ScriptedSampler([[5, 2], [1, 2], [7, 2]])
    checker = MathCheckerStub(reject={"99 / 3 = 33"})
    gen, stats = math_semantic_decode(backend, {"seq": 3}, prompt, _decode, checker, sampler, _cfg(),
                                      clock=lambda: 0.0)
    assert stats.n_rejects == 1
    assert backend.crop_calls == [2]           # cropped to step_start-1
    assert backend.refed_tokens == [12]        # re-fed boundary token at absolute s-1
    assert "99 / 3 = 33" not in checker.accepted   # rejected calc never committed
    assert "48 / 2 = 24" in checker.accepted


# --------------------------------------------------------------------------------------
# prose pass-through: a non-calculation line is accepted WITHOUT a check or rollback
# --------------------------------------------------------------------------------------
def test_prose_line_passes_through_without_check():
    prompt = [10, 11, 12]
    backend = StubBackend(len(prompt))
    # line1: [3,2] prose; line2: [1,2] valid calc; line3: [7,2] answer
    sampler = ScriptedSampler([[3, 2, 1, 2, 7, 2]])
    checker = MathCheckerStub()
    gen, stats = math_semantic_decode(backend, {"seq": 3}, prompt, _decode, checker, sampler, _cfg(),
                                      clock=lambda: 0.0)
    assert stats.n_checks == 1                 # only the ONE calc line was checked; prose skipped
    assert stats.n_rejects == 0
    assert backend.crop_calls == []            # no rollback for prose
    assert any("Let me compute" in t for t in checker.accepted)


# --------------------------------------------------------------------------------------
# terminal on the #### answer line
# --------------------------------------------------------------------------------------
def test_reaches_hash_answer_terminal():
    prompt = [10, 11, 12]
    backend = StubBackend(len(prompt))
    sampler = ScriptedSampler([[1, 2, 7, 2]])  # valid calc then answer
    checker = MathCheckerStub()
    gen, stats = math_semantic_decode(backend, {"seq": 3}, prompt, _decode, checker, sampler, _cfg(),
                                      clock=lambda: 0.0)
    text = _decode(gen)
    assert _is_hash_answer(text)
    assert stats.n_output_tokens == len(gen)


# --------------------------------------------------------------------------------------
# exact-rational correctness
# --------------------------------------------------------------------------------------
def test_answer_correct_exact_rational():
    assert answer_correct("72", "72")
    assert answer_correct("72.0", "72")           # value equality, not string equality
    assert not answer_correct("71", "72")
    assert not answer_correct(None, "72")


# --------------------------------------------------------------------------------------
# resume round-trip (shared JSONL helpers)
# --------------------------------------------------------------------------------------
def test_resume_keys_roundtrip(tmp_path):
    p = tmp_path / "gsm.jsonl"
    for r in [{"model": "llama", "method": "symbolic", "bucket": 1024, "item_id": "g0"},
              {"model": "llama", "method": "semantic", "bucket": 1024, "item_id": "g0"}]:
        append_row(str(p), r)
    keys = _load_completed_keys(str(p))
    assert ("llama", "symbolic", 1024, "g0") in keys
    assert ("llama", "semantic", 1024, "g0") in keys
