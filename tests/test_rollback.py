"""CPU-only tests for runner: JSONL resume + the KV-cache rollback index math (rev #8).

No torch. A stub backend records crop/decode calls so we can assert that after a reject
cycle the cache length is s-1 and the re-fed token is the boundary token at absolute
position s-1 (the off-by-one the rollback is built around).
"""
from __future__ import annotations

import json

from motivation_exp import runner as rn
from motivation_exp.runner import DecodeCfg, semantic_decode, _load_completed_keys, append_row


# --------------------------------------------------------------------------------------
# Stub backend: a growable "cache" of tokens; logits encode the next token to emit.
# --------------------------------------------------------------------------------------
class StubBackend:
    """Simulates KV cache growth + crop. ``decode_step`` returns a logits stand-in that the
    stub sampler maps deterministically, and records every crop length + re-fed token.
    """

    def __init__(self, prompt_len: int):
        self._len = prompt_len
        self.crop_calls: list[int] = []
        self.refed_tokens: list[int] = []   # token fed immediately after each crop
        self._just_cropped = False

    def seq_len(self) -> int:
        return self._len

    def decode_step(self, token_id: int):
        if self._just_cropped:
            self.refed_tokens.append(token_id)
            self._just_cropped = False
        # feeding a token advances the cache by one and "predicts" the next
        self._len += 1
        return {"fed": token_id, "seq": self._len}

    def crop(self, length: int):
        self.crop_calls.append(length)
        self._len = length
        self._just_cropped = True


# A scripted sampler: yields a fixed sequence of tokens per call, cycling by attempt so the
# first attempt produces a rejectable step and the retry produces an acceptable one.
class ScriptedSampler:
    def __init__(self, scripts: list[list[int]]):
        self.scripts = scripts
        self.attempt = 0
        self.pos = 0

    def __call__(self, logits, temperature, banned_first):
        seq = self.scripts[min(self.attempt, len(self.scripts) - 1)]
        tok = seq[self.pos]
        self.pos += 1
        if self.pos >= len(seq):
            self.pos = 0
            self.attempt += 1
        return tok


# token id 1 = "word", 2 = "." (period -> boundary), 9 = EOS/answer token
def _decode(ids):
    out = []
    for t in ids:
        out.append({1: "wren", 2: ".", 3: "wumpus", 9: "The answer is True."}.get(t, "?"))
    return " ".join(out).replace(" .", ".")


class CheckerStub:
    """Accepts only on the 2nd distinct step text; first step is rejected to force a rollback."""

    def __init__(self):
        self.candidate_set_size = 5
        self.accepted_texts: list[str] = []
        self._seen_reject = False

    def check_step(self, text):
        from motivation_exp.checker import CheckResult
        if not self._seen_reject:
            self._seen_reject = True
            return CheckResult(False, 0.1, None, None)  # reject first step -> rollback
        return CheckResult(True, 0.99, text, "mp")

    def accept(self, text):
        self.accepted_texts.append(text)


def test_rollback_crops_to_s_minus_1_and_refeeds_boundary_token():
    prompt = [10, 11, 12]  # prompt_len = 3 ; first step starts at absolute s = 3
    backend = StubBackend(prompt_len=len(prompt))
    # attempt 0: emit [word, .] -> "wren." (rejected). attempt 1: emit [word, .] again -> accepted.
    sampler = ScriptedSampler([[1, 2], [1, 2]])
    cfg = DecodeCfg(eot_id=9, max_new_tokens=8, per_step_cap=48)
    checker = CheckerStub()

    gen, stats = semantic_decode(backend, initial_logits={"seq": 3}, prompt_ids=prompt,
                                 decode_fn=_decode, checker=checker, sample_fn=sampler, cfg=cfg,
                                 clock=lambda: 0.0)

    # A reject happened -> exactly one crop, to s-1 = 2, and the re-fed token is prompt[s-1]=prompt[2]=12
    assert stats.n_rejects == 1
    assert backend.crop_calls == [2]                 # cropped to step_start-1 (s=3 -> 2)
    # the first decode_step after that crop re-fed the boundary token at absolute index s-1 == 12
    assert backend.refed_tokens == [12]


class AlwaysReject:
    candidate_set_size = 7

    def check_step(self, text):
        from motivation_exp.checker import CheckResult
        return CheckResult(False, 0.2, None, None)

    def __init__(self):
        self.accepted_texts = []

    def accept(self, text):
        self.accepted_texts.append(text)


def test_forced_accept_after_exhausting_retries_replays_best():
    prompt = [10, 11, 12]
    backend = StubBackend(prompt_len=len(prompt))
    sampler = ScriptedSampler([[1, 2]])  # every attempt emits the same step -> all rejected
    cfg = DecodeCfg(eot_id=9, max_new_tokens=2, per_step_cap=48, max_retries=3)
    checker = AlwaysReject()
    gen, stats = semantic_decode(backend, {"seq": 3}, prompt, _decode, checker, sampler, cfg,
                                 clock=lambda: 0.0)
    assert stats.n_forced_accepts == 1
    assert stats.n_rejects == 4           # attempts 0..3 all reject
    assert stats.duplicate_resample == 3  # attempts 1..3 reproduce the same step text
    assert backend.crop_calls == [2, 2, 2, 2]  # 3 retry crops + 1 forced-accept replay crop
    assert checker.accepted_texts == ["wren."]  # best attempt committed
    assert gen == [1, 2]


def test_semantic_decode_accepts_after_retry_and_reaches_answer():
    prompt = [10, 11, 12]
    backend = StubBackend(prompt_len=len(prompt))
    # step1 attempt0: [1,2] rejected; attempt1: [1,2] accepted -> then answer token 9 terminates
    sampler = ScriptedSampler([[1, 2], [1, 2], [9]])
    cfg = DecodeCfg(eot_id=9, max_new_tokens=8, per_step_cap=48)
    checker = CheckerStub()
    gen, stats = semantic_decode(backend, {"seq": 3}, prompt, _decode, checker, sampler, cfg,
                                 clock=lambda: 0.0)
    assert stats.n_rejects == 1
    assert stats.n_output_tokens == len(gen)
    assert 9 in gen  # reached the terminal answer token


# --------------------------------------------------------------------------------------
# resume logic
# --------------------------------------------------------------------------------------
def test_load_completed_keys_and_partial_line(tmp_path):
    p = tmp_path / "out.jsonl"
    rows = [
        {"model": "m", "method": "unguided", "bucket": 512, "item_id": "a"},
        {"model": "m", "method": "semantic", "bucket": 512, "item_id": "a"},
    ]
    for r in rows:
        append_row(str(p), r)
    # simulate a mid-write disconnect: trailing partial line
    with open(p, "a") as f:
        f.write('{"model": "m", "method": "sym')
    keys = _load_completed_keys(str(p))
    assert ("m", "unguided", 512, "a") in keys
    assert ("m", "semantic", 512, "a") in keys
    assert len(keys) == 2  # partial line ignored


def test_load_completed_keys_missing_file(tmp_path):
    assert _load_completed_keys(str(tmp_path / "nope.jsonl")) == set()


# --------------------------------------------------------------------------------------
# answer extraction
# --------------------------------------------------------------------------------------
def test_extract_answer_from_final_line():
    assert rn.extract_answer("Wren is a tumpus.\nThe answer is True.") == "True"
    assert rn.extract_answer("...\nThe answer is False.") == "False"
    assert rn.extract_answer("no verdict here") is None
