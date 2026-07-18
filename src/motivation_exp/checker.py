"""Semantic step-boundary verifier (SMC-style, simplified) for the semantic method.

At each sentence boundary the runner hands the just-completed step to :meth:`SemanticChecker.check_step`,
which accepts it if either
  (a) it *restates* a context fact/rule or an already-accepted step -- max cosine over the
      candidate set >= ``tau_restate``; the candidate set is **all context sentences** (grows
      with the bucket -- this is what makes latency scale) **+ previously accepted steps**; or
  (b) it is a valid *modus-ponens* continuation -- max cosine >= ``tau_mp`` against
      synthesized "expected next step" strings derived from the accepted-step frontier and
      the context rules (exact templates are fine; PrOntoQA is templatic).

The retry/rollback loop and the ``forced_accept`` flag live in the runner; :meth:`check_step`
reports a single check's outcome. The embedding model is injected as ``encode_fn`` -- a
callable ``list[str] -> np.ndarray`` returning L2-normalized rows -- so CPU tests use a stub
and Colab wraps ``BAAI/bge-small-en-v1.5`` (see :func:`sentence_transformer_encoder`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from . import grammar as gr

EncodeFn = Callable[[Sequence[str]], np.ndarray]


@dataclass
class CheckResult:
    """Outcome of ONE step check (the runner owns retries / forced_accept)."""

    accepted: bool
    cosine: float          # best similarity achieved (restate or mp, whichever drove it)
    matched: str | None    # the candidate/expected string that best matched
    kind: str | None       # "restate" | "mp" | None


class SemanticChecker:
    """Per-item semantic verifier holding the embedder and the growing candidate set."""

    def __init__(
        self,
        encode_fn: EncodeFn,
        tau_restate: float,
        tau_mp: float,
        predicate_guard: bool = True,
        log_decisions: bool = False,
        strip_connectives: bool = False,
    ) -> None:
        if tau_restate is None or tau_mp is None:
            raise ValueError("tau thresholds must be calibrated (not None) before checking")
        self.encode_fn = encode_fn
        self.tau_restate = float(tau_restate)
        self.tau_mp = float(tau_mp)
        # bge-small cannot reliably separate "X is slow" from "X is not slow", nor one novel
        # *pus concept from another (cosine > 0.9 for both). The guard requires the parsed
        # clause of the cosine-retrieved match to EQUAL the step's parsed clause (same subject,
        # predicate, is_property, polarity) -- it verifies the associative-match winner, it does
        # NOT replace the similarity search (which still runs over the full candidate set every
        # step -- the O(context) operation Panel B measures and CAM maps to). Paraphrases still
        # pass because they parse to the same Clause. Calibrate/verify in Phase 1.5.
        self.predicate_guard = bool(predicate_guard)

        # Diagnostics (behavior-neutral): count steps whose surface form fails parse_clause (the
        # prime false-reject channel for the UNGUIDED model -- connective-prefixed steps), and,
        # when log_decisions is on, keep a per-step record incl. a false-reject label computed by
        # re-parsing a connective-stripped copy. None of this affects the accept decision.
        self.log_decisions = bool(log_decisions)
        self.step_log: list[dict] = []
        self.n_unparsed_steps = 0
        # When on, normalize each step (strip leading discourse connectives) before embedding +
        # parsing, and store the normalized form in the candidate set. Closed-set, polarity- and
        # predicate-preserving (see grammar.strip_connectives), so the guard is unaffected.
        self.strip_connectives = bool(strip_connectives)

        # candidate set (restate target): context sentences + accepted steps. Each candidate's
        # parsed clause is cached HERE (parsed once at prefill/accept), so the per-step guard --
        # which scans the whole above-tau set every step -- never re-parses the same sentence.
        self._cand_texts: list[str] = []
        self._cand_clauses: list = []                        # parallel to _cand_texts (Clause | None)
        self._cand_embs: np.ndarray | None = None            # shape [n, d], L2-normalized

        # structured knowledge for modus-ponens synthesis
        self._rules: list[tuple[str, str, bool, bool]] = []  # (ante_concept, cons, cons_is_property, negated)
        self._frontier: set[tuple[str, str]] = set()         # (entity, concept) non-negated membership facts
        self._expected_cache: dict[str, np.ndarray] = {}     # expected-string -> emb row

    # ---- setup --------------------------------------------------------------------
    def prefill(self, context: Sequence[str]) -> None:
        """Embed all context sentences once and seed the rule set + fact frontier.

        Each context sentence is parsed ONCE here (cached in ``_cand_clauses``). Timed and
        reported separately by the runner (``embed_prefill_s``).
        """
        texts = list(context)
        self._cand_texts = list(texts)
        self._cand_embs = self._embed(texts) if texts else None
        self._cand_clauses = [gr.parse_clause(s) for s in texts]   # parse ONCE
        for c in self._cand_clauses:
            self._ingest_clause(c)

    def _ingest_clause(self, c) -> None:
        """Update the rule set / fact frontier from an already-parsed clause."""
        if c is None:
            return
        if c.kind == "rule":
            self._rules.append((c.subject, c.pred, c.is_property, c.negated))
        elif not c.is_property and not c.negated:
            # only non-negated concept-membership facts drive forward chaining
            self._frontier.add((c.subject, c.pred))

    # ---- per-step check -----------------------------------------------------------
    def check_step(self, step_text: str) -> CheckResult:
        """Verify one completed step against the candidate set and modus-ponens frontier."""
        step_text = step_text.strip()
        if not step_text or self._cand_embs is None:
            return CheckResult(False, 0.0, None, None)

        prepped = self._prep(step_text)      # normalized (connective-stripped) iff flag on
        emb = self._embed([prepped])[0]  # [d]
        step_clause = gr.parse_clause(prepped)
        if step_clause is None:
            self.n_unparsed_steps += 1

        # (a) restate: associative cosine match over the FULL candidate set (context + accepted
        #     steps). This full-set similarity search is the mechanism -- it always runs, and its
        #     cost grows with the bucket (Panel B / CAM). The guard only VERIFIES the winner.
        sims = self._cand_embs @ emb
        j_best = int(np.argmax(sims))
        cos_restate = float(sims[j_best])
        above = np.nonzero(sims >= self.tau_restate)[0]
        rj = self._guarded_pick(above, sims, step_clause, lambda k: self._cand_clauses[k])  # cached
        if rj is not None:
            return self._record(step_text, step_clause,
                                CheckResult(True, float(sims[rj]), self._cand_texts[rj], "restate"))

        # (b) modus-ponens: cosine vs synthesized expected-next-step strings (guarded likewise).
        #     _synthesize_expected returns (text, clause) pairs -- the clause is constructed, not
        #     re-parsed.
        expected = self._synthesize_expected()
        cos_mp, matched_mp = 0.0, None
        if expected:
            exp_texts = [t for t, _ in expected]
            exp_clauses = [c for _, c in expected]
            exp_embs = np.stack([self._expected_emb(t) for t in exp_texts])
            mp_sims = exp_embs @ emb
            i_best = int(np.argmax(mp_sims))
            cos_mp, matched_mp = float(mp_sims[i_best]), exp_texts[i_best]
            above_mp = np.nonzero(mp_sims >= self.tau_mp)[0]
            mi = self._guarded_pick(above_mp, mp_sims, step_clause, lambda k: exp_clauses[k])
            if mi is not None:
                return self._record(step_text, step_clause,
                                    CheckResult(True, float(mp_sims[mi]), exp_texts[mi], "mp"))

        # rejected: report the stronger of the two signals for logging
        if cos_restate >= cos_mp:
            return self._record(step_text, step_clause,
                                CheckResult(False, cos_restate, self._cand_texts[j_best], None))
        return self._record(step_text, step_clause, CheckResult(False, cos_mp, matched_mp, None))

    def _record(self, step_text, step_clause, result: CheckResult) -> CheckResult:
        """Optionally log a per-step decision with a false-reject label (diagnostics only)."""
        if not self.log_decisions:
            return result
        likely_false_reject = False
        if not result.accepted:
            # would a connective-stripped copy parse AND match a candidate / expected clause?
            norm = gr.parse_clause(gr.strip_connectives(step_text))
            if norm is not None:
                restate_hit = any(norm == cc for cc in self._cand_clauses if cc is not None)
                mp_hit = any(norm == c for _, c in self._synthesize_expected())
                likely_false_reject = restate_hit or mp_hit
        self.step_log.append({
            "step": step_text, "accepted": result.accepted, "kind": result.kind,
            "cosine": result.cosine, "parsed": step_clause is not None,
            "candidate_set_size": len(self._cand_texts),
            "likely_false_reject": likely_false_reject,
        })
        return result

    def _guarded_pick(self, above_idx, sims, step_clause, clause_of):
        """Among candidates whose cosine >= tau (``above_idx``), return the best index to accept.

        With the guard off: the highest-cosine candidate. With the guard on: scan ALL above-tau
        candidates (highest cosine first) and return the FIRST whose parsed clause equals the
        step's -- NOT just the argmax. This matters because bge-small cannot separate a negation
        from its positive, so a negated variant can outscore the true (paraphrased) match; if we
        only checked the top-1 we would falsely reject a valid step (hurting Panel A and wasting
        resamples on Panel B). Returns None if no above-tau candidate is parse-equal.
        """
        if above_idx.size == 0:
            return None
        order = above_idx[np.argsort(-sims[above_idx])]
        if not self.predicate_guard:
            return int(order[0])
        if step_clause is None:
            return None
        for k in order:
            if clause_of(int(k)) == step_clause:
                return int(k)
        return None

    def accept(self, step_text: str) -> None:
        """Add an accepted step to the candidate set and extend the fact frontier."""
        step_text = self._prep(step_text.strip())    # store the normalized form so later restatements match
        if not step_text:
            return
        emb = self._embed([step_text])  # [1, d]
        if self._cand_embs is None:
            self._cand_embs = emb
        else:
            self._cand_embs = np.vstack([self._cand_embs, emb])
        self._cand_texts.append(step_text)
        c = gr.parse_clause(step_text)      # parse ONCE at accept time; cache alongside
        self._cand_clauses.append(c)
        self._ingest_clause(c)

    def _prep(self, text: str) -> str:
        """Normalize a step for verification: strip leading discourse connectives iff enabled."""
        return gr.strip_connectives(text) if self.strip_connectives else text

    @property
    def candidate_set_size(self) -> int:
        return len(self._cand_texts)

    # ---- internals ----------------------------------------------------------------
    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        arr = np.asarray(self.encode_fn(list(texts)), dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        return arr

    def _expected_emb(self, text: str) -> np.ndarray:
        cached = self._expected_cache.get(text)
        if cached is None:
            cached = self._embed([text])[0]
            self._expected_cache[text] = cached
        return cached

    def _synthesize_expected(self) -> list[tuple[str, "gr.Clause"]]:
        """Expected next steps as (text, clause) pairs: apply each rule to each frontier fact.

        (entity is a C) + rule (every C is [not] a D | every C is [not] <prop>)
            => expected (entity is [not] a D)  or  (entity is [not] <prop>).
        The clause is CONSTRUCTED directly from the known semantic parts (not re-parsed), and the
        text is rendered via the shared grammar so phrasing matches the model's output.
        """
        out: list[tuple[str, gr.Clause]] = []
        seen: set[str] = set()
        for (entity, concept) in self._frontier:
            for (ante, cons, is_prop, rneg) in self._rules:
                if ante != concept:
                    continue
                sentence = gr.render_fact(entity, cons, is_prop, rneg)
                if sentence not in seen:
                    seen.add(sentence)
                    out.append((sentence, gr.Clause("fact", entity, cons, is_prop, rneg)))
        return out


def sentence_transformer_encoder(model) -> EncodeFn:
    """Wrap a sentence-transformers model into an ``encode_fn`` returning normalized rows.

    On Colab: ``SentenceTransformer('BAAI/bge-small-en-v1.5', model_kwargs={'torch_dtype':
    torch.float16}, device='cuda')``. Runs under inference mode via the library.
    """
    def encode_fn(texts: Sequence[str]) -> np.ndarray:
        return model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    return encode_fn
