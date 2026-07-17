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

import re
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from . import grammar as gr
from .datagen import _singularize

EncodeFn = Callable[[Sequence[str]], np.ndarray]

# Plural rule form ("Tumpuses are wumpuses.") is not one of the generation templates but
# can appear in context; parse it here for modus-ponens synthesis.
_PLURAL_RULE_RE = re.compile(r"^([A-Za-z][a-zA-Z]*)\s+are\s+(not\s+)?([A-Za-z][a-zA-Z]*)\.?$")


@dataclass
class CheckResult:
    """Outcome of ONE step check (the runner owns retries / forced_accept)."""

    accepted: bool
    cosine: float          # best similarity achieved (restate or mp, whichever drove it)
    matched: str | None    # the candidate/expected string that best matched
    kind: str | None       # "restate" | "mp" | None


# --------------------------------------------------------------------------------------
# Sentence parsing (structured triples for modus-ponens synthesis)
# --------------------------------------------------------------------------------------
def _parse_sentence(sentence: str) -> tuple | None:
    """Return ('fact', entity, concept, negated) or ('rule', ante, cons, negated) or None."""
    s = sentence.strip()
    m = gr._TEMPLATE_REGEXES["entity_pos"].match(s)
    if m:
        return ("fact", m.group(1), _singularize(m.group(2)), False)
    m = gr._TEMPLATE_REGEXES["entity_neg"].match(s)
    if m:
        return ("fact", m.group(1), _singularize(m.group(2)), True)
    m = gr._TEMPLATE_REGEXES["rule_pos"].match(s)
    if m:
        return ("rule", _singularize(m.group(1)), _singularize(m.group(2)), False)
    m = gr._TEMPLATE_REGEXES["rule_neg"].match(s)
    if m:
        return ("rule", _singularize(m.group(1)), _singularize(m.group(2)), True)
    m = _PLURAL_RULE_RE.match(s)
    if m:
        return ("rule", _singularize(m.group(1)), _singularize(m.group(3)), bool(m.group(2)))
    return None


class SemanticChecker:
    """Per-item semantic verifier holding the embedder and the growing candidate set."""

    def __init__(
        self,
        encode_fn: EncodeFn,
        tau_restate: float,
        tau_mp: float,
    ) -> None:
        if tau_restate is None or tau_mp is None:
            raise ValueError("tau thresholds must be calibrated (not None) before checking")
        self.encode_fn = encode_fn
        self.tau_restate = float(tau_restate)
        self.tau_mp = float(tau_mp)

        # candidate set (restate target): context sentences + accepted steps
        self._cand_texts: list[str] = []
        self._cand_embs: np.ndarray | None = None  # shape [n, d], L2-normalized

        # structured knowledge for modus-ponens synthesis
        self._rules: list[tuple[str, str, bool]] = []          # (ante, cons, negated)
        self._frontier: set[tuple[str, str, bool]] = set()     # (entity, concept, negated) facts
        self._expected_cache: dict[str, np.ndarray] = {}       # expected-string -> emb row

    # ---- setup --------------------------------------------------------------------
    def prefill(self, context: Sequence[str]) -> None:
        """Embed all context sentences once and seed the rule set + fact frontier.

        Timed and reported separately by the runner (``embed_prefill_s``).
        """
        texts = list(context)
        self._cand_texts = list(texts)
        self._cand_embs = self._embed(texts) if texts else None
        for s in texts:
            parsed = _parse_sentence(s)
            if parsed is None:
                continue
            if parsed[0] == "rule":
                self._rules.append((parsed[1], parsed[2], parsed[3]))
            else:  # seed frontier with initial entity facts
                self._frontier.add((parsed[1], parsed[2], parsed[3]))

    # ---- per-step check -----------------------------------------------------------
    def check_step(self, step_text: str) -> CheckResult:
        """Verify one completed step against the candidate set and modus-ponens frontier."""
        step_text = step_text.strip()
        if not step_text or self._cand_embs is None:
            return CheckResult(False, 0.0, None, None)

        emb = self._embed([step_text])[0]  # [d]

        # (a) restate: cosine vs all context sentences + accepted steps
        sims = self._cand_embs @ emb
        j = int(np.argmax(sims))
        cos_restate = float(sims[j])
        if cos_restate >= self.tau_restate:
            return CheckResult(True, cos_restate, self._cand_texts[j], "restate")

        # (b) modus-ponens: cosine vs synthesized expected-next-step strings
        expected = self._synthesize_expected()
        cos_mp, matched_mp = 0.0, None
        if expected:
            exp_embs = np.stack([self._expected_emb(e) for e in expected])
            mp_sims = exp_embs @ emb
            i = int(np.argmax(mp_sims))
            cos_mp, matched_mp = float(mp_sims[i]), expected[i]
            if cos_mp >= self.tau_mp:
                return CheckResult(True, cos_mp, matched_mp, "mp")

        # rejected: report the stronger of the two signals for logging
        if cos_restate >= cos_mp:
            return CheckResult(False, cos_restate, self._cand_texts[j], None)
        return CheckResult(False, cos_mp, matched_mp, None)

    def accept(self, step_text: str) -> None:
        """Add an accepted step to the candidate set and extend the fact frontier."""
        step_text = step_text.strip()
        if not step_text:
            return
        emb = self._embed([step_text])  # [1, d]
        if self._cand_embs is None:
            self._cand_embs = emb
        else:
            self._cand_embs = np.vstack([self._cand_embs, emb])
        self._cand_texts.append(step_text)

        parsed = _parse_sentence(step_text)
        if parsed is not None and parsed[0] == "fact":
            self._frontier.add((parsed[1], parsed[2], parsed[3]))

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

    def _synthesize_expected(self) -> list[str]:
        """Expected next steps: apply each context rule to each non-negated frontier fact.

        (entity is-a C) + rule (C -> D, negated?) => expected (entity is[-not]-a D).
        Negated frontier facts are conservatively skipped. Rendered via the shared templates.
        """
        out: list[str] = []
        seen: set[str] = set()
        for (entity, concept, neg) in self._frontier:
            if neg:
                continue
            for (ante, cons, rneg) in self._rules:
                if ante != concept:
                    continue
                tid = "entity_neg" if rneg else "entity_pos"
                sentence = gr.render_template(tid, [entity, cons])
                if sentence not in seen:
                    seen.add(sentence)
                    out.append(sentence)
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
