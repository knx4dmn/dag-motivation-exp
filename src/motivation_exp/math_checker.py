"""MathChecker: per-step verification for the GSM8K semantic arm (no embedder, no tau).

At each calculation step, verify (docs/gsm8k_motivation_plan.md, amendments A/B):
  (a) provenance-aware quantity retrieval -- every operand must trace to an ORIGINAL-problem
      sentence (relevance 1) or a derived intermediate; operands traceable only to distractor
      sentences (relevance 0), or found nowhere, are REJECTED. The scan is an UNCACHED, index-free
      LINEAR pass over the full context every time (software analog of a CAM associative search --
      the O(context) cost that Panel B measures). The relevance bit is ground truth from datagen,
      so this is an ORACLE UPPER BOUND on semantic verification.
  (b) exact arithmetic -- eval(expr) == result via fractions.Fraction (never float).

Drop-in for the runner's semantic arm: exposes check_step / accept / candidate_set_size, and
CheckResult carries a ``cosine`` field (1.0 iff accepted) so the existing forced-accept selection
works unchanged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from .gsm8k_datagen import extract_numbers, is_whitelist_constant, normalize_number, safe_eval, to_fraction

_RESULT_RE = re.compile(r"=\s*(-?\$?[\d,]+(?:\.\d+)?)")
_TOKEN_RE = re.compile(r"(-?\$?\d[\d,]*(?:\.\d+)?|[-+*/])")


def parse_calculation(step_text: str) -> tuple[str, str] | None:
    """Extract the (expr, result) of the calculation on a step line, or None if there is none.

    Robust to real CoT: tolerates unit words between the numbers ("2 bolts + 1 bolt = 3 bolts"),
    ``x``/``×`` for multiply, ``$`` and thousands commas. Anchors on the LAST ``= <number>`` and then
    walks BACKWARD collecting the maximal trailing `num (op num)+` run (words break the run), so
    stray earlier numbers are ignored. Returns None if no such calculation is present.
    """
    text = step_text.replace("×", "*").replace("÷", "/")
    text = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", " * ", text)   # 3 x 4 / 3x4 -> 3 * 4
    results = list(_RESULT_RE.finditer(text))
    if not results:
        return None
    m = results[-1]
    result = normalize_number(m.group(1))
    toks = _TOKEN_RE.findall(text[:m.start()])
    run, want_num = [], True
    for t in reversed(toks):
        is_op = t in "+-*/"
        if want_num:
            if is_op:
                break
            run.append(t); want_num = False
        else:
            if not is_op:
                break
            run.append(t); want_num = True
    run.reverse()
    if len(run) < 3 or run[0] in "+-*/" or run[-1] in "+-*/" or not any(t in "+-*/" for t in run):
        return None
    expr = " ".join(normalize_number(t) if t not in "+-*/" else t for t in run)
    return expr, result


@dataclass
class MathCheckResult:
    accepted: bool
    failed: str | None = None      # "provenance_distractor"|"provenance_missing"|"arithmetic"|"no_calc"
    detail: str | None = None      # the offending operand, or "expr=result"
    cosine: float = 0.0            # compat score for the runner's forced-accept selection
    kind: str | None = None        # "calc" when accepted


class MathChecker:
    """Provenance + arithmetic step checker for one GSM8K item."""

    def __init__(self) -> None:
        self._context: list[str] = []
        self._relevance: list[int] = []
        self._intermediates: set[str] = set()   # derived results (allowed cache; NOT context numbers)

    # ---- setup --------------------------------------------------------------------
    def prefill(self, context: Sequence[str], relevance: Sequence[int]) -> None:
        """Store the context + parallel relevance bits. No quantity index is built (red line B)."""
        self._context = list(context)
        self._relevance = list(relevance)
        if len(self._context) != len(self._relevance):
            raise ValueError("context and relevance length mismatch")
        self._intermediates = set()

    # ---- provenance (uncached linear scan) ----------------------------------------
    def _trace(self, num: str) -> str:
        """Return where a quantity comes from: 'problem' | 'intermediate' | 'distractor' | 'missing'.

        Fresh, index-free linear scan of the raw context every call (the CAM cost).

        TRACE PRECEDENCE (an operand that value-collides with several sources is legitimate if ANY
        legitimate source matches -- edge case 2): whitelist constant > derived intermediate >
        original-problem sentence > distractor. So a number appearing in BOTH a problem/intermediate
        AND a distractor sentence is NOT flagged distractor-referencing; only a distractor-ONLY match
        (no problem/intermediate source) rejects. Whitelist constants (half->2, %->100) are exempt;
        datagen keeps distractor numbers off the whitelist AND off the problem/gold-intermediate
        values, so these exemptions never hide a genuine distractor reference.
        """
        if is_whitelist_constant(num):
            return "constant"
        if num in self._intermediates:                 # derived intermediates take priority
            return "intermediate"
        found_distractor = False
        for sent, rel in zip(self._context, self._relevance):
            if num in extract_numbers(sent):
                if rel == 1:
                    return "problem"                   # a problem-sentence match wins over any distractor
                found_distractor = True
        return "distractor" if found_distractor else "missing"

    # ---- per-step check -----------------------------------------------------------
    def check_step(self, step_text: str) -> MathCheckResult:
        calc = parse_calculation(step_text)
        if calc is None:
            return MathCheckResult(False, "no_calc")
        expr, result = calc
        # (a) provenance of every operand
        for op in extract_numbers(expr):
            prov = self._trace(op)
            if prov == "distractor":
                return MathCheckResult(False, "provenance_distractor", op)
            if prov == "missing":
                return MathCheckResult(False, "provenance_missing", op)
        # (b) exact rational arithmetic
        try:
            if safe_eval(expr) != to_fraction(result):
                return MathCheckResult(False, "arithmetic", f"{expr}={result}")
        except (ValueError, ZeroDivisionError, SyntaxError):
            return MathCheckResult(False, "arithmetic", expr)
        return MathCheckResult(True, None, None, cosine=1.0, kind="calc")

    def accept(self, step_text: str) -> None:
        """Record a step's result as a derived intermediate (available to downstream provenance).

        MODE DISTINCTION (edge case 1):
        - Real runner (semantic arm): call this ONLY on a step the checker ACCEPTED (accepted-only
          semantics -- a rejected step is resampled and its result is discarded).
        - G0 post-hoc: call this on EVERY emitted step regardless of its verdict (record-all
          semantics -- there is no resampling, so downstream steps reference the model's own chain).
          This charges each step only its own first-order error and prevents one arithmetic failure
          from cascading into spurious 'missing' verdicts. See g0_gsm8k.posthoc_verdicts.
        """
        calc = parse_calculation(step_text)
        if calc is not None:
            self._intermediates.add(calc[1])

    @property
    def candidate_set_size(self) -> int:
        """Context sentences scanned per check -- the associative-search size (grows with bucket)."""
        return len(self._context)
