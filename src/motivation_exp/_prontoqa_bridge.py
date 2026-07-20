"""Bridge onto the cloned PrOntoQA generator, isolating its warts from our code.

Two upstream quirks this handles:

  1. ``run_experiment.py`` opens ``bad_patterns.txt`` via a RELATIVE path at module-import time
     (line ~674), so both the import AND every ``generate_question`` call must run with the
     process cwd set to the clone directory. We wrap both in a chdir context manager.
  2. ``generate_question`` returns ``(None,)*6`` on stochastic failure and is meant to be
     retried in a loop (the author's own driver does ``while True: ... if ok: break``). We do the
     retry here and only surface a successful 6-tuple.

Nothing here imports torch; it is CPU-only and drives the pure-Python generator.
"""
from __future__ import annotations

import contextlib
import importlib
import os
import sys
from typing import Sequence


class _WarnCountingSink:
    """A stdout replacement that counts (and discards) the generator's warning spam.

    ``run_experiment`` prints thousands of "Could not extend ontology ..." lines when a concept
    block is too small for a proof; unfiltered they truncate the real cell output. We aggregate
    the count instead of forwarding the text.
    """

    def __init__(self):
        self.warnings = 0

    def write(self, s):
        if "Could not extend" in s or "insufficient" in s:
            self.warnings += 1
        return len(s)

    def flush(self):
        pass


class ProntoQABridge:
    """Thin wrapper that imports and drives ``run_experiment`` from a clone directory."""

    def __init__(self, prontoqa_dir: str, seed: int = 0):
        self.dir = os.path.abspath(prontoqa_dir)
        if not os.path.isfile(os.path.join(self.dir, "run_experiment.py")):
            raise FileNotFoundError(f"run_experiment.py not found under {self.dir!r}")
        self._mod = None
        self._seed = seed
        # aggregate stats across all generate() calls
        self.total_calls = 0        # successful generate() calls
        self.total_tries = 0        # generate_question invocations (incl. retries)
        self.total_warnings = 0     # suppressed "could not extend ontology" warnings

    @contextlib.contextmanager
    def _cwd(self):
        prev = os.getcwd()
        os.chdir(self.dir)          # bad_patterns.txt is opened via a relative path upstream
        try:
            yield
        finally:
            os.chdir(prev)

    @property
    def module(self):
        """Import (once) and return the ``run_experiment`` module, under the clone cwd."""
        if self._mod is None:
            if self.dir not in sys.path:
                sys.path.insert(0, self.dir)
            with self._cwd():
                self._mod = importlib.import_module("run_experiment")
            # PrOntoQA draws from BOTH the stdlib `random` and `numpy.random` (theory.py/proof.py),
            # so seed both -> reproducible Phase 1 generation.
            self._mod.seed(self._seed)
            self._mod.np.random.seed(self._seed)
        return self._mod

    @property
    def morphology(self):
        return self.module.morphology

    def reserved_nouns(self) -> set[str]:
        """All nouns already registered upstream -- synthetic names must avoid these."""
        m = self.morphology
        return set(m.plural_nouns) | set(m.reverse_plural_nouns)

    def register_concepts(self, names: Sequence[str]) -> None:
        """Register synthetic concept nouns with the upstream morphology (idempotent)."""
        from . import datagen as dg

        dg.register_concepts(self.morphology, names)

    def generate(self, num_deduction_steps: int, concept_block: Sequence[str] | None,
                 *, max_tries: int = 500, suppress_output: bool = True, **kwargs):
        """Return one successful 6-tuple from ``generate_question``, retrying on None failures.

        ``concept_block`` is the disjoint block of (already-registered) concept names for this
        item, or None to use the upstream defaults. ``kwargs`` pass through (ontology,
        distractors, deduction_rule, formula_ordering, proof_width, no_adjectives, ...). The
        generator's per-try "could not extend ontology" warning spam is suppressed and counted
        (``total_warnings``); raises loudly with context if ``max_tries`` is exhausted.
        """
        gq = self.module.generate_question
        block = list(concept_block) if concept_block else None
        sink = _WarnCountingSink()
        redirect = contextlib.redirect_stdout(sink) if suppress_output else contextlib.nullcontext()
        with self._cwd(), redirect:
            for t in range(max_tries):
                self.total_tries += 1
                out = gq(num_deduction_steps, list(block) if block else None, **kwargs)
                if out[0] is not None:
                    self.total_calls += 1
                    self.total_warnings += sink.warnings
                    return out
        self.total_warnings += sink.warnings
        raise RuntimeError(
            f"generate_question failed after {max_tries} tries "
            f"(num_deduction_steps={num_deduction_steps}, block_size={len(block) if block else 0}, "
            f"kwargs={kwargs}). Widen the concept block (CONCEPT_BLOCK_SIZE) or lower N_HOPS."
        )
