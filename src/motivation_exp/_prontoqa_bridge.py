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


class ProntoQABridge:
    """Thin wrapper that imports and drives ``run_experiment`` from a clone directory."""

    def __init__(self, prontoqa_dir: str):
        self.dir = os.path.abspath(prontoqa_dir)
        if not os.path.isfile(os.path.join(self.dir, "run_experiment.py")):
            raise FileNotFoundError(f"run_experiment.py not found under {self.dir!r}")
        self._mod = None

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
                 *, max_tries: int = 500, **kwargs):
        """Return one successful 6-tuple from ``generate_question``, retrying on None failures.

        ``concept_block`` is the disjoint block of (already-registered) concept names for this
        item, or None to use the upstream defaults. ``kwargs`` pass through (ontology,
        distractors, deduction_rule, formula_ordering, proof_width, no_adjectives, ...).
        """
        gq = self.module.generate_question
        block = list(concept_block) if concept_block else None
        with self._cwd():
            for _ in range(max_tries):
                out = gq(num_deduction_steps, list(block) if block else None, **kwargs)
                if out[0] is not None:
                    return out
        raise RuntimeError(
            f"generate_question failed after {max_tries} tries "
            f"(num_deduction_steps={num_deduction_steps}, block_size={len(block) if block else 0})"
        )
