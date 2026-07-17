"""motivation_exp: symbolic vs. semantic structure-guided inference under growing context.

Modules
-------
- config    : central knobs, pins, seeds, frozen tau thresholds.
- datagen   : PrOntoQA adapter, distractor pool, bucketing, splits, validation, manifest.
- grammar   : per-item EBNF builder, XGrammar compile, few-shot exemplar synthesis.
- checker   : semantic step-boundary verifier (embedder + growing candidate set).
- runner    : the three custom decode loops, timing hooks, JSONL append/resume.
- plots     : the two-panel motivation figure from JSONL.

Only ``datagen``, ``grammar``, and ``plots`` are importable without GPU/model weights;
``checker`` and ``runner`` import their heavy deps lazily so CPU-only tests can exercise
the pure logic with stubs.
"""

__version__ = "0.1.0"
