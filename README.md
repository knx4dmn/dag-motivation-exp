# dag-motivation-exp

Preliminary motivation experiment for **symbolic vs. semantic structure-guided inference under
growing context**, on Colab T4 using PrOntoQA. Produces a two-panel figure: Panel A
(accuracy gap widens with context) and Panel B (semantic per-token latency grows with context;
symbolic stays flat).

**To run the experiment, open [`run_colab.ipynb`](run_colab.ipynb) on Colab and follow
[`RUNBOOK.md`](RUNBOOK.md).** The runbook covers setup, cell order, the G1/G2/G3 gates,
expected wall-clock, and every failure mode (OOM, wrong GPU, disconnect).

## Layout

- `src/motivation_exp/` — `datagen` (PrOntoQA adapter, bucketing, splits, validation, manifest),
  `grammar` (per-item EBNF + XGrammar + exemplar synthesis), `checker` (semantic step-boundary
  verifier), `runner` (three decode loops + KV-cache rollback + JSONL resume), `plots` (the figure),
  `config` (pins, seeds, frozen τ).
- `tests/` — CPU-only tests (`pytest tests/`), no GPU or model weights required.
- `scripts/fetch_prontoqa.sh` — clones the pinned PrOntoQA generator.

## Local development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[test]"
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q
```

The torch-dependent stack (`xgrammar`, `sentence-transformers`) is in the `[colab]` extra and is
installed on Colab under a constraints file that pins torch to the preinstalled build — see the
notebook's setup cell. Do **not** install `[colab]` locally expecting to run the model; the CPU
tests exercise all pure logic with stubs.
