# RUNBOOK — Motivation Experiment (Colab T4)

Step-by-step operator guide for `run_colab.ipynb`. You (the user) run everything on Colab;
the code is authored to execute nothing locally that needs a GPU or model weights.

The experiment produces a two-panel figure:
- **Panel A** — accuracy vs. context length; claim: the semantic−symbolic accuracy gap widens as context grows.
- **Panel B** — per-output-token decode latency vs. context length; claim: semantic latency grows with context, symbolic stays flat.

---

## Known hardware constraint (T4 + this model)

On the T4 (Turing, sm75), PyTorch **SDPA always runs the math backend** for Llama-3.2-3B:
the model uses GQA (24 query heads vs 8 KV heads), which the memory-efficient SDPA kernel
rejects (mismatched head counts); FlashAttention needs sm80+ so it's unavailable; cuDNN is
also rejected. The math backend materializes a full `(1, 24, L, L)` score tensor with an fp32
softmax upcast — a one-shot 8k prefill is ~6 GiB and OOMs. **This is why prefill is chunked**
(`config.PREFILL_CHUNK = 512`, incremental KV-cache build); decode steps are q_len=1 so their
score tensor is `(1, 24, 1, L)` and trivially small. Nothing you can toggle changes the backend
selection — chunked prefill is required, not optional. (This is stated in the figure caption too.)

## 0. Before you start (one-time)

1. **Push this repo to GitHub** and note the URL + a commit hash. Edit Cell 1's `SELF_REPO`
   and `SELF_COMMIT`.
2. **Pin the PrOntoQA commit.** Visit https://github.com/asaparov/prontoqa, copy the latest
   commit hash, and set `PRONTOQA_COMMIT` in Cell 1 (and/or `config.PRONTOQA_COMMIT`). The
   fetch script refuses to run with the placeholder.
3. **Hugging Face access.** Request access to `meta-llama/Llama-3.2-3B-Instruct` (gated) and
   have a token ready; Cell 1 calls `login()`.
4. **Select a T4 runtime.** Runtime → Change runtime type → T4 GPU. If Colab gives you an L4
   or A100, all latency numbers would be off-hardware — disconnect and retry (see below).

## 1. Cell order

Run top to bottom: **Cell 1 (setup) → Cell 2 (Phase 0) → Cell 3 (Phase 1) → Cell 4 (Phase 1.5) →
Cell 5 (Phase 2 pilot) → Cell 6 (Phase 3 full) → Cell 7 (Phase 4 plots).**

Every heavy cell writes to Google Drive and is safe to re-run after a disconnect — it resumes.

---

## 2. What each go/no-go gate looks like

**Phase 0 (Cell 2)** must print:
- `GPU: Tesla T4` (assert fails otherwise),
- `eot id: <n> = <|eot_id|>`,
- `8k smoke OK ... | peak VRAM X.XX GB (must be < ~15 GB)`. If the assert trips, the 8k
  bucket won't fit — see OOM below.

**Phase 1.5 (Cell 4)** prints a τ sweep table and a selected line:
`>>> SELECTED tau = 0.XX. Paste into config.py: TAU_RESTATE = 0.XX  TAU_MP = 0.XX`.
Paste those into `config.py` (and re-push) **before any reported run** — the checker refuses
`None` thresholds. τ is frozen into the data manifest automatically.

**Phase 2 (Cell 5)** prints the three gates:
- **G1** — `unguided@512 acc` should be in **[0.40, 0.80]**. Below 0.40 → the task is too hard:
  switch to Qwen2.5-3B (set `config.PRIMARY_MODEL`) or drop `N_HOPS` to 2. Above 0.80 → too
  easy: raise `N_HOPS` to 4. (Changing either requires re-running Phase 1.)
- **G2** — per bucket, `semantic ≥ symbolic` and the `gap` should not shrink as the bucket grows.
- **G3** — `semantic` latency should rise across buckets while `symbolic` stays roughly flat.

If **G2/G3 fail**, debug the checker before scaling up: confirm `candidate_set_size` grows with
bucket (it should — the candidate set is the full context), and that τ isn't so low the checker
accepts everything (vacuous) or so high it force-accepts everything. Both are visible in the
`overhead` fields of `results_pilot.jsonl`.

---

## 3. Expected wall-clock (T4)

| Phase | What | Rough time |
|---|---|---|
| 0 | deps + model load + 8k smoke | 5–10 min |
| 1 | generate 150 base + 200 pool, bucket, validate | 10–20 min |
| 1.5 | τ sweep on 10 items | 2–5 min |
| 2 | pilot: 20×5×3 = 300 generations | 1–2 GPU-hours |
| 3 | full: 100×5×3 = 1500 generations | 8–12 GPU-hours (semantic retries + long-bucket prefill dominate) |
| 4 | plots | < 1 min |

On the free tier, Phase 3 spreads over 2–3 days of reconnects; the resume logic absorbs this
(you just re-run Cell 6 each session). Colab Pro fits it in a day of babysat sessions.

---

## 4. Failure modes

**Wrong GPU (not T4).** Phase 0's assert stops you. Runtime → Disconnect and delete runtime →
reconnect until you get a T4. Never mix latency numbers across GPU types.

**OOM (usually at the 8k bucket).** Three mitigations are baked in: `logits_to_keep=1` on every
forward (removes the ~4.2 GB fp32 prefill-logits transient), `attention_mask=None` (keeps
decode-step score tensors small and takes the `is_causal` path), and **chunked prefill by
default** (`config.PREFILL_CHUNK = 512`) — required because SDPA runs the math backend for this
GQA model on sm75 (see "Known hardware constraint" above). If Phase 0's peak-VRAM assert still
trips or a run OOMs on the top bucket, escalate in this order:
1. **Lower the prefill chunk** — set `config.PREFILL_CHUNK = 256` (or 128). Smaller chunks →
   smaller per-chunk score tensor. Cheapest fix; no data regen.
2. **Cap the top bucket at 6k** — edit `config.BUCKETS` to `[512, 1024, 2048, 4096, 6144]` and
   re-run Phase 1 (data) → Phase 2/3. Do this if the KV cache itself (not activations) is the cause.

**Disconnect mid-run.** Expected on free Colab. Just reconnect and re-run the current phase's
cell. Completed `(model, method, bucket, item_id)` tuples are skipped; a half-written last line
is tolerated by the loader. No manual cleanup.

**Phase 1 data generation.** The generator is driven through `_prontoqa_bridge`, which handles
two upstream quirks: (a) `run_experiment.py` opens `bad_patterns.txt` by a relative path at import
time, so the bridge runs the import and every call with cwd set to the clone dir; (b)
`generate_question` returns `(None,)*6` on stochastic failure (~a few percent succeed per call),
so the bridge retries until success — this is the author's own intended usage, so slow-looking
generation is normal. The adapter targets the verified 6-tuple
`(question_text, query, formulas, chain_of_thought, str(answer), proof)`. Vocabulary isolation is
guaranteed structurally: each base item and each pool item gets its own **disjoint synthetic
concept block** (registered with the generator's morphology), so no distractor can share a concept
with any base item. If generation is slow, it's the retry loop; if it errors on `add_noun`
("already added"), a synthetic name collided with a reserved one — regenerate with a different
`GLOBAL_SEED` (the generator already filters reserved names, so this should not happen).

**XGrammar install/first-use error.** `xgrammar==0.1.34` pulls `apache-tvm-ffi` + `pydantic`
and is installed under the torch-pinning constraints file so it can't swap torch. If its CUDA
mask kernel misbehaves on T4, the documented fallback is Outlines (regex per-item patterns);
swap the symbolic path in `runner.run_symbolic`.

**torch got changed by an install.** Cell 1 asserts torch is unchanged after install. If it
trips, a dependency ignored the constraints file — re-run Cell 1; do not proceed with a swapped
torch (CUDA will break on T4).

---

## 5. Outputs (on Drive, under `MyDrive/motivation_exp/`)

- `env_manifest.json`, `pip_freeze.txt` — provenance (env hash, versions, 8k peak VRAM).
- `data/{bucket}/items.jsonl`, `data/manifest.json`, `data/calib.jsonl`, `data/exemplar.jsonl`.
- `results_pilot.jsonl`, `results_full.jsonl` — one JSONL line per item-run (schema in the plan §7).
- `motivation_figure.png` + `.pdf` — the deliverable figure. Caption states step-boundary check
  granularity, per-accepted-token / prefill-excluded latency definition, and single-T4/fp16.
