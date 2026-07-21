# Plan: GSM8K + GSM-IC distractors as the motivation task

**Status:** proposed, awaiting confirmation. No implementation until agreed.

## Decision & scope
PrOntoQA is frozen (failure-analysis material; nothing deleted). Verdict that closed it: with E
clean (query-entity scoped), the model's unconstrained distribution had ~0 overlap with the
derivable set (`rank==1 = 0` in all buckets) — a model↔task mismatch, not a checker bug. We switch
to **GSM8K** base problems with **GSM-IC-style distractors** (Shi et al., ICML 2023). Model stays
**Llama-3.2-3B-Instruct** (~60% GSM8K zero-shot → G1 sweet spot). Three-arm design
(unguided / symbolic / semantic), gates G1/G2/G3, pilot-before-full — all retained.

Why math should fit where PrOntoQA didn't: verification is **exact** (arithmetic + quantity
existence), so there is no embedding fuzziness / negation blindness / paraphrase brittleness, and
resampling can plausibly recover a correct calculation. The G0 gate (below) tests this assumption
before we build any guidance.

**Dataset (amendment C): GSM8K TEST split, not train.** Train is at high risk of pretraining
contamination — the model could recall answers and flatten the Panel A accuracy curve. Test split
keeps the task genuinely unsolved-from-memory.

## What carries over UNCHANGED (reuse as-is)
- Two-stage resumable **Phase 1a (generate→Drive) / 1b (bucket←Drive)**; per-item checkpointing.
- **Bucketing**: `DistractorPool` (pre-tokenize once, sum-based selection), token buckets
  512/1k/2k/4k/8k at ±5%, seeded, validation gate, manifest.
- **Three-arm runner**: `run_unguided/symbolic/semantic`, KV-cache rollback + resample scaffold,
  `logits_to_keep=1`, chunked prefill, `attention_mask=None`, `deterministic_decode`.
- **Latency accounting**: `torch.cuda.synchronize` regions, per-**accepted**-token metric, decomposition.
- **JSONL schema + resume** (skip completed `(model,method,bucket,item_id)`), chat-template prompting.
- **Gates** G1 (unguided acc ∈ [40,80]), G2 (semantic ≥ symbolic, gap non-shrinking), G3 (semantic
  latency grows, symbolic flat); **pilot→full** discipline; **plots** two-panel figure.
- **Config**: model, buckets, item counts, seeds. XGrammar for the symbolic baseline.

## 1. datagen spec (GSM8K + GSM-IC)
- **Base**: GSM8K (`openai/gsm8k`, `main` config, **test** split). Each item: a word problem ending
  in one question, and a solution with calculator annotations `<<a op b = c>>` and a final `#### N`.
- **Gold parse**: from the annotations extract `gold_steps` = ordered `["a op b = c", ...]`,
  `gold` = final number `N`, and `problem_quantities` = the numbers appearing in the problem text.
- **Distractors — template-generated (amendment D1)**: ≥ 8–10 **template families** (GSM-IC's three
  criteria: topic-related, in-range number, name/role-overlapping), **seeded**, names sampled from
  the item's own entity pool (proper names in the problem), numbers sampled in-range, **no duplicate
  sentence within an item**. They add unused quantities and never change `N`. Because names are
  per-item, distractors are generated **per item** (not a shared pool) and interleaved (gold
  sentences kept in order, question last) to hit each token bucket via the existing `DistractorPool`
  selection (pre-tokenize once, sum-based, ±5%).
- **Per-sentence relevance bit (amendment A)**: datagen emits `relevance: list[int]` parallel to
  `context` — 1 = original-problem sentence, 0 = injected distractor. This is the ground-truth
  provenance the semantic checker reads (framed as an **oracle upper bound** — a stand-in for a
  DAG-provided relevance signal).
- **Item**: `item_id, base_id, bucket, context(problem+distractors), relevance, question,
  gold(number), gold_steps, problem_quantities, token_count, seed`.
- **Splits**: reserve a fixed few-shot **exemplar** (real GSM8K CoT in calculation format). **No τ
  calibration** — the math checker is exact, so **Phase 1.5 is removed**.
- **Validation gate**: `eval(gold_steps) == gold` (exact rational); all original problem sentences
  survive injection; token count within tolerance; every distractor sentence carries relevance 0.
- **Structural note vs PrOntoQA**: base "generation" is just load+parse (fast, CPU, no retries), so
  Phase 1a = load+parse test items → Drive; Phase 1b = per-item distractor synth + bucket + validate.

## 2. checker spec (MathChecker — no embedder, no τ)
At each step boundary (a line containing `= <number>`, amendment D2), verify:
- **(a) Provenance-aware quantity retrieval — the O(context) CAM mechanism (amendments A + B).** For
  every operand in the step's expression, perform an **uncached, index-free LINEAR SCAN over the full
  context** to locate a sentence containing that quantity (software analog of a CAM associative
  search — NO precomputed set/index of context numbers; every check re-scans). Then:
  - operand traces to an **original-problem** sentence (relevance 1) → OK;
  - operand is a **derived intermediate** (result of a previously accepted step) → OK;
  - operand traces **only to distractor** sentences (relevance 0) → **REJECT**;
  - operand found nowhere → REJECT (hallucinated number).
  The relevance bit is ground truth from datagen, so this is an **oracle upper bound** on semantic
  verification (stand-in for DAG-provided relevance). Cost grows with the bucket (more sentences to
  scan) → Panel B / CAM. The latency accounting wraps this scan (`perf_counter`, same discipline).
  - **Small-constant exemption (implementation note, needs your ack).** Operands equal to a
    whitelist constant {0–10, 12, 100} are exempt from tracing — they are operation constants
    (`half`→2, `dozen`→12, `%`→100), not retrievable quantities; without this, valid `48/2` steps
    would be rejected and G0's step-pass rate would be artificially ~0. This is the SAME whitelist
    the symbolic arm allows. To keep it from hiding a distractor, **datagen generates distractor
    numbers to AVOID the whitelist** (range starts at 13, excludes 100), so a distractor quantity is
    always provenance-catchable — G0(b) stays valid. Flagging since amendment A didn't spell this out.
- **(b) Arithmetic check (amendment B).** Parse `<expr> = <result>`; assert `eval(expr) == result`
  using **`fractions.Fraction` (exact rational), not float equality** (GSM8K has divisions).
- Accept iff (a) ∧ (b); on accept, add `result` to derived-intermediates (the growing set).
- **Reject → resample** (same rollback/temperature scaffold; blind resample — frontier-guidance is
  NOT ported, it was PrOntoQA-specific). `CheckResult` records which sub-check failed
  (provenance-distractor / provenance-missing / arithmetic) for logging + G0.

## Implementation red lines (amendment B)
- **No caching / pre-indexing** of context quantities in the semantic arm. Every operand check is a
  fresh linear scan of the raw context sentences. This is deliberate: the associative-search cost IS
  the Panel B / CAM signal. The scan is the timed region.
- **Exact rational arithmetic** (`fractions.Fraction`) everywhere a computed value is compared.

## Two G0-mode edge cases (confirmed in code + tests)
1. **Record-all vs accepted-only intermediates.** The derived-intermediate set records a step's
   result **regardless of that step's own verdict in G0 post-hoc** (record-all), so an arithmetic
   failure can't cascade into spurious `missing` verdicts downstream (which could falsely trigger
   STOP-a); each step is charged only its own first-order error. The **real runner uses
   accepted-only** (a rejected step is resampled, its result discarded). The two modes are explicit
   in `MathChecker.accept`'s docstring and the two call sites.
2. **Value-collision trace precedence.** `_trace` precedence is whitelist > derived intermediate >
   original-problem sentence > distractor: an operand that value-collides with a legitimate source
   is NOT flagged distractor-referencing; only a distractor-ONLY match rejects. Since intermediates
   are unpredictable at datagen, this precedence handles runtime collisions; additionally datagen
   excludes distractor numbers that collide with the problem quantities **or the gold intermediate
   results** (known at datagen), tightening G0(b) attribution.

## G0 — overlap & attributability pre-flight gate  (runs AFTER checker, BEFORE grammar)
Purpose: catch a model↔task or checker↔task mismatch BEFORE building guidance/grammar (the exact
failure we hit late on PrOntoQA). **~40 items/bucket** (amendment 4), mid buckets **1k/2k**,
**unguided greedy only**, **observation only**. Prompt uses the reserved real GSM8K calculation-format
few-shot exemplar (amendment 3), not zero-shot.
1. Generate unguided CoT; run the checker **post-hoc** on every calculation step.
2. Report per bucket:
   - **parse rate** = fraction of items with ≥ 1 detected calculation line (amendment 3). If low,
     **fix prompting before interpreting (a)/(b)** — the model isn't producing checkable steps.
   - **(a) step-pass rate**, **decomposed into failure classes {missing, distractor-only, arithmetic}**
     (amendment 1). ``missing`` = operand traces nowhere (hallucinated / unparseable); ``distractor``
     = operand traces only to a distractor sentence; ``arithmetic`` = wrong `=`.
   - **(b) catchable fraction** among wrong-answer items (first bad step checker-catchable), reported
     **per bucket AND pooled across 1k/2k** (amendment 4; (b) conditions on wrong items, which are few
     per bucket).
3. **Verdict (decomposed thresholds; proposed — awaiting sign-off):**
   - **FIX PROMPTING** if parse rate < 0.70 in any bucket.
   - **STOP-on-(a)** [escalate model: Qwen2.5-3B → 7B int8] **ONLY if the `missing` class dominates**
     the pooled failures (`missing > distractor + arithmetic`) — that is the PrOntoQA-style mismatch.
     A low step-pass driven by `distractor + arithmetic` is **healthy signal**, not a stop.
   - **STOP-on-(b)** [pivot: HotpotQA-distractor / MuSiQue] if **pooled catchable < 1/3**.
   - **PROCEED** otherwise — including `step-pass < 0.50` when failures are the healthy
     distractor+arithmetic kind (it then shows up in (b)). Nominal bars stay (a) ≥ 0.50, (b) ≥ 1/3.
4. Log per-item, per-step verdicts (`g0_verdicts.jsonl`) as failure-analysis material either way.

## 3. grammar spec (symbolic baseline)
Static per-item calculation-format EBNF: output is a sequence of `<expr> = <number>` lines then a
final answer line (`#### <number>`), operators `+ - * /`. **Operands = ALL numbers in the
post-injection context** (problem + distractors) **plus a small-constant whitelist {0–10, 12, 100}**
(amendment A) — NOT `problem_quantities`, and **no relevance oracle on the symbolic arm**. This
enforces **syntactic/lexical** validity (well-formed calculations over any in-context number) but
neither relevance nor arithmetic correctness — the asymmetry that makes symbolic a meaningful
baseline, and it keeps G0(b)'s "references a distractor quantity = catchable" consistent (only the
semantic arm's oracle rejects distractor operands). Compiled once per item via XGrammar.

## 4. runner / plots — carry-over + FLAGGED domain changes
Reused unchanged: the three run loops, rollback/resample, timing, resume, prompting, plots skeleton.
Genuine math-domain changes (flagged, not silent):
- **`extract_answer`**: parse the final **number** (`#### N` / last number), not `True/False`.
- **checker**: new `MathChecker` (arithmetic + quantity retrieval); **drops the sentence-transformers
  / bge dependency** from the semantic path (no embeddings). `[colab]` extra loses `sentence-transformers`.
- **grammar**: calculation EBNF; **no exemplar-from-templates** (use a real GSM8K CoT exemplar).
- **Phase 1.5 (τ calibration): removed.**
- **Step-boundary detection**: calculation-line detector, not the period `split_sentences` rule.
- **frontier-guided machinery**: NOT ported (stays on the PrOntoQA branch).
- Plots caption: math domain, "retrieval + exact-arithmetic step check," correctness = exact numeric match.

## Resolved decisions (amendment D)
1. **Distractor source** = template-generate (D1): seeded, ≥ 8–10 template families, names from the
   item's entity pool, in-range numbers, no duplicate sentence within an item.
2. **Step boundary** = a line containing `= <number>`, with a per-step token cap (as today).
3. **Symbolic operands** = all post-injection context numbers + whitelist {0–10, 12, 100} (amendment A).
4. **Drop `sentence-transformers`** from the `[colab]` extra (the math checker needs no embedder).

## Implementation order (confirmed)
**1 datagen → 2 checker → G0 → 3 grammar → 4 runner/plots.** Pilot (G1/G2/G3) before full.
G0 thresholds above await your sign-off; they gate only the transition into steps 3–4, so datagen +
checker + the G0 harness are built first regardless.
