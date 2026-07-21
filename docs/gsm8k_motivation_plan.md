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
- **Base**: GSM8K (`openai/gsm8k`, `main` config, train split). Each item: a word problem ending in
  one question, and a solution with calculator annotations `<<a op b = c>>` and a final `#### N`.
- **Gold parse**: from the annotations extract `gold_steps` = ordered `["a op b = c", ...]`,
  `gold` = final number `N`, and `problem_quantities` = the numbers appearing in the problem text.
- **Distractors (GSM-IC, three criteria)**: topic-related, in-range number, name/role-overlapping
  sentences that are **irrelevant** to the solution (don't change `N`). We need MANY per item to
  reach 8k, so build a distractor **pool** (source TBD — see open questions) of such sentences and
  interleave them (gold sentences kept in order, question last) to hit each token bucket — exactly
  the PrOntoQA bucketing interface. No vocabulary-collision filter needed; instead the validation
  gate re-derives `N` from `gold_steps` and asserts injection didn't change it.
- **Item**: `item_id, base_id, bucket, context(problem+distractors), question, gold(number),
  gold_steps, problem_quantities, token_count, seed`.
- **Splits**: reserve a fixed few-shot **exemplar** (real GSM8K CoT in calculation format). **No τ
  calibration** — the math checker is exact, so **Phase 1.5 is removed**.
- **Validation gate**: `eval(gold_steps) == gold`; all gold problem sentences survive injection;
  token count within tolerance.

## 2. checker spec (MathChecker — no embedder, no τ)
At each step boundary (a calculation line), verify:
- **(a) Quantity retrieval — the O(context) CAM mechanism.** Every operand in the step's expression
  must exist in the context (problem + injected distractors) OR in the derived-intermediates set
  (results of previously accepted steps). This is a **full-context scan over all quantities, no
  top-k** — the associative match whose cost grows with the bucket (distractors add numbers) →
  Panel B / CAM. Exact numeric match with formatting tolerance (`1,000`≡`1000`, `3`≡`3.0`).
- **(b) Arithmetic check.** Parse `<expr> = <result>`; assert `eval(expr) == result` exactly.
- Accept iff (a) ∧ (b); on accept, add `result` to derived-intermediates (the growing set).
- **Reject → resample** (same rollback/temperature scaffold; blind resample — frontier-guidance is
  NOT ported, it was PrOntoQA-specific). Same latency accounting (`perf_counter` around the scan +
  arithmetic). CheckResult records which sub-check failed (retrieval vs arithmetic) for logging.
- Step segmentation: a "step" = a detected calculation line (newline- or `<<>>`-delimited), not a
  period-sentence — see open questions.

## G0 — overlap & attributability pre-flight gate  (runs AFTER checker, BEFORE grammar)
Purpose: catch a model↔task or checker↔task mismatch BEFORE building guidance/grammar (the exact
failure we hit late on PrOntoQA). ~20 items, mid buckets **1k/2k**, **unguided only**, no
accept/reject/resample — **observation only**.
1. Generate unguided CoT solutions; run the checker **post-hoc** on every emitted step.
2. Report per bucket:
   - **(a) step-pass rate** = fraction of unguided steps the checker would accept (model-distribution
     overlap with the checkable-valid set; the analog of PrOntoQA's `rank` diagnostic).
   - **(b) error attributability** = among wrong-answer items, fraction whose **first bad step** is
     checker-catchable (references a distractor quantity, or fails arithmetic) vs checker-invisible
     (wrong plan / wrong operation with clean arithmetic).
3. **Pre-committed thresholds:**
   - Proceed to steps 3–4 only if **(a) is clearly nonzero and majority-level** AND **(b) shows a
     meaningful catchable fraction**.
   - **If (a) ~ 0** → model-task mismatch again → **stop**, escalate model (Qwen2.5-3B, or 7B int8);
     do NOT build the rest of the pipeline.
   - **If (b) is low** → checker has no discriminative signal on this task's real failure modes →
     **stop and flag**; candidate pivot = multi-hop QA with distractor paragraphs (HotpotQA
     distractor setting / MuSiQue).
4. Log per-item, per-step verdicts so the G0 table becomes failure-analysis material either way.

## 3. grammar spec (symbolic baseline)
Static per-item calculation-format EBNF: output is a sequence of `<expr> = <number>` lines then a
final answer line (`#### <number>`), where expr operands are constrained to `problem_quantities`
(the in-context "vocabulary" analog) and operators `+ - * /`. Enforces **syntactic/lexical**
validity (well-formed calculations over in-problem numbers) but NOT arithmetic correctness (`=` may
be wrong) — the same asymmetry that made symbolic a meaningful baseline in PrOntoQA. Compiled once
per item via XGrammar.

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

## Implementation order (after confirmation)
**1 datagen → 2 checker → G0 → 3 grammar → 4 runner/plots.** Pilot (G1/G2/G3) before full.

## Open questions to confirm
1. **Distractor source** to reach 8k: (i) GSM-IC's released distractor sentences, (ii) harvest
   topic/name-matched sentences from other GSM8K problems, or (iii) template-generate per Shi et
   al.'s three criteria. Recommendation: (iii) for control + scale, seeded. Your call.
2. **Step boundary**: detect a calculation by `<<...>>` / an `= number` line / newline. Recommend
   "a line containing `= <number>`" as the boundary, with a per-step token cap as today.
3. **Symbolic operands**: constrain to `problem_quantities` only, or also allow small constants
   (e.g. 0–10, 100) the model may legitimately introduce? Recommend problem numbers + a small
   whitelist.
4. **Keep `sentence-transformers`** installed (unused) for branch symmetry, or drop it from `[colab]`?
