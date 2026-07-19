# Design: Frontier-Guided Resampling (variant 2 — frontier-constrained resample)

**Status:** proposed, awaiting agreement. No implementation until then.

## Motivation (from the diagnostics)
Detection is sound: 87/93 cat-3 rejects are true hallucinations (`wrong_combination` dominant —
the model recombines real in-context vocab into false steps), 6/93 are hop-skips that are
*secondary* damage from force-accepting an earlier hallucination, 0 independent checker bugs. The
failure is **blind temperature resampling**: it re-draws from the same distractor-poisoned
distribution → another hallucination → force-accept a wrong step → semantic ≤ symbolic.

Fix: on reject, the frontier already knows the derivable next steps (`_synthesize_expected`). Use
them to **constrain** the resample instead of praying. "The DAG tells you what's derivable next."

## Flow (semantic path only, on a step reject)
Inside `_emit_step`, when a step is rejected (attempt > 0), replacing today's blind temperature branch:
1. Roll back the KV cache to the step start and re-feed the boundary token (existing rollback; unchanged).
2. `E = checker.expected_steps()` — the derivable next-step surface strings synthesized from the
   current frontier × context rules (make `_synthesize_expected` public). Small by construction
   (distractor rules have disjoint concepts and never chain to the gold frontier).
3. If `E` non-empty: build a tiny per-reject grammar `G_E = root ::= (e1 | e2 | … | en) "\n"` over
   the literal `E` strings, compile it (reuse `grammar.compile_item_grammar` with a fresh matcher),
   and decode the step under `G_E`'s logit mask. The LM **chooses among** the derivable steps
   (greedy — see below); the emitted step parses to a clause in `E`, so the checker accepts it by
   construction (MP branch, parse-equal to an expected clause, cosine 1.0 ≥ τ).
4. Else fall back (Q1).

A `G_E`-constrained step is accepted by construction, so a frontier-guided reject resolves in **one**
guided attempt — the `max_retries` budget is only consumed by the blind fallback path.

## Q1 — Fallback semantics when E is empty or constrained decoding fails
Force-accept **remains the last resort**, unchanged; frontier guidance only makes it rare.

- **E empty** (frontier yields nothing derivable — usually a broken frontier from a prior
  forced-accept, or proof already complete which is handled earlier by the answer-line terminal):
  fall back to today's behavior for the remaining retries — blind temperature resample, then
  force-accept the best attempt after `max_retries`.
- **Constrained decode fails** (grammar can't complete within `per_step_cap`, or an XGrammar/compile
  error): fall back to blind resample for that attempt.
- **Forced-accept selection** (after `max_retries`): prefer an attempt whose clause ∈ `E` if any;
  else best-cosine (as today).

**Logging (all opt-in via the existing counters + `log_decisions`):** extend `SemanticStats` and the
JSONL overhead with
`n_frontier_guided` (rejects resolved by E-constrained decode),
`n_frontier_empty` (E was empty → blind fallback),
`n_constrained_fail` (E present but constrained decode failed → blind fallback), and keep
`n_forced_accepts`, now tagged with cause `{empty_E | constrained_fail | exhausted}`.
Rationale: force-accepting a wrong step is exactly what broke frontiers before, so every forced
accept must be auditable. If `n_forced_accepts` stays high, the method isn't working and the logs
show it immediately.

## Q2 — Latency accounting (charged to semantic per-accepted-token cost, same clock discipline)
Requirement: the new work must land in `decode_wall_s` (the denominator of `per_token_ms`) and be
decomposed for analysis, using the **same discipline** as today (`time.perf_counter` around CPU
regions; `torch.cuda.synchronize()` bracketing GPU forwards).

- **Automatically included:** all of `_emit_step` (rollback + frontier synth + grammar build +
  constrained decode) already runs *inside* the `decode_wall_s` region that wraps `semantic_decode`
  in `run_semantic`. So the guidance overhead is charged to per-accepted-token latency by
  construction — no denominator change.
- **Decomposition (new overhead fields), timed like the existing `check_total_s` / `grammar_compile_s`:**
  `frontier_synth_s` (perf_counter around `expected_steps()`),
  `grammar_build_s` (perf_counter around the per-reject `G_E` compile),
  `constrained_mask_s` (perf_counter around mask fill; the constrained forwards are already inside
  `backend.forward_s` with `cuda.synchronize()`). This lets us state "X% of semantic latency is
  frontier guidance," mirroring today's check/resample decomposition.
- **O(context) preserved (Panel B mechanism intact):** `expected_steps()` iterates frontier × rules
  = O(context) per step (same order as the restate cosine that already dominates Panel B) and yields
  a *small* `E`; `G_E` build is O(|E|) (negligible, not O(context)); the constrained mask is O(vocab)
  per token only during the (rare) guided resample. Net: guidance **replaces ≤3 wasted blind passes
  with 1 targeted pass**, so absolute latency drops while still **growing with context** (the per-step
  frontier synthesis + full-context associative match is the growing cost). CAM mapping unchanged.

## Q3 — Unguided and symbolic code paths untouched
The change is confined to the semantic path:
- Edits: `checker.py` (make `_synthesize_expected` public as `expected_steps()`; no logic change),
  `runner.py` `run_semantic` / `semantic_decode` / `_emit_step` (the guided-reject branch), and an
  **additive** `grammar.build_frontier_ebnf(E)` helper.
- `run_unguided` and `run_symbolic` do not call `semantic_decode`, the checker, or the new helper —
  their loops are byte-for-byte unchanged. The symbolic path keeps its static per-item compiled
  grammar; `build_item_ebnf` / `make_compiler` are reused read-only.
- Guard: a test asserts `run_unguided` / `run_symbolic` produce identical rows to `main` before the
  change (or, minimally, that their functions are untouched in the diff). The full-run defaults are
  unchanged; frontier guidance is gated by a new `DecodeCfg.frontier_guided` flag (default **False**)
  so the current semantic behavior and the full run are preserved until we opt in.

## Q4 — Is it really "guided choice" or snap-to-frontier? (|E| + chosen-rank logging)
If `|E| == 1` most of the time, variant 2 degenerates into **snap-to-frontier** in practice (the
structure dictates the step; the LM isn't choosing), which changes how we describe the method and
whether Panel A is "LM chose correctly under guidance" vs "structure dictated." We must measure this
BEFORE a reviewer does. Cheap to log now, expensive to discover later.

- **Every guided pass logs `|E|`** (the number of derivable next steps offered). Aggregated into a
  per-bucket `|E|` distribution in the overhead (`e_size_hist`, `e_size_mean`, `frac_E_gt1`).
- **When `|E| > 1`, log the chosen step's rank under the *unconstrained* LM.** Computed cheaply at
  the first branch position (the first token where the E candidates diverge): rank the chosen
  (constrained-greedy) token among the distinct allowed tokens there by the *unconstrained* logits
  already in hand. `rank == 1` ⇒ the LM would have picked it anyway (guidance passive);
  `rank > 1` ⇒ guidance actively overrode the LM's preference. Reported as a per-bucket distribution.
- Interpretation gate for the writeup: if `frac_E_gt1` is small and `rank == 1` dominates, we
  describe it as "structure-narrowed, LM-confirmed" (near snap); if `|E|>1` is common with `rank>1`
  frequent, it is genuinely "LM chooses among derivable steps under guidance."

## Temperature schedule & determinism
- **Constrained resample is GREEDY (temperature 0).** `G_E` already restricts to derivable steps, so
  greedy picks the most-probable derivable step — deterministic and reproducible, no benefit from
  sampling a set that is all-correct. The `temp_schedule` (0.7/0.9/1.1) is therefore **not used** on
  the guided path; it applies **only** to the blind fallback (E empty / constrained fail), unchanged.
- **Determinism:** greedy constrained decode is deterministic irrespective of RNG. The blind fallback
  still samples via `sample_fn`, which under `deterministic_decode` is seeded per item by the existing
  `_maybe_seed(cfg, item)` — so the whole method is reproducible for a controlled A/B.

## Contrast with symbolic (why this is the method the motivation wants)
- Symbolic: **static** per-item grammar (compiled once), constrains to any syntactically-valid
  in-vocab step → flat latency, no semantic derivability.
- Frontier-guided semantic: **dynamic** per-step constraint recomputed from the growing accepted
  chain → constrains to **semantically-derivable** steps → latency grows with context.
- Both now *guide* generation, but one is context-independent (flat) and one is context-dependent
  (growing): Panel A gap opens (semantic finally emits derivable steps) **and** Panel B gap opens
  (per-step frontier cost grows). Also fixes the 6 hop-skips for free (never force-accepting a wrong
  step keeps the frontier intact, preventing the cascade).

## Open risks / to decide
- Keep the LM **choosing among** derivable steps (this variant), not being handed one, or a reviewer
  says "the structure wrote the proof." Greedy-within-`G_E` still lets the LM select the branch.
- Multi-hop skips: one-hop `E` rejects a valid 2-hop-conclusion step and guides to the one-hop step
  (enforces step-by-step). Acceptable; optionally accept full-closure-derivable skips later (separate
  checker change, out of scope here).
- `G_E` must generate the `E` strings **exactly** (literal alternation) so acceptance is guaranteed;
  escape terminals as in `build_item_ebnf`.
