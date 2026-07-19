"""Offline (GPU-free) false-reject diagnostic for the semantic checker.

Motivation: G2 failed -- semantic never beats symbolic, and the reject rate rises with context.
A high reject rate is only a bug if the rejected steps were VALID. This tool measures the
checker's INTRINSIC false-reject rate on known-valid steps (the gold proof chain) as a function
of context size, with NO embedder and NO GPU.

Why this is embedding-independent (and why tau is irrelevant here):
  With the predicate guard, a valid step is accepted only if some candidate/expected clause is
  BOTH cosine >= tau AND parse-equal to the step. For a verbatim restatement the matching
  candidate is the step's own text (cosine 1.0); for a modus-ponens step the expected string is
  rendered in the same canonical form (cosine 1.0). So cosine >= tau is trivially satisfied for a
  valid step whose text matches a candidate/expected -- the ONLY things that can reject a valid
  step are (b) parse_clause failing on its surface form, or (c) the MP synthesis not reaching it.
  This replay reproduces exactly that accept logic (restate-clause-match OR mp-clause-match) over
  the BUCKETED context (gold + distractors), so any context-size effect shows up here.

Candidate-set note (hypothesis (a)): the checker matches against the FULL candidate set
(`sims = cand_embs @ emb`; `above = nonzero(sims >= tau)`), never a top-k -- confirmed by code.
Cosine is pairwise, so the true supporting sentence's score does not decay as distractors grow.
The only residual embedding risk is a gold step that is a PARAPHRASE (parse-equal to a candidate
but different surface text) -- we count those separately as ``paraphrase_restate`` since their
acceptance depends on the real embedder's cosine, not on the logic.

Usage:
    # over the data already on Drive (no GPU, no PrOntoQA needed):
    python diagnostics/offline_false_reject.py --data-dir <DRIVE>/data
    # or generate a fresh sample locally via the bridge:
    python diagnostics/offline_false_reject.py --generate --prontoqa-dir <clone> --n-base 20
"""
from __future__ import annotations

import argparse
from collections import defaultdict

from motivation_exp import grammar as gr
from motivation_exp.datagen import Item


def _ingest(clause, rules, frontier):
    if clause is None:
        return
    if clause.kind == "rule":
        rules.append((clause.subject, clause.pred, clause.is_property, clause.negated))
    elif not clause.is_property and not clause.negated:
        frontier.add((clause.subject, clause.pred))


def _expected_clauses(rules, frontier):
    exp = set()
    for (entity, concept) in frontier:
        for (ante, cons, is_prop, rneg) in rules:
            if ante == concept:
                exp.add(gr.Clause("fact", entity, cons, is_prop, rneg))
    return exp


def replay_gold_chain(item: Item) -> list[dict]:
    """Replay an item's gold chain through the checker's ACCEPT logic (embedding-free).

    Returns one record per gold step: whether the checker's logic would accept it, and if not,
    the cause. Gold steps are accepted-to-continue regardless (they are valid by construction),
    so a false reject on an early step does NOT cascade in this measurement -- we isolate the
    per-step intrinsic false-reject rate.
    """
    cand_clauses = [gr.parse_clause(s) for s in item.context]
    context_text = set(item.context)
    rules: list = []
    frontier: set = set()
    for c in cand_clauses:
        _ingest(c, rules, frontier)
    accepted_clauses = list(cand_clauses)

    records = []
    for step in item.gold_steps:
        sc = gr.parse_clause(step)
        restate_ok = sc is not None and any(sc == cc for cc in accepted_clauses if cc is not None)
        mp_ok = sc is not None and sc in _expected_clauses(rules, frontier)
        accept = restate_ok or mp_ok
        cause = None
        if not accept:
            cause = "b_step_unparsed" if sc is None else "c_derivation_gap"
        records.append({
            "step": step,
            "parsed": sc is not None,
            "restate": restate_ok,
            "mp": mp_ok,
            "accept": accept,
            "cause": cause,
            # paraphrase = accepted via restate but text not literally in context (cosine matters)
            "paraphrase_restate": bool(restate_ok and step not in context_text),
        })
        # accept the (valid) gold step to continue the chain
        accepted_clauses.append(sc)
        _ingest(sc, rules, frontier)
    return records


def diagnose(items_by_bucket: dict[int, list[Item]]) -> dict:
    """Aggregate the false-reject breakdown per bucket."""
    out = {}
    for bucket in sorted(items_by_bucket):
        items = items_by_bucket[bucket]
        n_steps = n_false = n_b_step = n_c = n_para = 0
        ctx_sents = ctx_unparsed = 0
        for it in items:
            # context parse health (contributes to (c): unparsed context drops candidates/rules)
            for s in it.context:
                ctx_sents += 1
                ctx_unparsed += gr.parse_clause(s) is None
            for r in replay_gold_chain(it):
                n_steps += 1
                if not r["accept"]:
                    n_false += 1
                    n_b_step += r["cause"] == "b_step_unparsed"
                    n_c += r["cause"] == "c_derivation_gap"
                n_para += r["paraphrase_restate"]
        out[bucket] = {
            "n_items": len(items), "n_gold_steps": n_steps,
            "false_rejects": n_false, "false_reject_rate": n_false / max(1, n_steps),
            "cause_b_step_unparsed": n_b_step, "cause_c_derivation_gap": n_c,
            "paraphrase_restate": n_para,
            "context_sentences": ctx_sents, "context_unparsed": ctx_unparsed,
            "context_unparsed_rate": ctx_unparsed / max(1, ctx_sents),
        }
    return out


def summarize_reject_dump(path: str) -> None:
    """Offline: summarize a reject_dump.jsonl (from Cell 10) -- category counts by bucket.

    category 1 = strip would recover it yet rejected (flag not reaching check time -> bug);
    category 2 = verbosity/paraphrase the connective list can't reduce;
    category 3 = well-formed but not derivable -> genuinely wrong (correct reject).
    """
    import json
    from collections import Counter, defaultdict
    rows = [json.loads(l) for l in open(path) if l.strip()]
    by_bucket = defaultdict(list)
    for r in rows:
        by_bucket[r.get("bucket", 0)].append(r)
    print(f"{'bucket':>8} {'rejects':>8} {'cat1':>6} {'cat2':>6} {'cat3':>6}")
    for b in sorted(by_bucket):
        c = Counter(r["category"] for r in by_bucket[b])
        print(f"{b:>8} {len(by_bucket[b]):>8} {c[1]:>6} {c[2]:>6} {c[3]:>6}")
    total = Counter(r["category"] for r in rows)
    print(f"OVERALL cat1={total[1]} cat2={total[2]} cat3={total[3]} of {len(rows)} rejects")
    if total[1]:
        print("!! cat1>0 => strip_connectives not reaching the check-time path (bug).")


def probe_parse_robustness() -> None:
    """Show that parse_clause fails on realistic LM phrasings of a VALID step, and that stripping
    connectives recovers them. This is the false-reject channel the gold replay cannot see (the
    unguided model phrases valid steps non-canonically).
    """
    variants = [
        "Wren is a wumpus.", "So Wren is a wumpus.", "Therefore, Wren is a wumpus.",
        "Thus Wren is a wumpus.", "Then Wren is a wumpus.", "Hence Wren is a wumpus.",
        "This means Wren is a wumpus.", "We know that Wren is a wumpus.", "So, Wren is a wumpus.",
        "Wren is therefore a wumpus.", "Wren is a wumpus, so it is not slow.",
    ]
    raw_ok = sum(gr.parse_clause(v) is not None for v in variants)
    norm_ok = sum(gr.parse_clause(gr.strip_connectives(v)) is not None for v in variants)
    print("\nparse_clause robustness on realistic LM phrasings of ONE valid step:")
    print(f"  raw parse:                {raw_ok}/{len(variants)} parse")
    print(f"  after strip_connectives:  {norm_ok}/{len(variants)} parse")
    print("  => an unguided model's connective-prefixed valid steps are FALSE-rejected;")
    print("     strip_connectives before parse_clause recovers them (proposed fix, not yet wired).")


def classify_step_log(records: list[dict]) -> dict:
    """Classify logged per-step decisions (checker.step_log) into false vs correct rejects.

    Buckets by candidate_set_size (a proxy for context length). Use after a diagnostic pilot run
    with SemanticChecker(log_decisions=True).
    """
    buckets = defaultdict(lambda: {"rejects": 0, "false_rejects": 0, "unparsed": 0, "n": 0})
    for r in records:
        size = r.get("candidate_set_size", 0)
        key = 512 if size < 40 else 1024 if size < 80 else 2048 if size < 160 else 4096 if size < 320 else 8192
        b = buckets[key]
        b["n"] += 1
        if not r["accepted"]:
            b["rejects"] += 1
            b["false_rejects"] += bool(r.get("likely_false_reject"))
        b["unparsed"] += not r.get("parsed", True)
    return dict(buckets)


def print_report(report: dict) -> None:
    print(f"\n{'bucket':>8} {'items':>6} {'steps':>6} {'FALSE-REJ':>10} {'rate':>6} "
          f"{'(b)step':>8} {'(c)deriv':>9} {'parphr':>7} {'ctx_unparsed':>13}")
    print("-" * 86)
    for b, r in report.items():
        print(f"{b:>8} {r['n_items']:>6} {r['n_gold_steps']:>6} {r['false_rejects']:>10} "
              f"{r['false_reject_rate']:>6.2f} {r['cause_b_step_unparsed']:>8} "
              f"{r['cause_c_derivation_gap']:>9} {r['paraphrase_restate']:>7} "
              f"{r['context_unparsed']:>6}/{r['context_sentences']:<6}")
    tot_b = sum(r["cause_b_step_unparsed"] for r in report.values())
    tot_c = sum(r["cause_c_derivation_gap"] for r in report.values())
    tot_f = sum(r["false_rejects"] for r in report.values())
    print("-" * 86)
    if tot_f == 0:
        print("VERDICT: 0 false rejects by checker logic -> valid steps are NOT rejected offline.")
        print("         The real-run rejects are the MODEL producing wrong steps (correct rejects),")
        print("         not a checker bug. tau/(a)/(b)/(c) are not the cause of G2.")
    else:
        dom = "b (parse failure)" if tot_b >= tot_c else "c (derivation gap)"
        print(f"VERDICT: {tot_f} false rejects; dominant cause = ({dom}). "
              f"(b)={tot_b}  (c)={tot_c}. tau is NOT the lever.")


# --------------------------------------------------------------------------------------
# data loading / generation
# --------------------------------------------------------------------------------------
def load_from_data_dir(data_dir: str, buckets) -> dict[int, list[Item]]:
    from motivation_exp import datagen as dg
    return {b: dg.load_items(f"{data_dir}/{b}/items.jsonl") for b in buckets}


def generate_sample(prontoqa_dir: str, n_base: int, targets, block_size: int) -> dict[int, list[Item]]:
    """Generate a fresh bucketed sample locally (whitespace token proxy) for the diagnostic."""
    import random
    from motivation_exp import datagen as dg, config as C
    from motivation_exp._prontoqa_bridge import ProntoQABridge

    br = ProntoQABridge(prontoqa_dir)
    n_pool = 300
    names = dg.make_concept_names(n_base * block_size + C.POOL_CONCEPT_SPACE, seed=C.GLOBAL_SEED,
                                  reserved=br.reserved_nouns())
    br.register_concepts(names)
    base_blocks = dg.partition_blocks(names[:n_base * block_size], block_size)
    pool_space = names[n_base * block_size:]

    def gen(block, i, pre):
        raw = br.generate(C.N_HOPS, block, ontology=C.ONTOLOGY, distractors=C.DISTRACTORS,
                          deduction_rule=C.DEDUCTION_RULE, no_adjectives=C.NO_ADJECTIVES)
        return dg.raw_prontoqa_adapter(raw, item_id=f"{pre}{i}", base_id=f"{pre}{i}", concepts_hint=block)

    bases = [gen(base_blocks[i], i, "base") for i in range(n_base)]
    pool_items = [gen(random.Random(C.GLOBAL_SEED + 1 + i).sample(pool_space, block_size), i, "pool")
                  for i in range(n_pool)]
    pool = dg.build_distractor_pool(pool_items, size=10**6, seed=C.GLOBAL_SEED)
    wc = lambda t: len(t.split())
    return dg.bucket_to_token_targets(bases, wc, pool, targets, base_seed=C.GLOBAL_SEED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", help="Drive data dir with {bucket}/items.jsonl (no GPU needed)")
    ap.add_argument("--buckets", default="512,1024,2048,4096,8192")
    ap.add_argument("--generate", action="store_true", help="generate a fresh sample via the bridge")
    ap.add_argument("--prontoqa-dir", default="prontoqa")
    ap.add_argument("--n-base", type=int, default=20)
    ap.add_argument("--block-size", type=int, default=16)
    args = ap.parse_args()

    if args.generate:
        # word-count proxy targets that yield increasing distractor counts (mimics 512..8k)
        targets = [100, 300, 800, 1600, 3200]
        items = generate_sample(args.prontoqa_dir, args.n_base, targets, args.block_size)
    else:
        buckets = [int(b) for b in args.buckets.split(",")]
        items = load_from_data_dir(args.data_dir, buckets)
    print_report(diagnose(items))
    probe_parse_robustness()


if __name__ == "__main__":
    main()
