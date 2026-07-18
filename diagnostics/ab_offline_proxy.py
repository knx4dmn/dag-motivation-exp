"""Offline A/B proxy: simulate the unguided model's connective-prefixed output on gold chains,
run the checker STRIP_CONNECTIVES OFF vs ON over increasing context sizes, report valid-step
accept rate / unparsed / rejects, and the negation+swap false-accept rate (must be 0)."""
import sys
import numpy as np
import random

PQ = sys.argv[1]
from motivation_exp._prontoqa_bridge import ProntoQABridge
from motivation_exp import datagen as dg, grammar as gr, config as C
from motivation_exp.checker import SemanticChecker

DIM = 4096
def stub(texts):
    o = np.zeros((len(texts), DIM), np.float32)
    for i, t in enumerate(texts):
        for w in t.lower().replace(".", " ").split():
            o[i, hash(w) % DIM] += 1.0
        n = np.linalg.norm(o[i]); o[i] /= n if n else 1
    return o

CONN = ["So ", "Therefore, ", "Thus ", "Then ", "Hence ", "This means ", "We know that "]
def model_phrasing(step, k):
    """Simulate the unguided LM: prefix most steps with a discourse connective."""
    if k % 4 == 0:
        return step                                   # ~25% canonical
    lead = CONN[k % len(CONN)]
    return lead + step[0].lower() + step[1:]

# ---- generate data ----
br = ProntoQABridge(PQ)
names = dg.make_concept_names(20 * 16 + C.POOL_CONCEPT_SPACE, seed=0, reserved=br.reserved_nouns())
br.register_concepts(names)
base_blocks = dg.partition_blocks(names[:20 * 16], 16)
pool_space = names[20 * 16:]
def gen(block, i, pre):
    raw = br.generate(3, block, ontology="fictional", distractors="relevant",
                      deduction_rule="ModusPonens", no_adjectives=True)
    return dg.raw_prontoqa_adapter(raw, item_id=f"{pre}{i}", base_id=f"{pre}{i}", concepts_hint=block)
bases = [gen(base_blocks[i], i, "base") for i in range(20)]
pool_items = [gen(random.Random(1 + i).sample(pool_space, 16), i, "pool") for i in range(300)]
pool = dg.build_distractor_pool(pool_items, size=10**6, seed=0)
wc = lambda t: len(t.split())
targets = [100, 300, 800, 1600, 3200]
buckets = dg.bucket_to_token_targets(bases, wc, pool, targets, base_seed=0)

# ---- SAFETY: strip is a no-op (parse-invariant) on every real gold step ----
n_gold = n_changed = 0
for its in buckets.values():
    for it in its:
        for s in it.gold_steps:
            n_gold += 1
            if gr.parse_clause(gr.strip_connectives(s)) != gr.parse_clause(s):
                n_changed += 1
print(f"SAFETY: strip_connectives changed the parsed clause on {n_changed}/{n_gold} real gold steps "
      f"(must be 0)\n")

# ---- A/B over buckets ----
all_concepts = sorted({c for it in bases for c in it.concepts})

def gold_derivable(chk, clause):
    """Embedding-free: is this clause a restate of a candidate OR a synthesized MP step?"""
    if clause is None:
        return False
    if any(clause == cc for cc in chk._cand_clauses if cc is not None):
        return True
    return any(clause == c for _, c in chk._synthesize_expected())

def run(items, flag):
    tot = acc = rej = unp = 0
    fa_neg = fa_swap = n_neg = n_swap = 0
    pr = random.Random(0)
    for it in items:
        chk = SemanticChecker(stub, 0.6, 0.6, predicate_guard=True, strip_connectives=flag)
        chk.prefill(it.context)
        for k, s in enumerate(it.gold_steps):
            ms = model_phrasing(s, k)
            before = chk.n_unparsed_steps
            r = chk.check_step(ms); tot += 1
            acc += r.accepted; rej += (not r.accepted); unp += (chk.n_unparsed_steps - before)
            # false-accept probes: only count perturbations that are GENUINELY NOT gold-derivable
            # (a random swap can accidentally hit a real next-hop -> that's a correct accept, skip).
            neg = gr.negate_sentence(s)
            others = [c for c in all_concepts if c not in s]
            sw = gr.swap_concept(s, pr.choice(others)) if others else None
            for probe, is_neg in [(neg, True), (sw, False)]:
                if not probe or probe == s:
                    continue
                pc = gr.parse_clause(gr.strip_connectives(probe))
                if gold_derivable(chk, pc):
                    continue                            # coincidentally valid -> not a wrong-probe
                accepted = chk.check_step(model_phrasing(probe, k)).accepted
                if is_neg: fa_neg += accepted; n_neg += 1
                else:      fa_swap += accepted; n_swap += 1
            chk.accept(ms)                              # advance chain with what the checker saw
    return dict(acc=acc/tot, rej=rej/tot, unp=unp, fa_neg=fa_neg/max(1,n_neg), fa_swap=fa_swap/max(1,n_swap))

print(f"{'bucket':>8} {'flag':>4} {'valid_accept':>13} {'reject_rate':>12} {'unparsed':>9} "
      f"{'FA_neg':>7} {'FA_swap':>8}")
print("-" * 70)
for b in targets:
    for flag in (False, True):
        m = run(buckets[b], flag)
        print(f"{b:>8} {'ON' if flag else 'OFF':>4} {m['acc']:>13.2f} {m['rej']:>12.2f} "
              f"{m['unp']:>9} {m['fa_neg']:>7.2f} {m['fa_swap']:>8.2f}")
    print()
