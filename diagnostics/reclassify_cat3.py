"""Offline reclassification of cat-3 (well-formed but rejected) semantic steps.

For each rejected step's parsed Clause, decide against the item's GOLD-DERIVABLE CLOSURE
(all context clauses + every modus-ponens-derivable entity fact):
  (i)   absent from the closure            -> true hallucination (typically a name/predicate swap)
  (ii)  present in the closure but rejected -> a correct conclusion rejected (hop-skip / broken
        prerequisite in the accepted-so-far chain)
  (iii) other (unparseable, etc.)

GPU-free. Usage:
  python diagnostics/reclassify_cat3.py <reject_dump.jsonl> <data_dir>
where data_dir has {bucket}/items.jsonl.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict

from motivation_exp import grammar as gr, datagen as dg


def gold_closure(context):
    """All context clauses + every MP-derivable entity fact (deductive closure), as a set of Clause."""
    clauses = [c for c in (gr.parse_clause(s) for s in context) if c is not None]
    G = set(clauses)                                           # restates of context facts + rules
    rules = [(c.subject, c.pred, c.is_property, c.negated) for c in clauses if c.kind == "rule"]
    frontier = {(c.subject, c.pred) for c in clauses
                if c.kind == "fact" and not c.is_property and not c.negated}
    changed = True
    while changed:
        changed = False
        for (e, con) in list(frontier):
            for (ante, cons, isprop, rneg) in rules:
                if ante != con:
                    continue
                nc = gr.Clause("fact", e, cons, isprop, rneg)
                if nc not in G:
                    G.add(nc); changed = True
                if not isprop and not rneg and (e, cons) not in frontier:
                    frontier.add((e, cons)); changed = True
    return G


def _lev1(a: str, b: str) -> bool:
    """True if edit distance(a, b) <= 1 (cheap: length diff <=1 + single mismatch)."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        return sum(x != y for x, y in zip(a, b)) == 1
    if la > lb:
        a, b, la, lb = b, a, lb, la                            # a is shorter
    i = j = 0; diff = 0
    while i < la and j < lb:
        if a[i] != b[j]:
            diff += 1; j += 1
            if diff > 1:
                return False
        else:
            i += 1; j += 1
    return True


def near_miss(clause, vocab) -> bool:
    """Is the clause's subject or predicate an edit-distance-1 neighbor of a real vocab token?"""
    for tok in (clause.subject, clause.pred):
        if tok not in vocab and any(_lev1(tok, v) for v in vocab):
            return True
    return False


def main():
    dump_path, data_dir = sys.argv[1], sys.argv[2]
    rows = [json.loads(l) for l in open(dump_path) if l.strip()]
    cat3 = [r for r in rows if r.get("category") == 3]

    # index items by id per bucket
    items = {}
    buckets = sorted({r["bucket"] for r in rows})
    for b in buckets:
        for it in dg.load_items(f"{data_dir}/{b}/items.jsonl"):
            items[it.item_id] = it
    closures, vocabs = {}, {}

    by_bucket = defaultdict(Counter)
    ii_examples, i_flavor = [], Counter()
    swap_examples = []
    for r in cat3:
        it = items.get(r["item"])
        if it is None:
            by_bucket[r["bucket"]]["iii"] += 1; continue
        if it.item_id not in closures:
            closures[it.item_id] = gold_closure(it.context)
            # vocab from the FULL bucketed context (gold + distractors + properties)
            e, c, p = dg.extract_vocabulary(it.context)
            vocabs[it.item_id] = ({x.lower() for x in c} | {x.lower() for x in p} | {x for x in e})
        K = gr.parse_clause(r.get("stripped") or r["raw"])
        if K is None:
            by_bucket[r["bucket"]]["iii"] += 1; continue
        if K in closures[it.item_id]:
            by_bucket[r["bucket"]]["ii"] += 1
            ii_examples.append((r["bucket"], r["item"], r["raw"]))
        else:
            by_bucket[r["bucket"]]["i"] += 1
            vocab = vocabs[it.item_id]
            toks = [K.subject, K.pred]
            in_vocab = [t for t in toks if (t if t[0].isupper() else t.lower()) in vocab]
            if len(in_vocab) == len(toks):
                i_flavor["wrong_combination"] += 1        # all tokens real, statement false
            elif near_miss(K, vocab):
                i_flavor["coined_near_miss"] += 1          # a token is edit-dist-1 of a real one
                if len(swap_examples) < 8:
                    swap_examples.append((r["bucket"], r["raw"], r["top3_candidates"][0] if r["top3_candidates"] else None))
            else:
                i_flavor["out_of_vocab"] += 1

    print(f"{'bucket':>8} {'cat3':>6} {'(i)halluc':>10} {'(ii)hopskip':>12} {'(iii)other':>11}")
    tot = Counter()
    for b in buckets:
        c = by_bucket[b]; tot.update(c)
        n = c["i"] + c["ii"] + c["iii"]
        if n:
            print(f"{b:>8} {n:>6} {c['i']:>10} {c['ii']:>12} {c['iii']:>11}")
    print("-" * 50)
    print(f"{'TOTAL':>8} {sum(tot.values()):>6} {tot['i']:>10} {tot['ii']:>12} {tot['iii']:>11}")
    print(f"\n(i) hallucination flavors: coined_near_miss={i_flavor['coined_near_miss']}  "
          f"wrong_combination(real vocab, false)={i_flavor['wrong_combination']}  "
          f"out_of_vocab={i_flavor['out_of_vocab']}")
    print("  near-miss examples [bucket | rejected | nearest context candidate]:")
    for b, raw, top in swap_examples:
        print(f"    b{b}: {raw!r}  ~  {top}")
    print(f"\n(ii) correct-conclusion-rejected: {len(ii_examples)}")
    for b, item, raw in ii_examples[:10]:
        print(f"  [b{b} {item}] {raw!r}")


if __name__ == "__main__":
    main()
