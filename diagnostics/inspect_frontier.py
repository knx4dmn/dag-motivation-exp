"""Offline inspection of the frontier-guided E set (no GPU, no model).

`E = checker.expected_steps()` depends only on the frontier/rules the checker builds from the
context + accepted steps -- pure logic, embedding-independent. So we can replay each item's GOLD
chain through the real SemanticChecker (with a dummy encoder) and, at every step, print E verbatim:
its size, whether the gold next step is in it, and how many members are about the item's own
(query) entity vs distractor entities. This shows directly whether E is well-formed and
gold-relevant or polluted with the distractor forward-closure.

Usage:  python diagnostics/inspect_frontier.py <data_dir> <bucket> [n_items]
"""
from __future__ import annotations

import sys
from collections import Counter

import numpy as np

from motivation_exp import datagen as dg, grammar as gr
from motivation_exp.checker import SemanticChecker


def _dummy_encode(texts):
    return np.zeros((len(texts), 8), dtype=np.float32)


def _query_entity(item) -> str | None:
    c = gr.parse_clause(item.question)
    if c and c.kind == "fact":
        return c.subject
    for s in item.gold_steps:                    # fall back to the chain's entity
        c = gr.parse_clause(s)
        if c and c.kind == "fact":
            return c.subject
    return None


def inspect(item, max_E_print=12):
    qent = _query_entity(item)
    print(f"\n{'='*90}\nitem {item.item_id}  bucket={item.bucket}  query={item.question!r}  "
          f"query_entity={qent}  gold_len={len(item.gold_steps)}  n_context={len(item.context)}")
    chk = SemanticChecker(_dummy_encode, 0.6, 0.6)
    chk.prefill(item.context, query_entity=qent)
    for i, gstep in enumerate(item.gold_steps):
        E = chk.expected_steps()
        gc = gr.parse_clause(gstep)
        gold_in_E = gstep in E or (gc is not None and any(gr.parse_clause(e) == gc for e in E))
        # entity breakdown of E
        ents = Counter()
        for e in E:
            ce = gr.parse_clause(e)
            ents[ce.subject if ce and ce.kind == "fact" else "(rule/none)"] += 1
        own = ents.get(qent, 0)
        print(f"\n  step {i}: GOLD next = {gstep!r}  (kind={gc.kind if gc else None})")
        print(f"    |E| = {len(E)}   gold_in_E = {gold_in_E}   "
              f"E-members about query_entity({qent}) = {own} / {len(E)}")
        top_ents = ", ".join(f"{k}:{v}" for k, v in ents.most_common(6))
        print(f"    E entity breakdown: {top_ents}")
        for e in E[:max_E_print]:
            print(f"      - {e!r}")
        if len(E) > max_E_print:
            print(f"      ... (+{len(E) - max_E_print} more)")
        chk.accept(gstep)


def main():
    data_dir, bucket = sys.argv[1], int(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    items = dg.load_items(f"{data_dir}/{bucket}/items.jsonl")[:n]
    print(f"inspecting {len(items)} items from bucket {bucket}")
    # summary across all items/steps
    sizes, gold_hits, gold_total = [], 0, 0
    for it in items:
        inspect(it)
        qent = _query_entity(it)
        chk = SemanticChecker(_dummy_encode, 0.6, 0.6); chk.prefill(it.context, query_entity=qent)
        for g in it.gold_steps:
            E = chk.expected_steps(); sizes.append(len(E))
            gc = gr.parse_clause(g)
            if gc and gc.kind == "fact":         # only MP-fact gold steps are expected to be in E
                gold_total += 1
                if g in E or any(gr.parse_clause(e) == gc for e in E):
                    gold_hits += 1
            chk.accept(g)
    print(f"\n{'='*90}\nSUMMARY bucket {bucket}: |E| mean={np.mean(sizes):.1f} max={max(sizes)} "
          f"min={min(sizes)} frac(|E|==1)={np.mean([s==1 for s in sizes]):.2f}")
    print(f"gold MP-fact step present in E: {gold_hits}/{gold_total}")


if __name__ == "__main__":
    main()
