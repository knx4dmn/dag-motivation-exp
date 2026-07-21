"""G0 pre-flight gate for the GSM8K motivation task (docs/gsm8k_motivation_plan.md).

Runs BEFORE any guidance is wired in: generate unguided CoT, run the MathChecker POST-HOC on every
calculation step (observation only), and report per bucket:
  (a) step-pass rate  = fraction of checkable steps the checker accepts (model↔checkable overlap);
  (b) catchable fraction = among WRONG-answer items, fraction with >=1 checker-rejected step
      (references a distractor quantity or fails arithmetic) vs checker-invisible (clean arithmetic,
      valid provenance, wrong plan).
Pre-committed PROCEED thresholds (awaiting sign-off): (a) >= 0.50 in BOTH 1k and 2k, (b) >= 1/3.

Core (segment / check / extract / aggregate) is GPU-free and unit-tested; ``run_g0`` does the
unguided generation on Colab.
"""
from __future__ import annotations

import re
from collections import defaultdict

from .gsm8k_datagen import normalize_number, to_fraction
from .math_checker import MathChecker, parse_calculation

_HASH_RE = re.compile(r"####\s*(-?\$?[\d,]+(?:\.\d+)?)")
_ANS_RE = re.compile(r"answer is\s*\$?(-?[\d,]+(?:\.\d+)?)", re.IGNORECASE)
_NUM_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")

# fixed few-shot exemplar (calculation format the checker + grammar expect)
EXEMPLAR_Q = ("Natalia sold clips to 48 of her friends in April, and then she sold half as many "
              "clips in May. How many clips did Natalia sell altogether in April and May?")
EXEMPLAR_COT = ("In May she sold 48 / 2 = 24 clips.\n"
                "Altogether she sold 48 + 24 = 72 clips.\n#### 72")
SYSTEM_PROMPT = ("Solve the math word problem step by step. Show each calculation on its own line in "
                 "the form 'a op b = c', then end with a line '#### <answer>'.")


def extract_number_answer(text: str) -> str | None:
    """Final numeric answer from model output: '#### N', else 'answer is N', else the last number."""
    m = _HASH_RE.search(text)
    if m:
        return normalize_number(m.group(1))
    m = _ANS_RE.search(text)
    if m:
        return normalize_number(m.group(1))
    nums = _NUM_RE.findall(text)
    return normalize_number(nums[-1]) if nums else None


def posthoc_verdicts(item, cot_text: str):
    """Segment CoT into calculation steps (lines with `= <number>`) and check each post-hoc.

    RECORD-ALL semantics (edge case 1): after checking a step we record its result as a derived
    intermediate **regardless of the step's own verdict** -- there is no resampling in G0, so a later
    step referencing an earlier (even wrong) result must trace to 'intermediate', not 'missing'. This
    charges each step only its own first-order error and prevents an arithmetic failure from
    cascading into spurious 'missing' verdicts that could falsely trigger STOP-a. (The real runner
    uses accepted-only semantics instead.) Returns list of (step_line, MathCheckResult)."""
    chk = MathChecker()
    chk.prefill(item.context, item.relevance)
    out = []
    for line in cot_text.splitlines():
        if parse_calculation(line) is None:
            continue
        out.append((line.strip(), chk.check_step(line)))   # verdict charged to THIS step only
        chk.accept(line)                                    # record its result regardless of verdict
    return out


def g0_item(item, cot_text: str) -> dict:
    """Per-item G0 record."""
    verdicts = posthoc_verdicts(item, cot_text)
    n_check = len(verdicts)
    n_pass = sum(v.accepted for _, v in verdicts)
    pred = extract_number_answer(cot_text)
    try:
        correct = pred is not None and to_fraction(pred) == to_fraction(item.gold)
    except (ValueError, ZeroDivisionError):
        correct = False
    first_reject = next(((s, v.failed) for s, v in verdicts if not v.accepted), None)
    return {"bucket": item.bucket, "item_id": item.item_id, "n_checkable": n_check, "n_pass": n_pass,
            "correct": correct, "has_reject": first_reject is not None, "first_reject": first_reject,
            "pred": pred, "gold": item.gold,
            "verdicts": [{"step": s, "accepted": v.accepted, "failed": v.failed} for s, v in verdicts]}


def g0_aggregate(rows) -> dict:
    """Per-bucket stats + a pooled entry.

    Reports (a) step-pass rate WITH its failure decomposition {missing, distractor, arithmetic}
    (amendment 1), a parse rate (fraction of items with >=1 detected calculation line, amendment 3),
    and (b) catchable fraction among wrong-answer items -- per bucket AND pooled (amendment 4).
    """
    by = defaultdict(list)
    for r in rows:
        by[r["bucket"]].append(r)

    def _fail_counts(rs):
        fm = fd = fa = 0
        for r in rs:
            for v in r["verdicts"]:
                if not v["accepted"]:
                    fm += v["failed"] == "provenance_missing"
                    fd += v["failed"] == "provenance_distractor"
                    fa += v["failed"] == "arithmetic"
        return fm, fd, fa

    out = {}
    for b, rs in by.items():
        steps = sum(r["n_checkable"] for r in rs)
        passed = sum(r["n_pass"] for r in rs)
        wrong = [r for r in rs if not r["correct"]]
        fm, fd, fa = _fail_counts(rs)
        parsed = sum(r["n_checkable"] >= 1 for r in rs)
        out[b] = {"n_items": len(rs), "acc": sum(r["correct"] for r in rs) / max(1, len(rs)),
                  "parse_rate": parsed / max(1, len(rs)),
                  "step_pass": passed / max(1, steps), "n_steps": steps,
                  "fail_missing": fm, "fail_distractor": fd, "fail_arith": fa,
                  "n_wrong": len(wrong), "catchable_frac": sum(r["has_reject"] for r in wrong) / max(1, len(wrong))}
    all_rows = [r for rs in by.values() for r in rs]
    wrong_all = [r for r in all_rows if not r["correct"]]
    fm, fd, fa = _fail_counts(all_rows)
    out["_pooled"] = {"n_wrong": len(wrong_all), "fail_missing": fm, "fail_distractor": fd, "fail_arith": fa,
                      "catchable_frac": sum(r["has_reject"] for r in wrong_all) / max(1, len(wrong_all))}
    return out


def g0_verdict(agg: dict, pass_a: float = 0.50, pass_b: float = 1 / 3, min_parse: float = 0.70) -> str:
    """PROCEED / STOP-a / STOP-b / FIX-PROMPTING per the decomposed-threshold rule (amendment 1)."""
    buckets = [b for b in agg if b != "_pooled"]
    if any(agg[b]["parse_rate"] < min_parse for b in buckets):
        return "FIX_PROMPTING"
    p = agg["_pooled"]
    # STOP-on-(a) ONLY if the missing class dominates the failures (PrOntoQA-style mismatch)
    if p["fail_missing"] > p["fail_distractor"] + p["fail_arith"]:
        return "STOP_A"
    if p["catchable_frac"] < pass_b:
        return "STOP_B"
    return "PROCEED"   # includes step_pass < pass_a when failures are healthy (distractor + arithmetic)


def print_g0(agg: dict, pass_a: float = 0.50, pass_b: float = 1 / 3, min_parse: float = 0.70) -> None:
    buckets = [b for b in agg if b != "_pooled"]
    print(f"{'bucket':>7} {'items':>6} {'acc':>6} {'parse':>6} {'(a)pass':>8}  "
          f"{'fail m/d/a':>12}  {'(b)catch':>9} {'n_wrong':>8}")
    for b in sorted(buckets):
        a = agg[b]
        print(f"{b:>7} {a['n_items']:>6} {a['acc']:>6.2f} {a['parse_rate']:>6.2f} {a['step_pass']:>8.2f}  "
              f"{a['fail_missing']:>3}/{a['fail_distractor']:>3}/{a['fail_arith']:<3}  "
              f"{a['catchable_frac']:>9.2f} {a['n_wrong']:>8}")
    p = agg["_pooled"]
    print(f"\npooled (b) catchable = {p['catchable_frac']:.2f} over {p['n_wrong']} wrong items "
          f"| pooled failures missing/distractor/arith = {p['fail_missing']}/{p['fail_distractor']}/{p['fail_arith']}")
    v = g0_verdict(agg, pass_a, pass_b, min_parse)
    msg = {
        "FIX_PROMPTING": f"FIX PROMPTING first: parse rate < {min_parse:.2f} in some bucket (few detected calculations).",
        "STOP_A": "STOP (a): the MISSING/hallucinated-quantity class dominates failures -> model-task mismatch; escalate (Qwen2.5-3B / 7B int8).",
        "STOP_B": f"STOP (b): pooled catchable {p['catchable_frac']:.2f} < {pass_b:.2f} -> checker not discriminative; pivot (HotpotQA-distractor / MuSiQue).",
        "PROCEED": "PROCEED to steps 3-4. (If (a) < 0.50, failures are the healthy distractor+arithmetic kind -> shows up in (b).)",
    }[v]
    print("VERDICT:", msg)


def run_g0(model, tokenizer, items, *, n_per_bucket: int = 20, max_new_tokens: int = 400):
    """Colab: unguided greedy CoT generation + post-hoc G0 records over ``items`` (grouped by bucket)."""
    import torch

    by_bucket = defaultdict(list)
    for it in items:
        by_bucket[it.bucket].append(it)

    rows = []
    for b, its in by_bucket.items():
        for it in its[:n_per_bucket]:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": EXEMPLAR_Q},
                {"role": "assistant", "content": EXEMPLAR_COT},
                {"role": "user", "content": " ".join(it.context + [it.question])},
            ]
            ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
            cot = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            rows.append(g0_item(it, cot))
    return rows
