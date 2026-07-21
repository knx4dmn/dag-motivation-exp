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

    Each step is accepted regardless of verdict so downstream provenance uses the model's OWN chain
    (its derived intermediates). Returns list of (step_line, MathCheckResult)."""
    chk = MathChecker()
    chk.prefill(item.context, item.relevance)
    out = []
    for line in cot_text.splitlines():
        if parse_calculation(line) is None:
            continue
        out.append((line.strip(), chk.check_step(line)))
        chk.accept(line)
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
    """Per-bucket (a) step-pass rate and (b) catchable fraction among wrong-answer items."""
    by = defaultdict(list)
    for r in rows:
        by[r["bucket"]].append(r)
    out = {}
    for b, rs in by.items():
        steps = sum(r["n_checkable"] for r in rs)
        passed = sum(r["n_pass"] for r in rs)
        wrong = [r for r in rs if not r["correct"]]
        catch = sum(r["has_reject"] for r in wrong)
        out[b] = {"n_items": len(rs), "acc": sum(r["correct"] for r in rs) / max(1, len(rs)),
                  "step_pass": passed / max(1, steps), "n_steps": steps,
                  "n_wrong": len(wrong), "catchable_frac": catch / max(1, len(wrong))}
    return out


def print_g0(agg: dict, pass_a: float = 0.50, pass_b: float = 1 / 3) -> None:
    print(f"{'bucket':>7} {'items':>6} {'acc':>6} {'(a)step_pass':>13} {'(b)catchable':>13} {'n_wrong':>8}")
    for b in sorted(agg):
        a = agg[b]
        print(f"{b:>7} {a['n_items']:>6} {a['acc']:>6.2f} {a['step_pass']:>13.2f} "
              f"{a['catchable_frac']:>13.2f} {a['n_wrong']:>8}")
    a_ok = all(agg[b]["step_pass"] >= pass_a for b in agg)
    b_ok = all(agg[b]["catchable_frac"] >= pass_b for b in agg)
    print(f"\nG0 thresholds: (a) step_pass >= {pass_a:.2f} in all -> {a_ok}; "
          f"(b) catchable >= {pass_b:.2f} in all -> {b_ok}")
    print("VERDICT:", "PROCEED to steps 3-4" if (a_ok and b_ok) else
          "STOP -- " + ("(a) low: model-task mismatch, escalate model" if not a_ok else
                        "(b) low: checker not discriminative, consider HotpotQA-distractor/MuSiQue"))


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
