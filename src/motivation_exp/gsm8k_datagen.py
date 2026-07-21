"""GSM8K + GSM-IC datagen: load/parse test-split problems, synth template distractors, bucket.

See docs/gsm8k_motivation_plan.md. CPU-only, importable without the datasets library (parsing works
on raw (question, answer) strings; ``load_gsm8k`` is a thin Colab loader). Reuses generic helpers
from :mod:`datagen` (token counter, manifest, splits, IO).

Key domain choices (amendments A-D): GSM8K **test** split; template-generated distractors (seeded,
per item, names from the item's entity pool, no dup within an item); a **per-sentence relevance
bit** (1=original problem, 0=distractor) that the semantic checker reads as an oracle; exact
rational validation (``fractions.Fraction``).
"""
from __future__ import annotations

import ast
import json
import os
import random
import re
from dataclasses import dataclass, field, asdict
from fractions import Fraction
from typing import Callable, Sequence

from .datagen import CountTokens, hf_token_counter  # reuse token-counter helper + type alias

# --------------------------------------------------------------------------------------
# Numbers + safe rational arithmetic
# --------------------------------------------------------------------------------------
_NUM_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def normalize_number(tok: str) -> str:
    """Normalize a numeric token: strip $ and thousands commas; keep sign/decimal."""
    return tok.replace("$", "").replace(",", "").strip()


def to_fraction(tok: str) -> Fraction:
    """Exact rational value of a numeric string ('1,000'->1000, '3.5'->7/2, '$5'->5)."""
    return Fraction(normalize_number(tok))


# Small constants the model may legitimately introduce (half->2, twice->2, dozen->12, %->100).
# Used by BOTH arms: symbolic grammar allows them as operands; the semantic provenance check
# exempts them from tracing. Distractor numbers are generated to AVOID this set, so a distractor
# quantity always remains provenance-catchable (keeps G0(b) valid).
CONST_WHITELIST = tuple(range(0, 11)) + (12, 100)
_WHITELIST_FRAC = {Fraction(n) for n in CONST_WHITELIST}


def is_whitelist_constant(num: str) -> bool:
    try:
        return to_fraction(num) in _WHITELIST_FRAC
    except (ValueError, ZeroDivisionError):
        return False


def extract_numbers(text: str) -> list[str]:
    """All numeric tokens in ``text``, normalized, de-duplicated in first-seen order."""
    out, seen = [], set()
    for m in _NUM_RE.finditer(text):
        n = normalize_number(m.group(0))
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def safe_eval(expr: str) -> Fraction:
    """Evaluate an arithmetic expression exactly (Fraction). Only +,-,*,/,parens,numbers allowed."""
    expr = normalize_number(expr) if _NUM_RE.fullmatch(expr.strip()) else expr.replace(",", "").replace("$", "")
    node = ast.parse(expr, mode="eval")

    def _ev(n) -> Fraction:
        if isinstance(n, ast.Expression):
            return _ev(n.body)
        if isinstance(n, ast.Constant):
            return Fraction(str(n.value))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = _ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            a, b = _ev(n.left), _ev(n.right)
            if isinstance(n.op, ast.Add): return a + b
            if isinstance(n.op, ast.Sub): return a - b
            if isinstance(n.op, ast.Mult): return a * b
            return a / b
        raise ValueError(f"disallowed expression node: {ast.dump(n)}")

    return _ev(node)


# --------------------------------------------------------------------------------------
# Sentence + name extraction
# --------------------------------------------------------------------------------------
_SENT_SPLIT = re.compile(r"(?<=[.?!])\s+")
_COMMON_CAPS = {
    "The", "A", "An", "In", "On", "At", "If", "How", "What", "When", "Each", "Every", "There",
    "This", "That", "He", "She", "It", "They", "We", "I", "Then", "So", "After", "For", "Of", "To",
    "Her", "His", "Their", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine",
    # months / weekdays are capitalized but are not person names
    "January", "February", "March", "April", "May", "June", "July", "August", "September",
    "October", "November", "December", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    "Saturday", "Sunday", "Altogether", "Total",
}


def split_problem(question: str) -> tuple[list[str], str]:
    """Split a GSM8K problem into (statement_sentences, final_question_sentence)."""
    parts = [p.strip() for p in _SENT_SPLIT.split(question.strip()) if p.strip()]
    if not parts:
        return [], question.strip()
    # the query is the last sentence ending in '?' (fallback: the last sentence)
    q_idx = max((i for i, p in enumerate(parts) if p.endswith("?")), default=len(parts) - 1)
    return parts[:q_idx] + parts[q_idx + 1:], parts[q_idx]


def extract_names(text: str) -> list[str]:
    """Heuristic proper names: capitalized alpha tokens, not sentence-initial, not common words."""
    names, seen = [], set()
    for sent in _SENT_SPLIT.split(text):
        toks = re.findall(r"[A-Za-z']+", sent)
        for i, t in enumerate(toks):
            if i == 0:
                continue
            if t[:1].isupper() and t.isalpha() and t not in _COMMON_CAPS and len(t) > 1:
                if t not in seen:
                    seen.add(t)
                    names.append(t)
    return names


# --------------------------------------------------------------------------------------
# Item model
# --------------------------------------------------------------------------------------
@dataclass
class GSMItem:
    item_id: str
    base_id: str
    context: list[str]              # problem + distractor sentences (bucketed order)
    relevance: list[int]            # parallel to context: 1=original problem, 0=distractor
    question: str                   # the final question sentence
    gold: str                       # normalized final numeric answer
    gold_steps: list[str]           # ["48/2 = 24", ...] (expr = result)
    problem_quantities: list[str]   # numbers in the original problem statements
    names: list[str]                # entity pool for distractor synthesis
    gold_context: list[str] = field(default_factory=list)  # original problem sentences
    n_steps: int = 0
    bucket: int | None = None
    token_count: int | None = None
    seed: int | None = None

    def to_json(self) -> dict:
        return asdict(self)


_STEP_RE = re.compile(r"<<([^>]+)>>")
_FINAL_RE = re.compile(r"####\s*(-?\$?[\d,]+(?:\.\d+)?)")


def parse_gsm8k_item(question: str, answer: str, item_id: str, base_id: str | None = None) -> GSMItem:
    """Parse a raw GSM8K (question, answer) pair into an unbucketed :class:`GSMItem`."""
    base_id = base_id or item_id
    steps = []
    for m in _STEP_RE.finditer(answer):
        body = m.group(1)
        if "=" in body:
            expr, res = body.split("=", 1)
            steps.append(f"{expr.strip()} = {normalize_number(res.strip())}")
    fm = _FINAL_RE.search(answer)
    if not fm:
        raise ValueError(f"no '#### <answer>' in GSM8K answer for {item_id!r}")
    gold = normalize_number(fm.group(1))

    statements, query = split_problem(question)
    return GSMItem(
        item_id=item_id, base_id=base_id,
        context=list(statements), relevance=[1] * len(statements),
        question=query, gold=gold, gold_steps=steps,
        problem_quantities=extract_numbers(" ".join(statements)),
        names=extract_names(question), gold_context=list(statements),
        n_steps=len(steps),
    )


# --------------------------------------------------------------------------------------
# Distractor synthesis (template-generated; GSM-IC three criteria)
# --------------------------------------------------------------------------------------
DISTRACTOR_TEMPLATES = [
    "{name} has {num} {obj}.",
    "{name} bought {num} {obj} at the store.",
    "There are {num} {obj} in {name}'s bag.",
    "{name} gave away {num} {obj} yesterday.",
    "{name}'s friend collected {num} {obj}.",
    "Last week {name} counted {num} {obj}.",
    "{name} saw {num} {obj} at the park.",
    "The shop sold {num} {obj} to {name}.",
    "{name} received {num} {obj} as a gift.",
    "{name} kept {num} {obj} at home.",
]
_OBJECTS = ["apples", "books", "marbles", "stickers", "coins", "pencils", "cards", "toys",
            "oranges", "balloons", "cookies", "flowers", "stamps", "buttons", "shells"]
_DEFAULT_NAMES = ["Alex", "Sam", "Jamie", "Taylor", "Jordan", "Riley", "Casey", "Morgan"]


def _num_range(base: GSMItem) -> tuple[int, int]:
    vals = []
    for q in base.problem_quantities:
        try:
            vals.append(int(abs(float(q))))
        except ValueError:
            pass
    hi = max(vals) * 2 if vals else 50
    # start above the constant whitelist (<=12) so every distractor quantity is provenance-catchable
    return 13, max(20, hi)


def generate_distractors(base: GSMItem, n: int, seed: int) -> list[str]:
    """Generate up to ``n`` unique, seeded distractor sentences (irrelevant; never change the answer).

    Numbers avoid the constant whitelist so distractor-use is always catchable by the semantic
    provenance check (G0(b)).
    """
    rng = random.Random(seed)
    names = base.names or _DEFAULT_NAMES
    lo, hi = _num_range(base)
    # amendment 2 (+ edge case): a distractor number must NOT collide with this item's problem
    # quantities NOR its gold intermediate results (any formatting variant) -- else a distractor
    # reference would trace to a legitimate quantity and corrupt G0(b) attribution.
    excluded_fracs = set()
    for q in base.problem_quantities:
        try:
            excluded_fracs.add(to_fraction(q))
        except (ValueError, ZeroDivisionError):
            pass
    for step in base.gold_steps:                       # gold intermediate results (known at datagen)
        if "=" in step:
            try:
                excluded_fracs.add(to_fraction(step.split("=", 1)[1]))
            except (ValueError, ZeroDivisionError):
                pass
    out, seen = [], set()
    attempts = 0
    while len(out) < n and attempts < n * 30:
        attempts += 1
        num = rng.randint(lo, hi)
        if is_whitelist_constant(str(num)) or Fraction(num) in excluded_fracs:
            continue
        s = rng.choice(DISTRACTOR_TEMPLATES).format(
            name=rng.choice(names), num=num, obj=rng.choice(_OBJECTS))
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# --------------------------------------------------------------------------------------
# Bucketing (per-item distractors; relevance tracked)
# --------------------------------------------------------------------------------------
def _assemble(context: Sequence[str], question: str) -> str:
    return " ".join(list(context) + [question])


def bucket_gsm_item(base: GSMItem, bucket: int, count_tokens: CountTokens, seed: int,
                    tol: float = 0.05, max_correction: int = 6) -> GSMItem:
    """Interleave seeded per-item distractors into the problem to hit ``bucket`` tokens (+/- tol).

    Original problem sentences keep their order (relevance 1); distractors (relevance 0) are inserted
    at fixed-seed random positions; the question stays last. Summed pre-tokenized lengths pick the
    count; a single full tokenize + bounded correction lands within tolerance.
    """
    rng = random.Random(seed)
    base_len = count_tokens(_assemble(base.gold_context, base.question))
    # generous distractor supply; ~ (bucket - base) / avg_distractor_len, with headroom
    est = max(4, int((bucket - base_len) / 6) + 16)
    distractors = generate_distractors(base, est * 2, seed)
    dlens = [count_tokens(d) for d in distractors]
    avg = (sum(dlens) / len(dlens)) if dlens else 8.0

    order = list(range(len(distractors)))
    rng.shuffle(order)
    need = max(0, bucket - base_len)
    cum, k = 0, 0
    for idx in order:
        if cum >= need:
            break
        cum += dlens[idx]
        k += 1

    def assemble_k(kk: int):
        r = random.Random(seed)                                  # deterministic positions for a given kk
        current = [(s, 1) for s in base.gold_context]            # (sentence, relevance)
        for idx in order[:kk]:
            current.insert(r.randint(0, len(current)), (distractors[idx], 0))
        ctx = [s for s, _ in current]
        rel = [rr for _, rr in current]
        return ctx, rel

    ctx, rel = assemble_k(k)
    exact = count_tokens(_assemble(ctx, base.question))
    for _ in range(max_correction):
        if bucket * (1 - tol) <= exact <= bucket * (1 + tol):
            break
        if exact < bucket * (1 - tol):
            if k >= len(order):
                break
            k = min(len(order), k + max(1, int((bucket - exact) / max(1.0, avg))))
        else:
            if k == 0:
                break
            k = max(0, k - max(1, int((exact - bucket) / max(1.0, avg))))
        ctx, rel = assemble_k(k)
        exact = count_tokens(_assemble(ctx, base.question))

    return GSMItem(
        item_id=f"{base.base_id}__b{bucket}", base_id=base.base_id,
        context=ctx, relevance=rel, question=base.question, gold=base.gold,
        gold_steps=base.gold_steps, problem_quantities=base.problem_quantities, names=base.names,
        gold_context=list(base.gold_context), n_steps=base.n_steps,
        bucket=bucket, token_count=exact, seed=seed,
    )


def bucket_gsm_to_targets(items: Sequence[GSMItem], count_tokens: CountTokens, targets: Sequence[int],
                          base_seed: int = 0, tol: float = 0.05,
                          log: Callable[[str], None] | None = None) -> dict[int, list[GSMItem]]:
    log = log or (lambda _m: None)
    out: dict[int, list[GSMItem]] = {b: [] for b in targets}
    for i, base in enumerate(items):
        for bucket in targets:
            seed = base_seed + i * 1000 + bucket
            it = bucket_gsm_item(base, bucket, count_tokens, seed, tol)
            out[bucket].append(it)
            if not (bucket * (1 - tol) <= (it.token_count or 0) <= bucket * (1 + tol)):
                log(f"[bucketing] WARN {base.base_id} b{bucket}: reached {it.token_count} (target {bucket})")
        if (i + 1) % 10 == 0 or i + 1 == len(items):
            log(f"[bucketing] {i + 1}/{len(items)} base items done")
    return out


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------
def validate_gsm_item(item: GSMItem, count_tokens: CountTokens, tol: float = 0.05) -> tuple[bool, str]:
    """Gate a bucketed GSM item: gold arithmetic exact, problem sentences survive, relevance consistent."""
    # 1) gold chain arithmetic (exact rational); last result == gold
    last = None
    for step in item.gold_steps:
        if "=" not in step:
            return False, f"malformed gold step: {step!r}"
        expr, res = step.split("=", 1)
        try:
            if safe_eval(expr) != to_fraction(res):
                return False, f"gold arithmetic mismatch: {step!r}"
        except (ValueError, ZeroDivisionError) as e:
            return False, f"gold step uneval: {step!r} ({e})"
        last = res.strip()
    if item.gold_steps and last is not None and to_fraction(last) != to_fraction(item.gold):
        return False, f"final gold step {last!r} != gold {item.gold!r}"

    # 2) original problem sentences survived injection
    joined = _assemble(item.context, item.question)
    for g in item.gold_context:
        if g not in joined:
            return False, f"problem sentence missing after injection: {g!r}"

    # 3) relevance parallel + distractors marked 0, problem marked 1
    if len(item.relevance) != len(item.context):
        return False, "relevance length mismatch"
    gold_set = set(item.gold_context)
    for s, r in zip(item.context, item.relevance):
        if (s in gold_set) != (r == 1):
            return False, f"relevance bit inconsistent for {s!r}"

    # 4) token tolerance
    if item.bucket is None or item.token_count is None:
        return False, "item not bucketed"
    lo, hi = item.bucket * (1 - tol), item.bucket * (1 + tol)
    if not (lo <= item.token_count <= hi):
        return False, f"token_count {item.token_count} outside [{lo:.0f},{hi:.0f}]"
    return True, "ok"


# --------------------------------------------------------------------------------------
# IO + loading
# --------------------------------------------------------------------------------------
def append_gsm_item(path: str, item: GSMItem) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(item.to_json()) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_gsm_items(path: str) -> list[GSMItem]:
    out: list[GSMItem] = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(GSMItem(**json.loads(line)))
            except json.JSONDecodeError:
                continue
    return out


def load_gsm8k(split: str = "test", limit: int | None = None) -> list[GSMItem]:
    """Colab loader: parse GSM8K via the ``datasets`` library into base GSMItems (unbucketed)."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    items = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        items.append(parse_gsm8k_item(row["question"], row["answer"], item_id=f"gsm{i:05d}"))
    return items
