"""PrOntoQA data generation, bucketing, splits, validation, and manifest.

This module is CPU-only and importable without model weights. Token counting is injected
as a ``Callable[[str], int]`` so tests can run with a whitespace counter and Colab can
pass the Llama tokenizer.

The highest-uncertainty piece is :func:`raw_prontoqa_adapter`: the exact return shape of
the upstream ``generate_question(...)`` is unknown until the repo is cloned, so the adapter
detects the schema defensively and fails loud rather than silently coercing.

The sentence-boundary rule (:func:`split_sentences`) is defined ONCE here and reused by the
checker and the grammar exemplar synthesizer so accuracy / step-boundary logic cannot
silently diverge across modules.
"""
from __future__ import annotations

import json
import random
import re
import statistics
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Iterable, Sequence

CountTokens = Callable[[str], int]

# --------------------------------------------------------------------------------------
# Shared sentence-boundary rule (single source of truth; see module docstring)
# --------------------------------------------------------------------------------------
_SENT_SPLIT_RE = re.compile(r"(?<=\.)\s+")


def split_sentences(text: str) -> list[str]:
    """Split PrOntoQA text into sentences on a period followed by whitespace.

    PrOntoQA is templatic: every fact/rule ends in a period. We keep the trailing period
    (so downstream string-search validation matches the original) and drop empty pieces.
    This exact rule is reused by the semantic checker's step detection.
    """
    text = text.strip()
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text)
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if not p.endswith("."):
            p = p + "."
        out.append(p)
    return out


# --------------------------------------------------------------------------------------
# Vocabulary extraction from templatic sentences
# --------------------------------------------------------------------------------------
# PrOntoQA sentence forms (fictional ontology), e.g.:
#   "Wren is a tumpus."                    -> entity Wren, concept tumpus
#   "Every tumpus is a wumpus."            -> concepts tumpus, wumpus
#   "Tumpuses are wumpuses."               -> concepts (plural) tumpus, wumpus
#   "Each wumpus is not a vumpus."         -> concepts wumpus, vumpus
# Entities are capitalized proper names appearing as a sentence subject; concepts are the
# lowercase class nouns. We extract conservatively via templates and normalize plurals.
_ENTITY_RE = re.compile(r"^([A-Z][a-zA-Z]*)\s+is\s+(?:a|an)\b")
_RULE_RE = re.compile(
    r"^(?:Every|Each|All)\s+([a-z][a-zA-Z]*)\s+is\s+(?:not\s+)?(?:a|an)\s+([a-z][a-zA-Z]*)\b"
)
_PLURAL_RULE_RE = re.compile(r"^([A-Za-z][a-zA-Z]*)\s+are\s+(?:not\s+)?([A-Za-z][a-zA-Z]*)\b")
_ENTITY_CONCEPT_RE = re.compile(
    r"^([A-Z][a-zA-Z]*)\s+is\s+(?:not\s+)?(?:a|an)\s+([a-z][a-zA-Z]*)\b"
)


def _singularize(word: str) -> str:
    """Plural->singular for templatic PrOntoQA nouns.

    Fictional concepts are singular ``-us`` / plural ``-uses`` (tumpus / tumpuses). We must
    NOT strip the ``s`` from an already-singular ``-us`` noun, so ``-us`` endings are held
    fixed and only genuine plural suffixes are reduced.
    """
    w = word.lower()
    if w.endswith("uses"):                       # tumpuses -> tumpus
        return w[:-2]
    if w.endswith("es") and w[:-2].endswith(("s", "x", "z", "ch", "sh")):
        return w[:-2]                            # sibilant plurals: boxes -> box
    if w.endswith("s") and not w.endswith(("us", "ss", "is", "as", "os")):
        return w[:-1]                            # regular plural: cats -> cat
    return w


def extract_vocabulary(sentences: Sequence[str]) -> tuple[list[str], list[str]]:
    """Return (entities, concepts) drawn only from the given sentences.

    Entities are proper names (capitalized subjects); concepts are class nouns, normalized
    to singular lowercase and de-duplicated while preserving first-seen order.
    """
    entities: list[str] = []
    concepts: list[str] = []
    seen_e: set[str] = set()
    seen_c: set[str] = set()

    def add_entity(name: str) -> None:
        if name and name not in seen_e:
            seen_e.add(name)
            entities.append(name)

    def add_concept(name: str) -> None:
        s = _singularize(name)
        if s and s not in seen_c:
            seen_c.add(s)
            concepts.append(s)

    for raw in sentences:
        s = raw.strip()
        m = _ENTITY_CONCEPT_RE.match(s)
        if m:
            add_entity(m.group(1))
            add_concept(m.group(2))
            continue
        m = _RULE_RE.match(s)
        if m:
            add_concept(m.group(1))
            add_concept(m.group(2))
            continue
        m = _PLURAL_RULE_RE.match(s)
        if m:
            add_concept(m.group(1))
            add_concept(m.group(2))
            continue
        m = _ENTITY_RE.match(s)
        if m:
            add_entity(m.group(1))
    return entities, concepts


# --------------------------------------------------------------------------------------
# Item model
# --------------------------------------------------------------------------------------
@dataclass
class Item:
    """One PrOntoQA example plus everything downstream stages need.

    ``context`` is the ordered list of gold fact/rule sentences (question/query excluded).
    ``question`` is the final query sentence (always placed last in a bucketed prompt).
    ``gold`` is the normalized boolean answer. ``gold_steps`` is the ordered proof chain.
    ``bucket``/``token_count``/``seed`` are set during bucketing.
    """

    item_id: str
    base_id: str
    context: list[str]
    question: str
    gold: bool
    entities: list[str]
    concepts: list[str]
    gold_steps: list[str] = field(default_factory=list)
    n_hops: int = 0
    bucket: int | None = None
    token_count: int | None = None
    seed: int | None = None
    # The original base context sentences (gold facts/rules), preserved through bucketing so
    # the validation gate can confirm they survived distractor injection intact.
    gold_context: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------------------
# Defensive adapter (highest-uncertainty external dependency)
# --------------------------------------------------------------------------------------
_TRUE_STRINGS = {"true", "t", "yes", "a"}
_FALSE_STRINGS = {"false", "f", "no", "b"}


def _normalize_gold(value: Any) -> bool:
    """Coerce a gold label (bool / 'True'/'False' / 'A'/'B' / 0/1) to bool; fail loud."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in _TRUE_STRINGS:
            return True
        if v in _FALSE_STRINGS:
            return False
    raise ValueError(f"Cannot normalize gold label: {value!r}")


def _as_sentence_list(value: Any) -> list[str]:
    """Accept a pre-split list of sentences or a single text blob; split blobs ourselves."""
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for chunk in value:
            out.extend(split_sentences(str(chunk)))
        return out
    return split_sentences(str(value))


def _first_present(d: dict, keys: Sequence[str]) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def raw_prontoqa_adapter(raw: Any, item_id: str, base_id: str | None = None) -> Item:
    """Normalize one upstream ``generate_question`` return value into an :class:`Item`.

    Tolerates three shapes and fails loudly on anything else:
      * dict: keys among {context/question_text/facts, query/question, answer/label/gold,
        chain_of_thought/proof/steps/gold_steps}.
      * tuple/list: (context, query, answer[, chain]).
      * object: attributes with the same names as the dict keys.

    Does NOT assume gold is a bool, and does NOT assume context is pre-split.
    """
    base_id = base_id or item_id

    context_val = query_val = gold_val = steps_val = None

    if isinstance(raw, dict):
        context_val = _first_present(raw, ["context", "question_text", "facts", "premises", "theory"])
        query_val = _first_present(raw, ["query", "question", "goal", "hypothesis"])
        gold_val = _first_present(raw, ["answer", "label", "gold", "target"])
        steps_val = _first_present(raw, ["chain_of_thought", "proof", "steps", "gold_steps", "chain"])
    elif isinstance(raw, (tuple, list)):
        if len(raw) < 3:
            raise ValueError(f"tuple form must be (context, query, answer[, chain]); got len {len(raw)}")
        context_val, query_val, gold_val = raw[0], raw[1], raw[2]
        steps_val = raw[3] if len(raw) > 3 else None
    else:  # object with attributes
        get = lambda names: next(
            (getattr(raw, n) for n in names if getattr(raw, n, None) is not None), None
        )
        context_val = get(["context", "question_text", "facts", "premises", "theory"])
        query_val = get(["query", "question", "goal", "hypothesis"])
        gold_val = get(["answer", "label", "gold", "target"])
        steps_val = get(["chain_of_thought", "proof", "steps", "gold_steps", "chain"])

    if context_val is None or query_val is None or gold_val is None:
        raise ValueError(
            f"Adapter could not locate context/query/answer in raw item {item_id!r}. "
            f"Got keys/type: {list(raw.keys()) if isinstance(raw, dict) else type(raw).__name__}"
        )

    context = _as_sentence_list(context_val)
    # The query may itself be a blob; keep it as a single sentence string.
    query_sentences = _as_sentence_list(query_val)
    if not query_sentences:
        raise ValueError(f"Empty query for item {item_id!r}")
    question = " ".join(query_sentences)

    gold = _normalize_gold(gold_val)
    gold_steps = _as_sentence_list(steps_val) if steps_val is not None else []

    # Vocabulary is extracted from context AND query so grammar terminals cover the query.
    entities, concepts = extract_vocabulary(list(context) + query_sentences)

    return Item(
        item_id=item_id,
        base_id=base_id,
        context=context,
        question=question,
        gold=gold,
        entities=entities,
        concepts=concepts,
        gold_steps=gold_steps,
        n_hops=len(gold_steps),
    )


# --------------------------------------------------------------------------------------
# Distractor pool + collision filter
# --------------------------------------------------------------------------------------
def build_distractor_pool(
    items: Iterable[Item], size: int, seed: int
) -> list[str]:
    """Assemble a shuffled pool of distractor sentences harvested from item contexts.

    These are homogeneous with gold sentences (same generator/syntax). Callers must still
    apply :func:`filter_colliding` per base question to drop vocabulary collisions.
    """
    pool: list[str] = []
    seen: set[str] = set()
    for it in items:
        for s in it.context:
            if s not in seen:
                seen.add(s)
                pool.append(s)
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:size] if size and size < len(pool) else pool


def _names_in(item: Item) -> set[str]:
    """Lowercased entity + concept names of an item, for collision detection."""
    names = {e.lower() for e in item.entities}
    names |= {c.lower() for c in item.concepts}
    return names


def filter_colliding(pool: Sequence[str], base: Item) -> list[str]:
    """Drop pool sentences that share any entity/concept name with ``base``'s ontology.

    A colliding distractor could change the gold answer, so it must never be injected.
    """
    base_names = _names_in(base)
    kept: list[str] = []
    for s in pool:
        ents, cons = extract_vocabulary([s])
        s_names = {n.lower() for n in ents} | {n.lower() for n in cons}
        if s_names.isdisjoint(base_names):
            kept.append(s)
    return kept


# --------------------------------------------------------------------------------------
# Bucketing
# --------------------------------------------------------------------------------------
def _assemble_prompt_text(context: Sequence[str], question: str) -> str:
    """Join context sentences then the query last, one space-separated blob."""
    return " ".join(list(context) + [question])


def bucket_one_item(
    base: Item,
    bucket: int,
    distractor_pool: Sequence[str],
    count_tokens: CountTokens,
    seed: int,
    tol: float = 0.05,
) -> Item:
    """Interleave distractors into ``base``'s context until the prompt hits ``bucket`` tokens.

    Gold sentences keep their relative order; distractors are inserted at uniformly random
    positions with a fixed, recorded seed. The query sentence always comes last. Grows the
    context until token count reaches ``bucket`` (within +/- ``tol``) or the pool is
    exhausted. Returns a NEW Item with bucket/token_count/seed populated.
    """
    rng = random.Random(seed)
    usable = filter_colliding(distractor_pool, base)
    rng.shuffle(usable)

    gold = list(base.context)
    target_hi = bucket * (1 + tol)
    current = list(gold)

    def tok(ctx: list[str]) -> int:
        return count_tokens(_assemble_prompt_text(ctx, base.question))

    di = 0
    # Grow until we reach the lower edge of tolerance; stop before exceeding the upper edge.
    while tok(current) < bucket * (1 - tol) and di < len(usable):
        insert_at = rng.randint(0, len(current))  # never after the (implicit) trailing query
        candidate = current[:insert_at] + [usable[di]] + current[insert_at:]
        if tok(candidate) > target_hi:
            di += 1  # this distractor overshoots; try the next (they vary in length)
            continue
        current = candidate
        di += 1

    return Item(
        item_id=f"{base.base_id}__b{bucket}",
        base_id=base.base_id,
        context=current,
        question=base.question,
        gold=base.gold,
        entities=base.entities,
        concepts=base.concepts,
        gold_steps=base.gold_steps,
        n_hops=base.n_hops,
        bucket=bucket,
        token_count=tok(current),
        seed=seed,
        gold_context=list(gold),
    )


def bucket_to_token_targets(
    items: Sequence[Item],
    count_tokens: CountTokens,
    distractor_pool: Sequence[str],
    targets: Sequence[int],
    base_seed: int = 0,
    tol: float = 0.05,
) -> dict[int, list[Item]]:
    """Bucket every base item to every target token count.

    Per-item insertion seeds are derived deterministically from ``base_seed`` and recorded
    on each produced item. Returns {bucket: [Item, ...]}.
    """
    out: dict[int, list[Item]] = {b: [] for b in targets}
    for i, base in enumerate(items):
        for bucket in targets:
            seed = base_seed + i * 1000 + bucket  # deterministic, unique per (item, bucket)
            out[bucket].append(
                bucket_one_item(base, bucket, distractor_pool, count_tokens, seed, tol)
            )
    return out


# --------------------------------------------------------------------------------------
# Validation gate
# --------------------------------------------------------------------------------------
def validate_item(item: Item, count_tokens: CountTokens, tol: float = 0.05) -> tuple[bool, str]:
    """Gate a bucketed item; return (ok, reason).

    Checks: every gold sentence survived injection intact (exact substring), the query is
    last, token count is within tolerance of the bucket, and the vocabulary covers the
    gold sentences.
    """
    if item.bucket is None or item.token_count is None:
        return False, "item not bucketed (bucket/token_count unset)"

    joined = _assemble_prompt_text(item.context, item.question)
    # Gold sentences (the original base context facts/rules) must all still be present verbatim.
    for g in item.gold_context or item.context:
        if g and g not in joined:
            return False, f"gold sentence missing after injection: {g!r}"
    # The query must be last.
    if not joined.rstrip().endswith(item.question.rstrip()):
        return False, "query sentence is not last in the assembled prompt"

    lo, hi = item.bucket * (1 - tol), item.bucket * (1 + tol)
    if not (lo <= item.token_count <= hi):
        return False, f"token_count {item.token_count} outside [{lo:.0f}, {hi:.0f}] for bucket {item.bucket}"

    if not item.entities and not item.concepts:
        return False, "empty vocabulary (grammar would be degenerate)"

    return True, "ok"


# --------------------------------------------------------------------------------------
# Splits (calibration + exemplar disjoint from buckets) -- R2 #1/#2
# --------------------------------------------------------------------------------------
def reserve_splits(
    items: Sequence[Item],
    n_calib: int,
    n_exemplar: int,
) -> tuple[list[Item], list[Item], list[Item]]:
    """Carve disjoint (bucket_items, calib_items, exemplar_items) from base items.

    The exemplar and calibration items are guaranteed absent from every bucket: they are
    removed from the front of the (already deterministically ordered) list before bucketing.
    Callers must bucket ONLY ``bucket_items``. Raises if there are too few items.
    """
    need = n_calib + n_exemplar
    if len(items) < need + 1:
        raise ValueError(f"need > {need} base items to reserve splits; got {len(items)}")
    exemplar = list(items[:n_exemplar])
    calib = list(items[n_exemplar : n_exemplar + n_calib])
    bucket_items = list(items[n_exemplar + n_calib :])
    return bucket_items, calib, exemplar


# --------------------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------------------
def write_manifest(
    buckets: dict[int, list[Item]],
    path: str,
    seed: int,
    *,
    calib_ids: Sequence[str] = (),
    exemplar_ids: Sequence[str] = (),
    tau_restate: float | None = None,
    tau_mp: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize dataset provenance + per-bucket stats to a JSON manifest and return it."""
    per_bucket = {}
    for b, its in buckets.items():
        counts = [it.token_count for it in its if it.token_count is not None]
        per_bucket[str(b)] = {
            "n_items": len(its),
            "mean_tokens": statistics.mean(counts) if counts else None,
            "std_tokens": statistics.pstdev(counts) if len(counts) > 1 else 0.0,
            "min_tokens": min(counts) if counts else None,
            "max_tokens": max(counts) if counts else None,
        }
    manifest = {
        "seed": seed,
        "buckets": per_bucket,
        "reserved": {
            "calibration_base_ids": list(calib_ids),
            "exemplar_base_ids": list(exemplar_ids),
        },
        "frozen_tau": {"tau_restate": tau_restate, "tau_mp": tau_mp},
    }
    if extra:
        manifest.update(extra)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def hf_token_counter(tokenizer) -> CountTokens:
    """Wrap an HF tokenizer into a ``count_tokens`` callable (no special tokens)."""
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False))
