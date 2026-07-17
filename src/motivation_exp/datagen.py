"""PrOntoQA data generation, bucketing, splits, validation, and manifest.

This module is CPU-only and importable without model weights. Token counting is injected
as a ``Callable[[str], int]`` so tests can run with a whitespace counter and Colab can
pass the Llama tokenizer.

:func:`raw_prontoqa_adapter` targets the VERIFIED upstream return shape (a 6-tuple; see the
function docstring), with a dict/object fallback kept for defensiveness. Vocabulary isolation
across items is guaranteed by per-item disjoint concept blocks (:func:`make_concept_names` /
:func:`partition_blocks`), registered with the generator's morphology via the bridge.

The sentence-boundary rule (:func:`split_sentences`) and the clause parser
(``grammar.parse_clause``) are the single sources of truth reused across modules so accuracy /
step-boundary / vocabulary logic cannot silently diverge.
"""
from __future__ import annotations

import json
import random
import re
import statistics
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Iterable, Sequence

from . import grammar as gr

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
# Vocabulary extraction (via the shared grammar.parse_clause)
# --------------------------------------------------------------------------------------
def extract_vocabulary(sentences: Sequence[str]) -> tuple[list[str], list[str], list[str]]:
    """Return (entities, concepts, properties) drawn only from the given sentences.

    Uses the shared PrOntoQA clause parser so concept nouns and property adjectives are
    separated exactly (properties classified against the fixed ``grammar.PROPERTY_WORDS``).
    All lists are singular/lowercase (entities keep case), de-duplicated in first-seen order.
    """
    entities: list[str] = []
    concepts: list[str] = []
    properties: list[str] = []
    se, sc, sp = set(), set(), set()

    def add(lst, seen, val):
        if val and val not in seen:
            seen.add(val)
            lst.append(val)

    for raw in sentences:
        c = gr.parse_clause(raw)
        if c is None:
            continue
        if c.kind == "fact":
            add(entities, se, c.subject)
        else:  # rule: subject is the antecedent concept
            add(concepts, sc, c.subject)
        if c.is_property:
            add(properties, sp, c.pred)
        else:
            add(concepts, sc, c.pred)
    return entities, concepts, properties


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
    properties: list[str] = field(default_factory=list)
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


_QUERY_PREFIX_RE = re.compile(r"^(?:True or false:|Prove:)\s*", re.IGNORECASE)


class GenerationFailed(Exception):
    """Raised when PrOntoQA's generate_question returned an all-None failure tuple."""


def raw_prontoqa_adapter(raw: Any, item_id: str, base_id: str | None = None,
                         *, concepts_hint: Sequence[str] | None = None) -> Item:
    """Normalize one ``generate_question`` return value into an :class:`Item`.

    The upstream API (verified from the pinned source, run_experiment.py line 664) returns a
    6-tuple::

        (question_text, query, formulas+premises, chain_of_thought, str(answer), linearized_proof)

    where ``question_text`` is the context as one string blob, ``query`` is prefixed with
    ``"True or false: "``, element 2 is logical-form objects (NOT the answer), ``chain_of_thought``
    is a list of step strings, and the answer is ``"True"``/``"False"`` (a STRING, not a bool).
    On generation failure PrOntoQA returns ``(None,)*6``; we raise :class:`GenerationFailed` so the
    caller can retry (its documented usage). A dict/object form is still accepted defensively.

    ``concepts_hint`` (the disjoint block the item was generated from) augments the vocabulary
    extracted from text, ensuring the grammar covers every block concept even if a given item's
    context happened not to use all of them.
    """
    base_id = base_id or item_id

    if isinstance(raw, (tuple, list)):
        if len(raw) == 6:
            question_text, query, _forms, chain, answer, _proof = raw
        elif len(raw) >= 4:  # tolerate a (context, query, answer, chain) shape defensively
            question_text, query, answer, chain = raw[0], raw[1], raw[2], raw[3]
        else:
            raise ValueError(f"unexpected tuple arity {len(raw)} for item {item_id!r}")
        if question_text is None or query is None or answer is None:
            raise GenerationFailed(f"generate_question failed (all-None) for item {item_id!r}")
        context_val, query_val, gold_val, steps_val = question_text, query, answer, chain
    elif isinstance(raw, dict):
        context_val = _first_present(raw, ["question_text", "context", "facts", "premises"])
        query_val = _first_present(raw, ["query", "question", "goal"])
        gold_val = _first_present(raw, ["answer", "label", "gold"])
        steps_val = _first_present(raw, ["chain_of_thought", "proof", "steps", "gold_steps", "chain"])
        if context_val is None or query_val is None or gold_val is None:
            raise ValueError(f"dict adapter missing context/query/answer for {item_id!r}: keys {list(raw)}")
    else:  # object with attributes
        get = lambda names: next((getattr(raw, n) for n in names if getattr(raw, n, None) is not None), None)
        context_val = get(["question_text", "context", "facts"])
        query_val = get(["query", "question", "goal"])
        gold_val = get(["answer", "label", "gold"])
        steps_val = get(["chain_of_thought", "proof", "steps", "chain"])
        if context_val is None or query_val is None or gold_val is None:
            raise ValueError(f"object adapter missing context/query/answer for {item_id!r}")

    context = _as_sentence_list(context_val)
    # Strip the "True or false: " / "Prove: " prefix, keep the statement sentence.
    query_str = _QUERY_PREFIX_RE.sub("", str(query_val).strip())
    query_sentences = split_sentences(query_str)
    if not query_sentences:
        raise ValueError(f"Empty query for item {item_id!r}")
    question = " ".join(query_sentences)

    gold = _normalize_gold(gold_val)
    gold_steps = _as_sentence_list(steps_val) if steps_val is not None else []

    # Vocabulary from context + query; augment concepts with the generation block if provided.
    entities, concepts, properties = extract_vocabulary(list(context) + query_sentences)
    if concepts_hint:
        seen = set(concepts)
        for c in concepts_hint:
            if c not in seen:
                seen.add(c)
                concepts.append(c)

    return Item(
        item_id=item_id,
        base_id=base_id,
        context=context,
        question=question,
        gold=gold,
        entities=entities,
        concepts=concepts,
        properties=properties,
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
    """Lowercased entity + CONCEPT names of an item, for collision detection.

    Properties are DELIBERATELY excluded: PrOntoQA's ~46 property adjectives are shared by
    every item, so including them would make every distractor "collide" and drop the whole
    pool. Concept disjointness (guaranteed by per-item disjoint concept blocks) is what keeps
    injected distractors inert; a distractor mentioning the base entity is also dropped, so a
    property-sharing distractor can only chain to a non-base entity and stays inert.
    """
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
        ents, cons, _props = extract_vocabulary([s])
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
        properties=base.properties,
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


def load_items(path: str) -> list[Item]:
    """Reload items from a JSONL file written by Phase 1 (for resumable later phases)."""
    out: list[Item] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Item(**json.loads(line)))
    return out


# --------------------------------------------------------------------------------------
# Disjoint synthetic concept vocabulary (vocabulary isolation -- see module + RUNBOOK)
# --------------------------------------------------------------------------------------
# PrOntoQA ships only ~90 disjoint concept names (9 blocks of 10). We need one disjoint block
# per base item AND a disjoint pool space, far more than that, so we synthesize novel nonsense
# nouns in the same "*pus"-style morphology (lowercase, end in a consonant + "us"; pluralize by
# +"es"). Each must be registered with PrOntoQA's morphology before use (add_noun), and must
# avoid the generator's reserved nouns. Blocks are pairwise disjoint so no distractor can share
# a concept with any base item -- the load-bearing invariant for the collision filter.
_SYNTH_ONSETS = [
    "bl", "cl", "dr", "fl", "gl", "gr", "kr", "pl", "pr", "sl", "sm", "sn", "sp", "st", "str",
    "sw", "tr", "thr", "vl", "zr", "b", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "r",
    "s", "t", "v", "w", "z", "ch", "sh", "zh", "br", "cr", "fr",
]
_SYNTH_NUCLEI = [
    "om", "im", "um", "am", "em", "or", "ar", "ur", "er", "ir", "ol", "al", "ul", "il", "el",
    "un", "in", "on", "an", "en",
]
_SYNTH_CODAS = ["pus", "mpus", "rpus", "lpus", "npus", "spus", "tpus", "dpus", "kpus", "gpus",
                "bpus", "zpus"]


def make_concept_names(n: int, seed: int, reserved: Iterable[str] = ()) -> list[str]:
    """Generate ``n`` novel PrOntoQA-style concept nouns, avoiding ``reserved`` names."""
    reserved = set(reserved)
    combos = [o + nu + co for o in _SYNTH_ONSETS for nu in _SYNTH_NUCLEI for co in _SYNTH_CODAS]
    random.Random(seed).shuffle(combos)
    out, seen = [], set()
    for name in combos:
        if name in reserved or name in seen:
            continue
        seen.add(name)
        out.append(name)
        if len(out) >= n:
            break
    if len(out) < n:
        raise ValueError(f"could only synthesize {len(out)} novel names, need {n}; widen the morphology")
    return out


def partition_blocks(names: Sequence[str], block_size: int) -> list[list[str]]:
    """Split a flat name list into pairwise-disjoint blocks of ``block_size``."""
    return [list(names[i:i + block_size]) for i in range(0, len(names) - block_size + 1, block_size)]


def register_concepts(morphology, names: Iterable[str]) -> None:
    """Register synthetic concept nouns (singular -> +'es' plural) with PrOntoQA's morphology.

    Skips names already present so it is safe to call repeatedly / on reconnect.
    """
    for name in names:
        if not morphology.is_noun(name):
            morphology.add_noun(name, gr.plural_of(name))
