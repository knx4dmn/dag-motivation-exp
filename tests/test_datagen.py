"""CPU-only tests for datagen: adapter shapes, vocab extraction, bucketing, validation, splits."""
from __future__ import annotations

import pytest

from motivation_exp import datagen as dg
from motivation_exp.datagen import Item


# A whitespace token counter stands in for the Llama tokenizer in CPU tests.
def wc(text: str) -> int:
    return len(text.split())


# --------------------------------------------------------------------------------------
# split_sentences
# --------------------------------------------------------------------------------------
def test_split_sentences_keeps_periods_and_drops_empty():
    text = "Wren is a tumpus.  Every tumpus is a wumpus. "
    assert dg.split_sentences(text) == ["Wren is a tumpus.", "Every tumpus is a wumpus."]


def test_split_sentences_appends_missing_terminal_period():
    assert dg.split_sentences("Wren is a tumpus") == ["Wren is a tumpus."]


# --------------------------------------------------------------------------------------
# vocabulary extraction
# --------------------------------------------------------------------------------------
def test_extract_vocabulary_entities_and_concepts():
    sents = [
        "Wren is a tumpus.",
        "Every tumpus is a wumpus.",
        "Wumpuses are vumpuses.",
    ]
    entities, concepts = dg.extract_vocabulary(sents)
    assert entities == ["Wren"]
    # singularized + de-duplicated, first-seen order
    assert concepts == ["tumpus", "wumpus", "vumpus"]


# --------------------------------------------------------------------------------------
# adapter: the three shapes + gold normalization
# --------------------------------------------------------------------------------------
DICT_RAW = {
    "context": ["Wren is a tumpus.", "Every tumpus is a wumpus."],
    "query": "Wren is a wumpus.",
    "answer": "True",
    "chain_of_thought": ["Wren is a tumpus.", "Wren is a wumpus."],
}
TUPLE_RAW = (
    "Wren is a tumpus. Every tumpus is a wumpus.",  # blob, not pre-split
    "Wren is a wumpus.",
    "False",
)


class ObjRaw:
    def __init__(self):
        self.facts = ["Wren is a tumpus.", "Every tumpus is a wumpus."]
        self.goal = "Wren is a wumpus."
        self.label = 1


def test_adapter_dict_shape():
    it = dg.raw_prontoqa_adapter(DICT_RAW, item_id="x1")
    assert it.gold is True
    assert it.context == ["Wren is a tumpus.", "Every tumpus is a wumpus."]
    assert it.question == "Wren is a wumpus."
    assert it.entities == ["Wren"]
    assert "tumpus" in it.concepts and "wumpus" in it.concepts
    assert it.n_hops == 2  # two gold steps


def test_adapter_tuple_shape_splits_blob():
    it = dg.raw_prontoqa_adapter(TUPLE_RAW, item_id="x2")
    assert it.gold is False
    assert it.context == ["Wren is a tumpus.", "Every tumpus is a wumpus."]
    assert it.question == "Wren is a wumpus."


def test_adapter_object_shape_and_int_gold():
    it = dg.raw_prontoqa_adapter(ObjRaw(), item_id="x3")
    assert it.gold is True  # label=1
    assert it.question == "Wren is a wumpus."


def test_adapter_fails_loud_on_missing_fields():
    with pytest.raises(ValueError):
        dg.raw_prontoqa_adapter({"context": ["a."], "query": "b."}, item_id="bad")  # no answer


def test_normalize_gold_ab_and_bad():
    assert dg._normalize_gold("A") is True
    assert dg._normalize_gold("B") is False
    with pytest.raises(ValueError):
        dg._normalize_gold("maybe")


# --------------------------------------------------------------------------------------
# distractor pool + collision filter
# --------------------------------------------------------------------------------------
def _base_item() -> Item:
    return dg.raw_prontoqa_adapter(DICT_RAW, item_id="base", base_id="base")


def test_filter_colliding_drops_shared_names():
    base = _base_item()  # names: Wren, tumpus, wumpus
    pool = [
        "Sprocket is a yumpus.",      # disjoint -> kept
        "Wren is a zumpus.",          # shares Wren -> dropped
        "Every yumpus is a tumpus.",  # shares tumpus -> dropped
        "Every zumpus is a jompus.",  # disjoint -> kept
    ]
    kept = dg.filter_colliding(pool, base)
    assert kept == ["Sprocket is a yumpus.", "Every zumpus is a jompus."]


# --------------------------------------------------------------------------------------
# bucketing + validation
# --------------------------------------------------------------------------------------
def test_bucket_reaches_target_within_tolerance_and_query_last():
    base = _base_item()
    # a large disjoint distractor pool
    pool = [f"Gadget{i} is a yumpus{i}." for i in range(400)]
    bucket = 60  # word-count target
    it = dg.bucket_one_item(base, bucket, pool, wc, seed=1, tol=0.05)
    assert it.bucket == 60
    assert 57 <= it.token_count <= 63  # within +/- 5%
    ok, reason = dg.validate_item(it, wc, tol=0.05)
    assert ok, reason
    # gold sentences preserved
    joined = " ".join(it.context + [it.question])
    for g in base.context:
        assert g in joined
    assert joined.rstrip().endswith(it.question)


def test_validate_flags_out_of_tolerance():
    base = _base_item()
    it = Item(
        item_id="t", base_id="base", context=base.context, question=base.question,
        gold=True, entities=base.entities, concepts=base.concepts,
        gold_context=base.context, bucket=1000, token_count=10,
    )
    ok, reason = dg.validate_item(it, wc)
    assert not ok and "outside" in reason


def test_bucket_to_token_targets_deterministic_seeds():
    bases = [_base_item(), dg.raw_prontoqa_adapter(DICT_RAW, item_id="base2", base_id="base2")]
    pool = [f"Gadget{i} is a yumpus{i}." for i in range(400)]
    out = dg.bucket_to_token_targets(bases, wc, pool, targets=[40, 60], base_seed=0)
    assert set(out.keys()) == {40, 60}
    assert len(out[40]) == 2 and len(out[60]) == 2
    # deterministic: rerun gives identical seeds/token counts
    out2 = dg.bucket_to_token_targets(bases, wc, pool, targets=[40, 60], base_seed=0)
    assert [i.seed for i in out[60]] == [i.seed for i in out2[60]]
    assert [i.token_count for i in out[60]] == [i.token_count for i in out2[60]]


# --------------------------------------------------------------------------------------
# splits disjointness
# --------------------------------------------------------------------------------------
def test_reserve_splits_disjoint():
    items = [dg.raw_prontoqa_adapter(DICT_RAW, item_id=f"i{i}", base_id=f"b{i}") for i in range(15)]
    bucket_items, calib, exemplar = dg.reserve_splits(items, n_calib=10, n_exemplar=1)
    bset = {i.base_id for i in bucket_items}
    cset = {i.base_id for i in calib}
    eset = {i.base_id for i in exemplar}
    assert len(eset) == 1 and len(cset) == 10
    assert bset.isdisjoint(cset) and bset.isdisjoint(eset) and cset.isdisjoint(eset)


def test_reserve_splits_raises_when_too_few():
    items = [dg.raw_prontoqa_adapter(DICT_RAW, item_id=f"i{i}", base_id=f"b{i}") for i in range(5)]
    with pytest.raises(ValueError):
        dg.reserve_splits(items, n_calib=10, n_exemplar=1)
