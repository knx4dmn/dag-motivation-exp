"""CPU-only tests for datagen: adapter (verified 6-tuple), vocab (concept vs property),
bucketing, validation, splits, and the synthetic disjoint-block generator."""
from __future__ import annotations

import pytest

from motivation_exp import datagen as dg
from motivation_exp.datagen import Item, GenerationFailed


def wc(text: str) -> int:
    return len(text.split())


# --------------------------------------------------------------------------------------
# split_sentences
# --------------------------------------------------------------------------------------
def test_split_sentences_keeps_periods_and_drops_empty():
    text = "Wren is a tumpus.  Every tumpus is a wumpus. "
    assert dg.split_sentences(text) == ["Wren is a tumpus.", "Every tumpus is a wumpus."]


# --------------------------------------------------------------------------------------
# vocabulary extraction: concepts vs properties
# --------------------------------------------------------------------------------------
def test_extract_vocabulary_separates_concepts_and_properties():
    sents = [
        "Wren is a tumpus.",
        "Every tumpus is a wumpus.",
        "Tumpuses are not slow.",   # slow is a property
        "Wren is not slow.",
    ]
    entities, concepts, properties = dg.extract_vocabulary(sents)
    assert entities == ["Wren"]
    assert concepts == ["tumpus", "wumpus"]
    assert properties == ["slow"]


def test_extract_vocabulary_each_every_plural_forms():
    sents = ["Each yumpus is shy.", "Impuses are tumpuses.", "Every impus is a zumpus."]
    entities, concepts, properties = dg.extract_vocabulary(sents)
    assert entities == []
    assert set(concepts) == {"yumpus", "impus", "tumpus", "zumpus"}
    assert properties == ["shy"]


# --------------------------------------------------------------------------------------
# adapter: the verified 6-tuple
# --------------------------------------------------------------------------------------
def _six_tuple(answer="True"):
    question_text = "Wren is a tumpus. Every tumpus is a wumpus. Tumpuses are not slow."
    query = "True or false: Wren is not slow."
    forms = ["<lf-objects>"]          # element 2: logical forms (ignored by adapter)
    chain = ["Wren is a tumpus.", "Tumpuses are not slow.", "Wren is not slow."]
    proof = ["<proof-objects>"]
    return (question_text, query, forms, chain, answer, proof)


def test_adapter_six_tuple_shape():
    it = dg.raw_prontoqa_adapter(_six_tuple("True"), item_id="x1")
    assert it.gold is True
    assert it.context == ["Wren is a tumpus.", "Every tumpus is a wumpus.", "Tumpuses are not slow."]
    assert it.question == "Wren is not slow."      # "True or false: " prefix stripped
    assert it.entities == ["Wren"]
    assert "tumpus" in it.concepts and "wumpus" in it.concepts
    assert it.properties == ["slow"]
    assert it.gold_steps[-1] == "Wren is not slow."
    assert it.n_hops == 3


def test_adapter_answer_is_string_false():
    it = dg.raw_prontoqa_adapter(_six_tuple("False"), item_id="x2")
    assert it.gold is False


def test_adapter_raises_generation_failed_on_all_none():
    with pytest.raises(GenerationFailed):
        dg.raw_prontoqa_adapter((None,) * 6, item_id="bad")


def test_adapter_concepts_hint_augments_vocab():
    it = dg.raw_prontoqa_adapter(_six_tuple(), item_id="x3", concepts_hint=["tumpus", "zzzpus"])
    assert "zzzpus" in it.concepts   # block concept not in context is still covered


def test_adapter_dict_fallback():
    raw = {"question_text": "Wren is a tumpus.", "query": "True or false: Wren is a tumpus.",
           "answer": "True", "chain_of_thought": ["Wren is a tumpus."]}
    it = dg.raw_prontoqa_adapter(raw, item_id="d1")
    assert it.gold is True and it.question == "Wren is a tumpus."


# --------------------------------------------------------------------------------------
# distractor pool + collision filter (properties excluded from collisions)
# --------------------------------------------------------------------------------------
def _base_item():
    return dg.raw_prontoqa_adapter(_six_tuple(), item_id="base", base_id="base")


def test_filter_colliding_drops_shared_concept_or_entity_keeps_shared_property():
    base = _base_item()  # entities: Wren; concepts: tumpus, wumpus; properties: slow
    pool = [
        "Sprocket is a yumpus.",        # disjoint -> kept
        "Wren is a yumpus.",            # shares entity Wren -> dropped
        "Every yumpus is a tumpus.",    # shares concept tumpus -> dropped
        "Every yumpus is not slow.",    # shares only PROPERTY slow -> KEPT (properties are global)
    ]
    kept = dg.filter_colliding(pool, base)
    assert kept == ["Sprocket is a yumpus.", "Every yumpus is not slow."]


# --------------------------------------------------------------------------------------
# bucketing + validation
# --------------------------------------------------------------------------------------
def test_bucket_reaches_target_and_carries_properties():
    base = _base_item()
    pool = [f"Gadget{i} is a yumpus{i}." for i in range(400)]
    it = dg.bucket_one_item(base, 60, pool, wc, seed=1, tol=0.05)
    assert 57 <= it.token_count <= 63
    assert it.properties == base.properties
    ok, reason = dg.validate_item(it, wc, tol=0.05)
    assert ok, reason
    assert " ".join(it.context + [it.question]).rstrip().endswith(it.question)


def test_validate_flags_out_of_tolerance():
    base = _base_item()
    it = Item(item_id="t", base_id="base", context=base.context, question=base.question, gold=True,
              entities=base.entities, concepts=base.concepts, properties=base.properties,
              gold_context=base.context, bucket=1000, token_count=10)
    ok, reason = dg.validate_item(it, wc)
    assert not ok and "outside" in reason


# --------------------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------------------
def test_reserve_splits_disjoint():
    items = [dg.raw_prontoqa_adapter(_six_tuple(), item_id=f"i{i}", base_id=f"b{i}") for i in range(15)]
    bucket_items, calib, exemplar = dg.reserve_splits(items, n_calib=10, n_exemplar=1)
    b = {i.base_id for i in bucket_items}; c = {i.base_id for i in calib}; e = {i.base_id for i in exemplar}
    assert len(e) == 1 and len(c) == 10
    assert b.isdisjoint(c) and b.isdisjoint(e) and c.isdisjoint(e)


# --------------------------------------------------------------------------------------
# synthetic disjoint-block generator
# --------------------------------------------------------------------------------------
def test_make_concept_names_novel_and_disjoint():
    reserved = {"wumpus", "tumpus", "impus"}
    names = dg.make_concept_names(200, seed=0, reserved=reserved)
    assert len(names) == 200
    assert len(set(names)) == 200            # unique
    assert set(names).isdisjoint(reserved)   # avoids reserved
    assert all(n.endswith("us") and n.islower() for n in names)  # *pus morphology


def test_partition_blocks_pairwise_disjoint():
    names = dg.make_concept_names(160, seed=0)
    blocks = dg.partition_blocks(names, 16)
    assert len(blocks) == 10
    seen = set()
    for blk in blocks:
        assert len(blk) == 16
        assert seen.isdisjoint(blk)
        seen |= set(blk)


def test_make_concept_names_raises_when_exhausted():
    with pytest.raises(ValueError):
        dg.make_concept_names(10_000_000, seed=0)


# --------------------------------------------------------------------------------------
# fast bucketing (DistractorPool) + loud failure + checkpoint IO
# --------------------------------------------------------------------------------------
def test_distractor_pool_precomputes_lengths_and_usable():
    base = _base_item()  # concepts: tumpus, wumpus
    sents = ["Sprocket is a yumpus.", "Every tumpus is a wumpus.", "Numo is a jompus."]
    dp = dg.DistractorPool(sents, wc)
    assert dp.lens == [4, 5, 4]                     # tokenized once each
    usable = dp.usable_indices(base)                # index 1 shares 'tumpus' -> excluded
    assert 1 not in usable and 0 in usable and 2 in usable


def test_bucketing_lands_within_tolerance_all_buckets():
    base = _base_item()
    pool = [f"Alpha{i} beta gamma delta epsilon." for i in range(2000)]  # 5 words each, no collisions
    out = dg.bucket_to_token_targets([base], wc, pool, targets=[100, 200, 400], base_seed=0)
    for b in (100, 200, 400):
        it = out[b][0]
        assert b * 0.95 <= it.token_count <= b * 1.05, (b, it.token_count)


def test_bucketing_warns_loudly_on_pool_exhaustion():
    base = _base_item()
    pool = ["Alpha beta gamma delta epsilon." for _ in range(3)]  # far too few for bucket 4096
    msgs = []
    dg.bucket_to_token_targets([base], wc, pool, targets=[4096], base_seed=0, log=msgs.append)
    assert any("pool likely exhausted" in m and base.base_id in m for m in msgs)


def test_append_and_load_items_resume(tmp_path):
    p = tmp_path / "raw.jsonl"
    a = _base_item()
    dg.append_item(str(p), a)
    dg.append_item(str(p), a)
    # tolerate a trailing partial line (mid-write disconnect)
    with open(p, "a") as f:
        f.write('{"item_id": "partial"')
    loaded = dg.load_items(str(p))
    assert len(loaded) == 2 and loaded[0].item_id == a.item_id


def test_load_items_missing_file_returns_empty(tmp_path):
    assert dg.load_items(str(tmp_path / "nope.jsonl")) == []
