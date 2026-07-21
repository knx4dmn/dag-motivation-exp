"""GSM8K generation loops (unguided / symbolic / semantic), timing, and JSONL resume.

Reuses the domain-agnostic infrastructure from :mod:`runner` -- the ``HFModelBackend`` (DynamicCache
+ chunked prefill + crop/re-feed rollback), the torch helpers, and the JSONL resume/append helpers --
and implements the GSM8K-specific step semantics, which differ from PrOntoQA in three ways:

  1. **Line boundary, not sentence period.** A reasoning step is one output LINE (ends at ``\\n``),
     since the model writes one calculation per line.
  2. **``#### <number>`` terminal.** The final-answer line, not a ``True/False`` clause.
  3. **Prose lines pass through.** Only calculation lines (``parse_calculation`` != None) are checked;
     a prose line (``no_calc``) is accepted without rollback. The semantic arm rejects (and resamples)
     ONLY on a distractor operand, a missing/hallucinated operand, or wrong arithmetic.

The semantic arm uses ACCEPTED-ONLY intermediate semantics (a rejected calc is resampled and its
result discarded) -- the opposite of G0's record-all mode. There is no frontier-guided path (the
symbolic grammar is format-only and static; semantic resampling is blind temperature sampling).

torch / transformers / xgrammar are imported lazily so the decode core + resume logic import and
unit-test on CPU with a stub backend.
"""
from __future__ import annotations

import time
from typing import Callable, Sequence

from . import gsm8k_grammar as gg
from .g0_gsm8k import EXEMPLAR_COT, EXEMPLAR_Q, SYSTEM_PROMPT, extract_number_answer
from .gsm8k_datagen import to_fraction
from .math_checker import parse_calculation
from .runner import (  # domain-agnostic infrastructure
    DecodeCfg,
    HFModelBackend,
    SemanticStats,
    _load_completed_keys,
    append_row,
    make_torch_helpers,
)

# Bare, grammar-conformant exemplar for the SYMBOLIC arm: the calculation grammar forbids prose, so
# its few-shot demo must be bare calculations (same reasoning as EXEMPLAR_COT, grammar-matched
# surface -- avoids the phrasing-mismatch that would fake Panel A's gap). unguided/semantic keep the
# prose EXEMPLAR_COT (the exact prompt G0 validated at parse rate 0.70-0.80).
EXEMPLAR_COT_BARE = "48 / 2 = 24\n48 + 24 = 72\n#### 72"


# --------------------------------------------------------------------------------------
# Answer / boundary / correctness (shared)
# --------------------------------------------------------------------------------------
def _has_line_boundary(text: str) -> bool:
    """A step is complete at a line break (one calculation per line)."""
    return "\n" in text


def _is_hash_answer(text: str) -> bool:
    """True once the terminal ``#### <number>`` answer line appears."""
    return "####" in text


def answer_correct(pred: str | None, gold: str) -> bool:
    """Exact-rational comparison of the extracted answer to the gold (never float)."""
    if pred is None:
        return False
    try:
        return to_fraction(pred) == to_fraction(gold)
    except (ValueError, ZeroDivisionError):
        return False


# --------------------------------------------------------------------------------------
# Semantic decode core (backend-agnostic; unit-tested with a stub)
# --------------------------------------------------------------------------------------
def math_semantic_decode(
    backend, initial_logits, prompt_ids: Sequence[int],
    decode_fn: Callable[[Sequence[int]], str], checker, sample_fn: Callable, cfg: DecodeCfg,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[list[int], SemanticStats]:
    """Line-by-line decode with per-calculation verification and KV-cache rollback.

    ``backend`` implements ``seq_len() / decode_step(token)->logits / crop(length)``. Rollback
    invariant (see runner.py): at a line boundary ``backend.seq_len() == len(all_ids) == s`` and
    ``logits`` predicts position ``s``; on reject we ``crop(s-1)`` and re-feed ``all_ids[s-1]`` so the
    resampled first token again lands at absolute position ``s``. Returns (generated_ids, stats).
    """
    all_ids = list(prompt_ids)
    gen: list[int] = []
    stats = SemanticStats()
    logits = initial_logits
    terminal = False

    while len(gen) < cfg.max_new_tokens and not terminal:
        step_start = backend.seq_len()
        toks, logits, info = _emit_math_line(
            backend, logits, all_ids, step_start, decode_fn, checker, sample_fn, cfg, stats, clock
        )
        gen.extend(toks)
        terminal = info["terminal"]

    stats.n_output_tokens = len(gen)
    stats.candidate_set_size = getattr(checker, "candidate_set_size", 0)
    return gen, stats


def _emit_math_line(backend, logits, all_ids, step_start, decode_fn, checker, sample_fn, cfg, stats, clock):
    """Generate one line with up to ``max_retries`` rollback resamples on a checker reject.

    Returns (line_tokens, next_logits, {"terminal": bool}). Prose (no_calc) lines and the answer
    line accept immediately; only distractor/missing/arithmetic calc failures trigger a resample.
    """
    attempts: list[tuple[float, list[int], str]] = []   # (score, tokens, text)
    banned_first: set[int] = set()

    for attempt in range(cfg.max_retries + 1):
        if attempt > 0:
            backend.crop(step_start - 1)                 # cache -> s-1
            logits = backend.decode_step(all_ids[step_start - 1])   # re-feed boundary token @ s-1
            stats.n_forward_passes += 1

        temp = 0.0 if attempt == 0 else cfg.temp_schedule[min(attempt - 1, len(cfg.temp_schedule) - 1)]
        line_toks, logits, forced_boundary, saw_eot = _generate_line_tokens(
            backend, logits, all_ids, banned_first, temp, sample_fn, decode_fn, cfg, stats
        )
        line_text = decode_fn(line_toks)
        if line_toks:
            banned_first.add(line_toks[0])

        # Terminal: EOS or the #### answer line -> accept without a provenance check.
        if saw_eot or _is_hash_answer(line_text):
            checker.accept(line_text)
            return line_toks, logits, {"terminal": True}

        # Prose (non-calculation) line -> pass through untouched (no check, no rollback).
        if parse_calculation(line_text) is None:
            checker.accept(line_text)                    # no-op (no result), keeps the API symmetric
            return line_toks, logits, {"terminal": False}

        if any(prev == line_text for (_, _, prev) in attempts):
            stats.duplicate_resample += 1

        t0 = clock()
        res = checker.check_step(line_text)
        stats.check_total_s += clock() - t0
        stats.n_checks += 1
        attempts.append((res.cosine, list(line_toks), line_text))

        if res.accepted and not forced_boundary:
            checker.accept(line_text)                    # accepted-only: record its result
            return line_toks, logits, {"terminal": False}

        stats.n_rejects += 1
        if forced_boundary:
            stats.no_boundary_forced += 1
        del all_ids[step_start:]

    # Retries exhausted -> forced-accept the best-scoring attempt by replaying its tokens.
    stats.n_forced_accepts += 1
    best = max(attempts, key=lambda a: a[0]) if attempts else (0.0, [], "")
    best_toks, best_text = best[1], best[2]
    backend.crop(step_start - 1)
    logits = backend.decode_step(all_ids[step_start - 1])
    stats.n_forward_passes += 1
    del all_ids[step_start:]
    for t in best_toks:
        all_ids.append(t)
        logits = backend.decode_step(t)
        stats.n_forward_passes += 1
    checker.accept(best_text)
    return best_toks, logits, {"terminal": _is_hash_answer(best_text)}


def _generate_line_tokens(backend, logits, all_ids, banned_first, temp, sample_fn, decode_fn, cfg, stats):
    """Sample tokens until a line break, the per-step cap, or EOS. Appends tokens to all_ids."""
    line_toks: list[int] = []
    forced_boundary = False
    saw_eot = False
    first = True
    while True:
        ban = banned_first if first else None
        tok = sample_fn(logits, temp, ban)
        first = False
        all_ids.append(tok)
        line_toks.append(tok)
        logits = backend.decode_step(tok)
        stats.n_forward_passes += 1

        if tok == cfg.eot_id:
            saw_eot = True
            break
        text = decode_fn(line_toks)
        if _has_line_boundary(text):
            break
        if len(line_toks) >= cfg.per_step_cap:
            forced_boundary = True
            break
    return line_toks, logits, forced_boundary, saw_eot


# --------------------------------------------------------------------------------------
# Prompt building (HF-only)
# --------------------------------------------------------------------------------------
def build_math_prompt_ids(tokenizer, item, exemplar_cot: str):
    """Chat-templated prompt: system + one exemplar turn + the item's (context + question)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": EXEMPLAR_Q},
        {"role": "assistant", "content": exemplar_cot},
        {"role": "user", "content": " ".join(list(item.context) + [item.question])},
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")


# --------------------------------------------------------------------------------------
# Row assembly
# --------------------------------------------------------------------------------------
def _base_row(item, method: str, model_name: str, gpu: str, env_hash: str) -> dict:
    return {
        "run_id": f"{model_name}:{method}:{item.bucket}:{item.item_id}",
        "model": model_name, "method": method, "bucket": item.bucket,
        "item_id": item.item_id, "seed": item.seed, "gold": item.gold,
        "gpu": gpu, "env_hash": env_hash, "timestamp": None,
    }


def _finish_row(row: dict, text: str, item, decode_wall: float, prefill_wall: float,
                n_tokens: int, overhead: dict) -> dict:
    pred = extract_number_answer(text)
    row.update({
        "correct": answer_correct(pred, item.gold), "pred": pred, "n_output_tokens": n_tokens,
        "decode_wall_s": decode_wall, "prefill_wall_s": prefill_wall,
        "per_token_ms": (decode_wall / n_tokens * 1000.0) if n_tokens else None,
        "overhead": overhead,
    })
    return row


# ======================================================================================
# HF / torch paths (lazy imports; Colab-only execution)
# ======================================================================================
def run_unguided(model, tokenizer, item, cfg: DecodeCfg, sync, sample_fn, *,
                 model_name: str, gpu: str = "", env_hash: str = "") -> dict:
    """Plain greedy decode (reference). No grammar, no checker."""
    _maybe_seed(cfg, item)
    prompt_ids = build_math_prompt_ids(tokenizer, item, EXEMPLAR_COT)
    backend = HFModelBackend(model, sync)
    sync(); t0 = time.perf_counter()
    logits = backend.prefill(prompt_ids, chunk_size=cfg.prefill_chunk_size)
    prefill_wall = time.perf_counter() - t0

    gen: list[int] = []
    sync(); t0 = time.perf_counter()
    while len(gen) < cfg.max_new_tokens:
        tok = sample_fn(logits, 0.0, None)
        gen.append(tok)
        if tok == cfg.eot_id:
            break
        logits = backend.decode_step(tok)
    decode_wall = time.perf_counter() - t0

    text = tokenizer.decode(gen, skip_special_tokens=True)
    overhead = {"grammar_compile_s": None, "embed_prefill_s": None, "check_total_s": None,
                "n_checks": 0, "n_rejects": 0, "n_forced_accepts": 0, "candidate_set_size": 0,
                "raw_forward_s": backend.forward_s}
    return _finish_row(_base_row(item, "unguided", model_name, gpu, env_hash),
                       text, item, decode_wall, prefill_wall, len(gen), overhead)


def run_symbolic(model, tokenizer, item, cfg: DecodeCfg, sync, compiled, *,
                 model_name: str, gpu: str = "", env_hash: str = "") -> dict:
    """Format-only grammar-constrained greedy decode via XGrammar primitives (no rollback).

    ``compiled`` is the shared, precompiled calculation grammar (item-independent) -- so no per-item
    compile time is charged and the symbolic latency stays flat.
    """
    import torch
    import xgrammar as xgr

    _maybe_seed(cfg, item)
    prompt_ids = build_math_prompt_ids(tokenizer, item, EXEMPLAR_COT_BARE)
    backend = HFModelBackend(model, sync)

    matcher = xgr.GrammarMatcher(compiled)                          # fresh per item
    vocab_size = model.config.vocab_size
    cpu_bitmask = xgr.allocate_token_bitmask(1, vocab_size)
    gpu_bitmask = torch.empty_like(cpu_bitmask, device=backend.device)

    sync(); t0 = time.perf_counter()
    logits = backend.prefill(prompt_ids, chunk_size=cfg.prefill_chunk_size)
    prefill_wall = time.perf_counter() - t0

    gen: list[int] = []
    mask_s = 0.0
    sync(); t0 = time.perf_counter()
    while len(gen) < cfg.max_new_tokens:
        tm = time.perf_counter()
        matcher.fill_next_token_bitmask(cpu_bitmask)
        gpu_bitmask.copy_(cpu_bitmask)
        xgr.apply_token_bitmask_inplace(logits, gpu_bitmask)
        mask_s += time.perf_counter() - tm
        tok = int(torch.argmax(logits))
        gen.append(tok)
        matcher.accept_token(tok)
        if matcher.is_terminated() or tok == cfg.eot_id:
            break
        logits = backend.decode_step(tok)
    decode_wall = time.perf_counter() - t0

    text = tokenizer.decode(gen, skip_special_tokens=True)
    overhead = {"grammar_compile_s": None, "embed_prefill_s": None, "check_total_s": None,
                "grammar_mask_s": mask_s, "n_checks": 0, "n_rejects": 0, "n_forced_accepts": 0,
                "candidate_set_size": 0, "raw_forward_s": backend.forward_s}
    return _finish_row(_base_row(item, "symbolic", model_name, gpu, env_hash),
                       text, item, decode_wall, prefill_wall, len(gen), overhead)


def run_semantic(model, tokenizer, item, cfg: DecodeCfg, sync, sample_fn, checker, *,
                 model_name: str, gpu: str = "", env_hash: str = "") -> dict:
    """Line-by-line greedy decode with per-calculation verify + KV-cache rollback."""
    _maybe_seed(cfg, item)
    prompt_ids = build_math_prompt_ids(tokenizer, item, EXEMPLAR_COT)
    backend = HFModelBackend(model, sync)

    sync(); t0 = time.perf_counter()
    logits = backend.prefill(prompt_ids, chunk_size=cfg.prefill_chunk_size)
    prefill_wall = time.perf_counter() - t0

    tc = time.perf_counter()
    checker.prefill(item.context, item.relevance)                  # no index built (red line B)
    check_prefill_s = time.perf_counter() - tc

    decode_fn = lambda ids: tokenizer.decode(ids, skip_special_tokens=True)
    prompt_list = prompt_ids[0].tolist()

    sync(); t0 = time.perf_counter()
    gen, stats = math_semantic_decode(backend, logits, prompt_list, decode_fn, checker, sample_fn, cfg)
    decode_wall = time.perf_counter() - t0

    text = tokenizer.decode(gen, skip_special_tokens=True)
    overhead = {"grammar_compile_s": None, "embed_prefill_s": check_prefill_s,
                "check_total_s": stats.check_total_s, "n_checks": stats.n_checks,
                "n_rejects": stats.n_rejects, "n_forced_accepts": stats.n_forced_accepts,
                "duplicate_resample": stats.duplicate_resample,
                "no_boundary_forced": stats.no_boundary_forced,
                "candidate_set_size": stats.candidate_set_size,
                "n_forward_passes": stats.n_forward_passes, "raw_forward_s": backend.forward_s}
    return _finish_row(_base_row(item, "semantic", model_name, gpu, env_hash),
                       text, item, decode_wall, prefill_wall, len(gen), overhead)


def _maybe_seed(cfg: DecodeCfg, item) -> None:
    if getattr(cfg, "deterministic_decode", False):
        import torch
        torch.manual_seed((item.seed or 0) & 0x7FFFFFFF)


def warmup(model, tokenizer, item, cfg, sync, sample_fn, n: int = 3) -> None:
    """Run ``n`` untimed short generations to absorb CUDA-init/compile jitter (after every load)."""
    for _ in range(n):
        run_unguided(model, tokenizer, item,
                     DecodeCfg(eot_id=cfg.eot_id, max_new_tokens=16), sync, sample_fn,
                     model_name="_warmup")


def run_all(model, tokenizer, buckets: dict, methods: Sequence[str], out_path: str, model_name: str,
            *, cfg: DecodeCfg, sync, sample_fn, compiler=None, checker_factory: Callable | None = None,
            gpu: str = "", env_hash: str = "", timestamp_fn: Callable[[], str] | None = None) -> None:
    """Drive every (model, method, bucket, item), interleaving methods round-robin per item.

    Resumable: skips any (model, method, bucket, item_id) already in ``out_path``. The symbolic
    grammar is compiled ONCE here (item-independent) and reused. A fresh MathChecker is built per
    item via ``checker_factory``.
    """
    import datetime

    ts = timestamp_fn or (lambda: datetime.datetime.now().isoformat())
    done = _load_completed_keys(out_path)

    compiled = gg.compile_calc_grammar(compiler) if ("symbolic" in methods and compiler is not None) else None

    any_item = next((its[0] for its in buckets.values() if its), None)
    if any_item is not None:
        warmup(model, tokenizer, any_item, cfg, sync, sample_fn)

    for bucket in sorted(buckets):
        for item in buckets[bucket]:
            for method in methods:
                key = (model_name, method, bucket, item.item_id)
                if key in done:
                    continue
                if method == "unguided":
                    row = run_unguided(model, tokenizer, item, cfg, sync, sample_fn,
                                       model_name=model_name, gpu=gpu, env_hash=env_hash)
                elif method == "symbolic":
                    row = run_symbolic(model, tokenizer, item, cfg, sync, compiled,
                                       model_name=model_name, gpu=gpu, env_hash=env_hash)
                elif method == "semantic":
                    row = run_semantic(model, tokenizer, item, cfg, sync, sample_fn, checker_factory(),
                                       model_name=model_name, gpu=gpu, env_hash=env_hash)
                else:
                    raise ValueError(f"unknown method: {method}")
                row["timestamp"] = ts()
                append_row(out_path, row)
                done.add(key)
