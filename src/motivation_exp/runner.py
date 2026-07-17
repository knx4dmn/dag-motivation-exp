"""Generation loops (unguided / symbolic / semantic), timing hooks, and JSONL resume.

Design note (rev #8 testability): the semantic method's KV-cache rollback is the load-bearing
mechanism, so its bookkeeping is written against a tiny **backend protocol** --
``seq_len() / decode_step(token) / crop(length)`` -- in :func:`semantic_decode`. A CPU stub
backend drives that function in tests and asserts the crop/re-feed indices; the real Colab
path (:class:`HFModelBackend`) plugs torch + ``torch.cuda.synchronize`` timing into the same
algorithm. Nothing in :func:`semantic_decode` imports torch.

torch / transformers are imported lazily inside the HF-only functions so this module (and its
resume + rollback logic) is importable and testable on CPU.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

from . import grammar as gr

# --------------------------------------------------------------------------------------
# Answer extraction / boundaries (shared)
# --------------------------------------------------------------------------------------
_ANSWER_SEARCH = re.compile(r"\b(True|False)\b")


def extract_answer(text: str) -> str | None:
    """Return 'True'/'False' from the final line of generated text, else None."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    for ln in reversed(lines):
        m = _ANSWER_SEARCH.search(ln)
        if m:
            return m.group(1)
    return None


def _has_sentence_boundary(step_text: str) -> bool:
    """A step is complete when its decoded text ends a sentence (trailing period)."""
    return step_text.rstrip().endswith(".")


def _is_answer(step_text: str) -> bool:
    """True if this step is the terminal answer line (contains True/False)."""
    return _ANSWER_SEARCH.search(step_text) is not None


# --------------------------------------------------------------------------------------
# Decode config
# --------------------------------------------------------------------------------------
@dataclass
class DecodeCfg:
    eot_id: int
    max_new_tokens: int = 256
    per_step_cap: int = 48
    temp_schedule: tuple[float, ...] = (0.7, 0.9, 1.1)
    max_retries: int = 3
    # REQUIRED on T4 (default 512): SDPA runs the math backend for this GQA model on sm75, so a
    # one-shot 8k prefill OOMs; chunking builds the KV cache incrementally. None disables it.
    prefill_chunk_size: int | None = 512


# --------------------------------------------------------------------------------------
# Semantic decode core (backend-agnostic; unit-tested with a stub)
# --------------------------------------------------------------------------------------
@dataclass
class SemanticStats:
    n_output_tokens: int = 0
    n_forward_passes: int = 0
    n_checks: int = 0
    n_rejects: int = 0
    n_forced_accepts: int = 0
    duplicate_resample: int = 0
    no_boundary_forced: int = 0
    check_total_s: float = 0.0
    candidate_set_size: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def semantic_decode(
    backend,
    initial_logits,
    prompt_ids: Sequence[int],
    decode_fn: Callable[[Sequence[int]], str],
    checker,
    sample_fn: Callable,
    cfg: DecodeCfg,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[list[int], SemanticStats]:
    """Token-by-token decode with sentence-boundary verification and KV-cache rollback.

    ``backend`` implements ``seq_len()``, ``decode_step(token)->logits``, ``crop(length)``.
    ``initial_logits`` predicts the first generated token (from the caller's prefill).
    ``sample_fn(logits, temperature, banned_first)`` -> int token id (greedy if temperature<=0).
    Returns (generated_token_ids, stats).

    Rollback invariant (see plan): at a step boundary ``backend.seq_len() == len(all_ids) == s``
    and ``logits`` predicts position ``s``. On reject we ``crop(s-1)`` and re-feed ``all_ids[s-1]``
    so the resampled first token again lands at absolute position ``s``.
    """
    all_ids = list(prompt_ids)
    gen: list[int] = []
    stats = SemanticStats()
    logits = initial_logits
    terminal = False

    while len(gen) < cfg.max_new_tokens and not terminal:
        step_start = backend.seq_len()  # == len(all_ids) == s
        step_toks, logits, info = _emit_step(
            backend, logits, all_ids, step_start, decode_fn, checker, sample_fn, cfg, stats, clock
        )
        gen.extend(step_toks)
        terminal = info["terminal"]

    stats.n_output_tokens = len(gen)
    stats.candidate_set_size = getattr(checker, "candidate_set_size", 0)
    return gen, stats


def _emit_step(
    backend, logits, all_ids, step_start, decode_fn, checker, sample_fn, cfg, stats, clock
):
    """Generate one reasoning step with up to ``max_retries`` rollback resamples.

    Returns (step_tokens, next_logits, {"terminal": bool, "forced": bool}).
    """
    attempts: list[tuple[float, list[int], str]] = []  # (score, tokens, text)
    banned_first: set[int] = set()

    for attempt in range(cfg.max_retries + 1):
        if attempt > 0:
            # Roll back to the step start and re-feed the boundary token (rev #1/#8).
            backend.crop(step_start - 1)          # cache -> s-1
            refeed = all_ids[step_start - 1]
            logits = backend.decode_step(refeed)  # cache -> s, logits predict position s
            stats.n_forward_passes += 1
        temp = 0.0 if attempt == 0 else cfg.temp_schedule[min(attempt - 1, len(cfg.temp_schedule) - 1)]

        step_toks, logits, forced_boundary, saw_eot = _generate_step_tokens(
            backend, logits, all_ids, banned_first, temp, sample_fn, decode_fn, cfg, stats
        )
        step_text = decode_fn(step_toks)
        if step_toks:
            banned_first.add(step_toks[0])

        # Terminal conditions: EOS, or the final answer line -> accept without checking.
        if saw_eot or _is_answer(step_text):
            checker_accept(checker, step_text)
            return step_toks, logits, {"terminal": True, "forced": False}

        # Duplicate-resample bookkeeping (rev #5).
        if any(prev_text == step_text for (_, _, prev_text) in attempts):
            stats.duplicate_resample += 1

        # Verify the completed step (single check; its cosine is the forced-accept score).
        t0 = clock()
        res = checker.check_step(step_text)
        stats.check_total_s += clock() - t0
        stats.n_checks += 1
        attempts.append((res.cosine, list(step_toks), step_text))

        if res.accepted and not forced_boundary:
            checker.accept(step_text)
            return step_toks, logits, {"terminal": False, "forced": False}

        # Reject: count, remove tentative tokens from all_ids, loop (crop happens next attempt).
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
    terminal = _is_answer(best_text)
    return best_toks, logits, {"terminal": terminal, "forced": True}


def _generate_step_tokens(
    backend, logits, all_ids, banned_first, temp, sample_fn, decode_fn, cfg, stats
):
    """Sample tokens until a sentence boundary, the per-step cap, or EOS.

    Returns (step_tokens, next_logits, forced_boundary, saw_eot). Appends tokens to all_ids.
    """
    step_toks: list[int] = []
    forced_boundary = False
    saw_eot = False
    first = True
    while True:
        ban = banned_first if first else None
        tok = sample_fn(logits, temp, ban)
        first = False
        all_ids.append(tok)
        step_toks.append(tok)
        logits = backend.decode_step(tok)
        stats.n_forward_passes += 1

        if tok == cfg.eot_id:
            saw_eot = True
            break
        text = decode_fn(step_toks)
        if _has_sentence_boundary(text):
            break
        if len(step_toks) >= cfg.per_step_cap:
            forced_boundary = True  # rev #6: force a boundary so one step can't eat the budget
            break
    return step_toks, logits, forced_boundary, saw_eot


def checker_accept(checker, text: str) -> None:
    if text.strip():
        checker.accept(text)


# --------------------------------------------------------------------------------------
# JSONL resume + append (pure; CPU-tested)
# --------------------------------------------------------------------------------------
def _completed_key(row: dict) -> tuple:
    return (row.get("model"), row.get("method"), row.get("bucket"), row.get("item_id"))


def _load_completed_keys(out_path: str) -> set[tuple]:
    """Return the set of completed (model, method, bucket, item_id) keys from an existing JSONL.

    Tolerates a trailing partial line (a mid-write disconnect). Missing file -> empty set.
    """
    keys: set[tuple] = set()
    if not os.path.exists(out_path):
        return keys
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # trailing partial line; ignore
            keys.add(_completed_key(row))
    return keys


def append_row(out_path: str, row: dict) -> None:
    """Append one JSONL row and fsync so a disconnect can't lose/corrupt it."""
    with open(out_path, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ======================================================================================
# HF / torch paths below (lazy imports; Colab-only execution)
# ======================================================================================
SYSTEM_PROMPT = (
    "You are a logical reasoning assistant. Given facts and rules, determine whether the "
    "final statement is True or False. Reason step by step, writing one sentence per line "
    "using only the given names and categories, then end with a line exactly of the form "
    "'The answer is True.' or 'The answer is False.'."
)


def build_prompt_ids(tokenizer, item, exemplar_item, exemplar_cot: str):
    """Build chat-templated prompt token ids: system + one exemplar turn + the item's question."""
    ex_ctx = " ".join(exemplar_item.context)
    it_ctx = " ".join(item.context)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{ex_ctx}\nQuestion: {exemplar_item.question} True or False?"},
        {"role": "assistant", "content": exemplar_cot},
        {"role": "user", "content": f"{it_ctx}\nQuestion: {item.question} True or False?"},
    ]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )


def make_torch_helpers():
    """Return (sync, sample_fn) bound to torch; call once per session."""
    import torch

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def sample_fn(logits, temperature: float, banned_first):
        # logits: 1D torch tensor over vocab. Greedy if temperature<=0.
        lg = logits.clone()
        if banned_first:
            idx = torch.tensor(sorted(banned_first), device=lg.device, dtype=torch.long)
            lg[idx] = float("-inf")
        if temperature <= 0:
            return int(torch.argmax(lg))
        # fp32 softmax over 128k vocab (fp16 underflows) then multinomial.
        probs = torch.softmax(lg.float() / temperature, dim=-1)
        return int(torch.multinomial(probs, num_samples=1))

    return sync, sample_fn


class HFModelBackend:
    """Manual-loop backend over an HF causal LM with a DynamicCache and fp16 forwards.

    ``logits_to_keep=1`` on every call (rev #1). Both ``cache_position`` AND ``attention_mask``
    are left as ``None``: with batch=1 and no padding an all-ones mask is a no-op that forces
    transformers to build an explicit 4D additive mask (SDPA ``is_causal=False`` -> math
    backend), which on Turing materializes a (1, heads, L, L) fp32 score tensor -- the 8k OOM.
    ``attention_mask=None`` takes the ``is_causal=True`` fast path and derives ``cache_position``
    from ``cache.get_seq_length()``, which is exactly what the crop/re-feed rollback relies on,
    so rollback semantics are unchanged.
    """

    def __init__(self, model, sync: Callable[[], None]):
        import torch
        from transformers import DynamicCache

        self._torch = torch
        self.model = model
        self.device = model.device
        self.sync = sync
        self.cache = DynamicCache()
        self.forward_s = 0.0

    def prefill(self, input_ids, chunk_size: int | None = 512):
        """Prefill the KV cache and return the last-token logits.

        ``chunk_size`` feeds the prompt in chunks so peak activation memory stays bounded
        (build the cache incrementally); attention stays causal across chunks because each
        chunk attends the cache. This is REQUIRED on T4, not a fallback: SDPA runs the math
        backend for this GQA model on sm75 (see config.PREFILL_CHUNK), so a one-shot 8k prefill
        materializes a (1, 24, 8192, 8192) score tensor and OOMs. The full chunked prefill is
        timed into ``forward_s`` (prefill is excluded from Panel B, so the metric is unchanged).
        """
        torch = self._torch
        input_ids = input_ids.to(self.device)
        self.sync()
        t0 = time.perf_counter()
        with torch.inference_mode():
            if chunk_size is None or input_ids.shape[1] <= chunk_size:
                out = self.model(
                    input_ids=input_ids, attention_mask=None, past_key_values=self.cache,
                    use_cache=True, logits_to_keep=1,
                )
            else:
                out = None
                for s in range(0, input_ids.shape[1], chunk_size):
                    out = self.model(
                        input_ids=input_ids[:, s:s + chunk_size], attention_mask=None,
                        past_key_values=self.cache, use_cache=True, logits_to_keep=1,
                    )
        self.sync()
        self.forward_s += time.perf_counter() - t0
        return out.logits[0, -1, :]

    def decode_step(self, token_id: int):
        torch = self._torch
        ids = torch.tensor([[int(token_id)]], device=self.device, dtype=torch.long)
        self.sync()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model(
                input_ids=ids, attention_mask=None, past_key_values=self.cache,
                use_cache=True, logits_to_keep=1,
            )
        self.sync()
        self.forward_s += time.perf_counter() - t0
        return out.logits[0, -1, :]

    def crop(self, length: int):
        self.cache.crop(length)

    def seq_len(self) -> int:
        return self.cache.get_seq_length()


def _base_row(item, method: str, model_name: str, gpu: str, env_hash: str) -> dict:
    return {
        "run_id": f"{model_name}:{method}:{item.bucket}:{item.item_id}",
        "model": model_name, "method": method, "bucket": item.bucket,
        "item_id": item.item_id, "seed": item.seed,
        "gold": "True" if item.gold else "False", "gpu": gpu, "env_hash": env_hash,
        "timestamp": None,  # stamped by caller (Date.now unavailable here)
    }


def _finish_row(row: dict, text: str, item, decode_wall: float, prefill_wall: float,
                n_tokens: int, overhead: dict) -> dict:
    pred = extract_answer(text)
    row.update({
        "correct": bool(pred is not None and (pred == "True") == item.gold),
        "pred": pred, "n_output_tokens": n_tokens,
        "decode_wall_s": decode_wall, "prefill_wall_s": prefill_wall,
        "per_token_ms": (decode_wall / n_tokens * 1000.0) if n_tokens else None,
        "overhead": overhead,
    })
    return row


def run_unguided(model, tokenizer, item, exemplar_item, exemplar_cot, cfg: DecodeCfg,
                 sync, sample_fn, *, model_name: str, gpu: str = "", env_hash: str = "") -> dict:
    """Plain greedy decode (reference). No logits processor, no checker."""
    prompt_ids = build_prompt_ids(tokenizer, item, exemplar_item, exemplar_cot)
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
    row = _base_row(item, "unguided", model_name, gpu, env_hash)
    return _finish_row(row, text, item, decode_wall, prefill_wall, len(gen), overhead)


def run_symbolic(model, tokenizer, item, exemplar_item, exemplar_cot, cfg: DecodeCfg,
                 sync, compiler, *, model_name: str, gpu: str = "", env_hash: str = "") -> dict:
    """Grammar-constrained greedy decode via XGrammar primitives (no rollback)."""
    import torch
    import xgrammar as xgr

    prompt_ids = build_prompt_ids(tokenizer, item, exemplar_item, exemplar_cot)
    backend = HFModelBackend(model, sync)

    t0 = time.perf_counter()
    ebnf = gr.build_item_ebnf(item.entities, item.concepts)
    compiled = gr.compile_item_grammar(ebnf, compiler)
    grammar_compile_s = time.perf_counter() - t0

    matcher = xgr.GrammarMatcher(compiled)                       # fresh per item
    vocab_size = model.config.vocab_size
    cpu_bitmask = xgr.allocate_token_bitmask(1, vocab_size)
    gpu_bitmask = torch.empty_like(cpu_bitmask, device=backend.device)  # rev #9: preallocate

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
    overhead = {"grammar_compile_s": grammar_compile_s, "embed_prefill_s": None,
                "check_total_s": None, "grammar_mask_s": mask_s, "n_checks": 0, "n_rejects": 0,
                "n_forced_accepts": 0, "candidate_set_size": 0, "raw_forward_s": backend.forward_s}
    row = _base_row(item, "symbolic", model_name, gpu, env_hash)
    return _finish_row(row, text, item, decode_wall, prefill_wall, len(gen), overhead)


def run_semantic(model, tokenizer, item, exemplar_item, exemplar_cot, cfg: DecodeCfg,
                 sync, sample_fn, checker, *, model_name: str, gpu: str = "", env_hash: str = "") -> dict:
    """Token-by-token greedy decode with sentence-boundary verify + KV-cache rollback."""
    prompt_ids = build_prompt_ids(tokenizer, item, exemplar_item, exemplar_cot)
    backend = HFModelBackend(model, sync)

    sync(); t0 = time.perf_counter()
    logits = backend.prefill(prompt_ids, chunk_size=cfg.prefill_chunk_size)
    prefill_wall = time.perf_counter() - t0

    te = time.perf_counter()
    checker.prefill(item.context)
    embed_prefill_s = time.perf_counter() - te

    decode_fn = lambda ids: tokenizer.decode(ids, skip_special_tokens=True)
    prompt_list = prompt_ids[0].tolist()

    sync(); t0 = time.perf_counter()
    gen, stats = semantic_decode(backend, logits, prompt_list, decode_fn, checker, sample_fn, cfg)
    decode_wall = time.perf_counter() - t0

    text = tokenizer.decode(gen, skip_special_tokens=True)
    overhead = {"grammar_compile_s": None, "embed_prefill_s": embed_prefill_s,
                "check_total_s": stats.check_total_s, "n_checks": stats.n_checks,
                "n_rejects": stats.n_rejects, "n_forced_accepts": stats.n_forced_accepts,
                "duplicate_resample": stats.duplicate_resample,
                "no_boundary_forced": stats.no_boundary_forced,
                "candidate_set_size": stats.candidate_set_size,
                "n_forward_passes": stats.n_forward_passes, "raw_forward_s": backend.forward_s}
    row = _base_row(item, "semantic", model_name, gpu, env_hash)
    return _finish_row(row, text, item, decode_wall, prefill_wall, len(gen), overhead)


def warmup(model, tokenizer, item, exemplar_item, exemplar_cot, cfg, sync, sample_fn, n: int = 3):
    """Run ``n`` untimed generations to absorb CUDA-init/compile jitter (rev #4).

    Called at the start of ``run_all`` -- i.e. after EVERY model load, including Colab
    reconnect-resumes -- so post-reconnect items don't carry warmup jitter into the medians.
    """
    for _ in range(n):
        run_unguided(model, tokenizer, item, exemplar_item, exemplar_cot,
                     DecodeCfg(eot_id=cfg.eot_id, max_new_tokens=16), sync, sample_fn,
                     model_name="_warmup")


def run_all(
    model, tokenizer, buckets: dict, methods: Sequence[str], out_path: str, model_name: str,
    *, exemplar_item, exemplar_cot: str, cfg: DecodeCfg, sync, sample_fn,
    compiler=None, checker_factory: Callable | None = None,
    gpu: str = "", env_hash: str = "", timestamp_fn: Callable[[], str] | None = None,
) -> None:
    """Drive every (model, method, bucket, item), interleaving methods round-robin per item.

    Resumable: skips any (model, method, bucket, item_id) already present in ``out_path``.
    Methods are interleaved within the session so thermal/clock drift doesn't bias one
    method's latency. A fresh ``SemanticChecker`` is built per item via ``checker_factory``.
    """
    import datetime

    ts = timestamp_fn or (lambda: datetime.datetime.now().isoformat())
    done = _load_completed_keys(out_path)

    # Warmup once per session (covers reconnect-resume), using any available item.
    any_item = next((its[0] for its in buckets.values() if its), None)
    if any_item is not None:
        warmup(model, tokenizer, any_item, exemplar_item, exemplar_cot, cfg, sync, sample_fn)

    for bucket in sorted(buckets):
        items = buckets[bucket]
        for item in items:
            for method in methods:  # round-robin over methods within each item
                key = (model_name, method, bucket, item.item_id)
                if key in done:
                    continue
                if method == "unguided":
                    row = run_unguided(model, tokenizer, item, exemplar_item, exemplar_cot,
                                       cfg, sync, sample_fn, model_name=model_name, gpu=gpu, env_hash=env_hash)
                elif method == "symbolic":
                    row = run_symbolic(model, tokenizer, item, exemplar_item, exemplar_cot,
                                       cfg, sync, compiler, model_name=model_name, gpu=gpu, env_hash=env_hash)
                elif method == "semantic":
                    checker = checker_factory()
                    row = run_semantic(model, tokenizer, item, exemplar_item, exemplar_cot,
                                       cfg, sync, sample_fn, checker, model_name=model_name, gpu=gpu, env_hash=env_hash)
                else:
                    raise ValueError(f"unknown method: {method}")
                row["timestamp"] = ts()
                append_row(out_path, row)
                done.add(key)
