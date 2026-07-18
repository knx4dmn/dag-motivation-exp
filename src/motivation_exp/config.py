"""Central configuration: buckets, thresholds, model ids, pins, seeds, frozen tau.

Everything that must stay constant across the pilot and full run lives here so that a
single source of truth is recorded in the run manifest. Values that are *calibrated*
(the semantic checker's tau thresholds) start as ``None`` and are frozen by Phase 1.5
before any reported run -- see ``TAU_RESTATE`` / ``TAU_MP`` below.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
PRIMARY_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
# Fallback only, triggered by gate G1 (see plan); NOT run by default.
FALLBACK_MODEL = "Qwen/Qwen2.5-3B-Instruct"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Short name used as the ``model`` field in JSONL rows and resume keys.
MODEL_SHORT = {
    PRIMARY_MODEL: "llama-3.2-3b-instruct",
    FALLBACK_MODEL: "qwen2.5-3b-instruct",
}

# Llama-3.2-Instruct stops on the chat turn terminator, NOT <|end_of_text|> (R2 #5).
STOP_TOKEN = "<|eot_id|>"

# --------------------------------------------------------------------------------------
# Dependency pins (mirrored in pyproject.toml; recorded in the manifest for provenance)
# --------------------------------------------------------------------------------------
PINS = {
    "transformers": "4.57.1",
    "xgrammar": "0.1.34",
    "sentence-transformers": "3.4.1",
    "numpy": ">=1.26,<3",
    "matplotlib": "3.9.2",
    # torch: use Colab preinstalled build; do NOT reinstall.
}
# VERIFIED LOCALLY: xgrammar==0.1.34 hard-requires apache-tvm-ffi, torch, transformers,
# pydantic, typing-extensions (importing xgrammar imports torch). sentence-transformers
# also hard-requires torch. Therefore BOTH must be installed with a constraints file that
# pins torch to Colab's preinstalled version, or pip will swap torch and break CUDA. See
# run_colab.ipynb's setup cell. The generated EBNF (grammar.build_item_ebnf) was confirmed
# to parse under xgrammar.Grammar.from_ebnf, including the no-entities and escaped cases.

# Pinned upstream commit for the PrOntoQA generator. VERIFY/UPDATE before a run: this is
# a placeholder for the latest github.com/asaparov/prontoqa commit at implementation time.
PRONTOQA_REPO = "https://github.com/asaparov/prontoqa"
PRONTOQA_COMMIT = "0a6412b6fddf46324a1cb96e066dd7b3d89b87d6"  # <-- user confirms the exact hash in the notebook.

# Your own package repo, cloned + `pip install -e .` on Colab. Placeholder for the user.
SELF_REPO = "https://github.com/knx4dmn/dag-motivation-exp"
SELF_COMMIT = "PLACEHOLDER_PIN_ME"

# --------------------------------------------------------------------------------------
# Experimental design (plan section 1)
# --------------------------------------------------------------------------------------
BUCKETS = [512, 1024, 2048, 4096, 8192]  # target prompt token counts (tokenizer-specific)
BUCKET_TOL = 0.05                        # +/- 5 percent of target
METHODS = ["unguided", "symbolic", "semantic"]

N_HOPS = 3               # num_deduction_steps (includes the axiom step); held constant
ONTOLOGY = "fictional"   # prevents world-knowledge shortcuts
DEDUCTION_RULE = "ModusPonens"
DISTRACTORS = "relevant"
NO_ADJECTIVES = True     # drop decorative adjective sentences -> cleaner proofs (conclusion stays property-based)
CONCEPT_BLOCK_SIZE = 16  # concepts per disjoint per-item block (validated success rate on T4-free gen)

PILOT_ITEMS_PER_BUCKET = 20
FULL_ITEMS_PER_BUCKET = 100
N_CALIB_ITEMS = 10       # held-out for Phase 1.5 tau sweep (disjoint from pilot + full)
N_EXEMPLAR_ITEMS = 1     # few-shot exemplar(s); independent ontology, never in a bucket
# Base questions to generate = full/bucket + spares (that fail validation) + calib + exemplar.
# Each base item gets its OWN disjoint concept block. Pool items instead sample blocks from a
# shared POOL_CONCEPT_SPACE that is disjoint from ALL base blocks -- so every distractor is
# concept-disjoint from every base item, without needing N_POOL_ITEMS*block distinct names.
N_BASE_ITEMS = 150
N_POOL_ITEMS = 1000          # independent ontologies for the distractor pool (generation is cheap)
POOL_CONCEPT_SPACE = 2000    # shared concept vocabulary the pool items sample their blocks from

# --------------------------------------------------------------------------------------
# Decoding (plan section 4)
# --------------------------------------------------------------------------------------
MAX_NEW_TOKENS = 256
N_WARMUP = 3             # untimed warmup generations after every model load (rev #4)

# Chunked prefill is REQUIRED on T4, not optional. Llama-3.2-3B uses GQA (24 query heads vs
# 8 KV heads); PyTorch's mem-efficient SDPA kernel rejects mismatched head counts, flash is
# unavailable on sm75, and cuDNN is rejected -- so SDPA ALWAYS falls back to the math backend
# for this model on this GPU. A single 8k prefill then materializes a (1, 24, 8192, 8192) score
# tensor whose fp32 softmax upcast is the ~6 GiB OOM. Feeding the prompt in PREFILL_CHUNK-token
# chunks (building the KV cache incrementally) caps the per-chunk score tensor at ~400 MB.
# Lower this if 8k still peaks too high. Prefill is excluded from Panel B, so the metric is
# unchanged; the full chunked prefill is still timed into prefill_wall_s.
PREFILL_CHUNK = 512

# Semantic checker rollback (plan section 4.3, rev #5/#6)
SEMANTIC_MAX_RETRIES = 3
TEMPERATURE_SCHEDULE = [0.7, 0.9, 1.1]   # escalating; index by attempt number
PER_STEP_TOKEN_CAP = 48                  # force a boundary if no period within this many tokens

# --------------------------------------------------------------------------------------
# Semantic thresholds -- FROZEN BY PHASE 1.5. None => uncalibrated (runner must refuse).
# --------------------------------------------------------------------------------------
TAU_RESTATE: float | None = None   # cosine >= this => step restates a context fact/rule
TAU_MP: float | None = None        # cosine >= this => valid modus-ponens continuation
TAU_GRID = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]  # swept in Phase 1.5
# Exact polarity/predicate match on the cosine-retrieved winner. Required because bge-small
# cannot separate "X is slow"/"X is not slow" or distinct novel *pus concepts (verified in
# Phase 1.5). It verifies the associative-match winner; it does NOT replace the full-candidate
# similarity search (the O(context) op Panel B measures / CAM maps to). Keep ON for reported runs.
PREDICATE_GUARD = True

# --------------------------------------------------------------------------------------
# RNG seeds
# --------------------------------------------------------------------------------------
GLOBAL_SEED = 0          # base seed; per-item insertion seeds are derived + recorded


@dataclass
class RunConfig:
    """Bundle of the knobs a single run needs; serialized into the manifest."""

    model: str = PRIMARY_MODEL
    buckets: list[int] = field(default_factory=lambda: list(BUCKETS))
    methods: list[str] = field(default_factory=lambda: list(METHODS))
    items_per_bucket: int = FULL_ITEMS_PER_BUCKET
    max_new_tokens: int = MAX_NEW_TOKENS
    tau_restate: float | None = TAU_RESTATE
    tau_mp: float | None = TAU_MP
    seed: int = GLOBAL_SEED

    def require_calibrated(self) -> None:
        """Raise if tau is unset -- no reported run may use uncalibrated thresholds (R2 #1)."""
        if self.tau_restate is None or self.tau_mp is None:
            raise RuntimeError(
                "tau thresholds are not calibrated. Run Phase 1.5 and freeze "
                "TAU_RESTATE / TAU_MP into config.py before any reported run."
            )
