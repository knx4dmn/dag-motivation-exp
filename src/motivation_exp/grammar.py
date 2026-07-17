"""Per-item EBNF construction, XGrammar compilation, and few-shot exemplar synthesis.

The symbolic method constrains generation to a sequence of reasoning-step sentences over
the item's in-context vocabulary, followed by a ``True``/``False`` answer line. This module
holds the ONE definition of the allowed sentence forms (:data:`STEP_TEMPLATES`) from which
three artifacts are derived so they cannot drift (the GAD failure mode, R2 #3):

  1. the EBNF grammar productions (:func:`build_item_ebnf`),
  2. a pure-Python conformance check (:func:`output_conforms`), used in CPU tests and as a
     cheap sanity gate, and
  3. the few-shot exemplar renderer (:func:`synthesize_exemplar_cot`).

XGrammar is imported lazily inside :func:`compile_item_grammar` / :func:`make_compiler` so
this module is importable (and testable) without the native package or a tokenizer.
"""
from __future__ import annotations

import re
from typing import Sequence

# --------------------------------------------------------------------------------------
# Step templates: the single source of truth for allowed sentence forms.
#
# A template is an ordered tuple of "parts". Each part is either a literal string or one of
# the slot markers NAME / CONCEPT / ART. NAME and CONCEPT slots carry vocabulary; ART is a
# derived article ("a"/"an") computed from the concept that immediately follows it.
# --------------------------------------------------------------------------------------
NAME, CONCEPT, ART = "NAME", "CONCEPT", "ART"

STEP_TEMPLATES: dict[str, tuple] = {
    # "Wren is a tumpus."
    "entity_pos": (NAME, " is ", ART, " ", CONCEPT, "."),
    # "Wren is not a tumpus."
    "entity_neg": (NAME, " is not ", ART, " ", CONCEPT, "."),
    # "Every tumpus is a wumpus."
    "rule_pos": ("Every ", CONCEPT, " is ", ART, " ", CONCEPT, "."),
    # "Every tumpus is not a wumpus."
    "rule_neg": ("Every ", CONCEPT, " is not ", ART, " ", CONCEPT, "."),
}

ANSWER_PREFIX = "The answer is "
ANSWER_RE = re.compile(r"^" + re.escape(ANSWER_PREFIX) + r"(True|False)\.$")


def _article(concept: str) -> str:
    return "an" if concept[:1].lower() in "aeiou" else "a"


# --------------------------------------------------------------------------------------
# Derivations from a template's parts
# --------------------------------------------------------------------------------------
def _slot_indices(parts: tuple) -> list[int]:
    """Positions of NAME/CONCEPT slots (the ones that consume a filler), in order."""
    return [i for i, p in enumerate(parts) if p in (NAME, CONCEPT)]


def _next_concept_filler_index(parts: tuple, art_pos: int) -> int:
    """Which filler index (into the NAME/CONCEPT filler list) the ART at ``art_pos`` refers to."""
    fi = 0
    for i, p in enumerate(parts):
        if p in (NAME, CONCEPT):
            if i > art_pos and p == CONCEPT:
                return fi
            fi += 1
    raise ValueError("ART slot has no following CONCEPT")


def render_template(template_id: str, fillers: Sequence[str]) -> str:
    """Render one sentence: ``fillers`` are the NAME/CONCEPT values in left-to-right order."""
    parts = STEP_TEMPLATES[template_id]
    out: list[str] = []
    fi = 0
    for i, p in enumerate(parts):
        if p == ART:
            concept_val = fillers[_next_concept_filler_index(parts, i)]
            out.append(_article(concept_val))
        elif p in (NAME, CONCEPT):
            out.append(fillers[fi])
            fi += 1
        else:
            out.append(p)
    return "".join(out)


def _template_regex(template_id: str) -> re.Pattern:
    parts = STEP_TEMPLATES[template_id]
    r = ["^"]
    for p in parts:
        if p == NAME:
            r.append(r"([A-Z][a-zA-Z]*)")
        elif p == CONCEPT:
            r.append(r"([a-z][a-zA-Z]*)")
        elif p == ART:
            r.append(r"(?:a|an)")
        else:
            r.append(re.escape(p))
    r.append("$")
    return re.compile("".join(r))


_TEMPLATE_REGEXES = {tid: _template_regex(tid) for tid in STEP_TEMPLATES}


def _escape_terminal(s: str) -> str:
    """Escape a string for a double-quoted GBNF/EBNF terminal."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _template_ebnf(template_id: str) -> str:
    parts = STEP_TEMPLATES[template_id]
    toks: list[str] = []
    for p in parts:
        if p == NAME:
            toks.append("name")
        elif p == CONCEPT:
            toks.append("concept")
        elif p == ART:
            toks.append("article")
        else:
            toks.append(f'"{_escape_terminal(p)}"')
    return " ".join(toks)


# --------------------------------------------------------------------------------------
# EBNF builder
# --------------------------------------------------------------------------------------
def build_item_ebnf(entities: Sequence[str], concepts: Sequence[str]) -> str:
    """Build a per-item EBNF (XGrammar/GBNF) over the item's vocabulary.

    Output is one-or-more reasoning steps (each a templatic sentence ending in a newline)
    followed by a single ``The answer is True.``/``False.`` line. NAME/CONCEPT terminals are
    drawn ONLY from ``entities``/``concepts``. Templates whose slots the vocabulary cannot
    fill (e.g. NAME templates when there are no entities) are dropped. The root can complete
    after the answer line, so XGrammar permits EOS there (rev #2).
    """
    if not concepts:
        raise ValueError("cannot build grammar with empty concept vocabulary")

    has_names = bool(entities)
    usable = [
        tid for tid in STEP_TEMPLATES
        if has_names or NAME not in STEP_TEMPLATES[tid]
    ]
    if not usable:
        raise ValueError("no usable step templates for this vocabulary")

    step_alts = " | ".join(f"({_template_ebnf(tid)})" for tid in usable)
    name_alts = " | ".join(f'"{_escape_terminal(e)}"' for e in entities) if has_names else '""'
    concept_alts = " | ".join(f'"{_escape_terminal(c)}"' for c in concepts)

    lines = [
        "root ::= step+ answer",
        'step ::= (' + step_alts + ') "\\n"',
        'answer ::= "' + _escape_terminal(ANSWER_PREFIX) + '" ("True" | "False") "."',
        "concept ::= " + concept_alts,
        'article ::= "a" | "an"',
    ]
    if has_names:
        lines.append("name ::= " + name_alts)
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------
# Pure-Python conformance check (mirrors the EBNF; used in tests + as a cheap gate)
# --------------------------------------------------------------------------------------
def _singular_set(concepts: Sequence[str]) -> set[str]:
    return {c.lower() for c in concepts}


def output_conforms(text: str, entities: Sequence[str], concepts: Sequence[str]) -> bool:
    """True iff ``text`` is a valid grammar output: step lines then an answer line, with
    every NAME/CONCEPT terminal present in the item's vocabulary.
    """
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if len(lines) < 2:
        return False
    *step_lines, answer_line = lines
    if not ANSWER_RE.match(answer_line):
        return False
    if not step_lines:
        return False

    ent_set = {e for e in entities}
    con_set = _singular_set(concepts)
    for ln in step_lines:
        if not _line_matches_a_template(ln, ent_set, con_set):
            return False
    return True


def _line_matches_a_template(line: str, ent_set: set[str], con_set: set[str]) -> bool:
    for tid, rx in _TEMPLATE_REGEXES.items():
        m = rx.match(line)
        if not m:
            continue
        # Group order follows the NAME/CONCEPT slot order in the template.
        slot_types = [STEP_TEMPLATES[tid][i] for i in _slot_indices(STEP_TEMPLATES[tid])]
        for val, typ in zip(m.groups(), slot_types):
            if typ == NAME and val not in ent_set:
                return False
            if typ == CONCEPT and val.lower() not in con_set:
                return False
        return True
    return False


# --------------------------------------------------------------------------------------
# Exemplar synthesis (from the SAME templates; dedicated item, never in a bucket -- R2 #2)
# --------------------------------------------------------------------------------------
def _normalize_step(sentence: str) -> str:
    """Parse a PrOntoQA chain sentence and re-render it in canonical template form.

    Guarantees the exemplar conforms to the grammar regardless of upstream phrasing quirks
    (article choice, spacing). Raises if the sentence matches no template.
    """
    s = sentence.strip()
    for tid, rx in _TEMPLATE_REGEXES.items():
        m = rx.match(s)
        if m:
            return render_template(tid, list(m.groups()))
    raise ValueError(f"exemplar step matches no known template: {sentence!r}")


def synthesize_exemplar_cot(exemplar_item) -> str:
    """Build the few-shot CoT text for a dedicated exemplar item from the shared templates.

    ``exemplar_item`` is an independent-ontology item that is NEVER placed in any bucket
    (R2 #2) -- passing a test item here would leak its own reasoning chain into the prompt
    and inflate accuracy for all methods. The returned text is grammar-conformant: normalized
    step lines followed by the answer line.
    """
    if not exemplar_item.gold_steps:
        raise ValueError("exemplar item must carry gold_steps to synthesize a CoT")
    steps = [_normalize_step(s) for s in exemplar_item.gold_steps]
    answer = f"{ANSWER_PREFIX}{'True' if exemplar_item.gold else 'False'}."
    return "".join(step + "\n" for step in steps) + answer


# --------------------------------------------------------------------------------------
# XGrammar compilation (lazy import; Colab-only path)
# --------------------------------------------------------------------------------------
def make_compiler(tokenizer, vocab_size: int):
    """Create a shared, cache-enabled XGrammar ``GrammarCompiler`` for a tokenizer.

    ``vocab_size`` MUST be ``model.config.vocab_size`` (e.g. 128256 for Llama-3.2), not
    ``len(tokenizer)`` -- a smaller value silently truncates the token bitmask.
    """
    import xgrammar as xgr

    tok_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
    return xgr.GrammarCompiler(tok_info, cache_enabled=True)


def compile_item_grammar(ebnf: str, compiler):
    """Compile an EBNF string with a shared ``GrammarCompiler`` (compilation cache keys on the string)."""
    return compiler.compile_grammar(ebnf)
