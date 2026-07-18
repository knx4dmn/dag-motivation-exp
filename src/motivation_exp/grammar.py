"""Per-item EBNF, XGrammar compilation, exemplar validation, and the shared clause parser.

Built against PrOntoQA's ACTUAL sentence forms (verified from the cloned generator at the
pinned commit). PrOntoQA ModusPonens reasoning mixes two predicate kinds:

  * concept membership -- takes an article/pluralizes: ``Sally is a tumpus.`` /
    ``Impuses are tumpuses.`` / ``Every impus is a zumpus.``
  * property predication -- an adjective, no article, never pluralized:
    ``Sally is not slow.`` / ``Tumpuses are not slow.`` / ``Each yumpus is shy.``

subjects are a proper name (fact) or ``Every``/``Each`` + concept, or a capitalized plural
concept (plural rule). PrOntoQA's property words are a FIXED, closed set (:data:`PROPERTY_WORDS`),
so a bare (article-less) predicate is classified exactly, not guessed.

One parser (:func:`parse_clause`) + one EBNF builder (:func:`build_item_ebnf`) + one conformance
check (:func:`output_conforms`) are derived from the same form definitions so the symbolic
grammar, the semantic checker, and the exemplar validation cannot drift.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

# --------------------------------------------------------------------------------------
# Fixed PrOntoQA vocabulary (from run_experiment.py at the pinned commit)
# --------------------------------------------------------------------------------------
# Property adjectives = the flattened property families (run_experiment.py lines ~356-368).
PROPERTY_WORDS = frozenset({
    "blue", "red", "brown", "orange", "small", "large", "metallic", "wooden", "luminous",
    "liquid", "transparent", "opaque", "nervous", "happy", "feisty", "shy", "bright", "dull",
    "sweet", "sour", "spicy", "bitter", "floral", "fruity", "earthy", "hot", "cold",
    "temperate", "kind", "mean", "angry", "amenable", "aggressive", "melodic", "muffled",
    "discordant", "loud", "slow", "moderate", "fast", "windy", "sunny", "overcast", "rainy",
    "snowy",
})
# Proper names (run_experiment.py line 321).
ENTITY_NAMES = ("Fae", "Rex", "Sally", "Max", "Alex", "Sam", "Polly", "Stella", "Wren")

ANSWER_PREFIX = "The answer is "
ANSWER_RE = re.compile(r"^" + re.escape(ANSWER_PREFIX) + r"(True|False)\.$")


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def plural_of(concept: str) -> str:
    """PrOntoQA pluralizes concept nouns by appending 'es' (wumpus -> wumpuses)."""
    return concept + "es"


def _singular_of(plural: str) -> str:
    return plural[:-2] if plural.endswith("es") else plural


# --------------------------------------------------------------------------------------
# Clause model + parser (shared by datagen extraction, the checker, and conformance)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Clause:
    """A parsed PrOntoQA sentence.

    kind='fact': ``subject`` is an entity name; the clause asserts subject has ``pred``.
    kind='rule': ``subject`` is the antecedent concept (singular); every such subject has ``pred``.
    ``is_property`` distinguishes a property adjective from a concept noun. ``negated`` flips it.
    """

    kind: str            # "fact" | "rule"
    subject: str         # entity name (fact) or antecedent concept, singular lowercase (rule)
    pred: str            # concept (singular lowercase) or property word
    is_property: bool
    negated: bool


_NAME = r"[A-Z][a-zA-Z]*"
_LWORD = r"[a-z][a-zA-Z]*"
_CWORD = r"[A-Z][a-zA-Z]*"

# fact:  "<Name> is [not] a|an <concept>."  |  "<Name> is [not] <property>."
_FACT_CONCEPT = re.compile(rf"^({_NAME}) is (not )?(?:an?) ({_LWORD})\.$")
_FACT_PROP = re.compile(rf"^({_NAME}) is (not )?({_LWORD})\.$")
# singular rule: "Every|Each <concept> is [not] a|an <concept>." | "... <property>."
_RULE_CONCEPT = re.compile(rf"^(?:Every|Each) ({_LWORD}) is (not )?(?:an?) ({_LWORD})\.$")
_RULE_PROP = re.compile(rf"^(?:Every|Each) ({_LWORD}) is (not )?({_LWORD})\.$")
# plural rule: "<Cplural> are [not] <cplural>." | "<Cplural> are [not] <property>."
_PLURAL = re.compile(rf"^({_CWORD}) are (not )?({_LWORD})\.$")


_LEADING_CONNECTIVE = re.compile(
    r"^(so|therefore|thus|then|hence|hereby|next|clearly|this means( that)?|"
    r"we know( that)?|it follows( that)?)[,]?\s+",
    re.IGNORECASE,
)
_MID_CONNECTIVE = re.compile(r"\b(therefore|thus|hence)\s+", re.IGNORECASE)


def strip_connectives(sentence: str) -> str:
    """Remove reasoning connectives an unguided LM adds to an otherwise-templatic step.

    e.g. "So Wren is a wumpus." / "Therefore, Wren is a wumpus." / "Wren is a wumpus, so it is
    not slow." -> "Wren is a wumpus." Diagnostic/normalization helper (see diagnostics/); NOT
    wired into :func:`parse_clause`'s accept path -- proposed fix pending review.
    """
    s = sentence.strip()
    prev = None
    while s != prev:                                   # peel stacked leading connectives
        prev = s
        s = _LEADING_CONNECTIVE.sub("", s).strip()
    s = re.split(r",\s*(?:so|therefore|thus|hence)\b", s, flags=re.IGNORECASE)[0].strip()  # compound tail
    s = _MID_CONNECTIVE.sub("", s).strip()
    if s and not s.endswith("."):
        s += "."
    return (s[0].upper() + s[1:]) if s else s


def parse_clause(sentence: str) -> Clause | None:
    """Parse one PrOntoQA sentence into a :class:`Clause`, or None if it matches no form.

    Predicate kind (concept vs property) is decided against the closed :data:`PROPERTY_WORDS`
    set, so an article-less predicate is classified exactly.
    """
    s = sentence.strip()
    # Order matters: try the article'd (concept) forms before the bare (property) forms.
    m = _FACT_CONCEPT.match(s)
    if m:
        return Clause("fact", m.group(1), m.group(3), False, bool(m.group(2)))
    m = _RULE_CONCEPT.match(s)
    if m:
        return Clause("rule", m.group(1), m.group(3), False, bool(m.group(2)))
    m = _FACT_PROP.match(s)
    if m and m.group(3) in PROPERTY_WORDS:
        return Clause("fact", m.group(1), m.group(3), True, bool(m.group(2)))
    m = _RULE_PROP.match(s)
    if m and m.group(3) in PROPERTY_WORDS:
        return Clause("rule", m.group(1), m.group(3), True, bool(m.group(2)))
    m = _PLURAL.match(s)
    if m:
        subj = _singular_of(m.group(1).lower())
        pred_raw = m.group(3)
        if pred_raw in PROPERTY_WORDS:
            return Clause("rule", subj, pred_raw, True, bool(m.group(2)))
        return Clause("rule", subj, _singular_of(pred_raw), False, bool(m.group(2)))
    return None


# --------------------------------------------------------------------------------------
# Rendering (used by the checker's modus-ponens synthesis)
# --------------------------------------------------------------------------------------
def render_fact(entity: str, pred: str, is_property: bool, negated: bool) -> str:
    """Render an entity fact in canonical singular form."""
    neg = "not " if negated else ""
    if is_property:
        return f"{entity} is {neg}{pred}."
    return f"{entity} is {neg}{_article(pred)} {pred}."


def render_clause(c: "Clause") -> str:
    """Render a parsed clause back to a canonical surface string (fact or Every-rule form)."""
    if c.kind == "fact":
        return render_fact(c.subject, c.pred, c.is_property, c.negated)
    neg = "not " if c.negated else ""
    if c.is_property:
        return f"Every {c.subject} is {neg}{c.pred}."
    return f"Every {c.subject} is {neg}{_article(c.pred)} {c.pred}."


def negate_sentence(sentence: str) -> str | None:
    """Return the polarity-flipped version of a clause (for τ discrimination tests), or None."""
    import dataclasses
    c = parse_clause(sentence)
    return None if c is None else render_clause(dataclasses.replace(c, negated=not c.negated))


def swap_concept(sentence: str, alt_concept: str) -> str | None:
    """Replace a concept-membership clause's predicate with ``alt_concept`` (τ tests), or None.

    Returns None for property clauses (a property swap is a different discrimination case).
    """
    import dataclasses
    c = parse_clause(sentence)
    if c is None or c.is_property:
        return None
    return render_clause(dataclasses.replace(c, pred=alt_concept))


# --------------------------------------------------------------------------------------
# EBNF builder
# --------------------------------------------------------------------------------------
def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _alt(words: Sequence[str]) -> str:
    return " | ".join(f'"{_esc(w)}"' for w in words)


def build_item_ebnf(
    entities: Sequence[str], concepts: Sequence[str], properties: Sequence[str] = ()
) -> str:
    """Build a per-item EBNF (XGrammar/GBNF) over the item's concept + property vocabulary.

    Output is one-or-more templatic reasoning steps (each ending in a newline) followed by a
    ``The answer is True.``/``False.`` line. Terminals are drawn ONLY from the item's vocabulary.
    Covers concept-membership and property predicates, singular (Name / Every / Each) and plural
    subjects, positive and negated. The root can complete after the answer, so XGrammar permits
    EOS there.
    """
    if not concepts and not properties:
        raise ValueError("cannot build grammar with empty concept+property vocabulary")

    concepts = list(dict.fromkeys(concepts))
    properties = list(dict.fromkeys(properties))
    has_names = bool(entities)
    has_prop = bool(properties)

    # predicate alternatives
    pred_sing_alts = []
    pred_plur_alts = []
    if concepts:
        pred_sing_alts.append('article " " concept')
        pred_plur_alts.append("cplural_low")
    if has_prop:
        pred_sing_alts.append("property")
        pred_plur_alts.append("property")
    pred_sing = " | ".join(pred_sing_alts)
    pred_plur = " | ".join(pred_plur_alts)

    clause_alts = []
    if has_names:
        clause_alts.append('name " is " ("not ")? (' + pred_sing + ')')
    if concepts:
        clause_alts.append('("Every " | "Each ") concept " is " ("not ")? (' + pred_sing + ')')
        clause_alts.append('cplural_cap " are " ("not ")? (' + pred_plur + ')')
    clause = " | ".join(f"({c})" for c in clause_alts)

    lines = [
        "root ::= step+ answer",
        'step ::= (' + clause + ') "\\n"',
        'answer ::= "' + _esc(ANSWER_PREFIX) + '" ("True" | "False") "."',
        'article ::= "a" | "an"',
    ]
    if concepts:
        lines.append("concept ::= " + _alt(concepts))
        lines.append("cplural_low ::= " + _alt([plural_of(c) for c in concepts]))
        lines.append("cplural_cap ::= " + _alt([plural_of(c).capitalize() for c in concepts]))
    if has_names:
        lines.append("name ::= " + _alt(entities))
    if has_prop:
        lines.append("property ::= " + _alt(properties))
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------
# Conformance check (mirrors the grammar; used in tests + exemplar validation)
# --------------------------------------------------------------------------------------
def output_conforms(
    text: str, entities: Sequence[str], concepts: Sequence[str], properties: Sequence[str] = ()
) -> bool:
    """True iff ``text`` is valid grammar output: step lines then an answer line, every
    terminal present in the item's vocabulary.
    """
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if len(lines) < 2:
        return False
    *step_lines, answer_line = lines
    if not ANSWER_RE.match(answer_line) or not step_lines:
        return False
    ent_set = set(entities)
    con_set = {c.lower() for c in concepts}
    prop_set = {p.lower() for p in properties}
    for ln in step_lines:
        c = parse_clause(ln)
        if c is None:
            return False
        if c.kind == "fact" and c.subject not in ent_set:
            return False
        if c.kind == "rule" and c.subject not in con_set:
            return False
        if c.is_property and c.pred not in prop_set:
            return False
        if not c.is_property and c.pred not in con_set:
            return False
    return True


# --------------------------------------------------------------------------------------
# Exemplar synthesis (dedicated item, never in a bucket -- R2 #2)
# --------------------------------------------------------------------------------------
def synthesize_exemplar_cot(exemplar_item) -> str:
    """Build the few-shot CoT for a dedicated exemplar item from its real gold steps.

    ``exemplar_item`` is an independent-ontology item that is NEVER placed in any bucket
    (R2 #2). The gold steps are PrOntoQA's own output and are used verbatim (the grammar is
    derived to match those forms), followed by the answer line. Raises if a step does not
    conform to the exemplar's own grammar -- pick a different exemplar in that case.
    """
    steps = list(exemplar_item.gold_steps)
    if not steps:
        raise ValueError("exemplar item must carry gold_steps to synthesize a CoT")
    props = getattr(exemplar_item, "properties", [])
    for s in steps:
        cot_line = s + "\nThe answer is True."  # per-line conformance harness
        if not output_conforms(cot_line, exemplar_item.entities, exemplar_item.concepts, props):
            raise ValueError(f"exemplar step does not conform to its grammar: {s!r}")
    answer = f"{ANSWER_PREFIX}{'True' if exemplar_item.gold else 'False'}."
    return "".join(s + "\n" for s in steps) + answer


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
    """Compile an EBNF string with a shared ``GrammarCompiler`` (cache keys on the string)."""
    return compiler.compile_grammar(ebnf)
