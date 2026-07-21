"""Format-only calculation grammar for the GSM8K symbolic arm (docs/gsm8k_motivation_plan.md).

Design decision (user sign-off, 2026-07): the symbolic baseline constrains the calculation
FORMAT, not the operand vocabulary. Amendment A originally said "operands = all context numbers +
whitelist", but a STATIC grammar cannot enumerate derived intermediates (e.g. ``24 + 48 = 72`` where
``24`` was computed in an earlier step and never appears in the problem text), so a context-only
operand rule would make every multi-step chain ungrammatical. Constraining only the format keeps the
grammar:

  * ITEM-INDEPENDENT and STATIC -- compiled once, reused for every item, so the symbolic arm's
    per-token latency stays FLAT with context length (the "symbolic flat" half of Panel B). A
    dynamic per-step grammar that admitted intermediates would recompile each step and grow with
    context, defeating the contrast.
  * a clean three-arm ladder: unguided (free) < symbolic (well-formed calculations, no relevance
    filter -- it may freely use a distractor number) < semantic (format + provenance + exact
    arithmetic). Symbolic guarantees only lexical well-formedness, never correctness.

Grammar (XGrammar / GBNF): one-or-more ``<expr> = <number>`` lines then a ``#### <number>`` answer.
Operands and results are free digit strings; EOS is permitted once the answer's number completes.
"""
from __future__ import annotations

# GBNF: operands/results are free numbers (a static grammar can't list runtime intermediates).
# `line+` -> at least one calculation (GSM8K problems always have >=1 op); the answer number is the
# final token so XGrammar allows EOS there (the symbolic loop also breaks on eot).
CALC_EBNF = (
    "root ::= line+ answer\n"
    'line ::= expr " = " number "\\n"\n'
    'expr ::= number (" " op " " number)+\n'
    'op ::= "+" | "-" | "*" | "/"\n'
    'number ::= [0-9]+ ("." [0-9]+)?\n'
    'answer ::= "#### " number\n'
)


def build_calc_ebnf() -> str:
    """Return the static, item-independent calculation-format EBNF (see module docstring)."""
    return CALC_EBNF


def compile_calc_grammar(compiler):
    """Compile the calculation grammar with a shared XGrammar ``GrammarCompiler`` (Colab-only).

    Compiled ONCE and reused across all items (the grammar is item-independent); the compiler's
    string cache makes a repeat call free. Mirrors ``grammar.compile_item_grammar``.
    """
    return compiler.compile_grammar(CALC_EBNF)


# --------------------------------------------------------------------------------------
# Pure-Python conformance mirror of CALC_EBNF (CPU-testable; no XGrammar needed)
# --------------------------------------------------------------------------------------
def _is_number(tok: str) -> bool:
    if tok.count(".") > 1:
        return False
    body = tok.replace(".", "", 1)
    return bool(body) and body.isdigit() and not tok.startswith(".") and not tok.endswith(".")


def _line_conforms(line: str) -> bool:
    """A calculation line: ``number (' ' op ' ' number)+ ' = ' number`` (>=1 operator)."""
    if " = " not in line:
        return False
    lhs, rhs = line.rsplit(" = ", 1)
    if not _is_number(rhs):
        return False
    parts = lhs.split(" ")
    if len(parts) < 3 or len(parts) % 2 == 0:            # num (op num)+ -> odd token count >=3
        return False
    for i, p in enumerate(parts):
        if i % 2 == 0:
            if not _is_number(p):
                return False
        elif p not in {"+", "-", "*", "/"}:
            return False
    return True


def calc_output_conforms(text: str) -> bool:
    """True iff ``text`` is valid symbolic output: >=1 calculation line then a ``#### <number>`` line.

    Mirrors :data:`CALC_EBNF` so tests can check conformance without compiling XGrammar.
    """
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if len(lines) < 2:
        return False
    *calc_lines, answer = lines
    if not answer.startswith("#### ") or not _is_number(answer[len("#### "):]):
        return False
    return bool(calc_lines) and all(_line_conforms(ln) for ln in calc_lines)
