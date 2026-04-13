"""
Plus task: learn the addition operation for natural numbers.

Language: P = {plus/3}, F = {s/1}, A = {0}, V = {x, y, z, v, w}

Target program:
  plus(0, x, x)
  plus(x, s(y), s(z)) ← plus(x, y, z)
  plus(s(x), y, s(z)) ← plus(y, x, z)

Background:
  plus(0, 0, 0)
"""

from __future__ import annotations

from ..logic.language import (
    Atom,
    Clause,
    Constant,
    FunctionSymbol,
    Language,
    Predicate,
    Variable,
    make_successor,
)
from ..problems.ilp_problem import ILPProblem

zero = Constant("0")


def _s(n: int):
    """Build s^n(0) term."""
    return make_successor(n, zero)


def _plus(a, b, c):
    """Build plus(a, b, c) atom."""
    return Atom("plus", (a, b, c))


def build_plus_problem(max_num: int = 4) -> ILPProblem:
    """Build the Plus ILP problem.

    Parameters
    ----------
    max_num : int
        Maximum natural number to include in examples.
        Positive examples are all correct additions a+b=c where a,b ≤ max_num.
    """

    language = Language(
        predicates=(Predicate("plus", 3),),
        functions=(FunctionSymbol("s", 1),),
        constants=(zero,),
        variables=(Variable("x"), Variable("y"), Variable("z"),
                   Variable("v"), Variable("w")),
    )

    # Background knowledge.
    background = [
        _plus(_s(0), _s(0), _s(0)),  # plus(0, 0, 0)
    ]

    # Positive examples: all correct additions up to max_num.
    positive = []
    for a in range(max_num + 1):
        for b in range(max_num + 1):
            c = a + b
            fact = _plus(_s(a), _s(b), _s(c))
            if fact not in background:
                positive.append(fact)

    # Negative examples: incorrect additions.
    negative = []
    for a in range(max_num + 1):
        for b in range(max_num + 1):
            c_correct = a + b
            # Add some wrong results.
            for c_wrong in [0, 1, max_num * 2]:
                if c_wrong != c_correct and c_wrong <= max_num * 2:
                    fact = _plus(_s(a), _s(b), _s(c_wrong))
                    if fact not in positive and fact not in background:
                        negative.append(fact)

    # Deduplicate negatives.
    negative = list(dict.fromkeys(negative))

    # Initial clause for beam search.
    initial_clauses = [
        Clause(Atom("plus", (Variable("x"), Variable("y"), Variable("z")))),
    ]

    return ILPProblem(
        positive_examples=positive,
        negative_examples=negative,
        background=background,
        language=language,
        initial_clauses=initial_clauses,
    )
