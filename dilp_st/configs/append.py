"""
Append task: learn the append function for lists.

Language: P = {app/3}, F = {f/2}, A = {a, b, c, *}, V = {x, y, z, v, w}

Target program:
  app([], x, x)
  app(x, [], x)
  app([x|y], z, [x|v]) ← app(y, z, v)

Background: (none — learn from scratch)
"""

from __future__ import annotations

from itertools import product as cartesian_product

from ..logic.language import (
    Atom,
    Clause,
    Constant,
    FunctionSymbol,
    Language,
    Predicate,
    Variable,
    make_list,
)
from ..problems.ilp_problem import ILPProblem

# Constants
a = Constant("a")
b = Constant("b")
nil = Constant("*")  # empty list sentinel


def _app(l1, l2, result):
    """Build app(l1, l2, result) atom."""
    return Atom("app", (l1, l2, result))


def _make_list(elements):
    """Build a list term from Python list of Constant elements."""
    return make_list(elements, nil)


def build_append_problem() -> ILPProblem:
    """Build the Append ILP problem."""

    language = Language(
        predicates=(Predicate("app", 3),),
        functions=(FunctionSymbol("f", 2),),
        constants=(a, b, nil),
        variables=(Variable("x"), Variable("y"), Variable("z"),
                   Variable("v"), Variable("w")),
    )

    # Generate all list pairs up to length 2 and their correct appends.
    elements = [a, b]
    all_lists = [[]]
    for length in range(1, 3):
        for combo in cartesian_product(elements, repeat=length):
            all_lists.append(list(combo))

    positive = []
    background = []

    # Base cases as background.
    for lst in all_lists:
        lt = _make_list(lst)
        bg1 = _app(nil, lt, lt)         # app([], x, x)
        bg2 = _app(lt, nil, lt)         # app(x, [], x)
        background.append(bg1)
        background.append(bg2)

    # Positive: non-trivial appends.
    for l1 in all_lists:
        for l2 in all_lists:
            result = l1 + l2
            if len(result) > 4:  # bound result length
                continue
            lt1 = _make_list(l1)
            lt2 = _make_list(l2)
            ltr = _make_list(result)
            fact = _app(lt1, lt2, ltr)
            if fact not in background:
                positive.append(fact)

    # Deduplicate.
    positive = list(dict.fromkeys(positive))
    background = list(dict.fromkeys(background))

    # Negative: wrong appends.
    negative = []
    for l1 in all_lists[:4]:
        for l2 in all_lists[:4]:
            correct = l1 + l2
            # Try a wrong result.
            wrong = list(reversed(l1)) + l2 if l1 else l2 + [a]
            if wrong != correct and len(wrong) <= 4:
                neg = _app(_make_list(l1), _make_list(l2), _make_list(wrong))
                if neg not in positive and neg not in background:
                    negative.append(neg)

    negative = list(dict.fromkeys(negative))

    # Initial clause.
    initial_clauses = [
        Clause(Atom("app", (Variable("x"), Variable("y"), Variable("z")))),
    ]

    return ILPProblem(
        positive_examples=positive,
        negative_examples=negative,
        background=background,
        language=language,
        initial_clauses=initial_clauses,
    )
