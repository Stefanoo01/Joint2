"""
Member task: learn the membership function for lists.

Language: P = {mem/2}, F = {f/2}, A = {a, b, c, *}, V = {x, y, z, v, w}

Target program:
  mem(x, [y|z]) ← mem(x, z)
  mem(x, [x|y])

Background:
  mem(a, [a]), mem(b, [b]), mem(c, [c])
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
    make_list,
)
from ..problems.ilp_problem import ILPProblem

# Constants
a = Constant("a")
b = Constant("b")
c = Constant("c")
nil = Constant("*")

# Variables
x = Variable("x")
y = Variable("y")


def _mem(elem, lst):
    """Helper: build mem(elem, lst) atom."""
    return Atom("mem", (elem, lst))


def build_member_problem() -> ILPProblem:
    """Build the Member ILP problem matching the paper specification."""

    language = Language(
        predicates=(Predicate("mem", 2),),
        functions=(FunctionSymbol("f", 2),),
        constants=(a, b, c, nil),
        variables=(Variable("x"), Variable("y"), Variable("z"),
                   Variable("v"), Variable("w")),
    )

    # Background knowledge: base cases.
    background = [
        _mem(a, make_list([a], nil)),  # mem(a, [a])
        _mem(b, make_list([b], nil)),  # mem(b, [b])
        _mem(c, make_list([c], nil)),  # mem(c, [c])
    ]

    # Positive examples.
    positive = [
        _mem(a, make_list([a, c], nil)),      # mem(a, [a, c])
        _mem(a, make_list([b, a], nil)),      # mem(a, [b, a])
        _mem(b, make_list([a, b], nil)),      # mem(b, [a, b])
        _mem(b, make_list([b, a], nil)),      # mem(b, [b, a])
        _mem(c, make_list([c, a], nil)),      # mem(c, [c, a])
        _mem(c, make_list([a, c], nil)),      # mem(c, [a, c])
        _mem(a, make_list([a, b, c], nil)),   # mem(a, [a, b, c])
        _mem(b, make_list([a, b, c], nil)),   # mem(b, [a, b, c])
        _mem(c, make_list([a, b, c], nil)),   # mem(c, [a, b, c])
    ]

    # Negative examples.
    negative = [
        _mem(c, make_list([b, a], nil)),      # mem(c, [b, a])
        _mem(c, make_list([a], nil)),          # mem(c, [a])
        _mem(b, make_list([a, c], nil)),      # mem(b, [a, c])
        _mem(a, make_list([b, c], nil)),      # mem(a, [b, c])
        _mem(b, make_list([a], nil)),          # mem(b, [a])
        _mem(a, make_list([b], nil)),          # mem(a, [b])
    ]

    # Initial clause for beam search.
    initial_clauses = [
        Clause(Atom("mem", (Variable("x"), Variable("y")))),  # mem(x, y)
    ]

    return ILPProblem(
        positive_examples=positive,
        negative_examples=negative,
        background=background,
        language=language,
        initial_clauses=initial_clauses,
    )
