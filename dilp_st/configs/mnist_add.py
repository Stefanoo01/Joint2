"""
MNIST Addition task configuration for Joint Training.

Language: 
  P = {add/3, digit/2, plus/3}
  F = {}
  A = {img1, img2, 0, 1, ..., 18}
  V = {x, y, z}

Target rule to learn:
  add(img1, img2, z) <- digit(img1, x), digit(img2, y), plus(x, y, z)

Background:
  plus(0, 0, 0), plus(0, 1, 1), ..., plus(9, 9, 18)
"""

from __future__ import annotations

from ..logic.language import (
    Atom,
    Clause,
    Constant,
    Language,
    Predicate,
    Variable,
)
from ..problems.ilp_problem import ILPProblem

# Constants
img1 = Constant("img1")
img2 = Constant("img2")
nums = [Constant(str(i)) for i in range(19)]

def _digit(img: Constant, val: Constant) -> Atom:
    return Atom("digit", (img, val))

def _plus(v1: Constant, v2: Constant, sum_val: Constant) -> Atom:
    return Atom("plus", (v1, v2, sum_val))

def _add(i1: Constant, i2: Constant, result: Constant) -> Atom:
    return Atom("add", (i1, i2, result))

def build_mnist_add_problem() -> ILPProblem:
    language = Language(
        predicates=(
            Predicate("add", 3),
            Predicate("digit", 2),
            Predicate("plus", 3),
        ),
        functions=(),
        constants=(img1, img2) + tuple(nums),
        variables=(Variable("x"), Variable("y"), Variable("z")),
    )

    # Background knowledge: all correct additions 0+0 to 9+9.
    background = []
    for a in range(10):
        for b in range(10):
            c = a + b
            background.append(_plus(nums[a], nums[b], nums[c]))

    # For compilation, the dILP static grounder needs "positive examples"
    # to trigger the backward chaining algorithm. We define all possible 
    # add(img1, img2, y) targets as positive so they exist in the graph.
    # At runtime, we extract their soft valuations instead of cross-entropy to 1.0.
    positive = []
    for s in range(19):
        positive.append(_add(img1, img2, nums[s]))

    # Negative examples: not needed for graph generation if positive contains all targets.
    negative = []

    # Initial clauses for beam search (the template structure).
    # We want to learn add(I1, I2, Y)
    initial_clauses = [
        Clause(Atom("add", (img1, img2, Variable("z")))),
    ]

    return ILPProblem(
        positive_examples=positive,
        negative_examples=negative,
        background=background,
        language=language,
        initial_clauses=initial_clauses,
    )
