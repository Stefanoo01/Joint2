"""
MNIST Sum Parity task configuration for Joint Training.

Language: 
  P = {sum_parity/3, digit/2, even_num/1, odd_num/1}
  F = {}
  A = {img1, img2, 0, 1, 2, ..., 9}
  V = {x, y}

Target rules to learn:
  sum_parity(img1, img2, 0) <- digit(img1, X), even_num(X), digit(img2, Y), even_num(Y)
  sum_parity(img1, img2, 0) <- digit(img1, X), odd_num(X), digit(img2, Y), odd_num(Y)
  sum_parity(img1, img2, 1) <- digit(img1, X), even_num(X), digit(img2, Y), odd_num(Y)
  sum_parity(img1, img2, 1) <- digit(img1, X), odd_num(X), digit(img2, Y), even_num(Y)

Background:
  even_num(0), even_num(2), ...
  odd_num(1), odd_num(3), ...
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
nums = [Constant(str(i)) for i in range(10)] # 0-9 digits
parity_ans = [Constant("0"), Constant("1")]  # 0=even, 1=odd

def _digit(img: Constant, val: Constant) -> Atom:
    return Atom("digit", (img, val))

def _even(val: Constant) -> Atom:
    return Atom("even_num", (val,))

def _odd(val: Constant) -> Atom:
    return Atom("odd_num", (val,))

def _sum_parity(i1: Constant, i2: Constant, result: Constant) -> Atom:
    return Atom("sum_parity", (i1, i2, result))

def build_mnist_sum_parity_problem() -> ILPProblem:
    language = Language(
        predicates=(
            Predicate("sum_parity", 3),
            Predicate("digit", 2),
            Predicate("even_num", 1),
            Predicate("odd_num", 1),
        ),
        functions=(),
        constants=(img1, img2) + tuple(nums),
        variables=(Variable("x"), Variable("y")),
    )

    # Background knowledge: parity of 0-9
    background = []
    for i in range(10):
        if i % 2 == 0:
            background.append(_even(nums[i]))
        else:
            background.append(_odd(nums[i]))

    # Target examples to compute soft valuations
    positive = []
    for p in parity_ans:
        positive.append(_sum_parity(img1, img2, p))

    negative = []

    # Initial clauses for beam search (the template structure).
    # sum_parity(img1, img2, C) <- digit(img1, X), P1(X), digit(img2, Y), P2(Y)
    initial_clauses = []
    
    X = Variable("x")
    Y = Variable("y")
    
    # We define the strict combination templates directly to avoid exploding the beam.
    for p_ans in parity_ans:
        for p1 in ["even_num", "odd_num"]:
            for p2 in ["even_num", "odd_num"]:
                head = Atom("sum_parity", (img1, img2, p_ans))
                body = (
                    Atom("digit", (img1, X)),
                    Atom(p1, (X,)),
                    Atom("digit", (img2, Y)),
                    Atom(p2, (Y,)),
                )
                initial_clauses.append(Clause(head, body))
                
                # We also include clauses with only ONE digit dependency 
                # (Shortcut logic template: sum_parity purely dictacted by one digit!)
                body_shortcut_1 = (
                    Atom("digit", (img1, X)),
                    Atom(p1, (X,)),
                )
                initial_clauses.append(Clause(head, body_shortcut_1))
                
                body_shortcut_2 = (
                    Atom("digit", (img2, Y)),
                    Atom(p2, (Y,)),
                )
                initial_clauses.append(Clause(head, body_shortcut_2))

    return ILPProblem(
        positive_examples=positive,
        negative_examples=negative,
        background=background,
        language=language,
        initial_clauses=initial_clauses,
    )
