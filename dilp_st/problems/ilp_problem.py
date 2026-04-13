"""
ILP Problem specification.

An ILP problem is a tuple Q = (E+, E−, B, L) where:
  - E+ are positive examples (ground atoms that should be entailed)
  - E− are negative examples (ground atoms that should not be entailed)
  - B  is background knowledge (ground atoms known to be true)
  - L  is the language (predicates, functions, constants, variables)
  - C0 are initial clauses for beam search
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from ..logic.language import Atom, Clause, Language


@dataclass
class ILPProblem:
    """Complete specification of an ILP problem instance."""
    positive_examples: List[Atom]
    negative_examples: List[Atom]
    background: List[Atom]
    language: Language
    initial_clauses: List[Clause]

    def __post_init__(self):
        # Validate that examples and background are ground.
        for a in self.positive_examples:
            assert a.is_ground(), f"Positive example must be ground: {a}"
        for a in self.negative_examples:
            assert a.is_ground(), f"Negative example must be ground: {a}"
        for a in self.background:
            assert a.is_ground(), f"Background atom must be ground: {a}"
