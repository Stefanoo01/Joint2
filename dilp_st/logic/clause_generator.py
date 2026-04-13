"""
Algorithm 1: Clause generation via beam search with refinement.

``beam_search(C0, problem, N_beam, T_beam)`` returns a pool of candidate
clauses that forms the search space for the differentiable optimiser.
"""

from __future__ import annotations

from typing import List, Set

from .language import Atom, Clause, Language
from .refinement import refine
from .unification import apply, is_unifiable, unify


def _clause_score(
    clause: Clause,
    positive: List[Atom],
    negative: List[Atom],
    background: List[Atom],
) -> float:
    """Heuristic score for a clause based on coverage.

    Higher is better:  +1 for each positive example the clause *could* entail
    (head unifies), −1 for each negative example.  This is a coarse heuristic;
    the differentiable optimiser does the real selection.
    """
    score = 0.0
    for e in positive:
        if is_unifiable(clause.head, e):
            score += 1.0
    for e in negative:
        if is_unifiable(clause.head, e):
            score -= 0.5  # penalise less harshly — the optimiser can handle noise
    return score


def beam_search(
    initial_clauses: List[Clause],
    language: Language,
    positive: List[Atom],
    negative: List[Atom],
    background: List[Atom],
    n_beam: int = 20,
    t_beam: int = 3,
    max_depth: int = 4,
    max_body: int = 3,
) -> List[Clause]:
    """Generate a pool of candidate clauses via beam search with refinement.

    Parameters
    ----------
    initial_clauses : list[Clause]
        Starting clauses C0 (typically just bare head atoms like ``p(x, y)``).
    language : Language
        The language L defining the search space.
    positive, negative, background : list[Atom]
        E+, E−, B from the ILP problem.
    n_beam : int
        Beam width — keep top-N candidates at each step.
    t_beam : int
        Number of refinement iterations.
    max_depth : int
        Maximum term depth allowed during refinement.
    max_body : int
        Maximum number of body atoms allowed.

    Returns
    -------
    list[Clause]
        The final pool of candidate clauses.
    """
    # Start with the initial clauses.
    beam: list[Clause] = list(initial_clauses)
    all_candidates: set[Clause] = set(beam)

    for _step in range(t_beam):
        new_candidates: list[Clause] = []
        for clause in beam:
            refined = refine(clause, language, max_depth=max_depth)
            for r in refined:
                if r.body_length > max_body:
                    continue
                if r not in all_candidates:
                    new_candidates.append(r)
                    all_candidates.add(r)

        if not new_candidates:
            break

        # Score and keep top-N from the *new* candidates.
        scored = [
            (c, _clause_score(c, positive, negative, background))
            for c in new_candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        beam = [c for c, _ in scored[:n_beam]]

    # Return all accumulated candidates (initial + refined), sorted for determinism.
    return sorted(all_candidates, key=lambda c: repr(c))
