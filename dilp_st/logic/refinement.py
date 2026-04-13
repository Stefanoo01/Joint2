"""
Refinement operators for clause specialisation (Eq. 1 of the paper).

Four operators:
  ρ_fun  — apply function symbols to variables
  ρ_sub  — substitute constants for variables
  ρ_rep  — replace a variable with another existing variable
  ρ_add  — add a body atom

The combined operator ``refine(clause, language)`` returns the union.
"""

from __future__ import annotations

from itertools import combinations
from typing import List, Set

from .language import (
    Atom,
    Clause,
    Constant,
    FunctionSymbol,
    FunctionTerm,
    Language,
    Predicate,
    Variable,
    term_depth,
)
from .unification import Substitution, apply


# Maximum term depth allowed during refinement to bound the search space.
DEFAULT_MAX_DEPTH = 4


def refine_fun(
    clause: Clause,
    language: Language,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> List[Clause]:
    """ρ_fun: substitute a variable *z* in *clause* with ``f(x1, ..., xn)``
    where ``f ∈ F`` and ``x1 ... xn`` are fresh variables."""
    results: list[Clause] = []
    clause_vars = clause.variables()
    # Pool of variable names to draw fresh ones from.
    all_var_names = {v.name for v in language.variables}
    used_var_names = {v.name for v in clause_vars}

    for z in clause_vars:
        for fs in language.functions:
            # Pick fresh variables not yet in the clause.
            available = sorted(all_var_names - used_var_names)
            if len(available) < fs.arity:
                continue  # not enough fresh variables
            fresh = [Variable(available[i]) for i in range(fs.arity)]
            new_term = FunctionTerm(fs.name, tuple(fresh))
            # Check depth bound
            if term_depth(new_term) > max_depth:
                continue
            subst: Substitution = {z.name: new_term}
            refined = apply(clause, subst)
            assert isinstance(refined, Clause)
            results.append(refined)
    return results


def refine_sub(clause: Clause, language: Language) -> List[Clause]:
    """ρ_sub: substitute a variable *z* with a constant *a ∈ A*."""
    results: list[Clause] = []
    clause_vars = clause.variables()
    for z in clause_vars:
        for a in language.constants:
            subst: Substitution = {z.name: a}
            refined = apply(clause, subst)
            assert isinstance(refined, Clause)
            results.append(refined)
    return results


def refine_rep(clause: Clause, language: Language) -> List[Clause]:
    """ρ_rep: replace a variable *z* with another variable *y* already in the
    clause (z ≠ y)."""
    results: list[Clause] = []
    clause_vars = sorted(clause.variables(), key=lambda v: v.name)
    for i, z in enumerate(clause_vars):
        for j, y in enumerate(clause_vars):
            if i == j:
                continue
            subst: Substitution = {z.name: y}
            refined = apply(clause, subst)
            assert isinstance(refined, Clause)
            results.append(refined)
    return results


def refine_add(clause: Clause, language: Language) -> List[Clause]:
    """ρ_add: add a body atom ``p(x1, ..., xn)`` using existing distinct
    variables from the clause."""
    results: list[Clause] = []
    clause_vars = sorted(clause.variables(), key=lambda v: v.name)

    for pred in language.predicates:
        # Generate all n-combinations of distinct variables for this predicate.
        if len(clause_vars) < pred.arity:
            continue
        for combo in combinations(clause_vars, pred.arity):
            # Also try all permutations of the combo for different arg orderings.
            for perm in _permutations(list(combo)):
                new_atom = Atom(pred.name, tuple(perm))
                # Avoid adding a body atom identical to one already present.
                if new_atom in clause.body:
                    continue
                # Avoid adding the head as a body atom in trivial cases.
                new_clause = Clause(clause.head, clause.body + (new_atom,))
                results.append(new_clause)
    return results


def refine(
    clause: Clause,
    language: Language,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> List[Clause]:
    """Combined refinement operator ρ_L = ρ_fun ∪ ρ_sub ∪ ρ_rep ∪ ρ_add."""
    result: list[Clause] = []
    result.extend(refine_fun(clause, language, max_depth))
    result.extend(refine_sub(clause, language))
    result.extend(refine_rep(clause, language))
    result.extend(refine_add(clause, language))
    # Deduplicate
    seen: set[Clause] = set()
    unique: list[Clause] = []
    for c in result:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _permutations(lst: list) -> list[tuple]:
    """All permutations of a list (for small arities this is fine)."""
    if len(lst) <= 1:
        return [tuple(lst)]
    result = []
    for i, el in enumerate(lst):
        rest = lst[:i] + lst[i + 1:]
        for perm in _permutations(rest):
            result.append((el,) + perm)
    return result
