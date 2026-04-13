"""
Algorithm 2: Ground atom enumeration via backward chaining.

``enumerate_ground_atoms(problem, clauses, T)`` returns an ordered list of
ground atoms required for T-step forward-chaining inference.

Optimised: pre-filters by predicate name so only compatible clause-atom pairs
are attempted for unification.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set

from .language import BOTTOM, TOP, Atom, Clause
from .unification import apply, unify


def enumerate_ground_atoms(
    positive: List[Atom],
    negative: List[Atom],
    background: List[Atom],
    clauses: List[Clause],
    T: int,
) -> List[Atom]:
    """Enumerate the set of ground atoms G needed for inference.

    Implements Algorithm 2 from the paper with predicate-based pre-filtering:
    1. G ← {⊥, ⊤} ∪ E+ ∪ E- ∪ B
    2. For T iterations, for each clause, for each atom in G *that shares
       the same predicate as the clause head*, try unification and add
       substituted body atoms to G.

    Parameters
    ----------
    positive : list[Atom]
        Positive examples E+.
    negative : list[Atom]
        Negative examples E−.
    background : list[Atom]
        Background knowledge B.
    clauses : list[Clause]
        The clause pool C (output of beam search).
    T : int
        Number of enumeration iterations (same as inference steps).

    Returns
    -------
    list[Atom]
        Ordered list of ground atoms.  Index 0 = ⊥, index 1 = ⊤,
        followed by the rest in discovery order.
    """
    # Pre-group clauses by head predicate for O(1) lookup.
    clauses_by_pred: Dict[str, List[Clause]] = defaultdict(list)
    for clause in clauses:
        if not clause.is_fact():  # facts have no body → nothing to enumerate
            clauses_by_pred[clause.head.predicate].append(clause)

    # Use a dict for O(1) membership + insertion-order preservation.
    g_set: dict[Atom, None] = {}
    g_set[BOTTOM] = None
    g_set[TOP] = None

    # Index atoms by predicate for efficient matching.
    pred_index: Dict[str, List[Atom]] = defaultdict(list)

    def _add_atom(a: Atom) -> bool:
        """Add atom to G if not already present. Returns True if new."""
        if a not in g_set:
            g_set[a] = None
            pred_index[a.predicate].append(a)
            return True
        return False

    for a in positive:
        _add_atom(a)
    for a in negative:
        _add_atom(a)
    for a in background:
        _add_atom(a)

    for _step in range(T):
        new_atoms: list[Atom] = []

        # For each predicate, only try clauses whose head matches.
        for pred_name, pred_clauses in clauses_by_pred.items():
            # Get current atoms with this predicate.
            atoms_with_pred = list(pred_index.get(pred_name, []))

            for clause in pred_clauses:
                for g in atoms_with_pred:
                    theta = unify(clause.head, g)
                    if theta is None:
                        continue
                    # Apply substitution to each body atom.
                    for body_atom in clause.body:
                        grounded = apply(body_atom, theta)
                        assert isinstance(grounded, Atom)
                        if grounded.is_ground() and grounded not in g_set:
                            new_atoms.append(grounded)

        if not new_atoms:
            break
        for a in new_atoms:
            _add_atom(a)

    return list(g_set.keys())
