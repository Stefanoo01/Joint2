"""
Tensor encoding: compile symbolic clauses + ground atoms into the index
tensor ``X`` (Eq. 4 of the paper).

The output is a static ``torch.LongTensor`` that drives the differentiable
inference — moved to GPU once and reused every forward pass.

Optimised: pre-groups atoms by predicate for O(1) lookup during encoding.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import torch

from ..logic.language import BOTTOM, TOP, Atom, Clause
from ..logic.unification import apply, is_unifiable, unify


def build_index_tensor(
    clauses: List[Clause],
    ground_atoms: List[Atom],
    device: torch.device = torch.device("cpu"),
) -> torch.LongTensor:
    """Build the index tensor ``X`` of shape ``[|C|, |G|, b]``.

    ``X[i, j, k]`` contains the index in *ground_atoms* of the k-th subgoal
    needed for clause ``clauses[i]`` to entail ``ground_atoms[j]``.

    Encoding follows Eq. 4:
    - If clause head and ground atom are unifiable and k < body length:
      ``X[i,j,k] = index(B_k θ)``
    - If unifiable but k >= body length:  ``X[i,j,k] = index(⊤)``
    - If not unifiable:  ``X[i,j,k] = index(⊥)``

    Parameters
    ----------
    clauses : list[Clause]
        Clause pool C.
    ground_atoms : list[Atom]
        Ordered ground atoms G (index 0 = ⊥, index 1 = ⊤).
    device : torch.device
        Target device for the tensor.

    Returns
    -------
    torch.LongTensor
        Index tensor of shape ``[|C|, |G|, b]``.
    """
    num_clauses = len(clauses)
    num_atoms = len(ground_atoms)
    max_body = max((c.body_length for c in clauses), default=0)
    # Ensure at least 1 for the body dimension
    max_body = max(max_body, 1)

    # Build atom → index lookup.
    atom_to_idx: Dict[Atom, int] = {a: i for i, a in enumerate(ground_atoms)}
    idx_bot = atom_to_idx[BOTTOM]  # should be 0
    idx_top = atom_to_idx[TOP]     # should be 1

    # Group atoms by predicate for fast lookup.
    atoms_by_pred: Dict[str, List[int]] = defaultdict(list)
    for j, a in enumerate(ground_atoms):
        if a is not BOTTOM and a is not TOP:
            atoms_by_pred[a.predicate].append(j)

    # Pre-allocate and fill with ⊥ index (default: not unifiable).
    X = torch.full((num_clauses, num_atoms, max_body), idx_bot, dtype=torch.long)

    for i, clause in enumerate(clauses):
        head_pred = clause.head.predicate
        # Only iterate over atoms sharing the same predicate as the clause head.
        candidate_indices = atoms_by_pred.get(head_pred, [])

        for j in candidate_indices:
            g_atom = ground_atoms[j]

            theta = unify(clause.head, g_atom)
            if theta is None:
                continue

            # Unifiable — fill body subgoal indices.
            for k in range(max_body):
                if k < clause.body_length:
                    body_grounded = apply(clause.body[k], theta)
                    assert isinstance(body_grounded, Atom)
                    if body_grounded in atom_to_idx:
                        X[i, j, k] = atom_to_idx[body_grounded]
                    else:
                        X[i, j, k] = idx_bot
                else:
                    X[i, j, k] = idx_top

    return X.to(device)


def build_fact_mask(
    clauses: List[Clause],
    ground_atoms: List[Atom],
    device: torch.device = torch.device("cpu"),
) -> torch.BoolTensor:
    """Build a boolean mask identifying which (clause, atom) pairs correspond
    to *unit clauses* (facts) whose head directly matches the atom.

    Shape: ``[|C|, |G|]``.  ``True`` means clause i is a fact that directly
    entails ground atom j.  This is handled separately in inference because
    a fact ``p(a)`` has no body — its valuation is always 1 when matched.
    """
    num_clauses = len(clauses)
    num_atoms = len(ground_atoms)
    mask = torch.zeros(num_clauses, num_atoms, dtype=torch.bool)

    # Group atoms by predicate.
    atoms_by_pred: Dict[str, List[int]] = defaultdict(list)
    for j, a in enumerate(ground_atoms):
        if a is not BOTTOM and a is not TOP:
            atoms_by_pred[a.predicate].append(j)

    for i, clause in enumerate(clauses):
        if not clause.is_fact():
            continue
        head_pred = clause.head.predicate
        for j in atoms_by_pred.get(head_pred, []):
            if is_unifiable(clause.head, ground_atoms[j]):
                mask[i, j] = True

    return mask.to(device)
