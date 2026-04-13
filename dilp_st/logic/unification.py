"""
First-order unification engine (Martelli–Montanari style).

Provides:
- ``unify(t1, t2) -> Optional[Substitution]`` — most general unifier
- ``is_unifiable(t1, t2) -> bool``
- ``apply(expr, subst)`` — apply substitution to term / atom / clause
"""

from __future__ import annotations

from typing import Dict, Optional

from .language import (
    Atom,
    Clause,
    Constant,
    FunctionTerm,
    Term,
    Variable,
)

# A substitution maps variable names → terms.
Substitution = Dict[str, Term]


def unify(t1: Term | Atom, t2: Term | Atom) -> Optional[Substitution]:
    """Compute the most general unifier for *t1* and *t2*.

    Returns ``None`` if the two expressions are not unifiable.
    Handles atoms transparently by wrapping them as pseudo-function-terms.
    """
    # Unwrap atoms into a canonical form for uniform handling.
    a, b = _to_unify_expr(t1), _to_unify_expr(t2)
    subst: Substitution = {}
    stack: list[tuple[_UExpr, _UExpr]] = [(a, b)]

    while stack:
        s, t = stack.pop()
        s = _walk(s, subst)
        t = _walk(t, subst)

        if s == t:
            continue

        if isinstance(s, str):
            if _occurs(s, t, subst):
                return None
            subst[s] = _from_uexpr(t)
            continue

        if isinstance(t, str):
            if _occurs(t, s, subst):
                return None
            subst[t] = _from_uexpr(s)
            continue

        if isinstance(s, tuple) and isinstance(t, tuple):
            if s[0] != t[0] or len(s[1]) != len(t[1]):
                return None
            for sa, ta in zip(s[1], t[1]):
                stack.append((sa, ta))
            continue

        # One is a ground constant string-key, the other compound — fail.
        return None

    return _resolve(subst)


def is_unifiable(t1: Term | Atom, t2: Term | Atom) -> bool:
    """Decision function σ̄  — return whether *t1* and *t2* are unifiable."""
    return unify(t1, t2) is not None


def apply(expr: Term | Atom | Clause, subst: Substitution) -> Term | Atom | Clause:
    """Apply substitution *subst* to *expr*, returning a new expression."""
    if isinstance(expr, Variable):
        return _apply_term(expr, subst)
    if isinstance(expr, Constant):
        return expr
    if isinstance(expr, FunctionTerm):
        return _apply_term(expr, subst)
    if isinstance(expr, Atom):
        return Atom(expr.predicate, tuple(_apply_term(a, subst) for a in expr.args))
    if isinstance(expr, Clause):
        new_head = apply(expr.head, subst)
        assert isinstance(new_head, Atom)
        new_body = tuple(apply(b, subst) for b in expr.body)
        assert all(isinstance(b, Atom) for b in new_body)
        return Clause(new_head, new_body)  # type: ignore[arg-type]
    raise TypeError(f"Cannot apply substitution to {type(expr)}")


# ---------------------------------------------------------------------------
# Internal representation for unification
#   Variable  → str (its name)
#   Constant  → tuple (name, ())
#   FuncTerm  → tuple (symbol, (child, ...))
#   Atom      → tuple (predicate, (child, ...))
# ---------------------------------------------------------------------------

_UExpr = str | tuple  # union of the two internal forms


def _to_unify_expr(t: Term | Atom) -> _UExpr:
    if isinstance(t, Variable):
        return t.name
    if isinstance(t, Constant):
        return ("__const__" + t.name, ())
    if isinstance(t, FunctionTerm):
        return (t.symbol, tuple(_to_unify_expr(a) for a in t.args))
    if isinstance(t, Atom):
        return ("__pred__" + t.predicate, tuple(_to_unify_expr(a) for a in t.args))
    raise TypeError(type(t))


def _from_uexpr(u: _UExpr) -> Term:
    """Convert internal representation back to a Term."""
    if isinstance(u, str):
        return Variable(u)
    name, children = u
    kids = tuple(_from_uexpr(c) for c in children)
    if name.startswith("__const__"):
        return Constant(name[len("__const__"):])
    # Everything else is a FunctionTerm (we never reconstruct Atom from here).
    return FunctionTerm(name, kids)


def _walk(u: _UExpr, subst: Substitution) -> _UExpr:
    """Chase variable bindings."""
    while isinstance(u, str) and u in subst:
        u = _to_unify_expr(subst[u])
    return u


def _occurs(var: str, expr: _UExpr, subst: Substitution) -> bool:
    """Occurs check: is *var* inside *expr*?"""
    expr = _walk(expr, subst)
    if isinstance(expr, str):
        return expr == var
    if isinstance(expr, tuple):
        return any(_occurs(var, c, subst) for c in expr[1])
    return False


def _resolve(subst: Substitution) -> Substitution:
    """Fully resolve a substitution so that no variable maps to another variable
    that is itself bound."""
    changed = True
    while changed:
        changed = False
        for k in list(subst):
            resolved = _full_apply_term(subst[k], subst)
            if resolved != subst[k]:
                subst[k] = resolved
                changed = True
    return subst


def _apply_term(t: Term, subst: Substitution) -> Term:
    if isinstance(t, Variable):
        if t.name in subst:
            return _apply_term(subst[t.name], subst)
        return t
    if isinstance(t, Constant):
        return t
    if isinstance(t, FunctionTerm):
        return FunctionTerm(t.symbol, tuple(_apply_term(a, subst) for a in t.args))
    raise TypeError(type(t))


def _full_apply_term(t: Term, subst: Substitution) -> Term:
    """Like _apply_term but ensures full resolution."""
    return _apply_term(t, subst)
