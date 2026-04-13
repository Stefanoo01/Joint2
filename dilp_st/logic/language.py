"""
Core symbolic data structures for first-order logic with function symbols.

Implements: Term (Constant, Variable, FunctionTerm), Atom, Clause, Language.
All objects are immutable (frozen dataclasses) and hashable for use in sets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Terms
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Constant:
    """A ground constant, e.g. `0`, `a`, `*`."""
    name: str

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class Variable:
    """A logic variable, e.g. `x`, `y`, `z`."""
    name: str

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class FunctionTerm:
    """A compound term built from a function symbol, e.g. `s(0)`, `f(x, y)`.

    ``symbol`` is the function name (e.g. ``"s"``), ``args`` is a tuple of
    sub-terms whose length equals the function arity.
    """
    symbol: str
    args: Tuple[Term, ...]

    @property
    def arity(self) -> int:
        return len(self.args)

    def __repr__(self) -> str:
        args_str = ", ".join(repr(a) for a in self.args)
        return f"{self.symbol}({args_str})"


# Union type for all term kinds.
Term = Constant | Variable | FunctionTerm


# ---------------------------------------------------------------------------
# Atoms & Clauses
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Atom:
    """An atom `p(t1, ..., tn)`. Ground when all terms are ground."""
    predicate: str
    args: Tuple[Term, ...]

    @property
    def arity(self) -> int:
        return len(self.args)

    def is_ground(self) -> bool:
        return all(_term_is_ground(a) for a in self.args)

    def variables(self) -> FrozenSet[Variable]:
        vs: set[Variable] = set()
        for a in self.args:
            _collect_vars(a, vs)
        return frozenset(vs)

    def __repr__(self) -> str:
        if not self.args:
            return self.predicate
        args_str = ", ".join(repr(a) for a in self.args)
        return f"{self.predicate}({args_str})"


# Special sentinel atoms
BOTTOM = Atom("⊥", ())  # always false
TOP = Atom("⊤", ())     # always true


@dataclass(frozen=True, slots=True)
class Clause:
    """A definite clause ``head ← body[0] ∧ ... ∧ body[n-1]``.

    A *fact* is a clause with an empty body.
    """
    head: Atom
    body: Tuple[Atom, ...] = ()

    @property
    def body_length(self) -> int:
        return len(self.body)

    def is_fact(self) -> bool:
        return len(self.body) == 0

    def variables(self) -> FrozenSet[Variable]:
        vs: set[Variable] = set()
        for a in self.head.args:
            _collect_vars(a, vs)
        for b in self.body:
            for a in b.args:
                _collect_vars(a, vs)
        return frozenset(vs)

    def __repr__(self) -> str:
        if self.body:
            body_str = ", ".join(repr(b) for b in self.body)
            return f"{self.head} ← {body_str}"
        return repr(self.head)


# ---------------------------------------------------------------------------
# Language
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Predicate:
    """A predicate symbol with its arity, e.g. `plus/3`."""
    name: str
    arity: int

    def __repr__(self) -> str:
        return f"{self.name}/{self.arity}"


@dataclass(frozen=True)
class FunctionSymbol:
    """A function symbol with its arity, e.g. `s/1`, `f/2`."""
    name: str
    arity: int

    def __repr__(self) -> str:
        return f"{self.name}/{self.arity}"


@dataclass(frozen=True)
class Language:
    """Language L = (P, F, A, V) specifying the ILP search space."""
    predicates: Tuple[Predicate, ...]
    functions: Tuple[FunctionSymbol, ...]
    constants: Tuple[Constant, ...]
    variables: Tuple[Variable, ...]

    def predicate_by_name(self, name: str) -> Optional[Predicate]:
        for p in self.predicates:
            if p.name == name:
                return p
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _term_is_ground(t: Term) -> bool:
    if isinstance(t, Constant):
        return True
    if isinstance(t, Variable):
        return False
    if isinstance(t, FunctionTerm):
        return all(_term_is_ground(a) for a in t.args)
    raise TypeError(f"Unknown term type: {type(t)}")


def _collect_vars(t: Term, acc: set[Variable]) -> None:
    if isinstance(t, Variable):
        acc.add(t)
    elif isinstance(t, FunctionTerm):
        for a in t.args:
            _collect_vars(a, acc)


def term_depth(t: Term) -> int:
    """Depth of a term tree (constants/variables have depth 0)."""
    if isinstance(t, (Constant, Variable)):
        return 0
    if isinstance(t, FunctionTerm):
        if not t.args:
            return 1
        return 1 + max(term_depth(a) for a in t.args)
    raise TypeError(f"Unknown term type: {type(t)}")


def atom_depth(a: Atom) -> int:
    """Maximum depth of any term in the atom."""
    if not a.args:
        return 0
    return max(term_depth(t) for t in a.args)


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def make_successor(n: int, zero: Constant | None = None) -> Term:
    """Build the term ``s^n(0)`` representing natural number *n*."""
    if zero is None:
        zero = Constant("0")
    t: Term = zero
    for _ in range(n):
        t = FunctionTerm("s", (t,))
    return t


def make_list(elements: Sequence[Term], nil: Constant | None = None) -> Term:
    """Build a list term ``f(a, f(b, f(c, *)))`` from elements [a, b, c].

    The sentinel ``*`` (or *nil*) marks the end of the list.
    """
    if nil is None:
        nil = Constant("*")
    t: Term = nil
    for e in reversed(elements):
        t = FunctionTerm("f", (e, t))
    return t
