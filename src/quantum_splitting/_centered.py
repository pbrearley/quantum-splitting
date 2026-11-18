"""Centered order conditions evaluated by polynomial integration."""

from ctypes import CDLL, c_int, c_void_p
from dataclasses import dataclass
from functools import cache

import numpy as np
from numpy.polynomial import Legendre, Polynomial
from numpy.typing import ArrayLike, NDArray

_Word = tuple[int, ...]
_ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class _Plan:
    alphabet: tuple[tuple[int, int], ...]
    words: tuple[_Word, ...]
    degrees: NDArray[np.int32]
    size: int
    arrays: tuple[NDArray, ...]


@cache
def _native_evaluator():
    from . import _iterated

    library = CDLL(_iterated.__file__)
    evaluate = library.iterated_eval
    evaluate.argtypes = [c_int] * 5 + [c_void_p] * 14
    evaluate.restype = None
    return evaluate


@cache
def _plan(order: int, m: int) -> _Plan:
    alphabet = tuple((k, degree - k - 1) for degree in range(2, order) for k in range(1, degree))
    weights = tuple(k + r + 1 for k, r in alphabet)
    targets = []

    def extend(word: _Word, degree: int) -> None:
        if degree % 2 and all(word < word[i:] + word[:i] for i in range(1, len(word))):
            targets.append((degree, word))
        for letter, weight in enumerate(weights):
            if degree + weight >= order:
                break
            extend(word + (letter,), degree + weight)

    extend((), 0)
    targets.sort()
    words = tuple(word for _, word in targets)
    factors = {()}
    factors.update(
        word[i:j] for word in words for i in range(len(word)) for j in range(i + 1, len(word) + 1)
    )
    factors = sorted(factors, key=lambda word: (len(word), word))
    positions = {word: index for index, word in enumerate(factors)}

    split_counts, splits = [0], []
    for word in factors:
        splits.extend((positions[word[:i]], positions[word[i:]]) for i in range(len(word) + 1))
        split_counts.append(len(splits))

    derivative_starts, derivative_splits = [0], []
    for word in words:
        derivative_splits.extend(
            (positions[word[:i]], positions[word[i:j]], positions[word[j:]])
            for i in range(len(word))
            for j in range(i + 1, len(word) + 1)
        )
        derivative_starts.append(len(derivative_splits))

    # On interval j, t = (j+u)/m. Store each Legendre factor P_r(2t-1)
    # as a polynomial in the local coordinate u.
    time_polys = np.zeros((m, len(alphabet), max(10, order)))
    legendre = tuple(Legendre.basis(r).convert(kind=Polynomial) for _, r in alphabet)
    for j in range(m):
        t = Polynomial([2 * j / m - 1, 2 / m])
        for i, factor in enumerate(legendre):
            local = factor(t).coef
            time_polys[j, i, : len(local)] = local

    def indices(values):
        return np.ascontiguousarray(values, dtype=np.int32)

    arrays = (
        indices([k for k, _ in alphabet]),
        indices([r for _, r in alphabet]),
        time_polys,
        indices([sum(weights[j] for j in word) for word in factors]),
        indices([positions[word[:-1]] for word in factors]),
        indices([word[-1] if word else -1 for word in factors]),
        indices(split_counts),
        indices(splits),
        indices([positions[word] for word in words]),
        indices(derivative_starts),
        indices(derivative_splits),
    )
    degrees = indices([degree for degree, _ in targets])
    return _Plan(alphabet, words, degrees, len(factors), arrays)


class CenteredConditions:
    """Odd weighted Lyndon conditions for the equal-A, palindromic family.

    Let ``u`` be the local coordinate in A interval ``j`` and let ``q[j]``
    be ``m * sum(b[:j+1]) - j - 1/2``. A generator ``(k, r)`` has weight
    ``k+r+1`` and integrand ``(q[j] + 1/2 - u)**k * P_r(2*(j+u)/m - 1)``
    with respect to A time. The evaluator integrates polynomials coefficient
    by coefficient and concatenates the resulting interval integrals.
    """

    def __init__(self, order: int, m: int) -> None:
        if order < 2 or order % 2:
            raise ValueError("order must be a positive even integer of at least 2")
        if m < 1:
            raise ValueError("m must be positive")
        self.order = order
        self.m = m
        self._plan = _plan(order, m)
        self.words = self._plan.words
        self.alphabet = self._plan.alphabet
        self.degrees = self._plan.degrees
        self._array_pointers = tuple(arr.ctypes.data for arr in self._plan.arrays)

    def evaluate(
        self, q: ArrayLike, derivatives: bool = False
    ) -> _ComplexArray | tuple[_ComplexArray, _ComplexArray]:
        """Return the residual and optionally its complex Jacobian with respect to q."""
        q = np.ascontiguousarray(q, dtype=np.complex128)
        if q.shape != (self.m,):
            raise ValueError(f"expected {self.m} cumulative offsets")
        result = np.empty(len(self.words), dtype=np.complex128)
        jacobian = np.empty((len(self.words), self.m), dtype=np.complex128) if derivatives else None
        _native_evaluator()(
            self.m,
            len(self.alphabet),
            self._plan.size,
            len(self.words),
            self.order - 1,
            *self._array_pointers,
            q.ctypes.data,
            result.ctypes.data,
            jacobian.ctypes.data if derivatives else None,
        )
        return (result, jacobian) if derivatives else result
