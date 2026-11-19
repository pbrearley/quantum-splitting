"""Double-precision word residuals and their analytic derivatives."""

from dataclasses import dataclass
from functools import cache
from itertools import product
from math import factorial

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._words import append_terms, derivative_splits, lyndon_words, word_index


@dataclass(frozen=True)
class _AppendPlan:
    destinations: NDArray[np.intp]
    starts: NDArray[np.intp]
    sources: NDArray[np.intp]
    powers: NDArray[np.intp]

    def apply(self, series: NDArray[np.complex128], exp_powers: NDArray[np.complex128]) -> None:
        """Append one exponential using a segmented sum over trailing runs."""
        terms = series[self.sources] * exp_powers[self.powers]
        series[self.destinations] += np.add.reduceat(terms, self.starts)


@dataclass(frozen=True)
class _ConditionPlan:
    degree: int
    a_append: _AppendPlan
    b_append: _AppendPlan
    indices: NDArray[np.intp]
    targets: NDArray[np.float64]
    derivative_starts: NDArray[np.intp]
    left_indices: NDArray[np.intp]
    right_indices: NDArray[np.intp]


def _make_append_plan(degree: int, letter: str) -> _AppendPlan:
    destinations, starts, sources, powers = [], [], [], []
    for destination, terms in append_terms(degree, letter):
        destinations.append(destination)
        starts.append(len(sources))
        for source, power in terms:
            sources.append(source)
            powers.append(power)
    return _AppendPlan(
        np.asarray(destinations, dtype=np.intp),
        np.asarray(starts, dtype=np.intp),
        np.asarray(sources, dtype=np.intp),
        np.asarray(powers, dtype=np.intp),
    )


def _word_plan(words: tuple[str, ...]) -> _ConditionPlan:
    degree = len(words[-1]) if words else 0
    starts, left, right = [], [], []
    for word in words:
        starts.append(len(left))
        for prefix, reversed_suffix in derivative_splits(word):
            left.append(prefix)
            right.append(reversed_suffix)
    return _ConditionPlan(
        degree,
        _make_append_plan(degree, "A"),
        _make_append_plan(degree, "B"),
        np.asarray([word_index(word) for word in words], dtype=np.intp),
        np.asarray([1.0 / factorial(len(word)) for word in words]),
        np.asarray(starts, dtype=np.intp),
        np.asarray(left, dtype=np.intp),
        np.asarray(right, dtype=np.intp),
    )


@cache
def _condition_plan(order: int) -> _ConditionPlan:
    return _word_plan(lyndon_words(order))


@cache
def _word_error_plan(order: int) -> _ConditionPlan:
    words = (
        "".join(letters)
        for degree in range(2, order + 1)
        for letters in product("AB", repeat=degree)
    )
    return _word_plan(tuple(words))


@cache
def _leading_error_plan(order: int) -> _ConditionPlan:
    degree = order + 1
    words = ("".join(letters) for letters in product("AB", repeat=degree))
    # The two pure words have zero residual identically in these coordinates.
    return _word_plan(tuple(word for word in words if "A" in word and "B" in word))


def _exponential_powers(
    coefficients: NDArray[np.complex128], degree: int
) -> NDArray[np.complex128]:
    powers = np.empty((coefficients.size, degree + 1), dtype=np.complex128)
    powers[:, 0] = 1
    for k in range(1, degree + 1):
        powers[:, k] = powers[:, k - 1] * (coefficients / k)
    return powers


class _ProductEvaluator:
    """Selected word coefficients in consistency-preserving coordinates."""

    def __init__(self, m: int, plan: _ConditionPlan) -> None:
        if m < 1:
            raise ValueError("m must be positive")
        self.m = m
        self.reference = 1.0 / (m + 1)
        positions = np.arange(m + 1)
        self._mirrors = np.minimum(positions, positions[::-1])
        self.weights: NDArray[np.float64] = np.bincount(self._mirrors).astype(float)
        self.weights.flags.writeable = False
        self.n_free = self.weights.size - 1

        # The final column of this Householder reflection is parallel to
        # the weights. The other columns are an orthonormal nullspace basis.
        normal = self.weights / np.linalg.norm(self.weights)
        normal[-1] += 1
        normal /= np.linalg.norm(normal)
        self.basis: NDArray[np.float64] = np.eye(self.weights.size)[:, :-1] - 2 * np.outer(
            normal, normal[:-1]
        )
        self.basis.flags.writeable = False
        self._plan = plan
        self._a_powers = _exponential_powers(
            np.asarray([1.0 / m], dtype=np.complex128), self._plan.degree
        )[0]

    def decode(self, y: ArrayLike) -> NDArray[np.complex128]:
        """Expand complex free coordinates into all B coefficients."""
        y = np.asarray(y, dtype=np.complex128)
        if y.shape != (self.n_free,):
            raise ValueError(f"expected {self.n_free} free coordinates")
        independent = self.reference + self.basis @ y
        return independent[self._mirrors]

    def encode(self, b: ArrayLike) -> NDArray[np.complex128]:
        """Project a full B sequence onto the free coordinates."""
        b = np.asarray(b, dtype=np.complex128)
        if b.shape != (self.m + 1,):
            raise ValueError(f"expected {self.m + 1} B coefficients")
        independent = (0.5 * (b + b[::-1]))[: self.weights.size]
        return self.basis.T @ (independent - self.reference)

    def _series(
        self,
        y: ArrayLike,
        before: NDArray[np.complex128] | None = None,
        after: NDArray[np.complex128] | None = None,
    ) -> NDArray[np.complex128]:
        """Build the product, optionally saving prefixes at B factors."""
        b = self.decode(y)
        plan = self._plan
        b_powers = _exponential_powers(b[: self.weights.size], plan.degree)
        series = np.zeros(1 << (plan.degree + 1), dtype=np.complex128)
        series[1] = 1
        for position, independent in enumerate(self._mirrors):
            if position:
                plan.a_append.apply(series, self._a_powers)
            if before is not None:
                before[position] = series
            plan.b_append.apply(series, b_powers[independent])
            if after is not None:
                after[position] = series
        return series

    def residual(self, y: ArrayLike) -> NDArray[np.complex128]:
        """Evaluate the complex differences from the exact word coefficients."""
        return self._series(y)[self._plan.indices] - self._plan.targets

    def residual_jacobian(
        self, y: ArrayLike
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        """Return the residual and the complex Jacobian in free coordinates."""
        plan = self._plan
        shape = (self.m + 1, 1 << (plan.degree + 1))
        before = np.empty(shape, dtype=np.complex128)
        after = np.empty_like(before)
        series = self._series(y, before, after)

        # The derivative at factor j inserts one B after its exponential.
        # By the palindrome, the suffix is the word reversal of the prefix
        # before the mirrored factor. Each B in a word gives one split.
        terms = after[:, plan.left_indices] * before[::-1, plan.right_indices]
        by_position = np.add.reduceat(terms, plan.derivative_starts, axis=1).T
        count = self.weights.size
        by_independent = by_position[:, :count].copy()
        mirrored = by_position[:, count:][:, ::-1]
        by_independent[:, : mirrored.shape[1]] += mirrored
        return series[plan.indices] - plan.targets, by_independent @ self.basis


class OrderConditions(_ProductEvaluator):
    """Odd Lyndon word order conditions for the equal-A, palindromic family.

    The complex free coordinates are expressed in an orthonormal basis for
    the weighted zero-sum space of the independent B coefficients. The
    uniform offset therefore enforces consistency throughout the solve.

    The logarithm of a palindromic product is odd. If all lower degrees
    match, the first possible discrepancy in the product equals the error
    in its logarithm at that degree and is a Lie polynomial. The Lyndon
    word coefficients determine that Lie polynomial by the unit triangular
    form of the Lyndon bracket basis. This lets us enforce the same order
    conditions directly in the product series, degree by degree.
    """

    def __init__(self, order: int, m: int) -> None:
        if order < 2:
            raise ValueError("order must be at least 2")
        super().__init__(m, _condition_plan(order))


class WordErrors(_ProductEvaluator):
    """Every word error from degree 2 through ``order``, as checked by ``verify``.

    Degree one is omitted, as the free coordinates enforce consistency.
    """

    def __init__(self, order: int, m: int) -> None:
        if order < 2:
            raise ValueError("order must be at least 2")
        super().__init__(m, _word_error_plan(order))


class LeadingError(_ProductEvaluator):
    """Squared word-basis error at degree `order + 1` and its gradient."""

    def __init__(self, order: int, m: int) -> None:
        if order < 2:
            raise ValueError("order must be at least 2")
        super().__init__(m, _leading_error_plan(order))

    def value(self, y: ArrayLike) -> float:
        """Return the squared Euclidean norm of the leading residual."""
        residual = self.residual(y)
        return float(np.vdot(residual, residual).real)

    def value_gradient(self, y: ArrayLike) -> tuple[float, NDArray[np.complex128]]:
        """Return the objective and gradient with `dE = Re(vdot(gradient, dy))`."""
        residual, jacobian = self.residual_jacobian(y)
        gradient = 2 * (jacobian.conj().T @ residual)
        return float(np.vdot(residual, residual).real), gradient
