"""Random-restart search for positive, palindromic splitting coefficients."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import count, islice, repeat
from math import ceil, isfinite, log10

import numpy as np
from numpy.typing import NDArray

from ._bounds import above_real_floor, real_floor
from ._centered_search import SearchConditions
from ._words import count_order_conditions
from .model import Formula
from .precision import RefinementError, _refine, verify
from .report import ProgressReporter

_DEFAULT_MAX_NFEV = 1600
_SCREEN_TOLERANCE = 1e-12


@dataclass(frozen=True)
class SearchResult:
    """Best candidate found within the search budget.

    ``residual`` is the maximum absolute word error through the requested
    order, evaluated with at least 40 working digits. ``attempts`` counts the
    starting points considered, and ``converged`` records whether the error
    meets the requested tolerance.
    """

    formula: Formula
    residual: float
    attempts: int
    converged: bool


@dataclass(frozen=True)
class _SearchOptions:
    order: int
    m: int
    seed: int
    initial: Formula | None
    min_real: Fraction
    max_nfev: int
    tolerance: float

    @property
    def verification_dps(self) -> int:
        return max(40, ceil(-log10(self.tolerance)) + 10)


@lru_cache(maxsize=4)
def _conditions(order: int, m: int) -> SearchConditions:
    return SearchConditions(order, m)


def _start(
    conditions: SearchConditions, options: _SearchOptions, index: int
) -> NDArray[np.complex128] | None:
    floor = float(options.min_real)
    if options.initial is not None and index % 2 == 0:
        # Finite decimal strings can still be outside the complex128 range.
        try:
            coefficients = np.array(
                [
                    complex(float(Fraction(real)), float(Fraction(imag)))
                    for real, imag in options.initial.b
                ],
                dtype=np.complex128,
            )
        except OverflowError:
            pass
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                start = conditions.encode(coefficients)
                if np.all(np.isfinite(start)):
                    if index == 0 and np.min(conditions.decode(start).real) > floor:
                        return start
                    rng = np.random.default_rng(np.random.SeedSequence((options.seed, index)))
                    start += 0.15 * (
                        rng.normal(size=conditions.n_free) + 1j * rng.normal(size=conditions.n_free)
                    )
                    if np.min(conditions.decode(start).real) > floor:
                        return start
    return conditions.start(options.seed, index, floor)


def _formula(options: _SearchOptions, full: NDArray[np.complex128]) -> Formula:
    half = full[: options.m // 2 + 1]
    pairs = tuple((format(value.real, ".17g"), format(value.imag, ".17g")) for value in half)
    return Formula(options.order, options.m, pairs)


def _verified_result(formula: Formula, options: _SearchOptions, attempts: int) -> SearchResult:
    dps = options.verification_dps
    tolerance = str(options.tolerance)
    check = verify(formula, dps=dps, tolerance=tolerance)
    result = SearchResult(formula, float(check.max_residual), attempts, check.passed)
    if result.converged or check.max_residual > _SCREEN_TOLERANCE:
        return result

    # Spend arbitrary-precision iterations only on an admissible nearby seed.
    try:
        refined = _refine(formula, digits=ceil(1 - log10(options.tolerance)), max_iterations=40)
    except RefinementError:
        return result
    if not above_real_floor(refined, options.min_real):
        return result
    refined_check = verify(refined, dps=dps, tolerance=tolerance)
    if refined_check.max_residual < check.max_residual:
        return SearchResult(
            refined, float(refined_check.max_residual), attempts, refined_check.passed
        )
    return result


def _attempt(options: _SearchOptions, index: int) -> SearchResult:
    if options.order > 2:
        conditions = _conditions(options.order, options.m)
        start = _start(conditions, options, index)
        if start is not None:
            free, _, _ = conditions.solve(start, options.max_nfev, float(options.min_real))
            formula = _formula(options, conditions.decode(free))
            if above_real_floor(formula, options.min_real):
                return _verified_result(formula, options, index + 1)

    uniform = (f"1/{options.m + 1}", "0")
    formula = Formula(options.order, options.m, (uniform,) * (options.m // 2 + 1))
    return _verified_result(formula, options, index + 1)


def search(
    order: int,
    m: int,
    *,
    attempts: int = 20,
    seed: int = 0,
    initial: Formula | None = None,
    min_real: float | str = 0.0,
    max_nfev: int = _DEFAULT_MAX_NFEV,
    tolerance: float = 1e-12,
    workers: int = 1,
    on_improvement: Callable[[SearchResult], None] | None = None,
    on_progress: Callable[[int, SearchResult], None] | None = None,
    verbose: bool = True,
    progress_interval: float = 15.0,
) -> SearchResult:
    """Search for a positive symmetric formula with ``m`` A factors.

    ``order`` must be even. The search requires at least as many free complex
    coefficients as order conditions. One attempt is a local search capped at
    ``max_nfev`` residual evaluations. A fixed ``seed`` indexes the same
    starts regardless of ``workers``. Set ``attempts=0`` to search until
    success or interruption.

    The stored real B coefficients must strictly exceed the exact decimal
    value of ``min_real``. If a supplied ``initial`` already meets that bound
    and ``tolerance``, return it directly. Otherwise each attempt screens the
    centered order conditions in double precision. For tolerances below
    ``1e-12``, a promising candidate is refined at arbitrary precision.
    Verification uses all words through ``order``, with higher working
    precision for tighter tolerances. The optional
    ``on_improvement`` callback runs in the calling process for each improved
    candidate, including an admissible initial formula. The optional
    ``on_progress`` callback runs in the calling process after each
    unsuccessful attempt with the number of attempts completed and the best
    candidate so far. Unless ``verbose`` is false, a header and a status line
    at most every ``progress_interval`` seconds are written to stderr. If the
    budget is exhausted, return the best candidate with ``converged=False``.
    """
    if type(order) is not int or order < 2 or order % 2:
        raise ValueError("search order must be an even integer of at least 2")
    if type(m) is not int or m < 1:
        raise ValueError("m must be a positive integer")
    if type(attempts) is not int or attempts < 0:
        raise ValueError("attempts must be a nonnegative integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(max_nfev) is not int or max_nfev < 1:
        raise ValueError("max_nfev must be a positive integer")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    floor = real_floor(min_real, m)
    if not isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite")
    if initial is not None and (initial.order != order or initial.m != m):
        raise ValueError("initial formula must have the requested order and m")

    if not isfinite(progress_interval) or progress_interval < 0:
        raise ValueError("progress_interval must be nonnegative and finite")

    free_coefficients = m // 2  # Palindromic B coefficients after enforcing their sum.
    order_conditions = count_order_conditions(order)
    if free_coefficients < order_conditions:
        raise ValueError(
            f"Order {order} with m={m} has {free_coefficients} free complex coefficients "
            f"for {order_conditions} complex order conditions. Use m >= {2 * order_conditions}."
        )

    options = _SearchOptions(order, m, seed, initial, floor, max_nfev, tolerance)
    reporter = (
        ProgressReporter(
            order,
            m,
            attempts,
            progress_interval,
            free_coefficients=free_coefficients,
            order_conditions=order_conditions,
        )
        if verbose
        else None
    )
    best = None
    if initial is not None and above_real_floor(initial, floor):
        best = _verified_result(initial, options, 0)
        if on_improvement is not None:
            on_improvement(best)
        if best.converged:
            return best

    done = 0

    def consider(results: Iterable[SearchResult]) -> SearchResult | None:
        nonlocal best, done
        for candidate in results:
            done += 1
            if best is None or candidate.residual < best.residual or candidate.converged:
                best = candidate
                if on_improvement is not None:
                    on_improvement(best)
            if candidate.converged:
                return candidate
            if reporter is not None:
                reporter(done, best.residual)
            if on_progress is not None:
                on_progress(done, best)
        return None

    # For m=1, consistency uniquely determines the palindrome.
    if m == 1:
        attempts = 1
    indices = iter(range(attempts)) if attempts else count()
    if workers == 1:
        result = consider(_attempt(options, index) for index in indices)
        if result is not None:
            return result
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            while batch := list(islice(indices, workers)):
                result = consider(pool.map(_attempt, repeat(options), batch))
                if result is not None:
                    return result

    assert best is not None
    return SearchResult(best.formula, best.residual, attempts, best.converged)
